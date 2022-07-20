import logging
import pdb
import numpy as np
import os
import torch
import matplotlib.pyplot as plt

from tqdm import tqdm
from utils import comm
from utils.timer import Timer, get_time_str
from collections import defaultdict
from data.datasets.evaluation import evaluate_python
from data.datasets.evaluation import generate_kitti_3d_detection,generate_nusc_3d_detection

from .visualize_infer import show_image_with_boxes, show_image_with_boxes_test
import json
import subprocess

def compute_on_dataset(model, data_loader, device, predict_folder, timer=None, vis=False, 
                        eval_score_iou=False, eval_depth=False, eval_trunc_recall=False):
    
    model.eval()
    cpu_device = torch.device("cpu")
    dis_ious = defaultdict(list)
    depth_errors = defaultdict(list)

    differ_ious = []
    infer_data={}
    
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(data_loader)):
            images, targets, image_ids = batch["images"], batch["targets"], batch["img_ids"]
            images = images.to(device)

            # extract label data for visualize
            vis_target = targets[0]
            targets = [target.to(device) for target in targets]

            if timer:
                timer.tic()
            output, eval_utils, visualize_preds = model(images, targets)
            output = output.to(cpu_device)

            if timer:
                torch.cuda.synchronize()
                timer.toc()

            dis_iou = eval_utils['dis_ious']
            gen_data= 'gen_pred_extra_kpts_2d' in visualize_preds.keys()
            if dis_iou is not None:
                for key in dis_iou: dis_ious[key] += dis_iou[key].tolist()
            if vis: show_image_with_boxes(vis_target.get_field('ori_img'), output, vis_target, 
                                    visualize_preds, img_id=image_ids[0],vis_scores=eval_utils['vis_scores'])#image_ids('000001',)

            # generate txt files for predicted objects
            predict_txt = image_ids[0] + '.txt'
            predict_txt = os.path.join(predict_folder, predict_txt)
            generate_kitti_3d_detection(output, predict_txt)
            if gen_data:
                infer_data[image_ids[0]]=[]
                for id in range(output.shape[0]):
                    pred_rot=output[id][12:13].numpy().tolist()
                    box=output[id][2:6].numpy().tolist()
                    dim=output[id][6:9].numpy().tolist()
                    pos=output[id][9:12].numpy().tolist()
                    score=output[id][13:14].numpy().tolist()
                    kpts_2d=visualize_preds['gen_pred_extra_kpts_2d'][id].cpu().numpy().tolist()
                    kpts_3d=visualize_preds['gen_pred_extra_kpts_3d'][id].cpu().numpy().tolist()
                    infer_data[image_ids[0]].append({
                        'kpts_2d':kpts_2d,
                        'kpts_3d':kpts_3d,
                        'pred_rot': pred_rot,
                        'box':box,
                        'dim':dim,
                        'pred_location':pos,
                        'score':score,
                        'cat':'Car',
                    })
               
    if gen_data:
        # generate data
        out_dir='gen_data'
        os.makedirs(out_dir,exist_ok=True)
        json.dump(infer_data,open(os.path.join(out_dir,'gen_data_infer.json'),'w'),indent=4)
        # output = torch.cat([clses, pred_alphas, pred_box2d, pred_dimensions, pred_locations, pred_rotys, scores], dim=1)
                        # 0:1     1:2           2:6         6:9             9:12              12:13     13:14
    # disentangling IoU
    for key, value in dis_ious.items():
        mean_iou = sum(value) / len(value)
        dis_ious[key] = mean_iou

    return dis_ious

def inference(
        model,
        data_loader,
        dataset_name,
        eval_types=("detections",),
        device="cuda",
        output_folder=None,
        metrics=['R40'],
        vis=False,
        eval_score_iou=False,
):
    device = torch.device(device)
    num_devices = comm.get_world_size()
    logger = logging.getLogger("DGDE.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    predict_folder = os.path.join(output_folder, 'data')
    os.makedirs(predict_folder, exist_ok=True)

    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    dis_ious = compute_on_dataset(model, data_loader, device, predict_folder, 
                                inference_timer, vis, eval_score_iou)
    comm.synchronize()

    for key, value in dis_ious.items():
        logger.info("{}, MEAN IOU = {:.4f}".format(key, value))

    if not comm.is_main_process():
        return None, None, None

    return None, None, None