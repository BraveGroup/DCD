import datetime
import logging
import time
import pdb
import os
import numpy as np

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from engine.inference import inference
from utils import comm
from utils.metric_logger import MetricLogger
from utils.comm import get_world_size
from torch.nn.utils import clip_grad_norm_

from torch.cuda.amp import autocast 
from torch.cuda.amp import GradScaler 

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])

        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)

        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}

    return reduced_losses

def do_eval(cfg, model, data_loaders_val, iteration):
    eval_types = ("detection",)
    dataset_name = cfg.DATASETS.TEST[0]

    if cfg.OUTPUT_DIR:
        output_folder = os.path.join(cfg.OUTPUT_DIR, dataset_name, "inference_{}".format(iteration))
        os.makedirs(output_folder, exist_ok=True)

    evaluate_metric, result_str, dis_ious = inference(
        model,
        data_loaders_val,
        dataset_name=dataset_name,
        eval_types=eval_types,
        device=cfg.MODEL.DEVICE,
        output_folder=output_folder,
    )
    comm.synchronize()
    return evaluate_metric, result_str, dis_ious

def do_train(
        cfg,
        distributed,
        model,
        data_loader,
        data_loaders_val,
        optimizer,
        scheduler,
        warmup_scheduler,
        checkpointer,
        device,
        arguments,
):
    logger = logging.getLogger("DGDE.trainer")
    logger.info("Start training")

    meters = MetricLogger(delimiter=" ", )
    max_iter = cfg.SOLVER.MAX_ITERATION
    start_iter = arguments["iteration"]

    # enable warmup
    if cfg.SOLVER.LR_WARMUP:
        assert warmup_scheduler is not None
        warmup_iters = cfg.SOLVER.WARMUP_STEPS
    else:
        warmup_iters = -1

    model.train()
    start_training_time = time.time()
    end = time.time()

    default_depth_method = cfg.MODEL.HEAD.OUTPUT_DEPTH
    grad_norm_clip = cfg.SOLVER.GRAD_NORM_CLIP

    if comm.get_local_rank() == 0:
        writer = SummaryWriter(os.path.join(cfg.OUTPUT_DIR, 'writer/{}/'.format(cfg.START_TIME)))
        best_mAP = 0
        best_result_str = None
        best_iteration = 0
        eval_iteration = 0
        record_metrics = ['Car_bev_', 'Car_3d_']
    
    is_gen=cfg.TEST.GENERATE_GMW
    if is_gen:
        logger.info('Start collecting the data for GMW')
        max_iter=start_iter+len(data_loader.dataset) // cfg.SOLVER.IMS_PER_BATCH

    if cfg.MODEL.FP16:
        scaler=GradScaler(init_scale=32.0,growth_interval=2000)

    for data, iteration in zip(data_loader, range(start_iter, max_iter)):
        data_time = time.time() - end

        images = data["images"].to(device)
        targets = [target.to(device) for target in data["targets"]]

        if is_gen:
            with torch.no_grad():
                loss_dict, log_loss_dict = model(images, targets)
        else:
            loss_dict, log_loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # reduce losses over all GPUs for logging purposes
        log_losses_reduced = sum(loss for key, loss in log_loss_dict.items() if key.find('loss') >= 0)
        meters.update(loss=log_losses_reduced, **log_loss_dict)
        
        optimizer.zero_grad()
        if not is_gen:
            if cfg.MODEL.FP16:
                scaler.scale(losses).backward()
            else:
                losses.backward()
        
        if grad_norm_clip > 0: clip_grad_norm_(model.parameters(), grad_norm_clip)
        
        if cfg.MODEL.FP16:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        if iteration < warmup_iters:
            warmup_scheduler.step(iteration)
        else:
            scheduler.step(iteration)

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        iteration += 1
        arguments["iteration"] = iteration

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if comm.get_rank() == 0:
            depth_errors_dict = {key: meters.meters[key].value for key in meters.meters.keys() if key.find('MAE') >= 0}
            writer.add_scalars('train_metric/depth_errors', depth_errors_dict, iteration)
            writer.add_scalar('stat/lr', optimizer.param_groups[0]["lr"], iteration)  # save learning rate

            for name, meter in meters.meters.items():
                if name.find('MAE') >= 0: continue
                if name in ['time', 'data']: writer.add_scalar("stat/{}".format(name), meter.value, iteration)
                else: writer.add_scalar("train_metric/{}".format(name), meter.value, iteration)

        if iteration % 10 == 0 or iteration == max_iter:
            # print('current scale proc:{}, scale:{}'.format(comm.get_rank(),scaler.get_scale()))
            if cfg.MODEL.FP16 and comm.get_rank() == 0:
                logger.info('current scale proc:{}, scale:{}'.format(comm.get_rank(),scaler.get_scale()))
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.8f} \n",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                )
            )

        if not is_gen:
            if iteration % cfg.SOLVER.SAVE_CHECKPOINT_INTERVAL == 0:
                logger.info('iteration = {}, saving checkpoint ...'.format(iteration))
                if comm.get_rank() == 0:
                    cur_epoch = iteration // arguments["iter_per_epoch"]
                    checkpointer.save("model_checkpoint_{}".format(cur_epoch), **arguments)
                                    
            if iteration == max_iter and comm.get_rank() == 0:
                checkpointer.save("model_final", **arguments)

        
    if is_gen:
        logger.info('Start generate Train data for GMW')
        # generate train data
        loss_func=model.heads.loss_evaluator
        import json
        out_dir='gen_data'
        os.makedirs(out_dir,exist_ok=True)
        json.dump(loss_func.gen_data,open(os.path.join(out_dir,'gen_data_train.json'),'w'),indent=4)
   
        # generate infer data
        logger.info('Start generate Infer data for GMW')          
        result_dict, result_str, dis_ious = do_eval(cfg, model, data_loaders_val, iteration)
        return
    
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    if comm.get_rank() == 0:
        logger.info(
            "Total training time: {} ({:.4f} s / it)".format(total_time_str)
        )
        logger.info(
            'Finish training DCD, please generate data for GMW'
        )
    exit()
