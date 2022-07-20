import time
import pdb
import fire

from . import kitti_common as kitti

import csv

def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [line.strip() for line in lines]

def evaluate(label_path,
             result_path,
             label_split_file,
             current_class=0,
             coco=False,
             score_thresh=-1,
             metric='R40'):
    
    from .eval import get_coco_eval_result, get_official_eval_result
    val_image_ids = _read_imageset_file(label_split_file)
    dt_annos = kitti.get_label_annos(result_path,val_image_ids)
    if score_thresh > 0:
        dt_annos = kitti.filter_annos_low_score(dt_annos, score_thresh)
    gt_annos = kitti.get_label_annos(label_path, val_image_ids)
    if coco:
        return get_coco_eval_result(gt_annos, dt_annos, current_class)
    else:
        return get_official_eval_result(gt_annos, dt_annos, current_class, metric=metric)

def generate_kitti_3d_detection(prediction, predict_txt):

    ID_TYPE_CONVERSION = {
        0: 'Car',
        1: 'Pedestrian',
        2: 'Cyclist',
    }

    with open(predict_txt, 'w', newline='') as f:
        w = csv.writer(f, delimiter=' ', lineterminator='\n')
        if len(prediction) == 0:
            w.writerow([])
        else:
            for p in prediction:
                p = p.numpy()
                p = p.round(4)
                type = ID_TYPE_CONVERSION[int(p[0])]
                row = [type, 0, 0] + p[1:].tolist()
                w.writerow(row)

    check_last_line_break(predict_txt)
def get_attr_name(attr_idx, label_name):
    """Get attribute from predicted index.

    This is a workaround to predict attribute when the predicted velocity
    is not reliable. We map the predicted attribute index to the one
    in the attribute set. If it is consistent with the category, we will
    keep it. Otherwise, we will use the default attribute.

    Args:
        attr_idx (int): Attribute index.
        label_name (str): Predicted category name.

    Returns:
        str: Predicted attribute name.
    """
    DefaultAttribute = {
        'car': 'vehicle.parked',
        'pedestrian': 'pedestrian.moving',
        'trailer': 'vehicle.parked',
        'truck': 'vehicle.parked',
        'bus': 'vehicle.moving',
        'motorcycle': 'cycle.without_rider',
        'construction_vehicle': 'vehicle.parked',
        'bicycle': 'cycle.without_rider',
        'barrier': 'none',
        'traffic_cone': 'none',
    }
    AttrMapping_rev2 = [
        'cycle.with_rider', 'cycle.without_rider', 'pedestrian.moving',
        'pedestrian.standing', 'pedestrian.sitting_lying_down',
        'vehicle.moving', 'vehicle.parked', 'vehicle.stopped', 'none'
    ]
    if label_name == 'car' or label_name == 'bus' \
        or label_name == 'truck' or label_name == 'trailer' \
            or label_name == 'construction_vehicle':
        if AttrMapping_rev2[attr_idx] == 'vehicle.moving' or \
            AttrMapping_rev2[attr_idx] == 'vehicle.parked' or \
                AttrMapping_rev2[attr_idx] == 'vehicle.stopped':
            return AttrMapping_rev2[attr_idx]
        else:
            return DefaultAttribute[label_name]
    elif label_name == 'pedestrian':
        if AttrMapping_rev2[attr_idx] == 'pedestrian.moving' or \
            AttrMapping_rev2[attr_idx] == 'pedestrian.standing' or \
                AttrMapping_rev2[attr_idx] == \
                'pedestrian.sitting_lying_down':
            return AttrMapping_rev2[attr_idx]
        else:
            return DefaultAttribute[label_name]
    elif label_name == 'bicycle' or label_name == 'motorcycle':
        if AttrMapping_rev2[attr_idx] == 'cycle.with_rider' or \
                AttrMapping_rev2[attr_idx] == 'cycle.without_rider':
            return AttrMapping_rev2[attr_idx]
        else:
            return DefaultAttribute[label_name]
    else:
        return DefaultAttribute[label_name]

# def decode_attr(cls,attr):
#     ID_TYPE_CONVERSION = [
#         'car' ,
#         'pedestrian',
#         'bicycle',
#         'motorcycle',
#         'barrier',
#         'bus',
#         'construction_vehicle',
#         'traffic_cone',
#         'trailer',
#         'truck',
#         'DontCare',
#     ]

#     attr_list=[]
#     for c,attr_idx in zip(clses,attr_idxes):
#         label_name=ID_TYPE_CONVERSION[c.item()]
#         attr_name=get_attr_name(attr_idx,label_name)
#         attr_list.append(attr_name)
#     return attr_list

def generate_nusc_3d_detection(prediction, predict_txt,attr_and_velo=True):
    ID_TYPE_CONVERSION = [
        'car' ,
        'pedestrian',
        'bicycle',
        'motorcycle',
        'barrier',
        'bus',
        'construction_vehicle',
        'traffic_cone',
        'trailer',
        'truck',
        'DontCare',
    ]

    with open(predict_txt, 'w', newline='') as f:
        w = csv.writer(f, delimiter=' ', lineterminator='\n')
        if len(prediction) == 0:
            w.writerow([])
        else:
            for p in prediction:
                p = p.numpy()
                p = p.round(4)
                type = ID_TYPE_CONVERSION[int(p[0])]
                if attr_and_velo:
                    attr=get_attr_name(int(p[-1]),type)
                    row = [type, 0, 0] + p[1:-1].tolist() + [attr]
                else:
                    row = [type, 0, 0] + p[1:].tolist()

                w.writerow(row)

    check_last_line_break(predict_txt)

def check_last_line_break(predict_txt):
    f = open(predict_txt, 'rb+')
    try:
        f.seek(-1, os.SEEK_END)
    except:
        pass
    else:
        if f.__next__() == b'\n':
            f.seek(-1, os.SEEK_END)
            f.truncate()
    f.close()

if __name__ == '__main__':
    fire.Fire()