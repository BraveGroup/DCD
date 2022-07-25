import math
import random
import pdb
import copy
import numpy as np

from PIL import Image, ImageOps
from data.datasets.kitti_utils import convertRot2Alpha, convertAlpha2Rot, refresh_attributes

class Compose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations
        self.PIL2Numpy = False

    def __call__(self, img, objs, calib):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img, mode="RGB")
            self.PIL2Numpy = True

        for a in self.augmentations:
            img, objs, calib = a(img, objs, calib)

        if self.PIL2Numpy:
            img = np.array(img)

        return img, objs, calib

class RandomHorizontallyFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img, objs, calib):
        if random.random() < self.p:
            # flip image
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img_w, img_h = img.size

            # flip labels
            if objs is not None:
                for idx, obj in enumerate(objs):       
                    # flip box2d
                    # obj.extra_kpts_2D[:,0]=img_w-obj.extra_kpts_2D[:,0]-1
                    # temp = obj.extra_kpts_2D[:29,:].copy()
                    # obj.extra_kpts_2D[:29,:]=obj.extra_kpts_2D[29:58,:]
                    # obj.extra_kpts_2D[29:58,:]=temp

                    w = obj.xmax - obj.xmin
                    obj.xmin = img_w - obj.xmax - 1
                    obj.xmax = obj.xmin + w
                    obj.box2d = np.array([obj.xmin, obj.ymin, obj.xmax, obj.ymax], dtype=np.float32)
                    
                    # flip roty
                    roty = obj.ry
                    roty = (-math.pi - roty) if roty < 0 else (math.pi - roty)
                    while roty > math.pi: roty -= math.pi * 2
                    while roty < (-math.pi): roty += math.pi * 2
                    obj.ry = roty

                    # projection-based 3D center flip
                    # center_loc = obj.t.copy()
                    # center_loc[1] -= obj.h / 2
                    # center2d, depth = calib.project_rect_to_image(center_loc.reshape(1, 3))
                    # center2d[:, 0] = img_w - center2d[:, 0] - 1
                    # center3d = flip_calib.project_image_to_rect(np.concatenate([center2d, depth.reshape(-1, 1)], axis=1))[0]
                    # center3d[1] += obj.h / 2
                    
                    # fliped 3D center
                    loc = obj.t.copy()
                    loc[0] = -loc[0]
                    obj.t = loc
                    # obj.extra_kpts_3D[:,0]*=-1##fucking bug
                    
                    obj.alpha = convertRot2Alpha(roty, obj.t[2], obj.t[0])
                    objs[idx] = obj

                    if hasattr(obj,"velo"):
                        obj.velo[0]*=-1
                    # obj.extra_kpts_2D, _ =calib.project_rect_to_image(obj.generate_extra_kpts_3d_loc())

            # flip calib
            P2 = calib.P.copy()
            P2[0, 2] = img_w - P2[0, 2] - 1
            P2[0, 3] = - P2[0, 3]
            calib.P = P2
            refresh_attributes(calib)

        return img, objs, calib

class RandomResize(object):
    def __init__(self, choice, multi_size):
        self.choice = choice
        self.multi_size=multi_size
        np.random.seed(63)
        self.max_num=100000
        self.choice_list=np.random.choice(len(self.multi_size),self.max_num).astype(np.int32)
        self.count=0

    def __call__(self, img, objs, calib):
        # resize image
        if self.choice==-1:
            choice=self.choice_list[(self.count//2) % self.max_num]
            self.count=self.count+1
        else:
            choice=self.choice
            
        new_size=self.multi_size[int(choice)]
        new_size=[int(new_size[0]),int(new_size[1])]

        ori_size=img.size
        scale_w,scale_h=float(new_size[0])/ori_size[0], float(new_size[1])/ori_size[1]

        img=img.resize(new_size)
        img_w, img_h = img.size

        # resize calib
        resize_matrix=np.eye(3)
        resize_matrix[0]*=scale_w
        resize_matrix[1]*=scale_h

        P2 = calib.P.copy()
        P2 = np.dot(resize_matrix,P2)
        calib.P = P2
        refresh_attributes(calib)

        # resize labels
        for idx, obj in enumerate(objs):
            obj.xmin*=scale_w
            obj.xmax*=scale_w
            obj.ymin*=scale_h
            obj.ymax*=scale_h
            obj.box2d = np.array([obj.xmin, obj.ymin, obj.xmax, obj.ymax], dtype=np.float32)

        return img, objs, calib