import numpy as np
import pdb
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
import torchvision.ops.roi_align as roi_align
from data.datasets.kitti_utils import convertAlpha2Rot
from torch.cuda.amp import autocast 

PI = np.pi

class Anno_Encoder():
        def __init__(self, cfg):
            device = cfg.MODEL.DEVICE
            self.INF = 100000000
            self.EPS = 1e-3

            # center related
            self.num_cls = len(cfg.DATASETS.DETECT_CLASSES)
            self.min_radius = cfg.DATASETS.MIN_RADIUS
            self.max_radius = cfg.DATASETS.MAX_RADIUS
            self.center_ratio = cfg.DATASETS.CENTER_RADIUS_RATIO
            self.target_center_mode = cfg.INPUT.HEATMAP_CENTER
            # if mode == 'max', centerness is the larger value, if mode == 'area', assigned to the smaller bbox
            self.center_mode = cfg.MODEL.HEAD.CENTER_MODE
            
            # depth related
            self.depth_mode = cfg.MODEL.HEAD.DEPTH_MODE
            self.depth_range = cfg.MODEL.HEAD.DEPTH_RANGE
            self.depth_ref = torch.as_tensor(cfg.MODEL.HEAD.DEPTH_REFERENCE).to(device=device)
            self.scale_depth_by_focal_lengths_factor = cfg.MODEL.HEAD.SCALE_DEPTH_BY_FOCAL_LENGTHS_FACTOR

            # dimension related
            self.dim_mean = torch.as_tensor(cfg.MODEL.HEAD.DIMENSION_MEAN).to(device=device)
            self.dim_std = torch.as_tensor(cfg.MODEL.HEAD.DIMENSION_STD).to(device=device)
            self.dim_modes = cfg.MODEL.HEAD.DIMENSION_REG

            # orientation related
            self.alpha_centers = torch.tensor([0, PI / 2, PI, - PI / 2]).to(device=device)
            self.fp16=cfg.MODEL.FP16
            self.multibin = (cfg.INPUT.ORIENTATION == 'multi-bin')
            self.orien_bin_size = cfg.INPUT.ORIENTATION_BIN_SIZE

            # offset related
            self.offset_mean = cfg.MODEL.HEAD.REGRESSION_OFFSET_STAT[0]
            self.offset_std = cfg.MODEL.HEAD.REGRESSION_OFFSET_STAT[1]

            # output info
            self.down_ratio = cfg.MODEL.BACKBONE.DOWN_RATIO


        @staticmethod
        def rad_to_matrix(rotys, N):
            device = rotys.device

            cos, sin = rotys.cos(), rotys.sin()

            i_temp = torch.tensor([[1, 0, 1],
                                 [0, 1, 0],
                                 [-1, 0, 1]]).to(dtype=torch.float32, device=device)
            # if N==0:
                # return 0.
            ry = i_temp.repeat(N, 1).view(N, -1, 3)

            ry[:, 0, 0] *= cos
            ry[:, 0, 2] *= sin
            ry[:, 2, 0] *= sin
            ry[:, 2, 2] *= cos

            return ry
        

        def decode_box2d_fcos(self, centers, pred_offset, pad_size=None, out_size=None):
            box2d_center = centers.view(-1, 2)
            box2d = box2d_center.new(box2d_center.shape[0], 4).zero_()
            # left, top, right, bottom
            box2d[:, :2] = box2d_center - pred_offset[:, :2]
            box2d[:, 2:] = box2d_center + pred_offset[:, 2:]

            # for inference
            if pad_size is not None:
                N = box2d.shape[0]
                out_size = out_size[0]
                # upscale and subtract the padding
                box2d = box2d * self.down_ratio - pad_size.repeat(1, 2)
                # clamp to the image bound
                box2d[:, 0::2].clamp_(min=0, max=out_size[0].item() - 1)
                box2d[:, 1::2].clamp_(min=0, max=out_size[1].item() - 1)

            return box2d

        def encode_box3d(self, rotys, dims, locs):
            '''
            construct 3d bounding box for each object.
            Args:
                    rotys: rotation in shape N
                    dims: dimensions of objects
                    locs: locations of objects

            Returns:

            '''
            if len(rotys.shape) == 2:
                    rotys = rotys.flatten()
            if len(dims.shape) == 3:
                    dims = dims.view(-1, 3)
            if len(locs.shape) == 3:
                    locs = locs.view(-1, 3)

            device = rotys.device
            N = rotys.shape[0]
            ry = self.rad_to_matrix(rotys, N)

            # l, h, w
            dims_corners = dims.reshape(-1, 1).repeat(1, 8)
            dims_corners = dims_corners * 0.5
            dims_corners[:, 4:] = -dims_corners[:, 4:]
            index = torch.tensor([[4, 5, 0, 1, 6, 7, 2, 3],
                                [0, 1, 2, 3, 4, 5, 6, 7],
                                [4, 0, 1, 5, 6, 2, 3, 7]]).repeat(N, 1).to(device=device)
            
            box_3d_object = torch.gather(dims_corners, 1, index)
            with autocast(enabled=False):
                box_3d = torch.matmul(ry, box_3d_object.view(N, 3, -1))
            box_3d += locs.unsqueeze(-1).repeat(1, 1, 8)

            return box_3d.permute(0, 2, 1)

        def decode_depth(self, depths_offset,calib_P):
            '''
            Transform depth offset to depth
            '''
            if self.depth_mode == 'exp':
                depth = depths_offset.exp()
            elif self.depth_mode == 'linear':
                depth = depths_offset * self.depth_ref[1] + self.depth_ref[0]
            elif self.depth_mode == 'inv_sigmoid':
                depth = torch.ones_like(depths_offset) / torch.sigmoid(depths_offset) - torch.ones_like(depths_offset)
            else:
                raise ValueError

            if self.depth_range is not None:
                depth = torch.clamp(depth, min=self.depth_range[0], max=self.depth_range[1])
            return depth

        def decode_location_flatten(self, points, offsets, depths, calibs, pad_size, batch_idxs):
        
            batch_size = len(calibs)
            gts = torch.unique(batch_idxs, sorted=True).tolist()
            locations = points.new_zeros(points.shape[0], 3).float()
            points = (points + offsets) * self.down_ratio - pad_size[batch_idxs]

            for idx, gt in enumerate(gts):
                corr_pts_idx = torch.nonzero(batch_idxs == gt).squeeze(-1)
                calib = calibs[gt]
                # concatenate uv with depth
                corr_pts_depth = torch.cat((points[corr_pts_idx], depths[corr_pts_idx, None]), dim=1)
                locations[corr_pts_idx] = calib.project_image_to_rect(corr_pts_depth)

            return locations

        def decode_depth_from_keypoints(self, pred_offsets, pred_keypoints, pred_dimensions, calibs, avg_center=False):
            # pred_keypoints: K x 10 x 2, 8 vertices, bottom center and top center
            assert len(calibs) == 1 # for inference, batch size is always 1
            
            calib = calibs[0]
            # we only need the values of y
            pred_height_3D = pred_dimensions[:, 1]
            pred_keypoints = pred_keypoints.view(-1, 10, 2)
            # center height -> depth
            if avg_center:
                updated_pred_keypoints = pred_keypoints - pred_offsets.view(-1, 1, 2)
                center_height = updated_pred_keypoints[:, -2:, 1]
                center_depth = calib.f_u * pred_height_3D.unsqueeze(-1) / (center_height.abs() * self.down_ratio * 2)
                center_depth = center_depth.mean(dim=1)
            else:
                center_height = pred_keypoints[:, -2, 1] - pred_keypoints[:, -1, 1]
                center_depth = calib.f_u * pred_height_3D / (center_height.abs() * self.down_ratio)
            
            # corner height -> depth
            corner_02_height = pred_keypoints[:, [0, 2], 1] - pred_keypoints[:, [4, 6], 1]
            corner_13_height = pred_keypoints[:, [1, 3], 1] - pred_keypoints[:, [5, 7], 1]
            corner_02_depth = calib.f_u * pred_height_3D.unsqueeze(-1) / (corner_02_height * self.down_ratio)
            corner_13_depth = calib.f_u * pred_height_3D.unsqueeze(-1) / (corner_13_height * self.down_ratio)
            corner_02_depth = corner_02_depth.mean(dim=1)
            corner_13_depth = corner_13_depth.mean(dim=1)
            # K x 3
            pred_depths = torch.stack((center_depth, corner_02_depth, corner_13_depth), dim=1)

            return pred_depths

        def decode_depth_from_keypoints_batch(self, pred_keypoints, pred_dimensions, calibs, batch_idxs=None):
            # pred_keypoints: K x 10 x 2, 8 vertices, bottom center and top center
            pred_height_3D = pred_dimensions[:, 1].clone()
            batch_size = len(calibs)
            if batch_size == 1:
                batch_idxs = pred_dimensions.new_zeros(pred_dimensions.shape[0])

            center_height = pred_keypoints[:, -2, 1] - pred_keypoints[:, -1, 1]
            corner_02_height = pred_keypoints[:, [0, 2], 1] - pred_keypoints[:, [4, 6], 1]
            corner_13_height = pred_keypoints[:, [1, 3], 1] - pred_keypoints[:, [5, 7], 1]

            pred_keypoint_depths = {'center': [], 'corner_02': [], 'corner_13': []}

            for idx, gt_idx in enumerate(torch.unique(batch_idxs, sorted=True).tolist()):
                calib = calibs[idx]
                corr_pts_idx = torch.nonzero(batch_idxs == gt_idx).squeeze(-1)
                center_depth = calib.f_u * pred_height_3D[corr_pts_idx] / (F.relu(center_height[corr_pts_idx]) * self.down_ratio + self.EPS)
                corner_02_depth = calib.f_u * pred_height_3D[corr_pts_idx].unsqueeze(-1) / (F.relu(corner_02_height[corr_pts_idx]) * self.down_ratio + self.EPS)
                corner_13_depth = calib.f_u * pred_height_3D[corr_pts_idx].unsqueeze(-1) / (F.relu(corner_13_height[corr_pts_idx]) * self.down_ratio + self.EPS)

                corner_02_depth = corner_02_depth.mean(dim=1)
                corner_13_depth = corner_13_depth.mean(dim=1)

                pred_keypoint_depths['center'].append(center_depth)
                pred_keypoint_depths['corner_02'].append(corner_02_depth)
                pred_keypoint_depths['corner_13'].append(corner_13_depth)

            for key, depths in pred_keypoint_depths.items():
                pred_keypoint_depths[key] = torch.clamp(torch.cat(depths), min=self.depth_range[0], max=self.depth_range[1])

            pred_depths = torch.stack([depth for depth in pred_keypoint_depths.values()], dim=1)
            return pred_depths

        def decode_dimension(self, cls_id, dims_offset):
            '''
            retrieve object dimensions
            Args:
                    cls_id: each object id
                    dims_offset: dimension offsets, shape = (N, 3)

            Returns:

            '''
            if self.dim_modes[0] == 'None':
                return dims_offset


            cls_id = cls_id.flatten().long()
            cls_dimension_mean = self.dim_mean[cls_id, :]

            if self.dim_modes[0] == 'exp':
                dims_offset = dims_offset.exp()

            if self.dim_modes[2]:
                cls_dimension_std = self.dim_std[cls_id, :]
                dimensions = dims_offset * cls_dimension_std + cls_dimension_mean
            else:
                dimensions = dims_offset * cls_dimension_mean
                
            return dimensions

        def decode_axes_orientation(self, vector_ori, locations):
            '''
            retrieve object orientation
            Args:
                    vector_ori: local orientation in [axis_cls, head_cls, sin, cos] format
                    locations: object location

            Returns: for training we only need roty
                             for testing we need both alpha and roty

            '''
            if self.multibin:
                pred_bin_cls = vector_ori[:, : self.orien_bin_size * 2].view(-1, self.orien_bin_size, 2)
                pred_bin_cls = torch.softmax(pred_bin_cls, dim=2)[..., 1]
                orientations = vector_ori.new_zeros(vector_ori.shape[0])
                for i in range(self.orien_bin_size):
                    mask_i = (pred_bin_cls.argmax(dim=1) == i)
                    s = self.orien_bin_size * 2 + i * 2
                    e = s + 2
                    pred_bin_offset = vector_ori[mask_i, s : e]
                    orientations[mask_i] = torch.atan2(pred_bin_offset[:, 0], pred_bin_offset[:, 1]) + self.alpha_centers[i]
            else:
                axis_cls = torch.softmax(vector_ori[:, :2], dim=1)
                axis_cls = axis_cls[:, 0] < axis_cls[:, 1]
                head_cls = torch.softmax(vector_ori[:, 2:4], dim=1)
                head_cls = head_cls[:, 0] < head_cls[:, 1]
                # cls axis
                orientations = self.alpha_centers[axis_cls + head_cls * 2]
                sin_cos_offset = F.normalize(vector_ori[:, 4:])
                orientations += torch.atan(sin_cos_offset[:, 0] / sin_cos_offset[:, 1])

            locations = locations.view(-1, 3)
            rays = torch.atan2(locations[:, 0], locations[:, 2])
            alphas = orientations
            rotys = alphas + rays

            larger_idx = (rotys > PI).nonzero()
            small_idx = (rotys < -PI).nonzero()
            if len(larger_idx) != 0:
                    rotys[larger_idx] -= 2 * PI
            if len(small_idx) != 0:
                    rotys[small_idx] += 2 * PI

            larger_idx = (alphas > PI).nonzero()
            small_idx = (alphas < -PI).nonzero()
            if len(larger_idx) != 0:
                    alphas[larger_idx] -= 2 * PI
            if len(small_idx) != 0:
                    alphas[small_idx] += 2 * PI

            return rotys, alphas
        def decode_kpts_2d(self,kpts_2d,bbox_2d):
            kpts_2d=kpts_2d.clone()
            bbox_2d_width=(bbox_2d[:,2]-bbox_2d[:,0])/2.
            bbox_2d_height=(bbox_2d[:,3]-bbox_2d[:,1])/2.
            kpts_2d[:,:,0]*=bbox_2d_width.unsqueeze(-1).expand(kpts_2d.shape[0],kpts_2d.shape[1])
            kpts_2d[:,:,1]*=bbox_2d_height.unsqueeze(-1).expand(kpts_2d.shape[0],kpts_2d.shape[1])
            return kpts_2d

        def get_up(self,matrix):
            """
            get the upper part of  matrix [b,n,n], not include the diag
            """
            b,n=matrix.shape[0] ,matrix.shape[1]
            upper=torch.zeros((b,int(n*(n-1)/2))).to(matrix.device)
            count=0
            for i in range(0,n):
                for j in range(i+1,n):
                    upper[:,count]=matrix[:,i,j]
                    count+=1
            return upper
        
        def decode_pairs_kpts_depth(self,kps,kps_3d,rot_y,K,training=False,kpts_2d_mask=None,gt_depth=None,weight=None):
            fx,cx,fy,cy=K[:,0,0],K[:,0,2],K[:,1,1],K[:,1,2]
            b1,b2,b3=K[:,0,3],K[:,1,3],K[:,2,3]

            num_joints=kps.shape[1]
            kpts_2d_norm=kps.clone()
            K_=K.unsqueeze(1).expand(-1,num_joints,-1,-1)
            kpts_2d_norm[:,:,0]=(kps[:,:,0]-K_[:,:,0,2])/K_[:,:,0,0] 
            kpts_2d_norm[:,:,1]=(kps[:,:,1]-K_[:,:,1,2])/K_[:,:,1,1]

            B=torch.zeros((kpts_2d_norm.shape[0],kpts_2d_norm.shape[1]*2,1)).to(kpts_2d_norm.device)
            C=torch.zeros((kpts_2d_norm.shape[0],kpts_2d_norm.shape[1]*2,1)).to(kpts_2d_norm.device)

            X=kps_3d[:,:,0:1]
            Y=kps_3d[:,:,1:2]
            Z=kps_3d[:,:,2:3]

            cosori = torch.cos(rot_y).unsqueeze(-1).expand_as(X)
            sinori = torch.sin(rot_y).unsqueeze(-1).expand_as(X)

            B[:,0::2] = X * cosori + Z * sinori
            B[:,1::2] = Y 
            C[:,0::2] = X*sinori - Z*cosori
            C[:,1::2] = X*sinori - Z*cosori
            B1=B.clone()
            B2=kpts_2d_norm.reshape(kpts_2d_norm.shape[0],-1,1) * C
            H_1=B1[:,1::2,:]
            H_2=B2[:,1::2,:]

            V=kpts_2d_norm[:,:,1:2]

            H1_1=H_1.expand(H_1.shape[0],H_1.shape[1],H_1.shape[1])
            H1_2=H_2.expand(H_1.shape[0],H_1.shape[1],H_1.shape[1])
            
            V1=V.expand_as(H1_1)
            
            if kpts_2d_mask is not None:
                kpts_2d_mask=kpts_2d_mask.unsqueeze(-1).expand_as(H1_1)
                depth_mask=kpts_2d_mask*kpts_2d_mask.permute(0,2,1)
                depth_mask=self.get_up(depth_mask)

            H_mat_new=(H1_1-H1_1.permute(0,2,1)) + (H1_2-H1_2.permute(0,2,1))

            V_mat=V1-V1.permute(0,2,1)

            Z_v_raw_new=H_mat_new.abs()/(V_mat).abs().clamp_min(1e-10)


            Z_v_raw=self.get_up(Z_v_raw_new)
            Z_v_raw=Z_v_raw.clamp_min(2.).clamp_max(80)

            if training:
                num_k=1500
                _,good_idx=torch.topk(self.get_up(V_mat).abs(),num_k,dim=-1) 
                depth_all=Z_v_raw.gather(-1,good_idx)
                if kpts_2d_mask is not None:
                    depth_mask=depth_mask.gather(-1,good_idx) 
            else:
                depth_all=Z_v_raw
            depth_all-=b3.unsqueeze(-1)

            if kpts_2d_mask is not None:
                return depth_all,depth_mask 
            else:
                return depth_all,None

        def decode_kpts_2d_img(self,kpts_2d,bbox_points,offset_3D,pad_size):
            return (kpts_2d+(bbox_points+offset_3D).unsqueeze(1).expand_as(kpts_2d))*4-pad_size

            
if __name__ == '__main__':
    pass