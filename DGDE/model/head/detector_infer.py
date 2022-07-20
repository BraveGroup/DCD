from logging import raiseExceptions
import torch
import pdb
import math
from model.anno_encoder import Anno_Encoder
from torch import nn
from shapely.geometry import Polygon
from torch.nn import functional as F
import cv2
import numpy as np
from model.layers.utils import (
	nms_hm,
	select_topk,
	select_point_of_interest,
)

from model.layers.utils import Converter_key2channel
from engine.visualize_infer import box_iou, box_iou_3d, box3d_to_corners

def make_post_processor(cfg):
	anno_encoder = Anno_Encoder(cfg)
	key2channel = Converter_key2channel(keys=cfg.MODEL.HEAD.REGRESSION_HEADS, channels=cfg.MODEL.HEAD.REGRESSION_CHANNELS)
	postprocessor = PostProcessor(cfg=cfg, anno_encoder=anno_encoder, key2channel=key2channel)
	
	return postprocessor

class PostProcessor(nn.Module):
	def __init__(self, cfg, anno_encoder, key2channel):
		
		super(PostProcessor, self).__init__()
		self.anno_encoder = anno_encoder
		self.key2channel = key2channel

		self.det_threshold = cfg.TEST.DETECTIONS_THRESHOLD
		self.max_detection = cfg.TEST.DETECTIONS_PER_IMG		
		self.eval_dis_iou = cfg.TEST.EVAL_DIS_IOUS
		self.eval_depth = cfg.TEST.EVAL_DEPTH
		self.extra_kpts_num =cfg.MODEL.HEAD.EXTRA_KPTS_NUM
		
		self.output_depth = cfg.MODEL.HEAD.OUTPUT_DEPTH
		self.pred_2d = cfg.TEST.PRED_2D

		self.pred_direct_depth = 'depth' in self.key2channel.keys
		self.depth_with_uncertainty = 'depth_uncertainty' in self.key2channel.keys
		self.regress_keypoints = 'corner_offset' in self.key2channel.keys
		self.keypoint_depth_with_uncertainty = 'corner_uncertainty' in self.key2channel.keys
		self.use_extra_kpts = ('extra_kpts_2d' in self.key2channel.keys) 
		self.use_only_extra_kpts= cfg.TEST.USE_ONLY_EXTRA_KPTS
		self.uncertainty_as_conf = cfg.TEST.UNCERTAINTY_AS_CONFIDENCE
		self.generate_data = cfg.TEST.GENERATE_GMW

		self.use_tta=cfg.DATASETS.USE_TTA
		self.tta_aug_params=cfg.DATASETS.TTA_AUG_PARAMS
		
		self.img_width=cfg.INPUT.WIDTH_TRAIN


	def prepare_targets(self, targets, test):
		pad_size = torch.stack([t.get_field("pad_size") for t in targets])
		calibs = [t.get_field("calib") for t in targets]
		size = torch.stack([torch.tensor(t.size) for t in targets])

		if test: return dict(calib=calibs, size=size, pad_size=pad_size)

		cls_ids = torch.stack([t.get_field("cls_ids") for t in targets])
		# regression locations (in pixels)
		target_centers = torch.stack([t.get_field("target_centers") for t in targets])
		# 3D infos
		dimensions = torch.stack([t.get_field("dimensions") for t in targets])
		rotys = torch.stack([t.get_field("rotys") for t in targets])
		locations = torch.stack([t.get_field("locations") for t in targets])
		# offset_2D = torch.stack([t.get_field("offset_2D") for t in targets])
		offset_3D = torch.stack([t.get_field("offset_3D") for t in targets])
		# reg mask
		extra_kpts_2d= torch.stack([t.get_field("extra_kpts_2d") for t in targets])
		extra_kpts_3d= torch.stack([t.get_field("extra_kpts_3d") for t in targets])
		reg_mask = torch.stack([t.get_field("reg_mask") for t in targets])
		Calib_P = torch.stack([t.get_field("Calib_P") for t in targets])
		target_varibales = dict(pad_size=pad_size, calib=calibs, size=size, cls_ids=cls_ids, target_centers=target_centers,
							dimensions=dimensions, rotys=rotys, locations=locations, offset_3D=offset_3D, reg_mask=reg_mask,
							extra_kpts_2d=extra_kpts_2d,extra_kpts_3d=extra_kpts_3d,Calib_P=Calib_P)

		return target_varibales


	def forward(self, predictions, targets, features=None, test=False, refine_module=None):
		pred_heatmap, pred_regression = predictions['cls'], predictions['reg']
		batch = pred_heatmap.shape[0]

		target_varibales = self.prepare_targets(targets, test=test)
		calib, pad_size = target_varibales['calib'], target_varibales['pad_size']
		img_size = target_varibales['size']

		# evaluate the disentangling IoU for each components in (location, dimension, orientation)
		dis_ious = self.evaluate_3D_detection(target_varibales, pred_regression) if self.eval_dis_iou else None

		# evaluate the accuracy of predicted depths
		depth_errors = self.evaluate_3D_depths(target_varibales, pred_regression) if self.eval_depth else None

		# max-pooling as nms for heat-map
		heatmap = nms_hm(pred_heatmap)
		visualize_preds = {'heat_map': pred_heatmap.clone()}

		# select top-k of the predicted heatmap
		scores, indexs, clses, ys, xs = select_topk(heatmap, K=self.max_detection)
		pred_bbox_points = torch.cat([xs.view(-1, 1), ys.view(-1, 1)], dim=1)
		pred_regression_pois = select_point_of_interest(batch, indexs, pred_regression).view(-1, pred_regression.shape[1])

		# thresholding with score
		scores = scores.view(-1)
		valid_mask = scores >= self.det_threshold

		# no valid predictions
		if valid_mask.sum() == 0:
			result = scores.new_zeros(0, 14)
			visualize_preds['keypoints'] = scores.new_zeros(0, 20)
			visualize_preds['proj_center'] = scores.new_zeros(0, 2)
			eval_utils = {'dis_ious': dis_ious, 'depth_errors': depth_errors, 'vis_scores': scores.new_zeros(0),
					'uncertainty_conf': scores.new_zeros(0), 'estimated_depth_error': scores.new_zeros(0)}
			return result, eval_utils, visualize_preds

		scores = scores[valid_mask]
		clses = clses.view(-1)[valid_mask]
		pred_bbox_points = pred_bbox_points[valid_mask]
		pred_regression_pois = pred_regression_pois[valid_mask]

		pred_2d_reg = F.relu(pred_regression_pois[:, self.key2channel('2d_dim')])
		pred_offset_3D = pred_regression_pois[:, self.key2channel('3d_offset')]
		pred_dimensions_offsets = pred_regression_pois[:, self.key2channel('3d_dim')]
		pred_orientation = torch.cat((pred_regression_pois[:, self.key2channel('ori_cls')], pred_regression_pois[:, self.key2channel('ori_offset')]), dim=1)
		visualize_preds['proj_center'] = pred_bbox_points + pred_offset_3D
		pred_box2d = self.anno_encoder.decode_box2d_fcos(pred_bbox_points, pred_2d_reg, pad_size, img_size)
		pred_box2d_medium = self.anno_encoder.decode_box2d_fcos(pred_bbox_points, pred_2d_reg)
		pred_dimensions = self.anno_encoder.decode_dimension(clses, pred_dimensions_offsets)
		# calib_P=target_varibales['Calib_P'][:,0,:,:]
		calib_P=torch.cat([torch.from_numpy(c.P).to(device=pred_dimensions.device).reshape(1,3,4) for c in calib]).float()

		if self.pred_direct_depth:
			pred_depths_offset = pred_regression_pois[:, self.key2channel('depth')].squeeze(-1)
			pred_direct_depths = self.anno_encoder.decode_depth(pred_depths_offset,calib_P)

		if self.depth_with_uncertainty:
			pred_direct_uncertainty = pred_regression_pois[:, self.key2channel('depth_uncertainty')].exp()
			visualize_preds['depth_uncertainty'] = pred_regression[:, self.key2channel('depth_uncertainty'), ...].squeeze(1)

		if self.regress_keypoints:
			pred_keypoint_offset = pred_regression_pois[:, self.key2channel('corner_offset')]
			pred_keypoint_offset = pred_keypoint_offset.view(-1, 10, 2)
			# solve depth from estimated key-points
			pred_keypoints_depths = self.anno_encoder.decode_depth_from_keypoints_batch(pred_keypoint_offset, pred_dimensions, calib)
			visualize_preds['keypoints'] = pred_keypoint_offset

		if self.keypoint_depth_with_uncertainty:
			pred_keypoint_uncertainty = pred_regression_pois[:, self.key2channel('corner_uncertainty')].exp()

		estimated_depth_error = None
		
		if self.pred_direct_depth and self.depth_with_uncertainty:
			pred_combined_depths = torch.cat((pred_direct_depths.unsqueeze(1), pred_keypoints_depths), dim=1)
			pred_combined_uncertainty = torch.cat((pred_direct_uncertainty, pred_keypoint_uncertainty), dim=1)
		else:
			pred_combined_depths = pred_keypoints_depths.clone()
			pred_combined_uncertainty = pred_keypoint_uncertainty.clone()
		
		depth_weights = 1 / pred_combined_uncertainty
		visualize_preds['min_uncertainty'] = depth_weights.argmax(dim=1)

		depth_weights = depth_weights / depth_weights.sum(dim=1, keepdim=True)
		pred_depths = torch.sum(pred_combined_depths * depth_weights, dim=1)
		
		estimated_depth_error = torch.sum(depth_weights * pred_combined_uncertainty, dim=1)

		batch_idxs = pred_depths.new_zeros(pred_depths.shape[0]).long()
		coarse_loc = self.anno_encoder.decode_location_flatten(pred_bbox_points, pred_offset_3D, pred_depths, calib, pad_size, batch_idxs)
		pred_rotys, pred_alphas = self.anno_encoder.decode_axes_orientation(pred_orientation, coarse_loc)

		clses = clses.view(-1, 1)
		pred_alphas = pred_alphas.view(-1, 1)
		pred_rotys = pred_rotys.view(-1, 1)
		scores = scores.view(-1, 1)
		
		#edge depths
		pred_depths=self.compute_pairs_kpts_depth(target_varibales,pred_regression_pois,\
			pred_bbox_points,pred_offset_3D,pred_rotys,visualize_preds)

		batch_idxs = pred_depths.new_zeros(pred_depths.shape[0]).long()
		pred_locations = self.anno_encoder.decode_location_flatten(pred_bbox_points, pred_offset_3D, pred_depths, calib, pad_size, batch_idxs)
		pred_locations[:, 1] += pred_dimensions[:, 1] / 2
		
		if self.generate_data:
			self.generate_infer_data(target_varibales,pred_regression_pois,pred_bbox_points,pred_offset_3D,\
				pred_keypoint_offset,pred_dimensions,visualize_preds)
		
		# change dimension back to h,w,l
		pred_dimensions = pred_dimensions.roll(shifts=-1, dims=1)

		vis_scores = scores.clone()
		if self.uncertainty_as_conf and estimated_depth_error is not None:
			uncertainty_conf = 1 - torch.clamp(estimated_depth_error, min=0.01, max=1)	
			scores = scores * uncertainty_conf.view(-1, 1)
			if torch.isnan(scores).any():
				mask=torch.isnan(scores)
				scores[mask]=0.0
		else:
			uncertainty_conf, estimated_depth_error = None, None
		
		# kitti output format
		result = torch.cat([clses, pred_alphas, pred_box2d, pred_dimensions, pred_locations, pred_rotys, scores], dim=1)

		eval_utils = {'dis_ious': dis_ious, 'depth_errors': depth_errors, 'uncertainty_conf': uncertainty_conf,
					'estimated_depth_error': estimated_depth_error, 'vis_scores': vis_scores}
		
		return result, eval_utils, visualize_preds
	
	def compute_pairs_kpts_depth(self,targets,pred_regression_pois,pred_bbox_points,pred_offset_3D,pred_rots,vis_pred):
		pred_extra_kpts_2d = pred_regression_pois[:, self.key2channel('extra_kpts_2d')].reshape((-1,self.extra_kpts_num+10,2))
		real_pred_2d=(pred_extra_kpts_2d+(pred_bbox_points+pred_offset_3D).unsqueeze(1).expand_as(pred_extra_kpts_2d))*4-targets["pad_size"]

		pred_extra_kpts_3d = pred_regression_pois[:, self.key2channel('extra_kpts_3d')].reshape((pred_extra_kpts_2d.shape[0],-1,3))

		Calib_P=torch.from_numpy(targets['calib'][0].P).to(pred_extra_kpts_2d.device).unsqueeze(0).expand(pred_extra_kpts_2d.shape[0],-1,-1)
		pairs_depths,_=self.anno_encoder.decode_pairs_kpts_depth(real_pred_2d,pred_extra_kpts_3d,pred_rots,Calib_P)
		vis_pred['pred_extra_kpts_2d']=real_pred_2d
		vis_pred['pred_extra_kpts_3d']=pred_extra_kpts_3d
		return pairs_depths.mean(1)

	def generate_infer_data(self,targets,pred_regression_pois,pred_bbox_points,pred_offset_3D,pred_keypoint_offset,pred_dim,vis_pred):
		pred_extra_kpts_2d = pred_regression_pois[:, self.key2channel('extra_kpts_2d')].reshape((-1,self.extra_kpts_num+10,2))
		real_pred_2d=(pred_extra_kpts_2d+(pred_bbox_points+pred_offset_3D).unsqueeze(1).expand_as(pred_extra_kpts_2d))*4-targets["pad_size"]

		pred_extra_kpts_3d = pred_regression_pois[:, self.key2channel('extra_kpts_3d')].reshape((pred_extra_kpts_2d.shape[0],-1,3))
		pred_ten_kpts_2d=(pred_keypoint_offset+(pred_bbox_points+pred_offset_3D).unsqueeze(1).expand_as(pred_keypoint_offset))*4-targets["pad_size"]
		pred_prj_center=(pred_bbox_points+pred_offset_3D).unsqueeze(1)*4-targets["pad_size"]

		kps=real_pred_2d
		kpts_2d_norm=torch.zeros_like(kps)
		K=targets['calib'][0].P[:,:3]
		kpts_2d_norm[:,:,0]=(kps[:,:,0]-K[0,2])/K[0,0] 
		kpts_2d_norm[:,:,1]=(kps[:,:,1]-K[1,2])/K[1,1]
		## kpts 3d
		kpts_3d=pred_extra_kpts_3d
		vis_pred['gen_pred_extra_kpts_2d']=kpts_2d_norm
		vis_pred['gen_pred_extra_kpts_3d']=kpts_3d
