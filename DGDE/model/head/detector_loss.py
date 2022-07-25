from logging import debug
import torch
import math
import torch.distributed as dist
import pdb
import cv2
from torch.nn import functional as F
from utils.comm import get_world_size
import numpy as np
from model.anno_encoder import Anno_Encoder
from model.layers.utils import select_point_of_interest
from model.utils import Uncertainty_Reg_Loss, Laplace_Loss

from model.layers.focal_loss import *
from model.layers.iou_loss import *
from model.head.depth_losses import *
from model.layers.utils import Converter_key2channel

def make_loss_evaluator(cfg):
	loss_evaluator = Loss_Computation(cfg=cfg)
	return loss_evaluator

class Loss_Computation():
	def __init__(self, cfg):
		
		self.anno_encoder = Anno_Encoder(cfg)
		self.key2channel = Converter_key2channel(keys=cfg.MODEL.HEAD.REGRESSION_HEADS, channels=cfg.MODEL.HEAD.REGRESSION_CHANNELS)
		
		self.max_objs = cfg.DATASETS.MAX_OBJECTS
		self.center_sample = cfg.MODEL.HEAD.CENTER_SAMPLE
		self.regress_area = cfg.MODEL.HEAD.REGRESSION_AREA
		self.heatmap_type = cfg.MODEL.HEAD.HEATMAP_TYPE
		self.corner_depth_sp = cfg.MODEL.HEAD.SUPERVISE_CORNER_DEPTH
		self.loss_keys = cfg.MODEL.HEAD.LOSS_NAMES

		self.world_size = get_world_size()
		self.dim_weight = torch.as_tensor(cfg.MODEL.HEAD.DIMENSION_WEIGHT).view(1, 3)
		self.uncertainty_range = cfg.MODEL.HEAD.UNCERTAINTY_RANGE

		# loss functions
		loss_types = cfg.MODEL.HEAD.LOSS_TYPE
		self.cls_loss_fnc = FocalLoss(cfg.MODEL.HEAD.LOSS_PENALTY_ALPHA, cfg.MODEL.HEAD.LOSS_BETA,cfg=cfg) # penalty-reduced focal loss
		self.iou_loss = IOULoss(loss_type=loss_types[2]) # iou loss for 2D detection

		# depth loss
		if loss_types[3] == 'berhu': self.depth_loss = Berhu_Loss()
		elif loss_types[3] == 'inv_sig': self.depth_loss = Inverse_Sigmoid_Loss()
		elif loss_types[3] == 'log': self.depth_loss = Log_L1_Loss()
		elif loss_types[3] == 'L1': self.depth_loss = F.l1_loss
		else: raise ValueError

		# regular regression loss
		self.reg_loss = loss_types[1]
		self.reg_loss_fnc = F.l1_loss if loss_types[1] == 'L1' else F.smooth_l1_loss
		self.keypoint_loss_fnc = F.l1_loss
		self.extra_kpts_2d_loss_fnc = RegWeightedL1Loss()
		self.extra_kpts_3d_loss_fnc = F.l1_loss
		self.kpts_2d_loc_loss_fnc =F.l1_loss
		self.kpts_3d_loc_loss_fnc =F.l1_loss

		# multi-bin loss setting for orientation estimation
		self.multibin = (cfg.INPUT.ORIENTATION == 'multi-bin')
		self.orien_bin_size = cfg.INPUT.ORIENTATION_BIN_SIZE
		self.trunc_offset_loss_type = cfg.MODEL.HEAD.TRUNCATION_OFFSET_LOSS

		self.loss_weights = {}
		for key, weight in zip(cfg.MODEL.HEAD.LOSS_NAMES, cfg.MODEL.HEAD.INIT_LOSS_WEIGHT): self.loss_weights[key] = weight

		# whether to compute corner loss
		self.compute_direct_depth_loss = 'depth_loss' in self.loss_keys
		self.compute_keypoint_depth_loss = 'keypoint_depth_loss' in self.loss_keys
		self.compute_pairs_kpts_depth_loss = 'pairs_kpts_depth_loss' in self.loss_keys
		self.compute_keypoint_depth_loss = 'keypoint_depth_loss' in self.loss_keys
		self.compute_weighted_depth_loss = 'weighted_avg_depth_loss' in self.loss_keys
		self.compute_corner_loss = 'corner_loss' in self.loss_keys
		self.separate_trunc_offset = 'trunc_offset_loss' in self.loss_keys
		
		self.pred_direct_depth = 'depth' in self.key2channel.keys
		self.depth_with_uncertainty = 'depth_uncertainty' in self.key2channel.keys
		self.compute_keypoint_corner = 'corner_offset' in self.key2channel.keys
		self.compute_extra_kpts_corner = 'extra_kpts_2d' in self.key2channel.keys
		self.corner_with_uncertainty = 'corner_uncertainty' in self.key2channel.keys

		self.uncertainty_weight = cfg.MODEL.HEAD.UNCERTAINTY_WEIGHT # 1.0
		self.keypoint_xy_weights = cfg.MODEL.HEAD.KEYPOINT_XY_WEIGHT # [1, 1]
		self.keypoint_norm_factor = cfg.MODEL.HEAD.KEYPOINT_NORM_FACTOR # 1.0
		self.modify_invalid_keypoint_depths = cfg.MODEL.HEAD.MODIFY_INVALID_KEYPOINT_DEPTH

		self.extra_kpts_num =cfg.MODEL.HEAD.EXTRA_KPTS_NUM

		self.fp16=cfg.MODEL.FP16
		self.batch_weight_factor=cfg.MODEL.BATCH_WEIGHT_FACTOR
		self.corner_loss_depth = cfg.MODEL.HEAD.CORNER_LOSS_DEPTH
		self.eps = 1e-5
		self.is_gen=cfg.TEST.GENERATE_GMW
		self.gen_data={
            'kpts_2d':[],
            'kpts_3d':[],
            'pred_rot':[],
            'gt_location':[],
			'pred_location':[],
            'weight_img':[],
			'img_idx':[],
        }

	def prepare_targets(self, targets):
		# clses
		heatmaps = torch.stack([t.get_field("hm") for t in targets])
		cls_ids = torch.stack([t.get_field("cls_ids") for t in targets])
		offset_3D = torch.stack([t.get_field("offset_3D") for t in targets])
		# 2d detection
		target_centers = torch.stack([t.get_field("target_centers") for t in targets])
		bboxes = torch.stack([t.get_field("2d_bboxes") for t in targets])
		# 3d detection
		keypoints = torch.stack([t.get_field("keypoints") for t in targets])
		extra_kpts_2d = torch.stack([t.get_field("extra_kpts_2d") for t in targets])
		extra_kpts_3d = torch.stack([t.get_field("extra_kpts_3d") for t in targets])
		Calib_P = torch.stack([t.get_field("Calib_P") for t in targets])
		find_pcl = torch.stack([t.get_field("find_pcl") for t in targets])

		keypoints_depth_mask = torch.stack([t.get_field("keypoints_depth_mask") for t in targets])
		extra_kpts_depth_mask = torch.stack([t.get_field("extra_kpts_depth_mask") for t in targets])

		dimensions = torch.stack([t.get_field("dimensions") for t in targets])
		locations = torch.stack([t.get_field("locations") for t in targets])
		rotys = torch.stack([t.get_field("rotys") for t in targets])
		alphas = torch.stack([t.get_field("alphas") for t in targets])
		orientations = torch.stack([t.get_field("orientations") for t in targets])
		# utils
		pad_size = torch.stack([t.get_field("pad_size") for t in targets])
		calibs = [t.get_field("calib") for t in targets]
		reg_mask = torch.stack([t.get_field("reg_mask") for t in targets])
		reg_weight = torch.stack([t.get_field("reg_weight") for t in targets])
		ori_imgs = torch.stack([t.get_field("ori_img") for t in targets])
		trunc_mask = torch.stack([t.get_field("trunc_mask") for t in targets])
		img_idx=[t.get_field("img_idx") for t in targets]
		ori_mask=torch.stack([t.get_field("ori_mask") for t in targets])


		return_dict = dict(cls_ids=cls_ids, target_centers=target_centers, bboxes=bboxes, keypoints=keypoints,extra_kpts_2d=extra_kpts_2d,
			extra_kpts_3d=extra_kpts_3d, Calib_P=Calib_P , dimensions=dimensions,locations=locations, rotys=rotys, alphas=alphas, calib=calibs, pad_size=pad_size,
			 reg_mask=reg_mask, reg_weight=reg_weight,offset_3D=offset_3D, ori_imgs=ori_imgs, trunc_mask=trunc_mask, orientations=orientations,
			  keypoints_depth_mask=keypoints_depth_mask,extra_kpts_depth_mask=extra_kpts_depth_mask,find_pcl=find_pcl,img_idx=img_idx,
			  ori_mask=ori_mask,
		)
		return heatmaps, return_dict

	def generate_data(self,targets_variables,pred_extra_kpts_2D_img,pred_extra_kpts_3D_real,reg_mask_gt,pred_rotys_3D,target_locations_3D,pred_locations_3D):
		kps=pred_extra_kpts_2D_img
		K=targets_variables['calib'][0].P[:,:3]
		num_joints=kps.shape[1]
		kpts_2d_norm=kps.clone()
		K_=torch.from_numpy(K).unsqueeze(0).unsqueeze(0).to(kps.device)
		kpts_2d_norm[:,:,0]=(kps[:,:,0]-K_[:,:,0,2])/K_[:,:,0,0] 
		kpts_2d_norm[:,:,1]=(kps[:,:,1]-K_[:,:,1,2])/K_[:,:,1,1]

		##kpts 3d
		kpts_3d=pred_extra_kpts_3D_real
		self.gen_data['kpts_2d'].append(kpts_2d_norm.detach().cpu().numpy().tolist())
		self.gen_data['kpts_3d'].append(kpts_3d.detach().cpu().numpy().tolist())
		reg_sum=reg_mask_gt.sum(-1)
		img_idx_list=[]

		##the img id 
		for i in range(reg_mask_gt.shape[0]):
			num=reg_sum[i].item()
			single_img_idx=targets_variables['img_idx'][i]
			for j in range(num):
				img_idx_list.append(single_img_idx)
		self.gen_data['img_idx'].append(img_idx_list)	
		self.gen_data['pred_rot'].append(pred_rotys_3D.detach().cpu().numpy().tolist())
		self.gen_data['gt_location'].append(target_locations_3D.detach().cpu().numpy().tolist())
		self.gen_data['pred_location'].append(pred_locations_3D.detach().cpu().numpy().tolist())

	
	def compute_pairs_kpts_loss(self,preds,pred_targets,batch_weight):
		#kpts l1 loss
		instance_num = pred_targets['extra_kpts_2d_mask'].shape[0]
		extra_kpts_2d_loss_all = self.loss_weights['extra_kpts_2d_loss'] * self.extra_kpts_2d_loss_fnc(preds['extra_kpts_2d'],
						pred_targets['extra_kpts_2d'],pred_targets['depth_3D']) * pred_targets['extra_kpts_2d_mask']# n*63*2
		extra_kpts_3d_loss_all = self.loss_weights['extra_kpts_3d_loss'] * self.extra_kpts_3d_loss_fnc(preds['extra_kpts_3d'],
						pred_targets['extra_kpts_3d'], reduction='none').sum(dim=2) * pred_targets['extra_kpts_3d_mask']
		extra_kpts_2d_loss = extra_kpts_2d_loss_all.sum() / torch.clamp(pred_targets['extra_kpts_2d_mask'].sum(), min=1) * (instance_num/batch_weight)
		extra_kpts_3d_loss = extra_kpts_3d_loss_all.sum() / torch.clamp(pred_targets['extra_kpts_3d_mask'].sum(), min=1) * (instance_num/batch_weight)

		#location loss	
		if self.compute_pairs_kpts_depth_loss:
			pred_pairs_kpts_depth, pairs_kpts_depth_mask = preds['pairs_kpt_depths_all'], preds['pairs_kpt_depths_mask'].bool()
			find_pcl_mask=pred_targets['find_pcl'].unsqueeze(-1).expand_as(pairs_kpts_depth_mask)
			invalid_pairs_kpts_depth_mask=(~pairs_kpts_depth_mask * find_pcl_mask).bool()
			valid_pairs_kpts_depth_mask= (pairs_kpts_depth_mask * find_pcl_mask).bool()
			target_pairs_kpts_depth = pred_targets['depth_3D'].unsqueeze(-1).expand_as(pairs_kpts_depth_mask)
			valid_pred_pairs_kpts_depth = pred_pairs_kpts_depth[valid_pairs_kpts_depth_mask]
			invalid_pred_pairs_kpts_depth = pred_pairs_kpts_depth[invalid_pairs_kpts_depth_mask].detach()
			# valid and non-valid
			valid_pairs_kpts_depth_loss = self.loss_weights['pairs_kpts_depth_loss'] * self.reg_loss_fnc(valid_pred_pairs_kpts_depth, 
													target_pairs_kpts_depth[valid_pairs_kpts_depth_mask], reduction='none')
			
			invalid_pairs_kpts_depth_loss = self.loss_weights['pairs_kpts_depth_loss'] * self.reg_loss_fnc(invalid_pred_pairs_kpts_depth, 
													target_pairs_kpts_depth[invalid_pairs_kpts_depth_mask], reduction='none')
			log_valid_pairs_kpts_depth_loss = valid_pairs_kpts_depth_loss.detach().mean()

			valid_pairs_kpts_depth_loss = valid_pairs_kpts_depth_loss.sum() / torch.clamp(valid_pairs_kpts_depth_mask.sum(), 1) * (instance_num/batch_weight)
			invalid_pairs_kpts_depth_loss = invalid_pairs_kpts_depth_loss.sum() / torch.clamp((invalid_pairs_kpts_depth_mask).sum(), 1)* (instance_num/batch_weight)

			# the gradients of invalid depths are not back-propagated
			if self.modify_invalid_keypoint_depths:
				pairs_kpts_depth_loss = valid_pairs_kpts_depth_loss + invalid_pairs_kpts_depth_loss
			else:
				pairs_kpts_depth_loss = valid_pairs_kpts_depth_loss


		pairs_kpts_mae=((preds['pairs_kpt_depths_all']-target_pairs_kpts_depth).abs()/target_pairs_kpts_depth)*valid_pairs_kpts_depth_mask
		pairs_all_mae=pairs_kpts_mae.sum()/ torch.clamp(valid_pairs_kpts_depth_mask.sum(), 1)
		return extra_kpts_2d_loss,extra_kpts_3d_loss,pairs_kpts_depth_loss,pairs_kpts_mae,pairs_all_mae,log_valid_pairs_kpts_depth_loss

	def prepare_predictions(self, targets_variables, predictions):
		pred_regression = predictions['reg']
		batch, channel, feat_h, feat_w = pred_regression.shape

		# 1. get the representative points
		targets_bbox_points = targets_variables["target_centers"] # representative points

		reg_mask_gt = targets_variables["reg_mask"]
		flatten_reg_mask_gt = reg_mask_gt.view(-1).bool()

		# the corresponding image_index for each object, used for finding pad_size, calib and so on
		batch_idxs = torch.arange(batch).view(-1, 1).expand_as(reg_mask_gt).reshape(-1)
		batch_idxs = batch_idxs[flatten_reg_mask_gt].to(reg_mask_gt.device) 

		valid_targets_bbox_points = targets_bbox_points.view(-1, 2)[flatten_reg_mask_gt].float()
		# valid_targets_bbox_points = valid_targets_bbox_points.half() if self.fp16 else valid_targets_bbox_points.float()
		# fcos-style targets for 2D
		target_bboxes_2D = targets_variables['bboxes'].view(-1, 4)[flatten_reg_mask_gt]
		target_bboxes_height = target_bboxes_2D[:, 3] - target_bboxes_2D[:, 1]
		target_bboxes_width = target_bboxes_2D[:, 2] - target_bboxes_2D[:, 0]

		#l,t,r,b
		target_regression_2D = torch.cat((valid_targets_bbox_points - target_bboxes_2D[:, :2], target_bboxes_2D[:, 2:] - valid_targets_bbox_points), dim=1)
		mask_regression_2D = (target_bboxes_height > 0) & (target_bboxes_width > 0)
		target_regression_2D = target_regression_2D[mask_regression_2D].float()

		# targets for 3D
		target_clses = targets_variables["cls_ids"].view(-1)[flatten_reg_mask_gt]
		target_depths_3D = targets_variables['locations'][..., -1].view(-1)[flatten_reg_mask_gt]
		target_rotys_3D = targets_variables['rotys'].view(-1)[flatten_reg_mask_gt]
		target_alphas_3D = targets_variables['alphas'].view(-1)[flatten_reg_mask_gt]
		target_offset_3D = targets_variables["offset_3D"].view(-1, 2)[flatten_reg_mask_gt]
		target_dimensions_3D = targets_variables['dimensions'].view(-1, 3)[flatten_reg_mask_gt]
		target_center= targets_variables['target_centers'].view(-1,2)[flatten_reg_mask_gt]
		target_pad_size=targets_variables['pad_size'].unsqueeze(1).expand_as(targets_variables['target_centers']).reshape(-1,2)[flatten_reg_mask_gt]
		target_ori_mask=targets_variables['ori_mask'].reshape(-1)[flatten_reg_mask_gt]

		
		target_orientation_3D = targets_variables['orientations'].view(-1, targets_variables['orientations'].shape[-1])[flatten_reg_mask_gt]
		target_locations_3D = self.anno_encoder.decode_location_flatten(valid_targets_bbox_points, target_offset_3D, target_depths_3D, 
										targets_variables['calib'], targets_variables['pad_size'], batch_idxs)
		
		
		target_corners_3D = self.anno_encoder.encode_box3d(target_rotys_3D, target_dimensions_3D, target_locations_3D).float()
		target_bboxes_3D = torch.cat((target_locations_3D, target_dimensions_3D, target_rotys_3D[:, None]), dim=1)

		target_trunc_mask = targets_variables['trunc_mask'].view(-1)[flatten_reg_mask_gt]
		obj_weights = targets_variables["reg_weight"].view(-1)[flatten_reg_mask_gt]
		target_find_pcl = targets_variables["find_pcl"].view(-1)[flatten_reg_mask_gt]

		# 2. extract corresponding predictions
		pred_regression_pois_3D = select_point_of_interest(batch, targets_bbox_points, pred_regression).view(-1, channel)[flatten_reg_mask_gt]
		
		pred_regression_2D = F.relu(pred_regression_pois_3D[mask_regression_2D, self.key2channel('2d_dim')]).float()
		pred_offset_3D = pred_regression_pois_3D[:, self.key2channel('3d_offset')].float()
		pred_dimensions_offsets_3D = pred_regression_pois_3D[:, self.key2channel('3d_dim')].float()
		pred_orientation_3D = torch.cat((pred_regression_pois_3D[:, self.key2channel('ori_cls')], 
									pred_regression_pois_3D[:, self.key2channel('ori_offset')]), dim=1)
		
		# decode the pred residual dimensions to real dimensions
		pred_dimensions_3D = self.anno_encoder.decode_dimension(target_clses, pred_dimensions_offsets_3D)
		
		# preparing outputs
		targets = { 'reg_2D': target_regression_2D, 'offset_3D': target_offset_3D, 'depth_3D': target_depths_3D, 'orien_3D': target_orientation_3D,
					'dims_3D': target_dimensions_3D, 'corners_3D': target_corners_3D, 'width_2D': target_bboxes_width, 'rotys_3D': target_rotys_3D,
					'cat_3D': target_bboxes_3D, 'trunc_mask_3D': target_trunc_mask, 'height_2D': target_bboxes_height,'location_3D':target_locations_3D,
					'find_pcl': target_find_pcl,'ori_mask':target_ori_mask
				}

		preds = {'reg_2D': pred_regression_2D, 'offset_3D': pred_offset_3D, 'orien_3D': pred_orientation_3D, 'dims_3D': pred_dimensions_3D}
		
		reg_nums = {'reg_2D': mask_regression_2D.sum(), 'reg_3D': flatten_reg_mask_gt.sum(), 'reg_obj': flatten_reg_mask_gt.sum()}
		weights = {'object_weights': obj_weights}
		
		targets['Calib_P'] = targets_variables["Calib_P"].view(flatten_reg_mask_gt.shape[0], 3, 4)[flatten_reg_mask_gt]

		# predict the depth with direct regression
		if self.pred_direct_depth:
			pred_depths_offset_3D = pred_regression_pois_3D[:, self.key2channel('depth')].squeeze(-1)
			pred_direct_depths_3D = self.anno_encoder.decode_depth(pred_depths_offset_3D,targets['Calib_P'])
			preds['depth_3D'] = pred_direct_depths_3D

		# predict the uncertainty of depth regression
		if self.depth_with_uncertainty:
			preds['depth_uncertainty'] = pred_regression_pois_3D[:, self.key2channel('depth_uncertainty')].squeeze(-1)
			
			if self.uncertainty_range is not None:
				preds['depth_uncertainty'] = torch.clamp(preds['depth_uncertainty'], min=self.uncertainty_range[0], max=self.uncertainty_range[1])

		# predict the keypoints
		if self.compute_keypoint_corner:
			# targets for keypoints
			target_corner_keypoints = targets_variables["keypoints"].view(flatten_reg_mask_gt.shape[0], -1, 3)[flatten_reg_mask_gt]
			targets['keypoints'] = target_corner_keypoints[..., :2]
			targets['keypoints_mask'] = target_corner_keypoints[..., -1]
			reg_nums['keypoints'] = targets['keypoints_mask'].sum()

			# mask for whether depth should be computed from certain group of keypoints
			target_corner_depth_mask = targets_variables["keypoints_depth_mask"].view(-1, 3)[flatten_reg_mask_gt]
			targets['keypoints_depth_mask'] = target_corner_depth_mask

			# predictions for keypoints
			pred_keypoints_3D = pred_regression_pois_3D[:, self.key2channel('corner_offset')]
			pred_keypoints_3D = pred_keypoints_3D.view(flatten_reg_mask_gt.sum().clamp_min(1), -1, 2)
			pred_keypoints_depths_3D = self.anno_encoder.decode_depth_from_keypoints_batch(pred_keypoints_3D, pred_dimensions_3D,
														targets_variables['calib'], batch_idxs)

			preds['keypoints'] = pred_keypoints_3D			
			preds['keypoints_depths'] = pred_keypoints_depths_3D
			if self.corner_with_uncertainty:
				preds['corner_offset_uncertainty'] = pred_regression_pois_3D[:, self.key2channel('corner_uncertainty')]

				if self.uncertainty_range is not None:
					preds['corner_offset_uncertainty'] = torch.clamp(preds['corner_offset_uncertainty'], min=self.uncertainty_range[0], max=self.uncertainty_range[1])

				# else:
				# 	print('keypoint depth uncertainty: {:.2f} +/- {:.2f}'.format(
				# 		preds['corner_offset_uncertainty'].mean().item(), preds['corner_offset_uncertainty'].std().item()))

		# predict the extra keypoints
		if self.compute_extra_kpts_corner:
			# targets for extra_kpts
			target_corner_extra_kpts = targets_variables["extra_kpts_2d"].view(flatten_reg_mask_gt.shape[0], -1, 3)[flatten_reg_mask_gt]
			targets['extra_kpts_2d'] = target_corner_extra_kpts[..., :2]
			

			targets['extra_kpts_3d'] = targets_variables["extra_kpts_3d"].view(flatten_reg_mask_gt.shape[0], -1, 3)[flatten_reg_mask_gt]

			find_pcl_mask= targets['find_pcl'].unsqueeze(-1).expand_as(target_corner_extra_kpts[..., 2])

			targets['extra_kpts_2d_mask'] = (target_corner_extra_kpts[..., 2] * find_pcl_mask).bool()
			reg_nums['extra_kpts_2d'] = targets['extra_kpts_2d_mask'].sum()

			targets['extra_kpts_3d_mask'] = find_pcl_mask
			reg_nums['extra_kpts_3d'] = targets['extra_kpts_3d_mask'].sum()
			
			# predictions for extra_kpts
			pred_extra_kpts_3D = pred_regression_pois_3D[:, self.key2channel('extra_kpts_3d')]
			pred_extra_kpts_2D = pred_regression_pois_3D[:, self.key2channel('extra_kpts_2d')]
			
			pred_extra_kpts_2D = pred_extra_kpts_2D.view(flatten_reg_mask_gt.sum(), -1, 2)#torch.Size([6, 48, 2])
			pred_extra_kpts_2D_real=pred_extra_kpts_2D

			pred_extra_kpts_3D = pred_extra_kpts_3D.view(flatten_reg_mask_gt.sum(), -1, 3)#torch.Size([6, 48, 3])
			pred_extra_kpts_3D_real=pred_extra_kpts_3D

			preds['extra_kpts_2d'] = pred_extra_kpts_2D_real		
			preds['extra_kpts_3d'] = pred_extra_kpts_3D_real		
			pred_ten_kpts_2D_img=self.anno_encoder.decode_kpts_2d_img(pred_keypoints_3D,target_center,target_offset_3D,target_pad_size.unsqueeze(1).expand_as(pred_keypoints_3D))
			target_ten_kpts_2D_img=self.anno_encoder.decode_kpts_2d_img(targets['keypoints'],target_center,target_offset_3D,target_pad_size.unsqueeze(1).expand_as(pred_keypoints_3D))
			pred_extra_kpts_2D_img=self.anno_encoder.decode_kpts_2d_img(pred_extra_kpts_2D_real,target_center,target_offset_3D,target_pad_size.unsqueeze(1).expand_as(pred_extra_kpts_2D_real))
			target_extra_kpts_2D_img=self.anno_encoder.decode_kpts_2d_img(targets['extra_kpts_2d'],target_center,target_offset_3D,target_pad_size.unsqueeze(1).expand_as(pred_extra_kpts_2D_real))

			pred_pairs_kpts_2D_img =pred_extra_kpts_2D_img
			target_pairs_kpts_2D_img =target_extra_kpts_2D_img
			
			pred_pairs_kpts_3D_real=pred_extra_kpts_3D_real
			target_pairs_kpts_3D_real=targets['extra_kpts_3d']

			kpts_mask=targets['extra_kpts_2d_mask']
			is_training=True
			preds['pairs_kpt_depths_gt'],preds['pairs_kpt_depths_mask']= self.anno_encoder.decode_pairs_kpts_depth(target_pairs_kpts_2D_img,target_pairs_kpts_3D_real,target_rotys_3D.unsqueeze(-1), targets['Calib_P'],is_training,kpts_mask,targets['depth_3D'])
			preds['pairs_kpt_depths_2d'],_= self.anno_encoder.decode_pairs_kpts_depth(pred_pairs_kpts_2D_img,target_pairs_kpts_3D_real,target_rotys_3D.unsqueeze(-1), targets['Calib_P'],is_training,kpts_mask,targets['depth_3D'])
			preds['pairs_kpt_depths_3d'],_= self.anno_encoder.decode_pairs_kpts_depth(target_pairs_kpts_2D_img,pred_pairs_kpts_3D_real,target_rotys_3D.unsqueeze(-1), targets['Calib_P'],is_training,kpts_mask,targets['depth_3D'])
			preds['pairs_kpt_depths_all'],_=self.anno_encoder.decode_pairs_kpts_depth(pred_pairs_kpts_2D_img,pred_pairs_kpts_3D_real,target_rotys_3D.unsqueeze(-1),targets['Calib_P'],is_training,kpts_mask,targets['depth_3D'])

		# compute the corners of the predicted 3D bounding boxes for the corner loss
		pred_combined_uncertainty = torch.cat((preds['depth_uncertainty'].unsqueeze(-1), preds['corner_offset_uncertainty']), dim=1).exp()
		pred_combined_depths = torch.cat((pred_direct_depths_3D.unsqueeze(-1), preds['keypoints_depths']), dim=1)
		
		if self.corner_loss_depth == 'edges':
			pred_corner_depth_3D = preds['pairs_kpt_depths_all'].mean(1)

		# compute the corners
		pred_locations_3D = self.anno_encoder.decode_location_flatten(valid_targets_bbox_points, pred_offset_3D, pred_corner_depth_3D, 
										targets_variables['calib'], targets_variables['pad_size'], batch_idxs)
		# decode rotys and alphas
		pred_rotys_3D, _ = self.anno_encoder.decode_axes_orientation(pred_orientation_3D, pred_locations_3D)
		# encode corners
		pred_corners_3D = self.anno_encoder.encode_box3d(pred_rotys_3D, pred_dimensions_3D, pred_locations_3D).float()
		# concatenate all predictions
		pred_bboxes_3D = torch.cat((pred_locations_3D, pred_dimensions_3D, pred_rotys_3D[:, None]), dim=1)

		preds.update({'corners_3D': pred_corners_3D, 'rotys_3D': pred_rotys_3D, 'cat_3D': pred_bboxes_3D})
		if self.is_gen:
			self.generate_data(targets_variables,pred_extra_kpts_2D_img,pred_extra_kpts_3D_real,reg_mask_gt,pred_rotys_3D,target_locations_3D,pred_locations_3D)
		return targets, preds, reg_nums, weights

	def __call__(self, predictions, targets):
		targets_heatmap, targets_variables = self.prepare_targets(targets)

		pred_heatmap = predictions['cls']
		pred_targets, preds, reg_nums, weights = self.prepare_predictions(targets_variables, predictions)

		batch_size=pred_heatmap.shape[0]
		batch_weight=batch_size * self.batch_weight_factor
		# heatmap loss
		if self.heatmap_type == 'centernet':
			hm_loss, num_hm_pos = self.cls_loss_fnc(pred_heatmap, targets_heatmap)
			hm_loss = self.loss_weights['hm_loss'] * hm_loss / batch_weight

		else: raise ValueError

		# synthesize normal factors
		num_reg_2D = reg_nums['reg_2D']
		num_reg_3D = reg_nums['reg_3D']
		num_reg_obj = reg_nums['reg_obj']
		
		trunc_mask = pred_targets['trunc_mask_3D'].bool()
		num_trunc = trunc_mask.sum()
		num_nontrunc = num_reg_obj - num_trunc

		# IoU loss for 2D detection
		if num_reg_2D > 0:
			reg_2D_loss, iou_2D = self.iou_loss(preds['reg_2D'].float(), pred_targets['reg_2D'])
			reg_2D_loss = self.loss_weights['bbox_loss'] * reg_2D_loss.sum() / batch_weight
			iou_2D = iou_2D.mean()
		else:
			reg_2D_loss= 0.0*hm_loss
			iou_2D = torch.zeros_like(reg_2D_loss)
		depth_MAE = (preds['depth_3D'] - pred_targets['depth_3D']).abs() / pred_targets['depth_3D']

		if num_reg_3D > 0:
			# direct depth loss
			if self.compute_direct_depth_loss:
				depth_3D_loss = self.loss_weights['depth_loss'] * self.depth_loss(preds['depth_3D'], pred_targets['depth_3D'], reduction='none')
				real_depth_3D_loss = depth_3D_loss.detach().sum() / batch_weight
				
				if self.depth_with_uncertainty:
					depth_3D_loss = depth_3D_loss * torch.exp(- preds['depth_uncertainty']) + \
							preds['depth_uncertainty'] * self.loss_weights['depth_loss']
	
				depth_3D_loss = depth_3D_loss.sum() / batch_weight
				
			# offset_3D loss
			offset_3D_loss = self.reg_loss_fnc(preds['offset_3D'], pred_targets['offset_3D'], reduction='none').sum(dim=1)

			# use different loss functions for inside and outside objects
			if self.separate_trunc_offset:
				#trunc offset loss
				if self.trunc_offset_loss_type == 'L1':
					trunc_offset_loss = offset_3D_loss[trunc_mask]
				
				elif self.trunc_offset_loss_type == 'log':
					trunc_offset_loss = torch.log(1 + offset_3D_loss[trunc_mask])
				trunc_offset_loss = self.loss_weights['trunc_offset_loss'] * trunc_offset_loss.sum() / batch_weight
				#offset loss
				if  (~trunc_mask).int().sum()==0:
					offset_3D_loss = 0.0* offset_3D_loss.sum()
				else:
					offset_3D_loss = self.loss_weights['offset_loss'] * offset_3D_loss[~trunc_mask].sum() / batch_weight
			else:
				offset_3D_loss = self.loss_weights['offset_loss'] * offset_3D_loss.sum() / batch_weight

			# orientation loss
			if self.multibin:
				ori_mask=pred_targets['ori_mask']
				if ori_mask.sum()>0:
					orien_3D_loss = self.loss_weights['orien_loss'] * \
									Real_MultiBin_loss(preds['orien_3D'][ori_mask], pred_targets['orien_3D'][ori_mask], num_bin=self.orien_bin_size)\
									/ batch_weight
				else:
					orien_3D_loss = torch.zeros_like(offset_3D_loss)

			# dimension loss
			dims_3D_loss = self.reg_loss_fnc(preds['dims_3D'], pred_targets['dims_3D'], reduction='none') * self.dim_weight.type_as(preds['dims_3D'])
			dims_3D_loss = self.loss_weights['dims_loss'] * dims_3D_loss.sum() / batch_weight

			with torch.no_grad(): 
				try:
					pred_IoU_3D = get_iou_3d(preds['corners_3D'], pred_targets['corners_3D']).mean()
				except:
					print('error of iou3d')
					print('pred 3D,target 3D',preds['corners_3D'], pred_targets['corners_3D'])
					exit()

			# corner loss
			if self.compute_corner_loss:
				# N x 8 x 3
				corner_3D_loss = self.loss_weights['corner_loss'] * \
						self.reg_loss_fnc(preds['corners_3D'], pred_targets['corners_3D'], reduction='none').sum()/ batch_weight

			if self.compute_keypoint_corner:
				# N x K x 3
				keypoint_loss = self.loss_weights['keypoint_loss'] * self.keypoint_loss_fnc(preds['keypoints'],
								pred_targets['keypoints'], reduction='none').sum(dim=2) * pred_targets['keypoints_mask']
				
				keypoint_loss = keypoint_loss.sum() / batch_weight

				if self.compute_keypoint_depth_loss:
					pred_keypoints_depth, keypoints_depth_mask = preds['keypoints_depths'], pred_targets['keypoints_depth_mask'].bool()
					target_keypoints_depth = pred_targets['depth_3D'].unsqueeze(-1).repeat(1, 3)
					
					valid_pred_keypoints_depth = pred_keypoints_depth[keypoints_depth_mask]
					invalid_pred_keypoints_depth = pred_keypoints_depth[~keypoints_depth_mask].detach()
					# valid and non-valid
					valid_keypoint_depth_loss = self.loss_weights['keypoint_depth_loss'] * self.reg_loss_fnc(valid_pred_keypoints_depth, 
															target_keypoints_depth[keypoints_depth_mask], reduction='none')
					
					invalid_keypoint_depth_loss = self.loss_weights['keypoint_depth_loss'] * self.reg_loss_fnc(invalid_pred_keypoints_depth, 
															target_keypoints_depth[~keypoints_depth_mask], reduction='none')
					
					# for logging
					if keypoints_depth_mask.int().sum()==0:
						log_valid_keypoint_depth_loss = 0.* invalid_keypoint_depth_loss.detach().sum()
					else:
						log_valid_keypoint_depth_loss = valid_keypoint_depth_loss.detach().sum()/batch_weight

					if self.corner_with_uncertainty:
						# center depth, corner 0246 depth, corner 1357 depth
						pred_keypoint_depth_uncertainty = preds['corner_offset_uncertainty']

						valid_uncertainty = pred_keypoint_depth_uncertainty[keypoints_depth_mask]
						invalid_uncertainty = pred_keypoint_depth_uncertainty[~keypoints_depth_mask]
						# print('valid uncert',valid_uncertainty)
						valid_keypoint_depth_loss = valid_keypoint_depth_loss * torch.exp(- valid_uncertainty) + \
												self.loss_weights['keypoint_depth_loss'] * valid_uncertainty

						invalid_keypoint_depth_loss = invalid_keypoint_depth_loss * torch.exp(- invalid_uncertainty)

					# average
					valid_keypoint_depth_loss = valid_keypoint_depth_loss.sum() / batch_weight
					invalid_keypoint_depth_loss = invalid_keypoint_depth_loss.sum() / batch_weight

					# the gradients of invalid depths are not back-propagated, only train the uncertainty
					if self.modify_invalid_keypoint_depths:
						keypoint_depth_loss = valid_keypoint_depth_loss + invalid_keypoint_depth_loss
					else:
						keypoint_depth_loss = valid_keypoint_depth_loss
				
				# compute the average error for each method of depth estimation
				keypoint_MAE = (preds['keypoints_depths'] - pred_targets['depth_3D'].unsqueeze(-1)).abs() \
									/ pred_targets['depth_3D'].unsqueeze(-1)
				
				center_MAE = keypoint_MAE[:, 0].mean()
				keypoint_02_MAE = keypoint_MAE[:, 1].mean()
				keypoint_13_MAE = keypoint_MAE[:, 2].mean()
				
				if self.compute_extra_kpts_corner:
					extra_kpts_2d_loss,extra_kpts_3d_loss,extra_kpts_depth_loss,extra_kpts_mae,extra_all_mae,log_valid_extra_kpts_depth_loss\
							=self.compute_pairs_kpts_loss(preds,pred_targets,batch_weight)

				if self.corner_with_uncertainty:
					if self.pred_direct_depth and self.depth_with_uncertainty:
						combined_depth = torch.cat((preds['depth_3D'].unsqueeze(1), preds['keypoints_depths']), dim=1)
						combined_uncertainty = torch.cat((preds['depth_uncertainty'].unsqueeze(1), preds['corner_offset_uncertainty']), dim=1).exp()
						combined_MAE = torch.cat((depth_MAE.unsqueeze(1), keypoint_MAE), dim=1)

					# the oracle MAE
					lower_MAE = torch.min(combined_MAE, dim=1)[0]
					# the hard ensemble
					hard_MAE = combined_MAE[torch.arange(combined_MAE.shape[0]), combined_uncertainty.argmin(dim=1)]
					# the soft ensemble
					combined_weights = 1 / combined_uncertainty
					combined_weights = combined_weights / combined_weights.sum(dim=1, keepdim=True)
					soft_depths = torch.sum(combined_depth * combined_weights, dim=1)
					soft_MAE = (soft_depths - pred_targets['depth_3D']).abs() / pred_targets['depth_3D']
					# the average ensemble
					mean_depths = combined_depth.mean(dim=1)
					mean_MAE = (mean_depths - pred_targets['depth_3D']).abs() / pred_targets['depth_3D']
					# average
					lower_MAE, hard_MAE, soft_MAE, mean_MAE = lower_MAE.mean(), hard_MAE.mean(), soft_MAE.mean(), mean_MAE.mean()
				
			depth_MAE = depth_MAE.mean()

		loss_dict = {
			'hm_loss':  hm_loss,
			'bbox_loss': reg_2D_loss,
			'dims_loss': dims_3D_loss,
			'orien_loss': orien_3D_loss,
		}

		log_loss_dict = {
			'2D_IoU': iou_2D.item(),
			'3D_IoU': pred_IoU_3D.item(),
		}

		MAE_dict = {}

		if self.separate_trunc_offset:
			loss_dict['offset_loss'] = offset_3D_loss
			loss_dict['trunc_offset_loss'] = trunc_offset_loss
		else:
			loss_dict['offset_loss'] = offset_3D_loss

		if self.compute_corner_loss:
			loss_dict['corner_loss'] = corner_3D_loss

		if self.pred_direct_depth:
			loss_dict['depth_loss'] = depth_3D_loss
			log_loss_dict['depth_loss'] = real_depth_3D_loss.item()

		if self.compute_keypoint_corner:
			loss_dict['keypoint_loss'] = keypoint_loss


		if self.compute_extra_kpts_corner:
			loss_dict['extra_kpts_2d_loss'] = extra_kpts_2d_loss
			loss_dict['extra_kpts_3d_loss'] = extra_kpts_3d_loss 
			loss_dict['extra_kpts_depth_loss'] = extra_kpts_depth_loss
			log_loss_dict['extra_kpts_depth_loss'] = log_valid_extra_kpts_depth_loss.item()
			MAE_dict.update({
				'extra_all_MAE':extra_all_mae.item(),
			})

		if self.compute_keypoint_depth_loss:
			loss_dict['keypoint_depth_loss'] = keypoint_depth_loss
			log_loss_dict['keypoint_depth_loss'] = log_valid_keypoint_depth_loss.item()
			

		# loss_dict ===> log_loss_dict
		for key, value in loss_dict.items():
			if key not in log_loss_dict:
				log_loss_dict[key] = value.item()

		# stop when the loss has NaN or Inf
		for v in loss_dict.values():
			if torch.isnan(v).sum() > 0:
				print(loss_dict)
				pdb.set_trace()
			if torch.isinf(v).sum() > 0:
				print(loss_dict)
				pdb.set_trace()

		log_loss_dict.update(MAE_dict)
		return loss_dict, log_loss_dict

def Real_MultiBin_loss(vector_ori, gt_ori, num_bin=4):
	gt_ori = gt_ori.view(-1, gt_ori.shape[-1]) # bin1 cls, bin1 offset, bin2 cls, bin2 offst

	cls_losses = 0
	reg_losses = 0
	reg_cnt = 0
	for i in range(num_bin):
		# bin cls loss
		cls_ce_loss = F.cross_entropy(vector_ori[:, (i * 2) : (i * 2 + 2)], gt_ori[:, i].long(), reduction='none')
		# regression loss
		valid_mask_i = (gt_ori[:, i] == 1)
		cls_losses += cls_ce_loss.sum()
		if valid_mask_i.sum() > 0:
			s = num_bin * 2 + i * 2
			e = s + 2
			pred_offset = F.normalize(vector_ori[valid_mask_i, s : e])
			reg_loss = F.l1_loss(pred_offset[:, 0], torch.sin(gt_ori[valid_mask_i, num_bin + i]), reduction='none') + \
						F.l1_loss(pred_offset[:, 1], torch.cos(gt_ori[valid_mask_i, num_bin + i]), reduction='none')

			reg_losses += reg_loss.sum()
			reg_cnt += valid_mask_i.sum()

	return cls_losses / num_bin + reg_losses











