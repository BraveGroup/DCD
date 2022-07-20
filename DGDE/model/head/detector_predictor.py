import torch
import pdb
import numpy as np
import torch
import pdb
import numpy as np
from torch import nn
from torch.nn import functional as F

from utils.registry import Registry
from model import registry
from model.layers.utils import sigmoid_hm
from model.make_layers import group_norm, _fill_fc_weights
from model.layers.utils import select_point_of_interest
from model.backbone.DCNv2.dcn_v2 import DCN


@registry.PREDICTOR.register("Base_Predictor")
class _predictor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(_predictor, self).__init__()
        # ("Car", "Cyclist", "Pedestrian")
        classes = cfg.DATASETS.MAX_CLASSES_NUM
        if classes!=len(cfg.DATASETS.DETECT_CLASSES):
            print('ATTENTION, classes!=len(cfg.DATASETS.DETECT_CLASSES)',classes,len(cfg.DATASETS.DETECT_CLASSES))
        # classes=10
        
        self.regression_head_cfg = cfg.MODEL.HEAD.REGRESSION_HEADS
        self.regression_channel_cfg = cfg.MODEL.HEAD.REGRESSION_CHANNELS
        self.output_width = cfg.INPUT.WIDTH_TRAIN // cfg.MODEL.BACKBONE.DOWN_RATIO
        self.output_height = cfg.INPUT.HEIGHT_TRAIN // cfg.MODEL.BACKBONE.DOWN_RATIO
        
        self.head_conv = cfg.MODEL.HEAD.NUM_CHANNEL

        self.active_func=cfg.MODEL.HEAD.ACTIVE_FUNC

        use_norm = cfg.MODEL.HEAD.USE_NORMALIZATION
        if use_norm == 'BN': self.norm_func = nn.BatchNorm2d
        elif use_norm == 'GN': self.norm_func = group_norm
        else: self.norm_func = nn.Identity

        # the inplace-abn is applied to reduce GPU memory and slightly increase the batch-size
        self.bn_momentum = cfg.MODEL.HEAD.BN_MOMENTUM
        self.abn_activision = 'leaky_relu'

        ##deeper head
        self.deeper_head= cfg.MODEL.HEAD.DEEPER_HEAD
        self.stacked_convs= cfg.MODEL.HEAD.STACKED_CONVS
        self.dcn_on_last_conv = cfg.MODEL.HEAD.DCN_ON_LAST_CONV
        self.in_channels = in_channels

        in_channels_head=self.head_conv if self.deeper_head else self.in_channels
        ###########################################
        ###############  Cls Heads ################
        ########################################### 

        if self.deeper_head:
            self._init_cls_head_prev()

        self.class_head = nn.Sequential(
                nn.Conv2d(in_channels_head, self.head_conv, kernel_size=3, padding=1, bias=False),
                self.norm_func(self.head_conv), self._get_active_func(),
                nn.Conv2d(self.head_conv, classes, kernel_size=1, padding=1 // 2, bias=True),
            )
        
        self.class_head[-1].bias.data.fill_(- np.log(1 / cfg.MODEL.HEAD.INIT_P - 1))

        ###########################################
        ############  Regression Heads ############
        ########################################### 
        
        if self.deeper_head:
            self._init_reg_head_prev()

        # init regression heads
        self.reg_features = nn.ModuleList()
        self.reg_heads = nn.ModuleList()

        # init regression heads
        for idx, regress_head_key in enumerate(self.regression_head_cfg):
            feat_layer = nn.Sequential(nn.Conv2d(in_channels_head, self.head_conv, kernel_size=3, padding=1, bias=False),
                                    self.norm_func(self.head_conv), self._get_active_func())
            
            self.reg_features.append(feat_layer)
            # init output head
            head_channels = self.regression_channel_cfg[idx]
            head_list = nn.ModuleList()
            # import pdb;pdb.set_trace()
            for key_index, key in enumerate(regress_head_key):
                key_channel = head_channels[key_index]
                output_head = nn.Conv2d(self.head_conv, key_channel, kernel_size=1, padding=1 // 2, bias=True)

                if key.find('uncertainty') >= 0 and cfg.MODEL.HEAD.UNCERTAINTY_INIT:
                    torch.nn.init.xavier_normal_(output_head.weight, gain=0.01)
                
                # since the edge fusion is applied to the offset branch, we should save the index of this branch
                if key == '3d_offset': self.offset_index = [idx, key_index]

                _fill_fc_weights(output_head, 0)
                head_list.append(output_head)

            self.reg_heads.append(head_list)

        ###########################################
        ##############  Edge Feature ##############
        ###########################################

        # edge feature fusion
        self.enable_edge_fusion = cfg.MODEL.HEAD.ENABLE_EDGE_FUSION
        self.edge_fusion_kernel_size = cfg.MODEL.HEAD.EDGE_FUSION_KERNEL_SIZE
        self.edge_fusion_relu = cfg.MODEL.HEAD.EDGE_FUSION_RELU

        if self.enable_edge_fusion:
            trunc_norm_func = nn.BatchNorm1d if cfg.MODEL.HEAD.EDGE_FUSION_NORM == 'BN' else nn.Identity
            trunc_activision_func = nn.ReLU(inplace=True) if self.edge_fusion_relu else nn.Identity()
            
            self.trunc_heatmap_conv = nn.Sequential(
                nn.Conv1d(self.head_conv, self.head_conv, kernel_size=self.edge_fusion_kernel_size, padding=self.edge_fusion_kernel_size // 2, padding_mode='replicate'),
                trunc_norm_func(self.head_conv), trunc_activision_func, nn.Conv1d(self.head_conv, classes, kernel_size=1),
            )
            
            self.trunc_offset_conv = nn.Sequential(
                nn.Conv1d(self.head_conv, self.head_conv, kernel_size=self.edge_fusion_kernel_size, padding=self.edge_fusion_kernel_size // 2, padding_mode='replicate'),
                trunc_norm_func(self.head_conv), trunc_activision_func, nn.Conv1d(self.head_conv, 2, kernel_size=1),
            )
    def _get_active_func(self):
        if self.active_func == 'relu':
            return nn.ReLU(inplace=True)
        elif self.active_func == 'leaky_relu':
            return nn.LeakyReLU(inplace=True)
        else:
            raise ValueError('No such activate func')

    def _init_cls_head_prev(self):
        """Initialize classification conv layers of the head."""
        self.cls_head_pre= nn.Sequential(
                nn.Conv2d(self.in_channels, self.head_conv, kernel_size=3, padding=1, bias=False),
                self.norm_func(self.head_conv), self._get_active_func(),
                DCN(self.head_conv, self.head_conv, kernel_size=(3,3), stride=1, padding=1, dilation=1, deformable_groups=1),
                self.norm_func(self.head_conv), self._get_active_func(),
            )


    def _init_reg_head_prev(self):
        """Initialize regression conv layers of the head."""
        self.reg_head_pre= nn.Sequential(
                nn.Conv2d(self.in_channels, self.head_conv, kernel_size=3, padding=1, bias=False),
                self.norm_func(self.head_conv), self._get_active_func(),
                DCN(self.head_conv, self.head_conv, kernel_size=(3,3), stride=1, padding=1, dilation=1, deformable_groups=1),
                self.norm_func(self.head_conv), self._get_active_func(),
            )


    def forward(self, features, targets):
        b, c, h, w = features.shape

        # output classification
        features_cls_pre = self.cls_head_pre(features) if self.deeper_head else features
        feature_cls = self.class_head[:-1](features_cls_pre)
        output_cls = self.class_head[-1](feature_cls)

        output_regs = []
        features_reg_pre = self.reg_head_pre(features) if self.deeper_head else features
        # output regression
        for i, reg_feature_head in enumerate(self.reg_features):
            reg_feature = reg_feature_head(features_reg_pre)

            for j, reg_output_head in enumerate(self.reg_heads[i]):
                output_reg = reg_output_head(reg_feature)

                # apply edge feature enhancement
                if self.enable_edge_fusion and i == self.offset_index[0] and j == self.offset_index[1]:
                    edge_indices = torch.stack([t.get_field("edge_indices") for t in targets]) # B x K x 2
                    edge_lens = torch.stack([t.get_field("edge_len") for t in targets]) # B
                    final_output_w = torch.stack([t.get_field("final_output_w") for t in targets]).float().reshape((-1,1,1)) # B
                    final_output_h = torch.stack([t.get_field("final_output_h") for t in targets]).float().reshape((-1,1,1)) # B

                    # normalize
                    grid_edge_indices = edge_indices.view(b, -1, 1, 2).float()
                    grid_edge_indices[..., 0] = grid_edge_indices[..., 0] / (final_output_w - 1) * 2 - 1
                    grid_edge_indices[..., 1] = grid_edge_indices[..., 1] / (final_output_h - 1) * 2 - 1

                    # apply edge fusion for both offset and heatmap
                    feature_for_fusion = torch.cat((feature_cls, reg_feature), dim=1)
                    grid_edge_indices=grid_edge_indices.type_as(feature_for_fusion)
                    edge_features = F.grid_sample(feature_for_fusion, grid_edge_indices, align_corners=True).squeeze(-1)

                    edge_cls_feature = edge_features[:, :self.head_conv, ...]
                    edge_offset_feature = edge_features[:, self.head_conv:, ...]
                    edge_cls_output = self.trunc_heatmap_conv(edge_cls_feature)
                    edge_offset_output = self.trunc_offset_conv(edge_offset_feature)
                    
                    for k in range(b):
                        edge_indice_k = edge_indices[k, :edge_lens[k]]
                        output_cls[k, :, edge_indice_k[:, 1], edge_indice_k[:, 0]] += edge_cls_output[k, :, :edge_lens[k]]
                        output_reg[k, :, edge_indice_k[:, 1], edge_indice_k[:, 0]] += edge_offset_output[k, :, :edge_lens[k]]
                
                output_regs.append(output_reg)

        output_cls = sigmoid_hm(output_cls)
        output_regs = torch.cat(output_regs, dim=1)

        return {'cls': output_cls.float(), 'reg': output_regs.float()}

def make_predictor(cfg, in_channels):
    func = registry.PREDICTOR[cfg.MODEL.HEAD.PREDICTOR]
    return func(cfg, in_channels)