from pickle import NONE
from numpy.matrixlib import matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from .yi2018cvpr.model import Net as FeatureExtractor
from .yi2018cvpr.config import get_config, print_usage
from lib.optimal_transport import RegularisedTransport
from lib.nonlinear_weighted_blind_pnp import NonlinearWeightedBlindPnP
import numpy as np
import json
import torch.utils.checkpoint as cp


def pairwiseL2Dist(x1, x2):
    """ Computes the pairwise L2 distance between batches of feature vector sets

    res[..., i, j] = ||x1[..., i, :] - x2[..., j, :]||
    since 
    ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a^T*b

    Adapted to batch case from:
        jacobrgardner
        https://github.com/pytorch/pytorch/issues/15253#issuecomment-491467128
    """
    x1_norm2 = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm2 = x2.pow(2).sum(dim=-1, keepdim=True)
    res = torch.baddbmm(
        x2_norm2.transpose(-2, -1),
        x1,
        x2.transpose(-2, -1),
        alpha=-2
    ).add_(x1_norm2).clamp_min_(1e-30).sqrt_()
    return res


class Sinkhorn(nn.Module):
    """
    BiStochastic Layer turns the input matrix into a bi-stochastic matrix.
    Parameter: maximum iterations max_iter
               a small number for numerical stability epsilon
    Input: input matrix s
    Output: bi-stochastic matrix s
    """
    def __init__(self, max_iter=20, epsilon=1e-10):
        super(Sinkhorn, self).__init__()
        self.max_iter = max_iter
        self.epsilon = epsilon

    def forward(self, s, nrows=None, ncols=None, exp=False, exp_alpha=20, dummy_row=False, dtype=torch.float32):
        batch_size = s.shape[0]

        if dummy_row:
            dummy_shape = list(s.shape)
            dummy_shape[1] = s.shape[2] - s.shape[1]
            s = torch.cat((s, torch.full(dummy_shape, 0.).to(s.device)), dim=1)
            new_nrows = ncols
            for b in range(batch_size):
                s[b, nrows[b]:new_nrows[b], :ncols[b]] = self.epsilon
            nrows = new_nrows

        row_norm_ones = torch.zeros(batch_size, s.shape[1], s.shape[1], device=s.device)  # size: row x row
        col_norm_ones = torch.zeros(batch_size, s.shape[2], s.shape[2], device=s.device)  # size: col x col
        for b in range(batch_size):
            row_slice = slice(0, nrows[b] if nrows is not None else s.shape[2])
            col_slice = slice(0, ncols[b] if ncols is not None else s.shape[1])
            row_norm_ones[b, row_slice, row_slice] = 1
            col_norm_ones[b, col_slice, col_slice] = 1

        # for Sinkhorn stacked on last dimension
        if len(s.shape) == 4:
            row_norm_ones = row_norm_ones.unsqueeze(-1)
            col_norm_ones = col_norm_ones.unsqueeze(-1)

        s += self.epsilon

        for i in range(self.max_iter):
            if exp:
                s = torch.exp(exp_alpha * s)
            if i % 2 == 1:
                # column norm
                # sum = torch.sum(torch.mul(s.unsqueeze(3), col_norm_ones.unsqueeze(1)), dim=2)
                sum = torch.einsum('bmnk,bknn->bmn',s.unsqueeze(3), col_norm_ones.unsqueeze(1))
            else:
                # row norm
                # sum = torch.sum(torch.mul(row_norm_ones.unsqueeze(3), s.unsqueeze(1)), dim=2)
                sum=torch.einsum('bmmk,bkmn->bmn',row_norm_ones.unsqueeze(3),s.unsqueeze(1))

            tmp = torch.zeros_like(s)
            for b in range(batch_size):
                row_slice = slice(0, nrows[b] if nrows is not None else s.shape[2])
                col_slice = slice(0, ncols[b] if ncols is not None else s.shape[1])
                tmp[b, row_slice, col_slice] = 1 / sum[b, row_slice, col_slice]
            s = s * tmp

        if dummy_row and dummy_shape[1] > 0:
            s = s[:, :-dummy_shape[1]]

        return s

class GMW(nn.Module):
    def __init__(self, args):
        super(GMW, self).__init__()
        ##4d,6d
        self.config_4d, _ = get_config()
        self.config_6d, _ = get_config()
        self.config_4d.in_channel = 4
        self.config_4d.gcn_in_channel = 4
        self.config_6d.in_channel = 6
        self.config_6d.gcn_in_channel = 6
        self.FeatureExtractor4d = FeatureExtractor(self.config_4d)
        self.FeatureExtractor6d = FeatureExtractor(self.config_6d)

        self.compute_dis = pairwiseL2Dist
        self.sinkhorn_lambda = 10.0
        self.sinkhorn_tolerance=1e-9
        self.sinkhorn = RegularisedTransport(self.sinkhorn_lambda, self.sinkhorn_tolerance)

        self.num_kpts=73
        self.up_mask=torch.zeros((self.num_kpts,self.num_kpts))
        for i in range(0,self.num_kpts):
            for j in range(i+1,self.num_kpts):
                self.up_mask[i,j]=1
        self.up_mask=self.up_mask.bool()

    def get_up(self,matrix):
        """
        get the upper part of  matrix [b,n,n], not include the diag
        """
        b,c=matrix.shape[0],matrix.shape[3]
        up_mask=self.up_mask.unsqueeze(0).unsqueeze(-1).expand_as(matrix).to(matrix.device)
        upper=matrix.masked_select(up_mask).reshape(b,-1,c)
        return upper

    def put_down(self,mat_raw,n):
        """
        matrix:[b,(n-1)(n-2)/2]
        """
        if len(mat_raw.shape)==3:
            b,k,c=mat_raw.shape
            matrix=torch.zeros((b,n,n,c)).to(mat_raw.device)
            count=0
            for i in range(1,n-1):
                for j in range(i+1,n):
                    matrix[:,i,j]=mat_raw[:,count]
                    count+=1
            return matrix
        else:
            raise ValueError('len must == 3')

    def edge_expand(self,f5d):
        """
        f5d:b,n,5
        """
        b,n,c=f5d.shape
        f5d=f5d.unsqueeze(-2).expand((b,n,n,c)) #b,n,n,5
        f5d_t=f5d.transpose(1,2)
        f5d=self.get_up(f5d)
        f5d_t=self.get_up(f5d_t)
        f10d=torch.cat((f5d,f5d_t),dim=-1) #b,k,10
        return f10d

    def graph_extract(self,feature):
        feature=feature.diagonal(offset=0,dim1=-2,dim2=-1)
        feature=1./feature
        return feature

    def graph_matching(self,f4d,f6d,FeatureExtractor4d,FeatureExtractor6d):
        # Extract features:
        f4d = FeatureExtractor4d(f4d.transpose(-2,-1)).transpose(-2,-1) # b x m x 128
        f6d = FeatureExtractor6d(f6d.transpose(-2,-1)).transpose(-2,-1) # b x n x 128
        
        # L2 Normalise:
        f4d = torch.nn.functional.normalize(f4d, p=2, dim=-1)
        f6d = torch.nn.functional.normalize(f6d, p=2, dim=-1)

        # Compute pairwise L2 distance matrix:
        M = self.compute_dis(f4d, f6d)
        diag_feat=self.graph_extract(M)
        
        # Sinkhorn:
        # Set replicated points to have a zero prior probability
        b, m, n = M.size()

        r = M.new_ones((b, m)) # bxm
        c = M.new_ones((b, n)) # bxn
        r/=m
        c/=n
        P=self.sinkhorn(M,r,c)
        return P, diag_feat
    

    def forward(self,kpts_2d,kpts_3d,pred_rot,args):
        """
        pred_rot.shape([128, 1])
        kpts_2d.shape([128, 57, 2])
        """
        f2d = kpts_2d
        f3d = kpts_3d    

        f4d = self.edge_expand(f2d)
        f6d = self.edge_expand(f3d)

        edge_P,reg_weights=self.graph_matching(f4d,f6d,self.FeatureExtractor4d,self.FeatureExtractor6d)
        return reg_weights,edge_P
