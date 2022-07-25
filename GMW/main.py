# BLINDPNP SOLVER WITH DECLARATIVE SINKHORN AND PNP NODES
# Dylan Campbell <dylan.campbell@anu.edu.au>
# Liu Liu <liu.liu@anu.edu.au>
# Stephen Gould <stephen.gould@anu.edu.au>
#
# Modified from PyTorch ImageNet example:
# https://github.com/pytorch/examples/blob/ee964a2eeb41e1712fe719b83645c79bcbd0ba1a/imagenet/main.py

import argparse
import os
import random
import shutil
import string
import time
import warnings
# from engine.visualize_infer import draw_kpts_2d

from numpy.lib.npyio import zipfile_factory
import numpy as np
import cv2
import math
import pickle
import statistics
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.tensorboard as tb
import json
from evaluation import evaluate_python
import matplotlib.pyplot as plt
import pdb
from model.model import GMW
from lib.losses import *
import utilities.geometry_utilities as geo
from utilities.dataset_utilities import Dataset
import multiprocessing
import torch.utils.checkpoint as cp
from tqdm import tqdm
from collections import OrderedDict
import torch.distributed as dist
np.set_printoptions(suppress=True)
# torch.manual_seed(2809)

parser = argparse.ArgumentParser(description='PyTorch GMW Training')
parser.add_argument('--dataset', dest='dataset', default='kitti', type=str,
                    help='dataset name')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N',
                    help='mini-batch size (default: 16)')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--log-dir', dest='log_dir', default='logs', type=str,
                    help='Directory for logging loss and accuracy')
parser.add_argument('--train_data_path', default='../DGDE/gen_data/gen_data_train.json', type=str,
                    help='train data path')
parser.add_argument('--val_data_path', default='../DGDE/gen_data/gen_data_infer.json', type=str,
                    help='val data path')
parser.add_argument('--kitti_path', default='../DGDE/dataset/kitti', type=str)
parser.add_argument('--val_freq', default=5, type=int,
                    help='validate frequency(default: 5)')
parser.add_argument('--reg_loss_start_epoch', default=50, type=int,
                    help='reg_loss_start_epoch(default: 50)')
parser.add_argument('--no_weight_change', action='store_true')                    
parser.add_argument('--cls_weight', default=1.0, type=float)
parser.add_argument('--reg_weight', default=0.0, type=float)
parser.add_argument('--lr_step', default='60,80', type=str)
parser.add_argument('--trainval', action='store_true')
parser.add_argument('--test_all', action='store_true')
parser.add_argument("--local_rank", type=int)

def get_dataset(args):
    if args.gpu is not None:
        train_dataset = Dataset('train', args, args.batch_size, preprocessed=True)
        val_dataset   = Dataset('valid', args, 1, preprocessed=True)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, 
            num_workers=args.workers, drop_last=True,
            collate_fn=None)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, 
            num_workers=args.workers, drop_last=False,
            collate_fn=None)
    else:
        train_dataset = Dataset('train', args, args.batch_size, preprocessed=True)
        val_dataset   = Dataset('valid', args, args.batch_size, preprocessed=True)

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = torch.utils.data.DataLoader(
                    train_dataset, args.batch_size, False,
        num_workers=args.workers, pin_memory=True, drop_last=True,sampler=train_sampler)

        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        val_loader = torch.utils.data.DataLoader(
                    val_dataset, args.batch_size, False,
        num_workers=args.workers, pin_memory=True, drop_last=False,sampler=val_sampler)

    return train_loader, val_loader

class GMW_data():
    def __init__(self,args) -> None:
        self.args=args
        self.thres=0.25
        self.split_file_path=os.path.join(args.kitti_path,'training/ImageSets/val.txt')
        self.data=json.load(open(args.val_data_path,'r'))
        self.result_dir=os.path.join(args.log_dir,'kitti_results_for_eval')
        os.makedirs(self.result_dir,exist_ok=True)
        self.kitti_cats = ['Car',]
        ##init dir
        if args.local_rank==0 or args.gpu is not None:
            if self.args.test_all:
                for i in range(7518):
                    img='{:06d}'.format(i)
                    f=open(os.path.join(self.result_dir,img+'.txt'),'w')
                    f.close()
            else:
                imgs=open(self.split_file_path,'r').readlines()
                for img in imgs:
                    img=img.replace('\n','')
                    f=open(os.path.join(self.result_dir,img+'.txt'),'w')
                    f.close()

    def write_detection_results(self,cls, result_dir, file_number, box,dim,pos,ori,score):
      '''One by one write detection results to KITTI format label files.
      '''
      if result_dir is None: return
      result_dir = result_dir 
      Px = pos[0]
      Py = pos[1]
      Pz = pos[2]
      l = dim[2]
      h = dim[0]
      w = dim[1]

      pi=np.pi
      if ori > 2 * pi:
          while ori > 2 * pi:
              ori -= 2 * pi
      if ori < -2 * pi:
          while ori < -2 * pi:
              ori += 2 * pi

      if ori > pi:
          ori = 2 * pi - ori
      if ori < -pi:
          ori = 2 * pi + pi

      alpha = ori - math.atan2(Px, Pz)
      # convert the object from cam2 to the cam0 frame

      output_str = cls + ' '
      output_str += '%.2f %.d ' % (-1, -1)
      output_str += '%.7f %.7f %.7f %.7f %.7f ' % (alpha, box[0], box[1], box[2], box[3])
      output_str += '%.7f %.7f %.7f %.7f %.7f %.7f %.7f %.7f \n' % (h, w, l, Px, Py, \
                                                                    Pz, ori, score)

      # Write TXT files
      if not os.path.exists(result_dir):
          os.makedirs(result_dir)
      pred_filename = result_dir + '/' + file_number + '.txt'
      with open(pred_filename, 'a') as det_file:
          det_file.write(output_str)
    
    def replace_location(self,new_loc,img_idx):
        b=img_idx.shape[0]
        for i in range(b):
            img_id='{:06d}'.format(int(img_idx[i][0]))
            idx=int(img_idx[i][1])
            result=self.data[img_id][idx]
            result['pred_location']=new_loc[i].cpu().numpy().tolist()
            #single write mode
            box=result['box']
            dim=result['dim']
            pos=result['pred_location']
            ori=result['pred_rot']
            score=result['score']
            if type(ori) == list:
                ori=ori[0]
            if type(score) == list:
                score=score[0]
            self.write_detection_results('Car', self.result_dir, img_id, box, dim, pos, ori, score)


    def eval_all_results(self):   
        results,_=evaluate_python(label_path=os.path.join(self.args.kitti_path,'training/label_2'), 
                    result_path=self.result_dir,
                    label_split_file=self.split_file_path,
                    current_class=0,
                    metric='R40')
        print(results)
        mAP_strict_moderate=float(results.split('\n')[3].split(',')[1])
        return mAP_strict_moderate,results

def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()
           
def main():
    best_mAP = 0.
    args = parser.parse_args()
    print(args)
    args.writer = tb.SummaryWriter(log_dir=args.log_dir) if args.log_dir else None
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    multiprocessing.set_start_method('spawn')

    model = GMW(args)
    if args.gpu is not None:
        print("Using GPU {} for training".format(args.gpu))
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        torch.distributed.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
        model= torch.nn.parallel.DistributedDataParallel(model,\
            find_unused_parameters=False,device_ids=[args.local_rank],output_device=args.local_rank)

    optimizer = torch.optim.AdamW(model.parameters(),
                                 lr=args.lr,
                                 betas=(0.9, 0.999),
                                 weight_decay=args.weight_decay
                                 )
    lr_step=[0.,0.]
    lr_step[0],lr_step[1]=args.lr_step.split(',')
    lr_step[0],lr_step[1]=int(lr_step[0]),int(lr_step[1])
    args.lr_step=lr_step
    def lr_lbmd(cur_epoch):
        cur_decay = 1.
        for decay_step in args.lr_step:
            if cur_epoch >= decay_step:
                cur_decay = cur_decay * 0.1
        
        return cur_decay

    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0, last_epoch=-1)

    # Optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_mAP = checkpoint['best_mAP']
            if args.gpu is not None:##single card
                ckpt=OrderedDict()
                for key,value in checkpoint['state_dict'].items():
                    ckpt[key.replace('module.','')]=checkpoint['state_dict'][key]
            else:
                ckpt=checkpoint['state_dict']  
            model.load_state_dict(ckpt)
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    train_loader, val_loader = get_dataset(args)

    kitti_eval=GMW_data(args)
    synchronize()
    if args.evaluate:
        validate(val_loader, model, 0, args,kitti_eval)
        synchronize()
        if args.local_rank==0 or args.gpu is not None:
            kitti_eval.eval_all_results()
        return

    for epoch in range(args.start_epoch+1, args.epochs+1): 
        if (epoch >= args.reg_loss_start_epoch) and not args.no_weight_change:
            args.reg_weight=1.0
            args.cls_weight=0.1
        model.train()
        train(train_loader, model, optimizer,scheduler, epoch, args)
        if epoch>0 and epoch % 5==0 and (args.local_rank == 0 or args.gpu is not None):
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_mAP': best_mAP,
                'optimizer' : optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            },False, dir=args.log_dir, filename='checkpoint_epoch_' + str(epoch))
        if epoch==args.epochs:
            validate(val_loader, model, epoch, args,kitti_eval)
            synchronize()
            if args.local_rank == 0 or args.gpu is not None:
                mAP_now,All_mAP=kitti_eval.eval_all_results()
                is_best = mAP_now > best_mAP
                best_mAP = max(mAP_now,best_mAP)            
                open(os.path.join(args.log_dir,'log.txt'),'a').write('\n{}\n\n'.format(All_mAP))
                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_mAP': best_mAP,
                    'optimizer' : optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                },is_best, dir=args.log_dir, filename='checkpoint_epoch_' + str(epoch))
        

    if args.writer:
        args.writer.close()

def graph_extract(feature):
        feature=feature.diagonal(offset=0,dim1=-2,dim2=-1)
        feature=1./feature
        return feature
        
def get_up(matrix):
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

def compute_reg_loss(pre_depths,edge_weight,gt_depth,good_idx=None):
    if good_idx is not None:
        pre_depths_=pre_depths.gather(-1,good_idx)
        edge_weight_=edge_weight.gather(-1,good_idx)
        edge_weight_=edge_weight_.softmax(dim=-1)
    Z_select_weighted=(pre_depths_*edge_weight_).sum(-1)
    reg_loss = (Z_select_weighted-gt_depth).abs().mean()
    return reg_loss,Z_select_weighted

def compute_z(kpts_2d,kpts_3d,pred_rot):
    extra_kpts_num=63

    B=torch.zeros((kpts_2d.shape[0],kpts_2d.shape[1]*2,1)).to(kpts_2d.device)
    C=torch.zeros((kpts_2d.shape[0],kpts_2d.shape[1]*2,1)).to(kpts_2d.device)

    X=kpts_3d[:,:,0:1]
    Y=kpts_3d[:,:,1:2]
    Z=kpts_3d[:,:,2:3]

    cosori = torch.cos(pred_rot).unsqueeze(-1).expand_as(X)
    sinori = torch.sin(pred_rot).unsqueeze(-1).expand_as(X)

    B[:,0::2] = X * cosori + Z * sinori
    B[:,1::2] = Y 
    C[:,0::2] = X*sinori - Z*cosori
    C[:,1::2] = X*sinori - Z*cosori
    B1=B.clone()
    B2=kpts_2d.reshape(kpts_2d.shape[0],-1,1) * C
    H_1=B1[:,1::2,:]
    H_2=B2[:,1::2,:]

    V=kpts_2d[:,:,1:2]

    H1_1=H_1.expand(H_1.shape[0],H_1.shape[1],H_1.shape[1])
    H1_2=H_2.expand(H_1.shape[0],H_1.shape[1],H_1.shape[1])
    
    V1=V.expand_as(H1_1)

    H_mat_new=(H1_1-H1_1.permute(0,2,1)) + (H1_2-H1_2.permute(0,2,1))

    V_mat=V1-V1.permute(0,2,1)
    
    Z_v_raw_new=H_mat_new.abs()/(V_mat).abs().clamp_min(1e-10)

    Z_v_raw=get_up(Z_v_raw_new)

    Z_v_raw=Z_v_raw.clamp_min(0.1).clamp_max(80.)

    ##get good idx
    K=1500
    _,good_idx=torch.topk(get_up(V_mat).abs(),K,dim=-1)
    
    return Z_v_raw,good_idx

def train(train_loader, model, optimizer,scheduler, epoch, args):
    batch_time = AverageMeter('Time', ':6.4f')
    data_time = AverageMeter('Data', ':6.4f')
    loss_meter = AverageMeter('Loss', ':6.4f')
    cls_loss_meter = AverageMeter('cls Loss', ':6.4f')    
    reg_loss_meter = AverageMeter('reg Loss', ':6.4f')
    
    cls_acc_meter = AverageMeter('cls_acc', ':6.4f')
    pos_cls_acc_meter = AverageMeter('pos_cls_acc', ':6.4f')
    Depth_MAE_meter = AverageMeter('Depth_MAE', ':6.4f')
    correspondence_probability_meter = AverageMeter('Outlier-Inlier Prob', ':6.4f')
    rotation_meter = AverageMeter('Rotation Error', ':6.4f')
    translation_meter = AverageMeter('Translation Error', ':6.4f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, loss_meter,cls_loss_meter,reg_loss_meter,Depth_MAE_meter],
        prefix="Epoch: [{}]".format(epoch),log_dir=args.log_dir)

    model.train()
    end = time.time()
    pred_best_idx_list=[]
    gt_best_idx_list=[] 
    for batch_index, (kpts_2d,kpts_3d,pred_rot,gt_location,img_idx) in enumerate(train_loader):
        data_time.update(time.time() - end) # Measure data loading time
        if args.gpu is not None:
            kpts_2d = kpts_2d.cuda(args.gpu, non_blocking=True)
            kpts_3d = kpts_3d.cuda(args.gpu, non_blocking=True)
            pred_rot = pred_rot.cuda(args.gpu, non_blocking=True)
            gt_location = gt_location.cuda(args.gpu, non_blocking=True)
        else:
            kpts_2d = kpts_2d.cuda( non_blocking=True)
            kpts_3d = kpts_3d.cuda(non_blocking=True)
            pred_rot = pred_rot.cuda(non_blocking=True)
            gt_location = gt_location.cuda(non_blocking=True)
        ###compute the edge depth candidates
        pre_depths,good_idx=compute_z(kpts_2d,kpts_3d,pred_rot)
        #cls loss
        reg_weights,edge_P=model(kpts_2d,kpts_3d,pred_rot,args)##b*57*2
        edge_P_gt=torch.eye(edge_P.shape[1]).expand_as(edge_P).to(edge_P.device)
        cls_loss=correspondenceLoss(edge_P,edge_P_gt)
        #reg loss
        reg_loss,pred_depth=compute_reg_loss(pre_depths,reg_weights,gt_location[:,-1],good_idx)
        #all loss
        loss=args.cls_weight*cls_loss + args.reg_weight* reg_loss
        # Compute gradient and do optimizer step
        optimizer.zero_grad()
        if not torch.isnan(loss).any():
            loss.backward()
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        ### meter wirte
        gt_depth=gt_location[:,2]
        Depth_MAE=((pred_depth-gt_depth).abs()/gt_depth).mean()
        loss_meter.update(loss.item(), kpts_2d.size(0))
        cls_loss_meter.update(cls_loss.item(), kpts_2d.size(0))
        reg_loss_meter.update(reg_loss.item(), kpts_2d.size(0))
        Depth_MAE_meter.update(Depth_MAE.item(),kpts_2d.size(0))

        if args.local_rank==0 or args.gpu is not None:
            if args.writer:
                global_step = epoch * len(train_loader) + batch_index
                args.writer.add_scalar('loss_train', loss.item(), global_step=global_step)
                
            if batch_index % args.print_freq == 0:
                progress.display(batch_index)

def validate(val_loader, model, epoch, args,kitti_eval):
    batch_time = AverageMeter('Time', ':6.4f')
    loss_meter = AverageMeter('Loss', ':6.4f')
    cls_loss_meter = AverageMeter('cls Loss', ':6.4f')
    reg_loss_meter = AverageMeter('reg Loss', ':6.4f')
    cls_acc_meter = AverageMeter('cls_acc', ':6.4f')
    pos_cls_acc_meter = AverageMeter('pos_cls_acc', ':6.4f')
    Depth_MAE_meter = AverageMeter('Depth_MAE', ':6.4f')
    correspondence_probability_meter = AverageMeter('Outlier-Inlier Prob', ':6.4f')
    rotation_meter = AverageMeter('Rotation Error', ':6.4f')
    translation_meter = AverageMeter('Translation Error', ':6.4f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, loss_meter,cls_loss_meter,reg_loss_meter,Depth_MAE_meter],
        prefix='Test: ')

    model.eval()
    with torch.no_grad():
        end = time.time()

        rotation_errors_theta0 = []
        translation_errors_theta0 = []

        start_time = time.time()
        for batch_index, (kpts_2d,kpts_3d,pred_rot,raw_location,dim,img_idx) in enumerate(val_loader):
            if args.gpu is not None:
                kpts_2d = kpts_2d.cuda(args.gpu, non_blocking=True)
                kpts_3d = kpts_3d.cuda(args.gpu, non_blocking=True)
                pred_rot = pred_rot.cuda(args.gpu, non_blocking=True)
                raw_location = raw_location.cuda(args.gpu, non_blocking=True)
                dim=dim.cuda(args.gpu, non_blocking=True)
            else:
                kpts_2d = kpts_2d.cuda( non_blocking=True)
                kpts_3d = kpts_3d.cuda(non_blocking=True)
                pred_rot = pred_rot.cuda(non_blocking=True)
                raw_location = raw_location.cuda(non_blocking=True)
                dim=dim.cuda(non_blocking=True)

            pre_depths,good_idx=compute_z(kpts_2d,kpts_3d,pred_rot)
            # Compute output
            raw_location=raw_location.clone()
                
            #cls loss
            reg_weights,edge_P=model(kpts_2d,kpts_3d,pred_rot,args)##b*57*2
            edge_P_gt=torch.eye(edge_P.shape[1]).expand_as(edge_P).to(edge_P.device)
            cls_loss=correspondenceLoss(edge_P,edge_P_gt)
            #reg loss
            reg_loss,pred_depth=compute_reg_loss(pre_depths,reg_weights,raw_location[:,-1],good_idx)
            loss=args.cls_weight*cls_loss+ args.reg_weight* reg_loss

            batch_time.update(time.time() - end)
            end = time.time()
            loss_meter.update(loss.item(), kpts_2d.size(0))
            cls_loss_meter.update(cls_loss.item(), kpts_2d.size(0))
            reg_loss_meter.update(reg_loss.item(), kpts_2d.size(0))
            
            raw_depth=raw_location[:,2]
            scale=pred_depth/raw_depth
            h=dim[:,0]
            raw_location[:,1]-=h/2
            pred_location=scale.unsqueeze(-1)*raw_location
            pred_location[:,1]+=h/2
            kitti_eval.replace_location(pred_location,img_idx)
            
            if args.local_rank==0 or args.gpu is not None:
                if args.writer:
                    global_step = epoch * len(val_loader) + batch_index
                    args.writer.add_scalar('loss_val', loss.item(), global_step=global_step)

                if batch_index % args.print_freq == 0:
                    progress.display(batch_index)

                print('Loss: {loss.avg:6.4f}, cls loss: {cls_loss.avg:6.4f}, reg loss: {reg_loss.avg:6.4f},Depth MAE:{Depth_MAE.avg:6.4f}'.format(loss=loss_meter,cls_loss=cls_loss_meter,reg_loss=reg_loss_meter,Depth_MAE=Depth_MAE_meter))

                if args.writer:
                    args.writer.add_scalar('loss_val', loss_meter.avg, global_step=epoch)
                    args.writer.add_scalar('cls_acc_val', cls_acc_meter.avg, global_step=epoch)
                    args.writer.add_scalar('pos_cls_acc_val', pos_cls_acc_meter.avg, global_step=epoch)
    return loss_meter.avg

def save_checkpoint(state,args, is_best=False, dir='', filename='checkpoint'):
    torch.save(state, os.path.join(dir, filename + '.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(dir, filename + '.pth.tar'), os.path.join(dir, 'model_best.pth.tar'))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="",log_dir=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.log_path=os.path.join(log_dir,'log.txt')
    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        open(self.log_path,'a').write(''.join(entries)+'\n')
        
    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

if __name__ == '__main__':
    main()
