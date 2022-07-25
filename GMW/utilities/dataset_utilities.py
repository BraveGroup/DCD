import numpy as np
import os
import pickle
import torch
from torch.functional import split
from tqdm import tqdm
import json
import pdb
import time

def load_data(args,dataset_split, preprocessed=False):
    final_data={}
    final_data={
        'kpts_2d':[],
        'kpts_3d':[],
        'pred_rot':[],
        'gt_location':[],
        'img_idx':[],
        'dim':[]
    }
    if dataset_split == "train":
        data=json.load(open(args.train_data_path,'r'))
        N=len(data['kpts_2d'])
        for i in range(N):
            K=len(data['kpts_2d'][i])
            for j in range(K):
                kpts_2d=np.array(data['kpts_2d'][i][j])
                kpts_3d=np.array(data['kpts_3d'][i][j])
                gt_loc=np.array(data['gt_location'][i][j])
                final_data['kpts_2d'].append(kpts_2d)
                final_data['kpts_3d'].append(kpts_3d)
                final_data['pred_rot'].append([data['pred_rot'][i][j]])
                final_data['gt_location'].append(gt_loc)
                final_data['img_idx'].append((0,0))
        for key in final_data.keys():
            final_data[key]=np.array(final_data[key],dtype=np.float32) 

    elif dataset_split == "valid":
        data=json.load(open(args.val_data_path,'r'))
        for img in data.keys():
            N=len(data[img])
            for i in range(N):
                a=data[img][i]
                kpts_2d=np.array(a['kpts_2d']).astype(np.float32).reshape((-1,2))[:73,:]
                kpts_3d=np.array(a['kpts_3d']).astype(np.float32).reshape((-1,3))[:73,:]
                pred_rot=np.array(a['pred_rot']).astype(np.float32)
                final_data['dim'].append(a['dim'])
                final_data['kpts_2d'].append(kpts_2d)    
                final_data['kpts_3d'].append(kpts_3d)    
                final_data['pred_rot'].append(pred_rot)
                final_data['gt_location'].append(np.array(a['pred_location']).astype(np.float32))    
                final_data['img_idx'].append((img,int(i)))
        for key in final_data.keys():
            final_data[key]=np.array(final_data[key],dtype=np.float32)

    return final_data

class Dataset(torch.utils.data.Dataset):
  def __init__(self, dataset_split, args, batch_size, preprocessed=True):
    self.batch_size = batch_size
    self.dataset_split=dataset_split
    self.data = load_data(args,dataset_split)
    self.len=len(self.data['kpts_2d']) 

  def __getitem__(self, index):
    if self.dataset_split == 'valid':
        return self.data["kpts_2d"][index], self.data["kpts_3d"][index], self.data["pred_rot"][index], self.data["gt_location"][index],self.data["dim"][index],self.data['img_idx'][index]
    else:
        return self.data["kpts_2d"][index], self.data["kpts_3d"][index], self.data["pred_rot"][index], self.data["gt_location"][index],self.data['img_idx'][index]

    
  def __len__(self):
    return self.len

