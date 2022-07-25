import torch
import pdb
from torch import nn

class Vanilla_FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(Vanilla_FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, prediction, target):
        positive_index = target.eq(1).float()
        negative_index = (target.lt(1) & target.ge(0)).float()

        loss = 0.
        positive_loss = torch.log(prediction) \
                        * torch.pow(target - prediction, self.gamma) * positive_index
                        
        negative_loss = torch.log(1 - prediction) \
                        * torch.pow(prediction, self.gamma) * negative_index

        positive_loss = positive_loss.sum() * self.alpha
        negative_loss = negative_loss.sum() * (1 - self.alpha)

        loss = - negative_loss - positive_loss

        return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=2, beta=4,cfg=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha ##hard and easy
        self.beta = beta   ##allivate neg 
        self.eps=1e-10
        # TYPE_ID_CONVERSION = {
		# 		'car': 0,
		# 		'pedestrian': 1,
		# 		'bicycle': 2,
		# 		'motorcycle': 3,
		# 		'barrier': 4,
		# 		'bus': 5,
		# 		'construction_vehicle':6,
		# 		'traffic_cone':7,
		# 		'trailer':8,
		# 		'truck':9,
		# 		'DontCare': 10,
		# 	}
        # self.pos_cls_weight=torch.tensor([1.,1.,1.,1.,1.,1.,3.,1.,3.,1,]).reshape(1,10,1,1).to(device=cfg.MODEL.DEVICE)
        # self.cls_weight=torch.tensor([1.,1.,1.5,1.5,1.5,1.,5.,1.,5.,2.]).reshape(1,10,1,1).to(device=cfg.MODEL.DEVICE)
        # self.cls_weight=torch.tensor([0.5,0.5,0.75,0.75,0.75,0.5,2.5,0.5,2.5,1]).reshape(1,10,1,1).to(device=cfg.MODEL.DEVICE)
        
        self.cls_num=cfg.DATASETS.MAX_CLASSES_NUM

        # self.pos_cls_weight=self.pos_cls_weight[:,:self.cls_num,:,:]
        # self.cls_weight=self.cls_weight[:,:self.cls_num,:,:]

    def forward(self, prediction, target):
        prediction=prediction.clamp(self.eps,1-self.eps)
        positive_index = target.eq(1).float()
        negative_index = (target.lt(1) & target.ge(0)).float()
        ignore_index = target.eq(-1).float() # ignored pixels

        negative_weights = torch.pow(1 - target, self.beta)
        loss = 0.

        positive_loss = torch.log(prediction) \
                        * torch.pow(1 - prediction, self.alpha) * positive_index 
        # if self.cls_num==10:
            # positive_loss*=self.pos_cls_weight
                        
        negative_loss = torch.log(1 - prediction) \
                        * torch.pow(prediction, self.alpha) * negative_weights * negative_index

        num_positive = positive_index.float().sum()
        positive_loss = positive_loss
        negative_loss = negative_loss
        
        # size=target.shape[-2]*target.shape[-1]
        # cls_mask= (target==0).sum((-2,-1)) < size #filter no target class
        # positive_loss = positive_loss[cls_mask,:,:].sum()
        # negative_loss = negative_loss[cls_mask,:,:].sum()

        loss = - negative_loss - positive_loss
        loss = loss.sum()

        return loss, num_positive

if __name__ == '__main__':
    focal_1 = Vanilla_FocalLoss(alpha=0.5)
    focal_2 = FocalLoss()

    pred = torch.rand(20)
    target = torch.randint(low=0, high=2, size=(20, 1)).view(-1)

    loss1 = focal_1(pred, target) * 2
    loss2 = focal_2(pred, target)

    print(loss1, loss2)






