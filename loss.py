import torch.nn as nn
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt

import numpy as np


class Subloss(nn.Module):
    def __init__(self, smooth=1e-15, p=2, reduction='mean'):
        super(Subloss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        # print(type(predict),type(target))
        # predict = np.ndarray(predict)
        # predict = torch.from_numpy(predict)
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = nn.Sigmoid()(predict)
        # pred = predict[0].cpu().detach().numpy().transpose(1,2,0)
        # targ = target[0].cpu().detach().numpy().transpose(1,2,0)
        # plt.imshow(pred, 'gray')
        # plt.show()
        # plt.imshow(targ,'gray')
        # plt.show()

        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)
        # print(predict.shape)
        # print(target.shape)
        num = torch.sum(torch.sub(predict, target).pow(self.p), dim=1)
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = num / den


        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        num = targets.size(0)
        smooth = 1

        probs = F.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE