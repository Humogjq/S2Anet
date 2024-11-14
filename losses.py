# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: clipped tensor
    """
    t = t.float()
    
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    
    return result

class Smoothloss(nn.Module):
    def __init__(self):
        super(Smoothloss, self).__init__()

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        H = y_pred.shape[-2]
        W = y_pred.shape[-1]

        loss = torch.abs(y_pred[:, :, 1:H - 1, 1:W - 1] - y_pred[:, :, 0:H - 2, 1:W - 1]) + \
               torch.abs(y_pred[:, :, 1:H - 1, 1:W - 1] - y_pred[:, :, 2:H, 1:W - 1]) + \
               torch.abs(y_pred[:, :, 1:H - 1, 1:W - 1] - y_pred[:, :, 1:H - 1, 0:W - 2]) + \
               torch.abs(y_pred[:, :, 1:H - 1, 1:W - 1] - y_pred[:, :, 1:H - 1, 2:W])

        M1 = torch.eq(y_true[:, :, 1:H - 1, 1:W - 1], y_true[:, :, 0:H - 2, 1:W - 1]).float()
        M2 = torch.eq(y_true[:, :, 1:H - 1, 1:W - 1], y_true[:, :, 2:H, 1:W - 1]).float()
        M3 = torch.eq(y_true[:, :, 1:H - 1, 1:W - 1], y_true[:, :, 1:H - 1, 0:W - 2]).float()
        M4 = torch.eq(y_true[:, :, 1:H - 1, 1:W - 1], y_true[:, :, 1:H - 1, 2:W]).float()

        mask = M1*M2*M3*M4
        print("smooth_loss:", torch.mean(loss*mask))

        return torch.mean(loss*mask)

class dice_loss(nn.Module):
    def __init__(self, eps=1e-12):
        super(dice_loss, self).__init__()
        self.eps = eps
    
    def forward(self, pred, gt):
        assert pred.size() == gt.size() and pred.size()[1] == 1
        
        N = pred.size(0)
        pred_flat = pred.view(N, -1)
        gt_flat = gt.view(N, -1)
        intersection = pred_flat * gt_flat
        dice = (2.0 * intersection.sum(1) + self.eps) / (pred_flat.sum(1) + gt_flat.sum(1) + self.eps)
        loss = 1.0 - dice.mean()
        
        return loss

class dice_disccup_loss(nn.Module):
    def __init__(self, eps=1e-12):
        super(dice_disccup_loss, self).__init__()
        self.eps = eps

    def forward(self, pred, gt):
        pred0 = pred[:, 0, :, :]
        gt0 = gt[:, 0, :, :]

        N0 = pred0.size(0)
        pred_flat0 = pred0.view(N0, -1)
        gt_flat0 = gt0.view(N0, -1)
        intersection0 = pred_flat0 * gt_flat0
        dice0 = (2.0 * intersection0.sum(1) + self.eps) / (pred_flat0.sum(1) + gt_flat0.sum(1) + self.eps)
        loss0 = 1.0 - dice0.mean()

        pred1 = pred[:, 1, :, :]
        gt1 = gt[:, 1, :, :]
        N1 = pred1.size(0)
        pred_flat1 = pred1.view(N1, -1)
        gt_flat1 = gt1.view(N1, -1)
        intersection1 = pred_flat1 * gt_flat1
        dice1 = (2.0 * intersection1.sum(1) + self.eps) / (pred_flat1.sum(1) + gt_flat1.sum(1) + self.eps)
        loss1 = 1.0 - dice1.mean()

        return 0.4 * loss0 + 0.6 * loss1

class focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, size_average=True):
        super(focal_loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average
    
    def forward(self, pred, gt):
        assert pred.size() == gt.size() and pred.size()[1] == 1
        
        pred_oh = torch.cat((pred, 1.0 - pred), dim=1)  # [b, 2, h, w]
        gt_oh = torch.cat((gt, 1.0 - gt), dim=1)  # [b, 2, h, w]
        pt = (gt_oh * pred_oh).sum(1)  # [b, h, w]
        focal_map = - self.alpha * torch.pow(1.0 - pt, self.gamma) * torch.log2(clip_by_tensor(pt, 1e-12, 1.0))  # [b, h, w]
        
        if self.size_average:
            loss = focal_map.mean()
        else:
            loss = focal_map.sum()
        
        return loss


# 构建损失函数，可扩展
def build_loss(loss):
    if loss == "mse":
        criterion = nn.MSELoss()
    elif loss == "l1":
        criterion = nn.L1Loss()
    elif loss == "smoothl1":
        criterion = nn.SmoothL1Loss()
    elif loss == "bce":
        criterion = focal_loss(alpha=1.0, gamma=0.0)
    elif loss == "focal":
        criterion = focal_loss(alpha=0.25, gamma=2.0)
    elif loss == "dice":
        criterion = dice_loss()
    elif loss == "smooth":
        criterion = Smoothloss()
    elif loss == "dice2":
        criterion = dice_disccup_loss()
    else:
        raise NotImplementedError('loss [%s] is not implemented' % loss)
    
    return criterion
