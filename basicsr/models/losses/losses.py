# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np

from basicsr.models.losses.loss_util import weighted_loss, SSIM

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


# @weighted_loss
# def charbonnier_loss(pred, target, eps=1e-12):
#     return torch.sqrt((pred - target)**2 + eps)


class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)


class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)


class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()


class KDLoss(nn.Module):
    """
    Args:
        loss_weight (float): Loss weight for KD loss. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0, temperature=0.15):
        super(KDLoss, self).__init__()

        self.loss_weight = loss_weight
        self.temperature = temperature

    def forward(self, S1_fea, S2_fea):

        """
        Args:
            S1_fea (List): contain shape (N, L) vector.
            S2_fea (List): contain shape (N, L) vector.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        loss_KD_dis = 0
        loss_KD_abs = 0
        for i in range(len(S1_fea)):
            S2_distance = F.log_softmax(S2_fea[i] / self.temperature, dim=1)
            S1_distance = F.softmax(S1_fea[i].detach() / self.temperature, dim=1)
            loss_KD_dis += F.kl_div(
                S2_distance, S1_distance, reduction='batchmean')
            loss_KD_abs += nn.L1Loss()(S2_fea[i], S1_fea[i].detach())
        return self.loss_weight * loss_KD_dis, self.loss_weight * loss_KD_abs

class FFTLoss(nn.Module):
    """L1 loss in frequency domain with FFT.

    Args:
        loss_weight (float): Loss weight for FFT loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(FFTLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. ' f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (..., C, H, W). Predicted tensor.
            target (Tensor): of shape (..., C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (..., C, H, W). Element-wise
                weights. Default: None.
        """

        pred_fft = torch.fft.fft2(pred, dim=(-2, -1))
        pred_fft = torch.stack([pred_fft.real, pred_fft.imag], dim=-1)
        target_fft = torch.fft.fft2(target, dim=(-2, -1))
        target_fft = torch.stack([target_fft.real, target_fft.imag], dim=-1)
        return self.loss_weight * l1_loss(pred_fft, target_fft, weight, reduction=self.reduction)

class L_grad_cosist(nn.Module):

    def __init__(self):
        super(L_grad_cosist, self).__init__()
        kernel_right = torch.FloatTensor( [[0,0,0],[0,1,-1],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor( [[0,0,0],[0,1, 0],[0,-1,0]]).cuda().unsqueeze(0).unsqueeze(0)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)

    def gradient_of_one_channel(self,x,y):
        D_org_right = F.conv2d(x , self.weight_right, padding="same")
        D_org_down = F.conv2d(x , self.weight_down, padding="same")
        D_enhance_right = F.conv2d(y , self.weight_right, padding="same")
        D_enhance_down = F.conv2d(y , self.weight_down, padding="same")
        return torch.abs(D_org_right),torch.abs(D_enhance_right),torch.abs(D_org_down),torch.abs(D_enhance_down)

    def gradient_Consistency_loss_patch(self,x,y):
        # B*C*H*W
        min_x = torch.abs(x.min(2,keepdim=True)[0].min(3,keepdim=True)[0]).detach()
        min_y = torch.abs(y.min(2,keepdim=True)[0].min(3,keepdim=True)[0]).detach()
        x = x - min_x
        y = y - min_y
        #B*1*1,3
        product_separte_color = (x*y).mean([2,3],keepdim=True)
        x_abs = (x**2).mean([2,3],keepdim=True)**0.5
        y_abs = (y**2).mean([2,3],keepdim=True)**0.5
        loss1 = (1-product_separte_color/(x_abs*y_abs+0.00001)).mean() + torch.mean(torch.acos(product_separte_color/(x_abs*y_abs+0.00001)))

        product_combine_color = torch.mean(product_separte_color,1,keepdim=True)
        x_abs2 = torch.mean(x_abs**2,1,keepdim=True)**0.5
        y_abs2 = torch.mean(y_abs**2,1,keepdim=True)**0.5
        loss2 = torch.mean(1-product_combine_color/(x_abs2*y_abs2+0.00001)) + torch.mean(torch.acos(product_combine_color/(x_abs2*y_abs2+0.00001)))
        return loss1 + loss2

    def forward(self, x, y):

        x_R1,y_R1, x_R2,y_R2  = self.gradient_of_one_channel(x[:,0:1,:,:],y[:,0:1,:,:])
        x_G1,y_G1, x_G2,y_G2  = self.gradient_of_one_channel(x[:,1:2,:,:],y[:,1:2,:,:])
        x_B1,y_B1, x_B2,y_B2  = self.gradient_of_one_channel(x[:,2:3,:,:],y[:,2:3,:,:])
        x = torch.cat([x_R1,x_G1,x_B1,x_R2,x_G2,x_B2],1)
        y = torch.cat([y_R1,y_G1,y_B1,y_R2,y_G2,y_B2],1)

        B,C,H,W = x.shape
        loss = self.gradient_Consistency_loss_patch(x,y)
        loss1 = 0
        loss1 += self.gradient_Consistency_loss_patch(x[:,:,0:H//2,0:W//2],y[:,:,0:H//2,0:W//2])
        loss1 += self.gradient_Consistency_loss_patch(x[:,:,H//2:,0:W//2],y[:,:,H//2:,0:W//2])
        loss1 += self.gradient_Consistency_loss_patch(x[:,:,0:H//2,W//2:],y[:,:,0:H//2,W//2:])
        loss1 += self.gradient_Consistency_loss_patch(x[:,:,H//2:,W//2:],y[:,:,H//2:,W//2:])

        return loss #+loss1#+torch.mean(torch.abs(x-y))#+loss1

class L_bright_cosist(nn.Module):

    def __init__(self):
        super(L_bright_cosist, self).__init__()

    def gradient_Consistency_loss_patch(self,x,y):
        # B*C*H*W
        min_x = torch.abs(x.min(2,keepdim=True)[0].min(3,keepdim=True)[0]).detach()
        min_y = torch.abs(y.min(2,keepdim=True)[0].min(3,keepdim=True)[0]).detach()
        x = x - min_x
        y = y - min_y
        #B*1*1,3
        product_separte_color = (x*y).mean([2,3],keepdim=True)
        x_abs = (x**2).mean([2,3],keepdim=True)**0.5
        y_abs = (y**2).mean([2,3],keepdim=True)**0.5
        loss1 = (1-product_separte_color/(x_abs*y_abs+0.00001)).mean() + torch.mean(torch.acos(product_separte_color/(x_abs*y_abs+0.00001)))

        product_combine_color = torch.mean(product_separte_color,1,keepdim=True)
        x_abs2 = torch.mean(x_abs**2,1,keepdim=True)**0.5
        y_abs2 = torch.mean(y_abs**2,1,keepdim=True)**0.5
        loss2 = torch.mean(1-product_combine_color/(x_abs2*y_abs2+0.00001)) + torch.mean(torch.acos(product_combine_color/(x_abs2*y_abs2+0.00001)))
        return loss1 + loss2
    def forward(self, x, y):

        B,C,H,W = x.shape
        loss = self.gradient_Consistency_loss_patch(x,y)
        loss1 = 0
        loss1 += self.gradient_Consistency_loss_patch(x[:,:,0:H//2,0:W//2],y[:,:,0:H//2,0:W//2])
        loss1 += self.gradient_Consistency_loss_patch(x[:,:,H//2:,0:W//2],y[:,:,H//2:,0:W//2])
        loss1 += self.gradient_Consistency_loss_patch(x[:,:,0:H//2,W//2:],y[:,:,0:H//2,W//2:])
        loss1 += self.gradient_Consistency_loss_patch(x[:,:,H//2:,W//2:],y[:,:,H//2:,W//2:])

        return loss #+loss1#+torch.mean(torch.abs(x-y))#+loss1

class L_recon(nn.Module):

    def __init__(self):
        super(L_recon, self).__init__()
        self.ssim_loss = SSIM()

    def forward(self, R_low, high):
        L1 = torch.abs(R_low - high).mean()
        L2 = (1- self.ssim_loss(R_low,high)).mean()
        return L1,L2


class L_color_zy(nn.Module):

    def __init__(self):
        super(L_color_zy, self).__init__()

    def forward(self, x, y):
        product_separte_color = (x * y).mean(1, keepdim=True)
        x_abs = (x ** 2).mean(1, keepdim=True) ** 0.5
        y_abs = (y ** 2).mean(1, keepdim=True) ** 0.5
        loss1 = (1 - product_separte_color / (x_abs * y_abs + 0.00001)).mean() + torch.mean(
            torch.acos(product_separte_color / (x_abs * y_abs + 0.00001)))

        return loss1

class L_exp(nn.Module):

    def __init__(self,patch_size):
        super(L_exp, self).__init__()
        # print(1)
        self.pool = nn.AvgPool2d(patch_size)
        # self.mean_val = mean_val
    def forward(self, x, mean_val ):

        b,c,h,w = x.shape
        x = torch.max(x,1,keepdim=True)[0]
        mean = self.pool(x)

        d = torch.mean(torch.pow(mean- torch.FloatTensor([mean_val] ).cuda(),2))
        return d

