# -*- coding: utf-8 -*-
"""
@author: Zixiang Zhao (zixiangzhao@stu.xjtu.edu.cn)

Pytorch implement for "Efficient and Model-Based Infrared and Visible Image Fusion via Algorithm Unrolling" (TCSVT 2021)

10.1109/TCSVT.2021.3075745

https://ieeexplore.ieee.org/document/9416456
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from args import channel,img_size,layer_numb
import kornia
#定义初始化滤波 
Laplace = kornia.filters.Laplacian(19)#高通滤波  
Blur = kornia.filters.BoxBlur((11, 11))#低通滤波  
# =============================================================================
# 网络搭建
# =============================================================================
class BCL(nn.Module):
    def __init__(self):
        super(BCL, self).__init__()
        self.kernel = nn.Parameter(torch.randn(size=[64, 1, 3, 3]))
        self.zero_pad=nn.ReflectionPad2d(1)
        self.rest=nn.Sequential(
            nn.BatchNorm2d(1),
            nn.PReLU()
            )
        self.eta=nn.Parameter(
                nn.init.normal_(
                        torch.empty(1).cuda(),mean=0.1,std=0.03
                        )) 
        self.theta=nn.Parameter(
                nn.init.normal_(
                        torch.empty(1).cuda(),mean=1e-3,std=1e-4
                        )) 
    def forward(self,x_in,img):
        x_in_2=self.zero_pad(x_in)
        x_1=F.conv2d(x_in_2,self.kernel,padding=0)
        x_2=self.zero_pad(x_1)
        kernel2 = torch.rot90(self.kernel, 2, [-1,-2]).transpose(0,1)
        x_3=F.conv2d(x_2,kernel2,padding=0)
        x_out=self.rest(x_3)
        x_out_2=x_in-self.eta*(x_out-self.theta*(img-x_in))
        return x_out_2,img,self.eta.item(),self.theta.item()

class DCL(nn.Module):
    def __init__(self):
        super(DCL, self).__init__()
        self.kernel = nn.Parameter(torch.randn(size=[64, 1, 3, 3]))
        self.zero_pad=nn.ReflectionPad2d(1)
        self.rest=nn.Sequential(
            nn.BatchNorm2d(1),
            nn.PReLU()
            )
        self.eta=nn.Parameter(
                nn.init.normal_(
                        torch.empty(1).cuda(),mean=0.1,std=0.03
                        )) 
        self.theta=nn.Parameter(
                nn.init.normal_(
                        torch.empty(1).cuda(),mean=1e-3,std=1e-4
                        )) 
    def forward(self,x_in,img):
        x_in_2=self.zero_pad(x_in)
        x_1=F.conv2d(x_in_2,self.kernel,padding=0)
        x_2=self.zero_pad(x_1)
        kernel2 = torch.rot90(self.kernel, 2, [-1,-2]).transpose(0,1)
        x_3=F.conv2d(x_2,kernel2,padding=0)
        x_out=self.rest(x_3)
        x_out_2=x_in-self.eta*(x_out-self.theta*(img-x_in))
        return x_out_2,img,self.eta.item(),self.theta.item()  
    
class Encoder_Base(nn.Module):
    def __init__(self,size=img_size):
        super(Encoder_Base, self).__init__()
        self.numb=layer_numb
        self.conv1 = nn.ModuleList([BCL() for i in range(self.numb)])
    def forward(self,img):
        img_blur=Blur(img)
        eta_list_base=[]
        theta_list_base=[]
        for layer in self.conv1:
            img_blur, img, eta, theta = layer(img_blur,img)
            eta_list_base.append(eta)
            theta_list_base.append(theta)
        return img_blur,eta_list_base,theta_list_base


class Encoder_Detail(nn.Module):
    def __init__(self,size=img_size):
        super(Encoder_Detail, self).__init__()
        self.numb=layer_numb
        self.conv2 = nn.ModuleList([DCL() for i in range(self.numb)])
    def forward(self,img):
        img_laplace=Laplace(img)
        eta_list_detail=[]
        theta_list_detail=[]
        for layer in self.conv2:
            img_laplace, img, eta, theta = layer(img_laplace,img)
            eta_list_detail.append(eta)
            theta_list_detail.append(theta)
        return img_laplace,eta_list_detail,theta_list_detail

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 1, 3, padding=0, bias=False), # in_channels, out_channels, kernel_size
            nn.BatchNorm2d(1),
            nn.Sigmoid()
            )
    def forward(self, fm_b, fm_d):
        fm=fm_b+fm_d
        return self.decoder(fm)
