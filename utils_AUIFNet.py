# -*- coding: utf-8 -*-
"""
@author: Zixiang Zhao (zixiangzhao@stu.xjtu.edu.cn)

Pytorch implement for "Efficient and Model-Based Infrared and Visible Image Fusion via Algorithm Unrolling" (TCSVT 2021)

10.1109/TCSVT.2021.3075745

https://ieeexplore.ieee.org/document/9416456
"""
import numpy as np
import torch
from AUIFNet import Encoder_Base,Encoder_Detail,Decoder
import torch.nn.functional as F

device='cuda'

def output_img(x):
    return x.cpu().detach().numpy()[0,0,:,:]

def l1_addition(y1,y2,window_width=1):
      ActivityMap1 = y1.abs()
      ActivityMap2 = y2.abs()

      kernel = torch.ones(2*window_width+1,2*window_width+1)/(2*window_width+1)**2
      kernel = kernel.to(device).type(torch.float32)[None,None,:,:]
      kernel = kernel.expand(y1.shape[1],y1.shape[1],2*window_width+1,2*window_width+1)
      ActivityMap1 = F.conv2d(ActivityMap1, kernel, padding=window_width)
      ActivityMap2 = F.conv2d(ActivityMap2, kernel, padding=window_width)
      WeightMap1 = ActivityMap1/(ActivityMap1+ActivityMap2)
      WeightMap2 = ActivityMap2/(ActivityMap1+ActivityMap2)
      return WeightMap1*y1+WeightMap2*y2

def Test_fusion(img_test1,img_test2,addition_mode='Sum'):
    Encoder_Base_Test = Encoder_Base().to(device)
    Encoder_Base_Test.load_state_dict(torch.load(
            "Models\TCSVT_Encoder_Base.model"
            ))
    Encoder_Detail_Test = Encoder_Detail().to(device)
    Encoder_Detail_Test.load_state_dict(torch.load(
            "Models\TCSVT_Encoder_Detail.model"
            ))
    Decoder_Test = Decoder().to(device)
    Decoder_Test.load_state_dict(torch.load(
            "Models\TCSVT_Decoder.model"
            ))
    
    Encoder_Base_Test.eval()
    Encoder_Detail_Test.eval()
    Decoder_Test.eval()
    
    img_test1 = np.array(img_test1, dtype='float32')/255# 将其转换为一个矩阵
    img_test1 = torch.from_numpy(img_test1.reshape((1, 1, img_test1.shape[0], img_test1.shape[1])))
    
    img_test2 = np.array(img_test2, dtype='float32')/255 # 将其转换为一个矩阵
    img_test2 = torch.from_numpy(img_test2.reshape((1, 1, img_test2.shape[0], img_test2.shape[1])))
    
    img_test1=img_test1.cuda()
    img_test2=img_test2.cuda()
    
    with torch.no_grad():
        B_K_IR,_,_=Encoder_Base_Test(img_test1)
        B_K_VIS,_,_=Encoder_Base_Test(img_test2)
        D_K_IR,_,_=Encoder_Detail_Test(img_test1)
        D_K_VIS,_,_=Encoder_Detail_Test(img_test2)
        
    if addition_mode=='Sum':      
        F_b=(B_K_IR+B_K_VIS)
        F_d=(D_K_IR+D_K_VIS)

    elif addition_mode=='Average':
        F_b=(B_K_IR+B_K_VIS)/2         
        F_d=(D_K_IR+D_K_VIS)/2

    elif addition_mode=='l1_norm':
        F_b=l1_addition(B_K_IR,B_K_VIS)
        F_d=l1_addition(D_K_IR,D_K_VIS)
        
    with torch.no_grad():
        Out = Decoder_Test(F_b,F_d)
     
    return output_img(Out) 
