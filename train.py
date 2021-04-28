# -*- coding: utf-8 -*-
"""
@author: Zixiang Zhao (zixiangzhao@stu.xjtu.edu.cn)

Pytorch implement for "Efficient and Model-Based Infrared and Visible Image Fusion via Algorithm Unrolling" (TCSVT 2021)

10.1109/TCSVT.2021.3075745

https://ieeexplore.ieee.org/document/9416456
"""
import torchvision
from torchvision import transforms
import torchvision.utils as vutils
import numpy as np
import torch
import time
from torch import nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import os
import scipy.io as scio
import kornia
from AUIFNet import Encoder_Base,Encoder_Detail,Decoder
from args import train_data_path,train_path,device,batch_size,channel,lr,is_cuda,log_interval,img_size,layer_numb,epochs

Train_Image_Number=len(os.listdir(train_data_path+'FLIR\\'))
Iter_per_epoch=(Train_Image_Number % batch_size!=0)+Train_Image_Number//batch_size
# =============================================================================
# Preprocessing and dataset establishment  
# =============================================================================
transforms = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.Grayscale(1),
        transforms.ToTensor(),
        ])
                          
Data = torchvision.datasets.ImageFolder(train_data_path,transform=transforms)
dataloader = torch.utils.data.DataLoader(Data, batch_size,shuffle=True)
# =============================================================================
# Models
# =============================================================================
Encoder_Base_Train=Encoder_Base()
Encoder_Detail_Train=Encoder_Detail()
Decoder_Train=Decoder()

if is_cuda:
    Encoder_Base_Train=Encoder_Base_Train.cuda()
    Encoder_Detail_Train=Encoder_Detail_Train.cuda()
    Decoder_Train=Decoder_Train.cuda()
 
 
print(Encoder_Base_Train)
print(Encoder_Detail_Train)
print(Decoder_Train)


optimizer1 = optim.Adam(Encoder_Base_Train.parameters(), lr = lr)
optimizer2 = optim.Adam(Encoder_Detail_Train.parameters(), lr =0.1* lr)
optimizer3 = optim.Adam(Decoder_Train.parameters(), lr = lr)


scheduler1 = torch.optim.lr_scheduler.MultiStepLR(
        optimizer1, [41, 81], gamma=0.1)
scheduler2 = torch.optim.lr_scheduler.MultiStepLR(
        optimizer2, [41, 81], gamma=0.1)
scheduler3 = torch.optim.lr_scheduler.MultiStepLR(
        optimizer3, [41, 81], gamma=0.1)

MSELoss = nn.MSELoss()
SmoothL1Loss=nn.SmoothL1Loss()
L1Loss=nn.L1Loss()
SSIMLoss = kornia.losses.SSIM(3, reduction='mean')

# =============================================================================
# Training
# =============================================================================
print('============ Training Begins ===============')
print('The total number of images is %d,\n Need to cycle %d times.'%(Train_Image_Number,Iter_per_epoch))

name=['Encoder_Base_Train','Encoder_Detail_Train','Decoder_Train']

for iteration in range(epochs):

    Encoder_Base_Train.train()
    Encoder_Detail_Train.train()
    Decoder_Train.train()
    
   
    data_iter_input = iter(dataloader)
    
    for step in range(Iter_per_epoch):
        img_input,_ =next(data_iter_input)
        
          
        if is_cuda:
            img_input=img_input.cuda()
        
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        # =====================================================================
        # Calculate loss  
        # =====================================================================
        B_K,eta_B,theta_B=Encoder_Base_Train(img_input)
        D_K,eta_D,theta_D=Encoder_Detail_Train(img_input)
        img_recon=Decoder_Train(B_K,D_K)
        # Total loss
        mse=MSELoss(img_input,img_recon)
        ssim=SSIMLoss(img_input,img_recon)
        loss = mse + 5*ssim
        # Update
        loss.backward()
        optimizer1.step() 
        optimizer2.step()
        optimizer3.step()
        # =====================================================================
        # Print 
        # =====================================================================
        los = loss.item()
        mse_l=mse.item()
        ssim_l=ssim.item()
        if (step + 1) % log_interval == 0:          
            print('Epoch/step: %d/%d, loss: %.7f, lr: %f' %(iteration+1, step+1, los, optimizer1.state_dict()['param_groups'][0]['lr']))
            print('MSELoss:%.7f\nSSIMLoss:%.7f'%(mse_l,ssim_l))

    scheduler1.step()
    scheduler2.step()
    scheduler3.step()

# Save models
Encoder_Base_Train.eval()
Encoder_Base_Train.cpu()
Encoder_Detail_Train.eval()
Encoder_Detail_Train.cpu()
Decoder_Train.eval()
Decoder_Train.cpu()

name=['Encoder_Base_Train','Encoder_Detail_Train','Decoder_Train']
for i in range(3):
    save_model_filename =str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" +name[i] + ".model"
    save_model_path = os.path.join(train_path, save_model_filename)
    if i == 0:
        torch.save(Encoder_Base_Train.state_dict(), save_model_path)
    elif i == 1:
        torch.save(Encoder_Detail_Train.state_dict(), save_model_path)
    elif i == 2:
        torch.save(Decoder_Train.state_dict(), save_model_path)
    print("\nDone, trained model saved at", save_model_path)

 