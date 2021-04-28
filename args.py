# -*- coding: utf-8 -*-
"""
@author: Zixiang Zhao (zixiangzhao@stu.xjtu.edu.cn)

Pytorch implement for "Efficient and Model-Based Infrared and Visible Image Fusion via Algorithm Unrolling" (TCSVT 2021)

10.1109/TCSVT.2021.3075745

https://ieeexplore.ieee.org/document/9416456
"""

# =============================================================================
# Hyper-parameter setting 
# =============================================================================
Train_data_choose='FLIR'
if Train_data_choose=='FLIR':
    train_data_path = '.\\Datasets\\Train_data_FLIR\\'
    log_interval = 12
    epochs = 80

train_path = '.\\Train_result\\'
device = "cuda"
channel=64
lr = 1*1e-2
is_cuda = True
img_size=128
layer_numb=10
batch_size=32
# =============================================================================



