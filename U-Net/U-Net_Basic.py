#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 16:53:24 2018

@author: eunsook
"""
import torch
import torch.nn as nn
import torch.utils as utils
import torch.nn.init as init
import torch.utils.data as data
import torchvision.utils as v_utils
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.datasets as dset

def conv_block(in_dim, out_dim, act_fn):
    model = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
            act_fn,
    )
    return model

#%%
def conv_trans_block(in_dim, out_dim, act_fn):
    model = nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_dim),
            act_fn,
    )
    return model
#%%
def maxpool():
    pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    return pool
#%%
def conv_block_2(in_dim, out_dim, act_fn):
    model = nn.Sequential(
            conv_block(in_dim, out_dim, act_fn),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim)
    )
    return model
#%%
def conv_block_3(in_dim, out_dim, act_fn):
    model = nn.Sequential(
            conv_block(in_dim, out_dim, act_fn),
            conv_block(out_dim, out_dim, act_fn),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
    )
    return model
#%%
class UnetGenerator(nn.Module):
    
    def __init__(self, in_dim, out_dim, num_filter):
        super(UnetGenerator, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filter = num_filter
        act_fn = nn.LeakyReLU(0.2, inplace=True)
        
        print("\n----------Init U-Net-----------\n")
        
        self.down_1 = conv_block_2(self.in_dim, self.num_filter, act_fn)
        self.pool_1 = maxpool()
        self.down_2 = conv_block_2(self.num_filter*1, self.num_filter*2, act_fn)
        self.pool_2 = maxpool()
        self.down_3 = conv_block_2(self.num_filter*2, self.num_filter*4, act_fn)
        self.pool_3 = maxpool()
        self.down_4 = conv_block_2(self.num_filter*4, self.num_filter*8, act_fn)
        self.pool_4 = maxpool()
        
        self.bridge = conv_block_2(self.num_filter*8, self.num_filter*16, act_fn)
        
        self.trans_1 = conv_trans_block(self.num_filter*16, self.num_filter*8, act_fn)
        self.up_1 = conv_block_2(self.num_filter*16, self.num_filter*8, act_fn)
        self.trans_2 = conv_trans_block(self.num_filter*8, self.num_filter*4, act_fn)
        self.up_2 = conv_block_2(self.num_filter*8, self.num_filter*4, act_fn)
        self.trans_3 = conv_trans_block(self.num_filter*4, self.num_filter*2, act_fn)
        self.up_3 = conv_block_2(self.num_filter*4, self.num_filter*2, act_fn)
        self.trans_4 = conv_trans_block(self.num_filter*2, self.num_filter*1, act_fn)
        self.up_4 = conv_block_2(self.num_filter*2, self.num_filter*1, act_fn)
        
        self.out = nn.Sequential(
            nn.Conv2d(self.num_filter, self.out_dim, 3, 1, 1),
            nn.Tanh(),
        )
        
    def forward(self, input):
        print("\n--------forward-------\n")
        
        down_1 = self.down_1(input)
        pool_1 = self.pool_1(down_1)
        down_2 = self.down_2(pool_1)
        pool_2 = self.pool_2(down_2)
        down_3 = self.down_3(pool_2)
        pool_3 = self.pool_3(down_3)
        down_4 = self.down_4(pool_3)
        pool_4 = self.pool_4(down_4)
        
        bridge = self.bridge(pool_4)
        
        trans_1 = self.trans_1(bridge)
        concat_1 = torch.cat([trans_1, down_4], dim=1)
        up_1 = self.up_1(concat_1)
        trans_2 = self.trans_2(up_1)
        concat_2 = torch.cat([trans_2, down_3], dim=1)
        up_2 = self.up_2(concat_2)
        trans_3 = self.trans_3(up_2)
        concat_3 = torch.cat([trans_3, down_2], dim=1)
        up_3 = self.up_3(concat_3)
        trans_4 = self.trans_4(up_3)
        concat_4 = torch.cat([trans_4, down_1], dim=1)
        up_4 = self.up_4(concat_4)
        
        out = self.out(up_4)
        
        return out
    
#%% main.py

import numpy as np
import matplotlib.pyplot as plt
import argparse
#%%
if __name__ == "__main__":
#%%
    #parser = argparse.ArgumentParser()
    #parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    #parser.add_argument("--num_gpu", type=int, default=1, help="number of gpus")
    #args = parser.parse_args()
    
    #hyperparameters
    batch_size = 1 #args.batch_size
    img_size = 256
    lr = 0.0002
    epoch = 100
    
    #input pipeline
    #image augmentation
    img_dir = "./maps"
    img_data = dset.ImageFolder(root=img_dir, transform = transforms.Compose([
                                                transforms.Scale(size=img_size),
                                                transforms.CenterCrop(size=(img_size,img_size*2)),
                                                transforms.ToTensor(),
                                                ]))
    
    img_batch = data.DataLoader(img_data, batch_size=batch_size, 
                                shuffle=True, num_workers=2)
    
    # initiate Generator
    '''
    if args.network == "unet":
        #generator = nn.DataParallel(UnetGenerator(3, 3, 64), device_ids=[i for i in range(args.num_gpu)]).cuda()
        generator = UnetGenerator(3, 3, 64)
    else:
        generator = UnetGenerator(3, 3, 64)
        print("\n--------not selected--------\n")
    '''
    generator = UnetGenerator(3, 3, 64)
    # loss function & optimizer
    recon_loss_func = nn.MSELoss()
    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
    
    #training
    file = open('./unet_mse_loss', 'w')
    for i in range(epoch):
        for _, (image, label) in enumerate(img_batch):
            satel_image, map_image = torch.chunk(image, chunks=2, dim=3)
            
            gen_optimizer.zero_grad()
            
#            x = Variable(satel_image).cuda(0)
#            y_ = Variable(map_image).cuda(0)
            x = Variable(satel_image)
            y_ = Variable(map_image)
            y = generator.forward(x)
            
            loss = recon_loss_func(y, y_)
            file.write(str(loss)+"\n")
            loss.backward()
            gen_optimizer.step()
            
            if _ % 400 == 0:
                print(i)
                print(loss)
                v_utils.save_image(x.cpu().data, "./result/original_image_{}_{}.png".format(i, _))
                v_utils.save_image(y_.cpu().data, "./result/label_image_{}_{}.png".format(i, _))
                v_utils.save_image(y.cpu().data, "./result/gen_image_{}_{}.png".format(i,_))
                torch.save(generator, './model/{}.pkl'.format("unet"))
        
        