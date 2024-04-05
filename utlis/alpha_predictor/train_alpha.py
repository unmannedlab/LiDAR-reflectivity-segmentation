# -*- coding: utf-8 -*-
"""
Created on Mon May 15 14:40:21 2023

@author: DSE-01
"""

import os
import time
import numpy as np
import alpha_model
import matplotlib.pyplot as plt
import math
from torch.utils.tensorboard import SummaryWriter
from torchviz import make_dot
from torch.utils.data import TensorDataset
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision



writer = SummaryWriter("exp1")
def train():
    root = '/path/to/dataset/'
    #train_list = 'train.txt'
    #val_list = 'val.txt'
    #retrain = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epoch = 100
    cudnn.benchmak = False
    cudnn.deterministic = True
    cudnn.enabled = True
    torch.cuda.set_device(0)
    models = alpha_model.alpha()
    models.to(device)
    batch_size = 1024
    
    #optimizer = torch.optim.SGD(models.parameters(), lr = 0.00001,momentum = 0.9)
    #train_dataset = dataset.dataset(root,train_list)
    #val_dataset = dataset.dataset(root,val_list)
    
    train_data = np.load(root+'train_data.npy',allow_pickle = True)
    train_gt = np.arccos(np.load(root+'train_GT.npy',allow_pickle = True))
    train_data_tensor = torch.Tensor(train_data)
    train_gt_tensor = torch.Tensor(train_gt)
    train_dataset = TensorDataset(train_data_tensor,train_gt_tensor)
    
    val_data = np.load(root+'val_data.npy',allow_pickle = True)
    val_gt = np.arccos(np.load(root+'val_GT.npy',allow_pickle = True))
    val_data_tensor = torch.Tensor(val_data)
    val_gt_tensor = torch.Tensor(val_gt)
    val_dataset = TensorDataset(val_data_tensor,val_gt_tensor)
    
    trainloader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size = batch_size,
                                              num_workers = 2,
                                              shuffle = True,
                                              pin_memory = False,
                                              drop_last = True)
    valloader = torch.utils.data.DataLoader(val_dataset,
                                              batch_size = batch_size,
                                              num_workers = 2,
                                              shuffle = True,
                                              pin_memory = False,
                                              drop_last = True)
    lossfn = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(models.parameters(),lr = 0.01,weight_decay = float(0.025*math.sqrt(batch_size/(len(trainloader)*epoch))))
    #if retrain:
    #    models.load_state_dict(torch.load('path/to/model.pth')['model_state_dict'])
    print(models)
    los = []
    vallo = []
    best_los = 100
    for i in range(epoch):
        print('Epoch :',i)
        running_loss = 0.0
        val_loss = 0.0
        j = 0
        for batch_idx, (data,target) in tqdm(enumerate(trainloader),total = len(trainloader)):
            optimizer.zero_grad()
            models.train()
            data = data.to(device)
            target = target.to(device).unsqueeze(1)
            out = models(data)
            loss = lossfn(out,target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            writer.add_scalar('Loss/train', loss.item(), i)
            j=j+1
        los.append(running_loss/len(trainloader))
        print('train loss:',running_loss/len(trainloader))

        ## validaton
        for batch_idx, (data, target) in tqdm(enumerate(valloader),total = len(valloader)):
            models.eval()
            data = data.to(device)
            target = target.to(device).unsqueeze(1)
            out = models(data)
            loss = lossfn(out,target)
            val_loss += loss.item()
        vallo.append(val_loss/len(valloader))
        #val_r2.append(r2_val/len(valloader))
        writer.add_scalar('Loss/test', loss.item(), i)
        print('val loss',val_loss/len(valloader))
        with open(root+'run3.txt','a') as file:
            file.write('Epoch: '+str(i)+' Training loss = '+str(running_loss/len(trainloader))+' Validation loss = '+str(val_loss/len(valloader))+'\n')
        if val_loss/len(valloader) < best_los:
            torch.save({'epoch': i,
                          'model_state_dict': models.state_dict(),
                          'optimizer_state_dict': optimizer.state_dict(),
                          'loss':val_loss/len(valloader)},root+'/model/best_model_mega_dot.pth')
            best_los = val_loss/len(valloader)
    print(min(vallo))
    plt.plot(list(range(epoch-10)),los[10:],label = 'training loss')
    plt.plot(list(range(epoch-10)),vallo[10:],label = 'validation loss')
    plt.savefig(root+'train_dot.png')
    
train()
