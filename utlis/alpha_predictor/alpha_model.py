# -*- coding: utf-8 -*-
"""
Created on Mon May 15 14:24:46 2023

@author: DSE-01
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class alpha(nn.Module):  # sub-classing nn.module class
    def __init__(self):
        super(alpha,self).__init__()  
        self.l1 = nn.Linear(3,3)
        self.l2 = nn.Linear(3,16)
        self.l3 = nn.Linear(16,32)
        self.l4 = nn.Linear(32,64)
        self.l5 = nn.Linear(64,1)
    
    def forward(self,inp):
        l1_out = torch.tanh(self.l1(inp))
        l2_out = torch.tanh(self.l2(l1_out))
        l3_out = torch.tanh(self.l3(l2_out))
        #l3_out = self.dp(l3_out)
        l4_out = F.relu(self.l4(l3_out))
        l5_out = F.relu(self.l5(l4_out))
        
        #f_out = torch.cos(l5_out)
        
        return l5_out
        