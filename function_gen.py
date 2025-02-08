# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 08:08:49 2025

@author: jtm44
"""

import numpy as np
import matplotlib.pyplot as plt
import torch

def heat_source(N,x,y,T,delta):
    heat=np.zeros((N,N))
    centre=np.array([x,y]).reshape(2,)
    for i in range(N):
        for j in range(N):
            coord=np.array([i,j]).reshape(2,)
            d=np.sum(np.square(coord-centre))
            heat[i,j]=T*np.exp(-d/delta**2)
    return heat

def gen_temps(N,heats):
    heat=np.zeros((N,N))
    for i in range(heats):
        heats+=heat_source(N,np.random.rand()*N,np.random.rand()*N,np.random.rand()*80+20,20)
    return heats

def initial_sample(heats,inital_samples,size):
    x=np.random.randint(0,size,(inital_samples,2))
    y=heats[x[:,0],x[:,1]].reshape(-1,)
    return [torch.Tensor(x),torch.Tensor(y)]

def increase_data(next_x,train_x,train_y,heats):
    next_x=torch.Tensor([int(i) for i in next_x]).reshape(1,2)

    train_x = torch.cat((train_x,next_x), dim=0)

    train_x = torch.unique(train_x, dim=0, return_counts=False)

    train_y=torch.Tensor(heats[train_x[:,0].numpy().astype(int),train_x[:,1].numpy().astype(int)])
    
    return [train_x,train_y]
    




