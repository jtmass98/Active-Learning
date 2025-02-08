# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 08:20:42 2025

@author: jtm44
"""

import function_gen as fg
import gps
import numpy as np
import matplotlib.pyplot as plt
import importlib
import torch
#%%
####generate the data
size=100   ##size of the grid 
heats=fg.gen_temps(size,20) #generate the ground truth temperature distribution
inital_samples=1   #the number of temperature readings initially
samples=fg.initial_sample(heats, inital_samples,size)  #measure the initial sample
model = gps.MyGP(samples[0], samples[1])  #create the GP modelling the temperature

##find the maximum and create lists to track progress
max_heats = np.unravel_index(np.argmax(heats), heats.shape)
heatmaps=[]
sample_list=[]
#%%

for k in range (20):
    ###optimise the GP and findthe optimise the expected improvement
    next_x,maxi=gps.next_sample(samples,model,size)
    ##add the new sample to the samples
    samples=fg.increase_data(next_x,samples[0],samples[1],heats)
    ###update the model training data
    model.update_train_data(samples[0], samples[1])
    
    ###generate the heatmaps for tracking progress and  generating animations
    heatmapi=model.heatmap(N)
    heatmaps.append(heatmapi)
    sample_list.append(next_x)
    print('est max: ', maxi)
    print('True max: ',max_heats, heats[max_heats] )
    
    
#%%
###plot the final version and compare with the ground truth

plt.figure()
plt.title('Prediction')
plt.imshow(heatmaps[-1],origin="lower")
plt.scatter(samples[0][:,1].numpy(),samples[0][:,0].numpy(),color='red')
plt.scatter([next_x[1]],[next_x[0]],marker='x')

plt.figure()
plt.title('Truth')
plt.imshow(heats,origin="lower")
plt.scatter(samples[0][:,1].numpy(),samples[0][:,0].numpy(),color='red')
plt.scatter([next_x[1]],[next_x[0]],marker='x')

#%%

##save the step by step figures to create an animation

sample_listx=np.array(sample_list)
plt.imshow(heats, cmap="hot")
plt.savefig('Figures/2/True_heatmap'+'.png')
plt.close()
for i in range(len(heatmaps)):
    plt.figure()
    plt.imshow(heatmaps[i], cmap="hot")
    plt.scatter(sample_listx[:i,1],sample_listx[:i,0])
    plt.savefig('Figures/2/heatmaps'+str(i)+'.png')
    plt.close()
    # np.save('Figures/heatmaps'+str(i)+'.npy',heatmaps[i])
    