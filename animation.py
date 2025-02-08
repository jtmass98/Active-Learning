# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 08:13:35 2025

@author: jtm44
"""

import imageio as imageio
import os
import numpy as np

frames=[]
for i in range(31):
    frames.append(imageio.imread('Figures/1/heatmaps'+str(i)+'.png'))





# Parameters  # Change this to your folder
output_gif = "heatmap_animation_1.gif"
duration = 0.1  # Seconds per frame


# Read images and save as GIF
imageio.mimsave(output_gif, frames, duration=duration)

print(f"GIF saved as {output_gif}")