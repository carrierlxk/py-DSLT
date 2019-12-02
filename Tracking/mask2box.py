#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 16:16:37 2019

@author: xiankai
"""

import numpy  as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torch
import json
import matplotlib.patches as patches

def visualize_tracking_result(img, bbox, fig_n):
    """
    visualize tracking result
    """
    fig = plt.figure(fig_n)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    r = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth = 3, edgecolor = "#00ff00", zorder = 1, fill = False)
    ax.imshow(img)
    ax.add_patch(r)
    plt.ion()
    plt.show()
    plt.pause(0.00001)
    plt.clf()
    
im1 = Image.open('/media/xiankai/Data/segmentation/train/Annotations/0a7b27fde9/00025.png')
rgb1 = Image.open('/media/xiankai/Data/segmentation/train/JPEGImages/0a7b27fde9/00025.jpg')
for n in range(1,6):
    #plt.figure(0)
    #plt.imshow(im1)
    im1 = np.array(im1)
    
    index = np.argwhere(im1 == n) #cv2.minAreaRect(im1)#
    if index.size>0:
        rows = index[:, 0]
        clos = index[:, 1]
        # 解析左上角行列号
        left_top_r = np.min(rows)  # y
        left_top_c = np.min(clos)  # x
    
        # 解析右下角行列号
        right_bottom_r = np.max(rows)
        right_bottom_c = np.max(clos)
        location = np.array([left_top_c, left_top_r, right_bottom_c-left_top_c, right_bottom_r-left_top_r])
        visualize_tracking_result(rgb1, location, 1)
        cv2.rectangle(np.array(rgb1), (location[0], location[1]), (location[0]+location[2], location[1]+location[3]), (0, 255, 255), 5)
#plt.figure(1)
#plt.imshow(rgb1)
#visualize_tracking_result(rgb1, location, 1)