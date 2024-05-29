# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 18:12:21 2022

@author: Orange
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_channel_ROIs(image_example, channel_boxes_final, channel_prediction_final, channel_ROIs, filename , detected_ROI_image_path ):
    
    x_min = channel_boxes_final[:,0]
    x_max = channel_boxes_final[:,1]
    y_min = channel_boxes_final[:,2]
    y_max = channel_boxes_final[:,3]
    
    x_min_crop = channel_ROIs[:,0]
    x_max_crop = channel_ROIs[:,1]
    y_min_crop = channel_ROIs[:,2]
    y_max_crop = channel_ROIs[:,3]
    
    
    n_channels = len(x_min)
    n_channels_crop = len(x_min_crop)

    plt.ion()
    
    plt.figure()
    plt.imshow(image_example, cmap = 'gray', origin='upper')
    plt.axis('off')
    plt.title(filename)


    for i in range(n_channels):
        
        # Plot the ROI box
        ROI_coord_X = [x_min[i], x_max[i], x_max[i], x_min[i], x_min[i]]
        ROI_coord_Y = [y_min[i], y_min[i], y_max[i], y_max[i], y_min[i]]
        
        plt.plot(ROI_coord_X, ROI_coord_Y,'r')
        plt.text(x_min[i], y_max[i], 'p ' + str(np.round(channel_prediction_final[i],decimals=2)), bbox=dict(facecolor='yellow', alpha=0.5), fontsize=8)

    for i in range(n_channels_crop):
        
        # Plot the ROI box
        ROI_coord_X = [x_min_crop[i], x_max_crop[i], x_max_crop[i], x_min_crop[i], x_min_crop[i]]
        ROI_coord_Y = [y_min_crop[i], y_min_crop[i], y_max_crop[i], y_max_crop[i], y_min_crop[i]]
        
        plt.plot(ROI_coord_X, ROI_coord_Y,'b')
        plt.text(x_min_crop[i], y_max_crop[i], 'ROI ' + str(i), bbox=dict(facecolor='yellow', alpha=0.5), fontsize=8)

    plt.savefig(detected_ROI_image_path + filename[0:-4] + '_ROIs.jpg', dpi=100)
    plt.close()


    