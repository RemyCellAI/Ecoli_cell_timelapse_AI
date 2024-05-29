# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 15:17:49 2022

@author: Orange
"""


import numpy as np
import matplotlib.pyplot as plt

def plot_result_crop(images, filenames, bb_contours, data, n_frames, timeinterval, ROI_number, save_path, display_result=True  ):

    for j in range(n_frames):
        file_name = filenames[j]
        img = images[j,:,:]
        n_cells = int(data[j,3,0])
        
        h, w = np.shape(img)
        aspect_ratio = w/h
        
        plt.ioff()
        fig = plt.figure()
        fig.set_size_inches((aspect_ratio*10, 10), forward=False)
        plt.imshow(img, cmap = 'gray', origin='upper')
        plt.axis('off')
        plt.title(file_name)
        
        for k in range(n_cells):
            cell_obj = bb_contours[j][0][0][k,1]
            cell_cl = bb_contours[j][0][0][k,2]
            x_min = bb_contours[j][0][0][k,3]
            x_max = bb_contours[j][0][0][k,4]
            y_min = bb_contours[j][0][0][k,5]
            y_max = bb_contours[j][0][0][k,6]
            
            x_cent = (x_min + x_max)/2
            y_cent = (y_min + y_max)/2
            
            contour_coord = bb_contours[j][1][k]
            cell_length = data[j,4+k,2]
        
            # Plot the ROI box
            ROI_coord_X = [x_min, x_max, x_max, x_min, x_min]
            ROI_coord_Y = [y_min, y_min, y_max, y_max, y_min]
            
            plt.plot(ROI_coord_X, ROI_coord_Y,'r')
    
            # Plot the contour
            plt.plot(contour_coord[:,0] + 1, contour_coord[:,1] + 1,'oy')    
            
            # Cell length:
            plt.plot( [x_min , x_cent], [y_max+9 , y_cent] ,'y')
            plt.text(x_min, y_max+10,  str(np.round(cell_length,decimals=1)) + ' mu', bbox=dict(facecolor='yellow', alpha=0.8), fontsize=24)
            
            # Detection probabilities:
            plt.text(x_min, y_max + 15,  'obj ' + str(np.round(cell_obj,decimals=1)) , bbox=dict(facecolor='yellow', alpha=0.8), fontsize=24)
            plt.text(x_min, y_max + 20,  'cl ' + str(np.round(cell_cl,decimals=1)) , bbox=dict(facecolor='yellow', alpha=0.8), fontsize=24)
    
    
        # Clock:
        hrs = int((j*timeinterval)//60)
        mins = int((j*timeinterval) - hrs * 60)
        plt.text(10, 10, str(hrs) + ' hrs : ' + str(mins) + ' min' , bbox=dict(facecolor='yellow', alpha=0.8), fontsize=44)

                
        plt.savefig(save_path + file_name[0:-4] + '_ROI0' + str(ROI_number) + '.jpg', dpi=100)
        plt.close(fig)
