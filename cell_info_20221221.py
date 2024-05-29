# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 14:52:00 2022

@author: Orange
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 11:52:57 2022

@author: Orange
"""

# -*- coding: utf-8 -*-
"""
Created on Sat May 14 17:43:05 2022

@author: Orange
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread, imshow



       
from ellipse_fit import ls_ellipse, polyToParams
    
def cell_info(X, Y, measurement_threshold):
    """
    INPUT:
    X = X values of the segmentation wrt the origin of the complete time lapse image
    Y = Y values of the segmentation wrt the origin of the complete time lapse image
    measurement_threshold = fraction of the outer ends used to calculate the mean border
    
    OUTPUT:
    cell_measurement = [yc, xc, cell height, cell length]
    data_rot = cell contour rotated to lay it flat
    angle = angle of the cell (used to rotate the cell contour)
    
    """
    cell_measurement = np.zeros((4),dtype='float32')
    cell_measurement[0] = np.mean(Y)
    cell_measurement[1] = np.mean(X)

    data = np.zeros([len(X),2])
    data[:,1] = X - np.mean(X)
    data[:,0] = Y - np.mean(Y)
    
    # Get the angle
    
    v, flag = ls_ellipse(data[:,1],data[:,0]) # ls_ellipse will return a zero is the determinant was zero
    # if flag == 1: # If the determinant was zeros, the cell will be ignored by keeping the measurements zero:
        
    try:
        ccx,ccy,axesA,axesB,deg,inve = polyToParams(v,printMe=True)
        angle = - deg * np.pi/180
        
        # plt.figure()
        # plt.plot(data[:,0], data[:,1],'s')
        # plt.plot(X ,Y,'r')
        
        rot_mat = np.zeros([2,2])
        rot_mat[0,0] = np.cos(angle)
        rot_mat[0,1] = np.sin(angle)
        rot_mat[1,0] = -np.sin(angle)
        rot_mat[1,1] = np.cos(angle)
        
        data_rot = np.dot(data , rot_mat.T)
        data_rot[:,1] = data_rot[:,1] + np.mean(X) 
        data_rot[:,0] = data_rot[:,0] + np.mean(Y)
        data_rot = data_rot.astype(int)
        
        # data[:,1] = data[:,1] + np.mean(X) 
        # data[:,0] = data[:,0] + np.mean(Y)
    
        cutoff = np.round(measurement_threshold * len(data_rot[:,1])).astype(int)
        sort_data = np.sort(data_rot[:,0])
        cell_measurement[2] = np.mean(sort_data[-cutoff:,]) - np.mean(sort_data[0:cutoff])
        sort_data = np.sort(data_rot[:,1])
        cell_measurement[3] = np.mean(sort_data[-cutoff:,]) - np.mean(sort_data[0:cutoff])
    except:
        print('Could not fit an ellipse in the contour. Singular Matrix? The cell is skipped.')
        data_rot = 0
        angle = 0
    # else:
    #     data_rot = 0
    #     angle = 0


    return cell_measurement, data_rot, angle

def correct_cell_number(results, roi):
    # Correct the number of cells and cell numbers now that cells have been removed from the list:
    frames = np.unique(results[roi][:,1]).astype(int)
    for frame in frames:
        index = results[roi][:,1] == frame
        # n_cells = np.sum(index)
        cell_numbers = np.unique(results[roi][index,3])
        n_cells = len(cell_numbers)
        
        results[roi][index,2] = np.ones((np.sum(index),)) * n_cells
        
        temp = results[roi][index,3]
        for cell_number in range(n_cells):
            cell_index = temp == cell_numbers[cell_number]
            temp[cell_index] = cell_number
        results[roi][index,3] = temp
    return results