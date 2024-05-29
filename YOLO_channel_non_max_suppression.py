# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 17:57:02 2022

@author: Orange
"""

import numpy as np


def YOLO_channel_non_max_suppression(object_predictions, ROI_dims, cell_threshold, iou_threshold):
    """
    Non Max Supression
    
    INPUT:
    object_predictions      object probabilities
    ROI_boxes:              box details (total number of boxes, 4) --> [x_c, y_c, width, height]

    cell_threshold:         all predictions with a lower probability will be discarded  
    iou_threshold:          all IoUs with a higher value will be discarded
    
    OUTPUT:
    cell_prediction_final:  probabilities of the final selection
    cell_boxes_final:       ROIs of the final selection [x_min, x_max, y_min, y_max] DIFFERENT THAN THE CELL NMS!!!
    """
    
    # Calculate the min max values of the boxes
    xmin = np.zeros((len(ROI_dims),1))
    xmax = np.zeros((len(ROI_dims),1))
    ymin = np.zeros((len(ROI_dims),1))
    ymax = np.zeros((len(ROI_dims),1))
    
    # Calculate the predictions
    prediction = object_predictions # objectivity * class
    
    # ROI_dims --> [x_c, y_c, width, height]
    # ROI_boxes --> [x_min, x_max, y_min, y_max]
    for i in range(len(ROI_dims)):
    
            xmin[i] = ROI_dims[i,0] - ROI_dims[i,2]/2
            xmax[i] = ROI_dims[i,0] + ROI_dims[i,2]/2
            ymin[i] = ROI_dims[i,1] - ROI_dims[i,3]/2
            ymax[i] = ROI_dims[i,1] + ROI_dims[i,3]/2

    ROI_boxes = np.stack((xmin,xmax,ymin,ymax),axis=1)
    ROI_boxes = np.squeeze(ROI_boxes,-1)
    
    # Set the counter
    k = 0

    # Extract the boxes that predict cells only
    cell_index = prediction > cell_threshold    
    cell_prediction = prediction[cell_index]   
    cell_box_dim = ROI_boxes[cell_index,:]

    # Allocate space for the final results
    cell_prediction_final = np.zeros(len(cell_prediction))
    cell_boxes_final = np.zeros([len(cell_prediction),4])

    while len(cell_prediction) > 0: # while there are still cell boxes left in the list

        # Find the box with the max cell prediction and create an array with the rest of the boxes
        max_index = cell_prediction == np.max(cell_prediction)
        
        
        # Make sure there is only 1 max value treated
        true_index = np.argwhere(max_index)
        if (len(true_index) > 1):
            max_index[true_index[1:]] = False
        
        # Extract the max prediction and box
        max_prediction = cell_prediction[max_index]
        max_box = cell_box_dim[max_index,:]
        
        # # Plot for debug
        # from cell_utils import jaccard_coef, get_contour, plot_ROIs, plot_ROIs_and_pred
        # plot_ROIs_and_pred(img, max_box[:,2], max_box[:,3], max_box[:,4], max_box[:,5],max_prediction)

        other_boxes = cell_box_dim
        
        # IoU
        xi1 = np.zeros(len(other_boxes))
        yi1 = np.zeros(len(other_boxes))
        xi2 = np.zeros(len(other_boxes))
        yi2 = np.zeros(len(other_boxes))
        max_box_temp = np.ones([len(other_boxes),4]) * max_box
        
        xi1 = np.amax([max_box_temp[:,0] , other_boxes[:,0]], axis=0)
        yi1 = np.amax([max_box_temp[:,2] , other_boxes[:,2]], axis=0)
        xi2 = np.amin([max_box_temp[:,1] , other_boxes[:,1]], axis=0)
        yi2 = np.amin([max_box_temp[:,3] , other_boxes[:,3]], axis=0)
        
        
        inter_width = xi2 - xi1
        inter_height = yi2 - yi1
        
        # if negative: IoU is zero
        inter_width[inter_width < 0] = 0
        inter_height[inter_height < 0] = 0    
        inter_area = inter_width * inter_height
        
        # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
        ## (≈ 3 lines)
        max_box_area = (max_box[0,1] - max_box[0,0]) * (max_box[0,3] - max_box[0,2])
        other_box_area = (other_boxes[:,1] - other_boxes[:,0]) * (other_boxes[:,3] - other_boxes[:,2])
        union_area = max_box_area + other_box_area - inter_area
        
        # compute the IoU
        iou = inter_area / union_area

   
        # Store the max box results and remove it and the overlapping 
        # boxes from the original list:    
        cell_prediction_final[k] = max_prediction
        cell_boxes_final[k,:] = max_box # DIFFERENT THAN THE CELL NMS WHERE x_cent and y_cent ARE ADDED!!

        # remove_boxes_index = (iou > iou_threshold) + max_index
        keep_boxes_index = (iou <= iou_threshold)
        cell_prediction = cell_prediction[keep_boxes_index]
        cell_box_dim = cell_box_dim[keep_boxes_index]
        
        k += 1
            

    # Remove the allocated space that was not needed
    cell_boxes_final = cell_boxes_final[cell_prediction_final > 0].astype(int) # [x_c, y_c, width, height]
    cell_prediction_final = cell_prediction_final[cell_prediction_final > 0].astype('float64')


    return cell_prediction_final, cell_boxes_final