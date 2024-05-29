# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 18:00:07 2023

@author: Orange
"""

import numpy as np

def NMS_objects(object_predictions, ROI_dims, prediction_threshold, iou_threshold):
    # Calculate the min max values of the boxes
    # xmin = np.zeros((len(ROI_dims),1))
    # xmax = np.zeros((len(ROI_dims),1))
    # ymin = np.zeros((len(ROI_dims),1))
    # ymax = np.zeros((len(ROI_dims),1))
    
    
    # Calculate the bounding boxes
    # ROI_dims --> [x_c, y_c, width, height]
    # ROI_boxes --> [x_min, x_max, y_min, y_max]
    
    xmin = ROI_dims[:,0] - ROI_dims[:,2]/2
    xmax = ROI_dims[:,0] + ROI_dims[:,2]/2
    ymin = ROI_dims[:,1] - ROI_dims[:,3]/2
    ymax = ROI_dims[:,1] + ROI_dims[:,3]/2
    
    ROI_boxes = np.stack((xmin,xmax,ymin,ymax),axis=1)
    
    
    # Extract the boxes that predict cells only
    cell_index = object_predictions > prediction_threshold
    obj_prediction = object_predictions[cell_index]
    bb_dim = ROI_boxes[cell_index,:]
    
    # Allocate space for the final results
    obj_prediction_final = np.zeros(len(obj_prediction)).astype('float32')
    bb_final = np.zeros([len(obj_prediction),6]).astype('float32') # x_cent, y_cent, x_min, x_max, y_min, y_max
    # Set the counter
    k = 0
    while len(obj_prediction) > 0: # while there are still cell boxes left in the list
    
        # Find the box with the max cell prediction and create an array with the rest of the boxes
        max_index = obj_prediction == np.max(obj_prediction)
        
        
        # Make sure there is only 1 max value treated
        true_index = np.argwhere(max_index)
        if (len(true_index) > 1):
            max_index[true_index[1:]] = False
        
        # Extract the max prediction and box
        max_prediction = obj_prediction[max_index][0]
        max_box = bb_dim[max_index,:][0]
        
        # Remove this max ROI from the list:
        obj_prediction = obj_prediction[~max_index]
        bb_dim =  bb_dim[~max_index,:]
        
        
        # IoU
        xi1 = np.zeros(len(bb_dim))
        yi1 = np.zeros(len(bb_dim))
        xi2 = np.zeros(len(bb_dim))
        yi2 = np.zeros(len(bb_dim))
        max_box_temp = np.ones([len(bb_dim),4]) * max_box
        
        # Calculate the overlap of max_box and the rest:
        xi1 = np.amax([max_box_temp[:,0] , bb_dim[:,0]], axis=0)
        yi1 = np.amax([max_box_temp[:,2] , bb_dim[:,2]], axis=0)
        xi2 = np.amin([max_box_temp[:,1] , bb_dim[:,1]], axis=0)
        yi2 = np.amin([max_box_temp[:,3] , bb_dim[:,3]], axis=0)
        
        inter_width = xi2 - xi1
        inter_height = yi2 - yi1
        
        # When the bounding boxes do not overlap, inter_width or inter_height becomes negative.
        inter_width[inter_width < 0] = 0
        inter_height[inter_height < 0] = 0
        inter_area = inter_width * inter_height

        
        # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
        max_box_area = (max_box[1] - max_box[0]) * (max_box[3] - max_box[2])
        other_box_area = (bb_dim[:,1] - bb_dim[:,0]) * (bb_dim[:,3] - bb_dim[:,2])
        union_area = max_box_area + other_box_area - inter_area
        
        # Compute the IoU
        iou = inter_area / union_area
        
        # If the current bounding box is completely covered by another bounding box, set the iou = 1:
        complete_overlap_index = inter_area == max_box_area
        iou[complete_overlap_index] = 1
            
       
        # Store the max box results and remove it and the overlapping 
        # boxes from the original list:    
        obj_prediction_final[k] = max_prediction
        
        bb_final[k,0] = (max_box[0] + max_box[1])/2 # x_cent
        bb_final[k,1] = (max_box[2] + max_box[3])/2 # y_cent
        bb_final[k,2:] = max_box
    
        # Filter out the bounding boxes that exceed the overlap threshold:
        keep_boxes_index = (iou <= iou_threshold)
        obj_prediction = obj_prediction[keep_boxes_index]
        bb_dim = bb_dim[keep_boxes_index,:]
        
        k += 1
            
    
    # Remove the allocated space that was not needed
    bb_final = bb_final[obj_prediction_final > 0].astype(int) # --> [x_cent, y_cent, x_min, x_max, y_min, y_max]
    obj_prediction_final = obj_prediction_final[obj_prediction_final > 0].astype('float32')

    return obj_prediction_final, bb_final
