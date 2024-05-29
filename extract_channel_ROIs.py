# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 14:50:05 2022

@author: Orange
"""
import numpy as np
import tensorflow as tf

from feature_proposal_utils_march2023 import apply_channel_detection
#from feature_proposal_utils import apply_channel_detection
from YOLO_channel_non_max_suppression import YOLO_channel_non_max_suppression
# from detection_non_max_suppression_20231223 import NMS_objects
from plot_channel_ROIs import plot_channel_ROIs


def get_channel_ROIs(fpn_model, anchors_fpn, fpn_dim, mu_fpn, sigma_fpn, prediction_threshold_fpn, iou_threshold_fpn, img_example, channel_ROI_dims, growth_direction, detected_ROI_image_path, img_example_file_name):
    
    """ Detect the channels"""
    h, w = np.shape(img_example)

       
    # SCALE THE DATA
    X = np.copy(img_example)
    X = np.reshape(X,w * h)
    X[X==0] = 1
    X = np.log(X)
    X = (X-mu_fpn)/(6*sigma_fpn)
    
    X = np.reshape(X,(w,h))
    X = np.expand_dims(X,-1)
    X = np.expand_dims(X,0)
    
    #  Maxpool 512x512 --> 128x128
    X = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(X)
    X = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(X)
    
    # plt.figure()
    # plt.imshow(np.squeeze(X))
    # print(np.shape(X))
    
    # Detect
    object_predictions, ROI_dims = apply_channel_detection(X, fpn_dim, anchors_fpn, fpn_model)
    object_predictions = np.squeeze(object_predictions)
    ROI_dims = np.squeeze(ROI_dims) # [x_c, y_c, width, height]

    # NON MAX SUPPRESION
    channel_prediction_final, channel_boxes_final = YOLO_channel_non_max_suppression(object_predictions, ROI_dims, prediction_threshold_fpn, iou_threshold_fpn)
    # channel_prediction_final, channel_boxes_final = NMS_objects(object_predictions, ROI_dims, prediction_threshold_fpn, iou_threshold_fpn)
    
    
    
    # channel_boxes_final --> ROI_boxes --> [x_min, x_max, y_min, y_max]

    """ Define the final ROIs """
    x_cent = ( channel_boxes_final[:,0] + channel_boxes_final[:,1] ) / 2
    y_cent = ( channel_boxes_final[:,2] + channel_boxes_final[:,3] ) / 2
    
    channel_ROIs = np.ndarray([len(x_cent),4])
    
    channel_ROIs[:,2] = np.round(y_cent) - channel_ROI_dims[0]/2
    channel_ROIs[:,3] = np.round(y_cent) + channel_ROI_dims[0]/2
    
    # if growth_direction == 'left_to_right':
    #     channel_ROIs[:,0] = np.round(x_cent) - np.floor(channel_ROI_dims[1]*3/4)
    #     channel_ROIs[:,1] = np.round(x_cent) + np.ceil(channel_ROI_dims[1]/4)
    # elif growth_direction == 'right_to_left':
    #     channel_ROIs[:,0] = np.round(x_cent) - np.floor(channel_ROI_dims[1]/3)
    #     channel_ROIs[:,1] = np.round(x_cent) + np.ceil(channel_ROI_dims[1]*2/3)
    
    if growth_direction == 'left_to_right':
        channel_ROIs[:,0] = channel_boxes_final[:,0] - 50
        channel_ROIs[:,1] = channel_ROIs[:,0] + 256
    elif growth_direction == 'right_to_left':
        channel_ROIs[:,1] = channel_boxes_final[:,1] + 60
        channel_ROIs[:,0] = channel_ROIs[:,1] - 256
        
    
    """ Filter out ROIs close to borders """
    index_y = (channel_ROIs[:,2] > 0) * (channel_ROIs[:,3] < h)
    channel_ROIs = channel_ROIs[index_y,:]
    
    """ Move the ROIs that reach outside the image """
    index_x1 = (channel_ROIs[:,0] < 0)
    index_x2 = (channel_ROIs[:,1] > w-1)
    channel_ROIs[index_x1,0:2] = [0,channel_ROI_dims[1]]
    channel_ROIs[index_x2,0:2] = [w-channel_ROI_dims[1],w]
    channel_ROIs = channel_ROIs.astype('uint16')
    
    # Sort the ROIs from top to bottom
    y_cent = (channel_ROIs[:,2] + channel_ROIs[:,3])/2
    index = np.argsort(y_cent).astype('uint8')
    channel_ROIs = channel_ROIs[index,:]
    
    
    plot_channel_ROIs(img_example, channel_boxes_final, channel_prediction_final, channel_ROIs, img_example_file_name, detected_ROI_image_path )


    return channel_ROIs


def crop_channels(img_stack, channel_ROIs, channel_ROI_dims):
    frames = np.shape(img_stack)[0]
    n_ROIs = np.shape(channel_ROIs)[0]
    cropped_channels = np.zeros((n_ROIs * frames,channel_ROI_dims[0],channel_ROI_dims[1]), dtype='uint16')
    
    for i in range(n_ROIs):
        start = 0 + i*frames
        end = (i+1)*frames
        cropped_channels[start:end,:,:] = img_stack[:,channel_ROIs[i,2]:channel_ROIs[i,3],channel_ROIs[i,0]:channel_ROIs[i,1]]

    return cropped_channels
    
    
    
    
    
    
    