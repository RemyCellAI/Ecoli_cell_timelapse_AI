# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 21:15:21 2022

@author: Orange
"""

import tensorflow as tf
import keras
from keras import backend as K
import numpy as np

# UTILS


def YOLO_cell_head(y_true, y_pred, img_dim, anchors):
    
    """
    INPUT: 
        Y_pred: model output in parametrized format [none, w, h, f]
        Y_true: ground truth in original image dimensions [none, w, h, f]
        anchors: anchor dimensions used 
    
    """
    m = tf.shape(y_pred)[0] 
    fh = tf.shape(y_pred)[1]
    fw = tf.shape(y_pred)[2]
    
    y_pred = tf.cast(y_pred,tf.double)
    y_true = tf.cast(y_true,tf.double)
    
    grid_cell_dim_x = img_dim[2]/ fw
    grid_cell_dim_y = img_dim[1]/ fh
    
    # PREDICTIONS
    y_pred_list = calculate_boxes_from_output_multiple_anchor(y_pred, anchors, grid_cell_dim_x, grid_cell_dim_y, img_dim)
    
    # GROUND TRUTH
    y_true_list = extract_from_true_label_multiple_anchor(y_true, anchors, img_dim)

    return y_true_list, y_pred_list
    

def YOLO_cell_loss_hyper(img_dim, anchors, LAMBDA_COORD, LAMBDA_NOOBJ, LAMBDA_OBJ): 

    def YOLO_cell_loss(y_true, y_pred):
        
        y_true_list, y_pred_list = YOLO_cell_head(y_true, y_pred, img_dim, anchors)

        p0_pred = y_pred_list[:,0]
        bx_pred = y_pred_list[:,1]
        by_pred = y_pred_list[:,2]
        bw_pred = y_pred_list[:,3]
        bh_pred = y_pred_list[:,4]

        
        
        p0_true = y_true_list[:,0]
        bx_true = y_true_list[:,1]
        by_true = y_true_list[:,2]
        bw_true = y_true_list[:,3]
        bh_true = y_true_list[:,4]

        

        # CALCULATE IOUs
        iou = iou_from_box_data(bx_true, by_true, bw_true, bh_true, bx_pred, by_pred, bw_pred, bh_pred)
        
        # CALCULATE THE PROBABILITIES
        p_pred_iou = p0_pred * iou


        # GET THE DETECTION MASK
        detection_mask = tf.identity(p0_true)
        one_ = tf.constant(1, dtype=tf.float64)
        detection_mask_inv = tf.math.abs(tf.multiply(tf.math.negative(one_) , (tf.identity(p0_true) - one_)))
        

        # COORDINATES LOSS
        COORDINATES_LOSS = tf.multiply(LAMBDA_COORD , tf.reduce_sum( tf.multiply(detection_mask , ( tf.sqrt(tf.square(bx_pred - bx_true)) + tf.sqrt(tf.square(by_pred - by_true)) ) )))
    
        # DIMENSIONS LOSS
        DIMENSIONS_LOSS = tf.multiply(LAMBDA_COORD , tf.reduce_sum( tf.multiply(detection_mask , ( tf.sqrt(tf.square(bw_pred - bw_true)) + tf.sqrt(tf.square(bh_pred - bh_true)) ))))
    
        # OBJECTIVENESS LOSS
        OBJECTIVENESS_LOSS = tf.multiply(LAMBDA_OBJ , tf.reduce_sum(tf.multiply(detection_mask , tf.sqrt(tf.square(p_pred_iou - p0_true)))))
        
        # NO OBJECT LOSS
        NO_OBJECT_LOSS = tf.multiply(LAMBDA_NOOBJ ,  tf.reduce_sum( tf.multiply(detection_mask_inv , tf.sqrt(tf.square(p0_pred - p0_true)))) ) # the no obj probability is used independently from the iou
    
        
        total_loss = COORDINATES_LOSS + DIMENSIONS_LOSS + OBJECTIVENESS_LOSS + NO_OBJECT_LOSS
    


        return total_loss
    return YOLO_cell_loss



def index_for_keras_greater_than(data, threshold):
    threshold_array = tf.multiply(tf.ones_like(data), threshold) # create a tensor all ones
    mask = tf.cast(tf.greater(data, threshold_array),tf.double) # boolean tensor casted into a double tensor
    return tf.where(mask)
    
def index_for_keras_less_than(data, threshold):
    threshold_array = tf.multiply(tf.ones_like(data), threshold) # create a tensor all ones
    mask = tf.cast(tf.less(data, threshold_array),tf.double) # boolean tensor casted into a double tensor
    return tf.where(mask)
    
def index_for_keras_equal(data, value):
    index = tf.cast(tf.math.equal(data,value),tf.double)
    return tf.where(index)

def calculate_boxes_from_output_multiple_anchor(y_pred, anchors, grid_cell_dim_x, grid_cell_dim_y, img_dim):
    
    # GET THE GRID COORDINATES
    m = tf.shape(y_pred)[0] 
    fh = tf.shape(y_pred)[1]
    fw = tf.shape(y_pred)[2]
    
    x = tf.linspace(0, fw-1, fw)
    y = tf.linspace(0, fh-1, fh)
    grid_x, grid_y = tf.meshgrid(x, y)
    grid_x = tf.tile(K.reshape(grid_x, [fw*fh]),[m])
    grid_y = tf.tile(K.reshape(grid_y, [fw*fh]),[m])
    
    # EXTRACT THE PARAMETERS AND CONVERT THE BOX DIMENSIONS TO THE ACTUAL VALUES
    anchor_number = 0
    h_a, w_a = anchors[anchor_number,:]
    p0_pred, tx, ty, tw, th = extract_from_label(y_pred,anchor_number)
    y_pred_list0 = calculate_box_data_training(p0_pred, tx, ty, tw, th, grid_x, grid_y, grid_cell_dim_x, grid_cell_dim_y, w_a, h_a)

    anchor_number = 1
    h_a, w_a = anchors[anchor_number,:]
    p0_pred, tx, ty, tw, th = extract_from_label(y_pred,anchor_number)
    y_pred_list1 = calculate_box_data_training(p0_pred, tx, ty, tw, th, grid_x, grid_y, grid_cell_dim_x, grid_cell_dim_y, w_a, h_a)

    anchor_number = 2
    h_a, w_a = anchors[anchor_number,:]
    p0_pred, tx, ty, tw, th = extract_from_label(y_pred,anchor_number)
    y_pred_list2 = calculate_box_data_training(p0_pred, tx, ty, tw, th, grid_x, grid_y, grid_cell_dim_x, grid_cell_dim_y, w_a, h_a)

    anchor_number = 3
    h_a, w_a = anchors[anchor_number,:]
    p0_pred, tx, ty, tw, th = extract_from_label(y_pred,anchor_number)
    y_pred_list3 = calculate_box_data_training(p0_pred, tx, ty, tw, th, grid_x, grid_y, grid_cell_dim_x, grid_cell_dim_y, w_a, h_a)

    # print('y_pred_list0 : ' )
    # print(tf.shape(y_pred_list0))
    # print('y_pred_list1 : ')
    # print(tf.shape(y_pred_list1))
    # print('y_pred_list2 : ')
    # print(tf.shape(y_pred_list2))
    # print('y_pred_list3 : ')
    # print(tf.shape(y_pred_list3))
    
    y_pred_list = tf.concat([y_pred_list0, y_pred_list1,y_pred_list2,y_pred_list3],0)

    #for i in range(len(anchors)-1):
     # anchor_number = i + 1
     # h_a, w_a = anchors[anchor_number,:]
     # p0_pred, tx, ty, tw, th, c_pred = extract_from_label(y_pred,anchor_number)
     # y_pred_list_ = calculate_box_data(p0_pred, tx, ty, tw, th, c_pred, grid_x, grid_y, grid_cell_dim_x, grid_cell_dim_y, w_a, h_a)
     # y_pred_list = tf.concat([y_pred_list,y_pred_list_],0)

    return y_pred_list    



def extract_from_true_label_multiple_anchor(y_true, anchors, img_dim):

    anchor_number = 0
    p0_true, bx_true, by_true, bw_true, bh_true = extract_from_label(y_true, anchor_number)
    y_true_list0 = tf.stack((p0_true, bx_true, by_true, bw_true, bh_true),axis=-1)

    anchor_number = 1
    p0_true, bx_true, by_true, bw_true, bh_true = extract_from_label(y_true, anchor_number)
    y_true_list1 = tf.stack((p0_true, bx_true, by_true, bw_true, bh_true),axis=-1)

    anchor_number = 2
    p0_true, bx_true, by_true, bw_true, bh_true = extract_from_label(y_true, anchor_number)
    y_true_list2 = tf.stack((p0_true, bx_true, by_true, bw_true, bh_true),axis=-1)

    anchor_number = 3
    p0_true, bx_true, by_true, bw_true, bh_true = extract_from_label(y_true, anchor_number)
    y_true_list3 = tf.stack((p0_true, bx_true, by_true, bw_true, bh_true),axis=-1)

    y_true_list = tf.concat([y_true_list0, y_true_list1, y_true_list2, y_true_list3],0)


    #for i in range(len(anchors)-1):
     # anchor_number = i + 1
     # p0_true, bx_true, by_true, bw_true, bh_true, c_true = extract_from_label(y_true, anchor_number)
     # y_true_list_ = tf.stack((p0_true, bx_true, by_true, bw_true, bh_true, c_true),axis=-1)
     # y_true_list = tf.concat([y_true_list, y_true_list_],0)

    return y_true_list

def extract_from_label(y,anchor_number):
    index = anchor_number * 5

    p0_pred = y[:,:,:,index + 0]
    tx = y[:,:,:,index + 1]
    ty = y[:,:,:,index + 2]
    tw = y[:,:,:,index + 3]
    th = y[:,:,:,index + 4]

    
         
    # Flatten the results of the batch
    p0_pred = K.flatten(p0_pred)
    tx = K.flatten(tx)
    ty = K.flatten(ty)
    tw = K.flatten(tw)
    th = K.flatten(th)

    
    return p0_pred, tx, ty, tw, th


def calculate_box_data_training(p0_pred, tx, ty, tw, th, grid_x, grid_y, grid_cell_dim_x, grid_cell_dim_y, w_a, h_a):
    
    """
    Here the object prediction is stacked together with the box prediction. This is how the list is used for the loss function during training.
    """

    # Calculate boxes
    bx_pred = tf.multiply((K.sigmoid(tx)+ grid_x),grid_cell_dim_x)
    by_pred = tf.multiply((K.sigmoid(ty)+ grid_y),grid_cell_dim_y)
    bw_pred = tf.multiply(w_a , K.exp(tw))
    bh_pred = tf.multiply(h_a , K.exp(th))

    # Calculate objectiveness and class
    p0_pred = K.sigmoid(p0_pred)


    y_pred_list = tf.stack((p0_pred, bx_pred, by_pred, bw_pred, bh_pred),axis=-1)

    return y_pred_list

def calculate_box_data(p0_pred, tx, ty, tw, th, grid_x, grid_y, grid_cell_dim_x, grid_cell_dim_y, w_a, h_a):
    """
    Here the object prediction is output seperately from the box prediction. In the class function for data processing they are used seperately for the NMS script.
    """
    # Calculate boxes
    bx_pred = tf.multiply((K.sigmoid(tx)+ grid_x),grid_cell_dim_x)
    by_pred = tf.multiply((K.sigmoid(ty)+ grid_y),grid_cell_dim_y)
    bw_pred = tf.multiply(w_a , K.exp(tw))
    bh_pred = tf.multiply(h_a , K.exp(th))
    
    # Calculate objectiveness and class
    p0_pred = K.sigmoid(p0_pred)

    
    box_data = tf.stack((bx_pred, by_pred, bw_pred, bh_pred),axis=-1)
    
    return p0_pred, box_data

def iou_from_box_data(bx_true, by_true, bw_true, bh_true, bx_pred, by_pred, bw_pred, bh_pred):
    # Get the xmin, xmax, ymin, ymax for the iou step
    xmin = bx_pred - tf.math.divide_no_nan(bw_pred , 2)
    xmax = bx_pred + tf.math.divide_no_nan(bw_pred , 2)
    ymin = by_pred - tf.math.divide_no_nan(bh_pred , 2)
    ymax = by_pred + tf.math.divide_no_nan(bh_pred , 2) 
    
    xmin_true = bx_true - tf.math.divide_no_nan(bw_true , 2)   
    xmax_true = bx_true + tf.math.divide_no_nan(bw_true , 2)  
    ymin_true = by_true - tf.math.divide_no_nan(bh_true , 2)  
    ymax_true = by_true + tf.math.divide_no_nan(bh_true , 2)  
    
    # IoU
    
    xi1 = tf.stack([xmin_true , xmin], axis=-1)
    xi2 = tf.stack([xmax_true , xmax], axis=-1)
    yi1 = tf.stack([ymin_true , ymin], axis=-1)
    yi2 = tf.stack([ymax_true , ymax], axis=-1)
    
    xi1 = tf.reduce_max(xi1,axis=1)  
    xi2 = tf.reduce_min(xi2,axis=1)
    yi1 = tf.reduce_max(yi1,axis=1)
    yi2 = tf.reduce_min(yi2,axis=1)

    inter_width = xi2 - xi1
    inter_height = yi2 - yi1
    
    zeros = tf.zeros_like(inter_width) # create a tensor all ones
    mask = tf.cast(tf.greater(inter_width, zeros),tf.double) # boolean tensor casted into a double tensor
    inter_width = tf.multiply(inter_width, mask) + zeros
       
    zeros = tf.zeros_like(inter_height) # create a tensor all ones
    mask = tf.cast(tf.greater(inter_height, zeros),tf.double) # boolean tensor casted into a double tensor
    inter_height = tf.multiply(inter_height, mask) + zeros
    
    ####

    inter_area = tf.multiply(inter_width, inter_height)   

    box_area = tf.multiply((xmax - xmin) , (ymax - ymin))
    box_area_true = tf.multiply((xmax_true - xmin_true) , (ymax_true - ymin_true))

    union_area = box_area_true + box_area - inter_area
    
    # compute the IoU
    return tf.math.divide_no_nan( inter_area , union_area ) 


def IoU_metric_simple_hyper(img_dim, anchors):
   
    def IoU_metric_simple(y_true, y_pred):
        
        y_pred = tf.cast(y_pred,tf.double)
        y_true = tf.cast(y_true,tf.double)
        
        y_true_list, y_pred_list = YOLO_cell_head(y_true, y_pred, img_dim, anchors)

        bx_pred = y_pred_list[:,1]
        by_pred = y_pred_list[:,2]
        bw_pred = y_pred_list[:,3]
        bh_pred = y_pred_list[:,4]
        
        p0_true = y_true_list[:,0]
        bx_true = y_true_list[:,1]
        by_true = y_true_list[:,2]
        bw_true = y_true_list[:,3]
        bh_true = y_true_list[:,4]

        
        # IoU's of all grid cells
        iou = iou_from_box_data(bx_true, by_true, bw_true, bh_true, bx_pred, by_pred, bw_pred, bh_pred)
        
        # Evaluate the IoU of the locations of the ground truth only:
        index = tf.where(p0_true)
        iou = tf.gather_nd(iou,index)
        
        average_iou = tf.math.reduce_mean(iou)
        return average_iou
    return IoU_metric_simple

def centr_dist_hyper(img_dim, anchors):
   
    def centr_dist(y_true, y_pred):
        
        y_pred = tf.cast(y_pred,tf.double)
        y_true = tf.cast(y_true,tf.double)
        
        y_true_list, y_pred_list = YOLO_cell_head(y_true, y_pred, img_dim, anchors)

        p_pred = y_pred_list[:,0]
        bx_pred = y_pred_list[:,1]
        by_pred = y_pred_list[:,2]
        bw_pred = y_pred_list[:,3]
        bh_pred = y_pred_list[:,4]

        
        
        p0_true = y_true_list[:,0]
        bx_true = y_true_list[:,1]
        by_true = y_true_list[:,2]
        bw_true = y_true_list[:,3]
        bh_true = y_true_list[:,4]

        
        delta_dist = tf.sqrt( tf.square(bx_pred - bx_true) + tf.square(by_pred - by_true) )
        
        # Evaluate the IoU of the locations of the ground truth only:
        index = tf.where(p0_true)
        delta_dist = tf.gather_nd(delta_dist,index)
        
        average_delta_dist = tf.math.reduce_mean(delta_dist)
        return average_delta_dist
    return centr_dist



def NOOBJ_metric_hyper(img_dim, anchors):
   
    def NOOBJ_metric(y_true, y_pred):
        
        y_pred = tf.cast(y_pred,tf.double)
        y_true = tf.cast(y_true,tf.double)
        
        y_true_list, y_pred_list = YOLO_cell_head(y_true, y_pred, img_dim, anchors)

        p0_pred = y_pred_list[:,0]
        p0_true = y_true_list[:,0]

        
        # GET THE DETECTION MASK
        one_ = tf.constant(1, dtype=tf.float64)
        detection_mask_inv = tf.math.abs(tf.multiply(tf.math.negative(one_) , (tf.identity(p0_true) - one_)))
        
        index = tf.where(detection_mask_inv)
        NOOBJ_all = tf.gather_nd(p0_pred,index)
        NOOBJ = tf.math.reduce_mean(NOOBJ_all)

        return NOOBJ
    return NOOBJ_metric

def OBJ_metric_hyper(img_dim, anchors):
   
    def OBJ_metric(y_true, y_pred):
        
        y_pred = tf.cast(y_pred,tf.double)
        y_true = tf.cast(y_true,tf.double)
        
        y_true_list, y_pred_list = YOLO_cell_head(y_true, y_pred, img_dim, anchors)

        p0_pred = y_pred_list[:,0]
        p0_true = y_true_list[:,0]

        

        index = tf.where(p0_true)
        OBJ_all = tf.gather_nd(p0_pred,index)
        
        OBJ = tf.math.reduce_mean(OBJ_all)

        return OBJ
    return OBJ_metric


def d_w_hyper(img_dim, anchors):
   
    def d_w(y_true, y_pred):
        
        y_pred = tf.cast(y_pred,tf.double)
        y_true = tf.cast(y_true,tf.double)
        
        y_true_list, y_pred_list = YOLO_cell_head(y_true, y_pred, img_dim, anchors)

        p_pred = y_pred_list[:,0]
        bx_pred = y_pred_list[:,1]
        by_pred = y_pred_list[:,2]
        bw_pred = y_pred_list[:,3]
        bh_pred = y_pred_list[:,4]
        
        
        p0_true = y_true_list[:,0]
        bx_true = y_true_list[:,1]
        by_true = y_true_list[:,2]
        bw_true = y_true_list[:,3]
        bh_true = y_true_list[:,4]

        
        delta_w = tf.sqrt( tf.square(bw_pred - bw_true))
        
        # Evaluate the IoU of the locations of the ground truth only:
        index = tf.where(p0_true)
        delta_w = tf.gather_nd(delta_w,index)
        
        average_delta_w = tf.math.reduce_mean(delta_w)
        return average_delta_w
    return d_w


def d_h_hyper(img_dim, anchors):
   
    def d_h(y_true, y_pred):
        
        y_pred = tf.cast(y_pred,tf.double)
        y_true = tf.cast(y_true,tf.double)
        
        y_true_list, y_pred_list = YOLO_cell_head(y_true, y_pred, img_dim, anchors)

        p_pred = y_pred_list[:,0]
        bx_pred = y_pred_list[:,1]
        by_pred = y_pred_list[:,2]
        bw_pred = y_pred_list[:,3]
        bh_pred = y_pred_list[:,4]
        
        
        p0_true = y_true_list[:,0]
        bx_true = y_true_list[:,1]
        by_true = y_true_list[:,2]
        bw_true = y_true_list[:,3]
        bh_true = y_true_list[:,4]

        
        delta_h = tf.sqrt( tf.square(bh_pred - bh_true))
        
        # Evaluate the IoU of the locations of the ground truth only:
        index = tf.where(p0_true)
        delta_h = tf.gather_nd(delta_h,index)
        
        average_delta_h = tf.math.reduce_mean(delta_h)
        return average_delta_h
    return d_h

def calculate_predictions(y_pred, anchors, img_dim):
    
    # GET THE GRID COORDINATES
    m = tf.shape(y_pred)[0] 
    fh = tf.shape(y_pred)[1]
    fw = tf.shape(y_pred)[2]
    
    grid_cell_dim_x = tf.cast(img_dim[1]/ fw ,dtype=tf.float32)
    grid_cell_dim_y = tf.cast(img_dim[0]/ fh ,dtype=tf.float32)
    
    x = tf.linspace(0, fw-1, fw)
    y = tf.linspace(0, fh-1, fh)
    grid_x, grid_y = tf.cast(tf.meshgrid(x, y),dtype=tf.float32)
    grid_x = tf.cast(tf.tile(K.reshape(grid_x, [fw*fh]),[m]),dtype=tf.float32)
    grid_y = tf.cast(tf.tile(K.reshape(grid_y, [fw*fh]),[m]),dtype=tf.float32)
    
    # EXTRACT THE PARAMETERS AND CONVERT THE BOX DIMENSIONS TO THE ACTUAL VALUES
    anchor_number = 0
    h_a, w_a = anchors[anchor_number,:]
    p0_pred, tx, ty, tw, th = extract_from_label(y_pred,anchor_number)
    object_predictions_anchor0, ROI_dims_anchor0 = calculate_box_data(p0_pred, tx, ty, tw, th, grid_x, grid_y, grid_cell_dim_x, grid_cell_dim_y, w_a, h_a)

    anchor_number = 1
    h_a, w_a = anchors[anchor_number,:]
    p0_pred, tx, ty, tw, th = extract_from_label(y_pred,anchor_number)
    object_predictions_anchor1, ROI_dims_anchor1  = calculate_box_data(p0_pred, tx, ty, tw, th, grid_x, grid_y, grid_cell_dim_x, grid_cell_dim_y, w_a, h_a)
    
    anchor_number = 2
    h_a, w_a = anchors[anchor_number,:]
    p0_pred, tx, ty, tw, th = extract_from_label(y_pred,anchor_number)
    object_predictions_anchor2, ROI_dims_anchor2  = calculate_box_data(p0_pred, tx, ty, tw, th, grid_x, grid_y, grid_cell_dim_x, grid_cell_dim_y, w_a, h_a)
    
    anchor_number = 3
    h_a, w_a = anchors[anchor_number,:]
    p0_pred, tx, ty, tw, th = extract_from_label(y_pred,anchor_number)
    object_predictions_anchor3, ROI_dims_anchor3  = calculate_box_data(p0_pred, tx, ty, tw, th, grid_x, grid_y, grid_cell_dim_x, grid_cell_dim_y, w_a, h_a)
    
    # y_pred_list = tf.concat([y_pred_list0, y_pred_list1,y_pred_list2,y_pred_list3],-1)


    return object_predictions_anchor0, ROI_dims_anchor0, object_predictions_anchor1, ROI_dims_anchor1, object_predictions_anchor2, ROI_dims_anchor2, object_predictions_anchor3, ROI_dims_anchor3


def apply_cell_detection(X, anchors, img_dim, detection_model):
    
    # Use the model
    y_pred = detection_model.predict(X)
    print('predicted stack dimensions: ' + str(np.shape(y_pred)))
    # Translate the outcome into predictions
    object_predictions_anchor0, ROI_dims_anchor0, object_predictions_anchor1, ROI_dims_anchor1, object_predictions_anchor2, ROI_dims_anchor2, object_predictions_anchor3, ROI_dims_anchor3 = calculate_predictions(y_pred, anchors, img_dim)

    # Take slices and reshape:
    # m_pred, h_pred, w_pred, d_pred = np.shape(y_pred)
    # n_anchors = np.shape(anchors)[0]
    
    # object_predictions = np.stack((y_pred_list[:,0],y_pred_list[:,5],y_pred_list[:,10],y_pred_list[:,15]),-1)
    # ROI_dims = np.stack((y_pred_list[:,1:5],y_pred_list[:,6:10],y_pred_list[:,11:15],y_pred_list[:,16:20]),-1)

    return object_predictions_anchor0, ROI_dims_anchor0, object_predictions_anchor1, ROI_dims_anchor1, object_predictions_anchor2, ROI_dims_anchor2, object_predictions_anchor3, ROI_dims_anchor3