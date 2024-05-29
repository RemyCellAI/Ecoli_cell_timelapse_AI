# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 16:38:46 2022

@author: Orange
"""


import tensorflow as tf
import keras
from keras import backend as K
import numpy as np

# UTILS


def fpn_head(y_true, y_pred, img_dim, anchors):
    
    """
    INPUT: 
        Y_pred: model output in parametrized format [none, w, h, f]
        Y_true: ground truth in original image dimensions [none, w, h, f]
        anchors: anchor dimensions used 
    
    """
    
    y_pred = tf.cast(y_pred,tf.double)
    y_true = tf.cast(y_true,tf.double)
    
   
    # PREDICTIONS
    y_pred_list = calculate_predictions(y_pred, anchors, img_dim)
    
    # GROUND TRUTH
    y_true_list = extract_from_true_label(y_true, anchors, img_dim)

    return y_true_list, y_pred_list
    

def fpn_loss_hyper(img_dim, anchors, LAMBDA_COORD, LAMBDA_NOOBJ, LAMBDA_OBJ): 

    def fpn_loss(y_true, y_pred):
        
        y_true_list, y_pred_list = fpn_head(y_true, y_pred, img_dim, anchors)

        p_obj_pred = y_pred_list[:,0]
        p_bkgrd_pred = y_pred_list[:,1]
        bx_pred = y_pred_list[:,2]
        by_pred = y_pred_list[:,3]
        bw_pred = y_pred_list[:,4]
        bh_pred = y_pred_list[:,5]

        
        
        p_obj_true = y_true_list[:,0]
        p_bkgrd_true = y_true_list[:,1]
        bx_true = y_true_list[:,2]
        by_true = y_true_list[:,3]
        bw_true = y_true_list[:,4]
        bh_true = y_true_list[:,5]



        # GET THE DETECTION MASK
        detection_mask = tf.identity(p_obj_true)
        detection_mask_inv = tf.identity(p_bkgrd_true)

        # COORDINATES LOSS
        COORDINATES_LOSS = tf.multiply(LAMBDA_COORD , tf.reduce_sum( tf.multiply(detection_mask , ( tf.sqrt(tf.square(bx_pred - bx_true)) + tf.sqrt(tf.square(by_pred - by_true)) ) )))
    
        # DIMENSIONS LOSS
        DIMENSIONS_LOSS = tf.multiply(LAMBDA_COORD , tf.reduce_sum( tf.multiply(detection_mask , ( tf.sqrt(tf.square(bw_pred - bw_true)) + tf.sqrt(tf.square(bh_pred - bh_true)) ))))
    
        # OBJECTIVENESS LOSS
        OBJECTIVENESS_LOSS = tf.multiply(LAMBDA_OBJ , tf.reduce_sum(tf.multiply(detection_mask , tf.sqrt(tf.square(p_obj_pred - p_obj_true)))))
        
        # NO OBJECT LOSS
        NO_OBJECT_LOSS = tf.multiply(LAMBDA_NOOBJ ,  tf.reduce_sum( tf.multiply(detection_mask_inv , tf.sqrt(tf.square(p_bkgrd_pred - p_bkgrd_true)))) ) # the no obj probability is used independently from the iou
    

        
        total_loss = COORDINATES_LOSS + DIMENSIONS_LOSS + OBJECTIVENESS_LOSS + NO_OBJECT_LOSS
    


        return total_loss
    return fpn_loss



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
    w_a, h_a = anchors[anchor_number,:] ### <---- this should be h_a, w_a, swapped it during training?
    p_obj, p_bkgrd, tx, ty, tw, th = extract_from_label(y_pred,anchor_number)
    y_pred_list = calculate_box_data(p_obj, p_bkgrd, tx, ty, tw, th, grid_x, grid_y, grid_cell_dim_x, grid_cell_dim_y, w_a, h_a)

    return y_pred_list    



def extract_from_true_label(y_true, anchors, img_dim):
    
    anchor_number = 0
    w_a, h_a = anchors[anchor_number,:]
    p_obj, p_bkgrd, bx_true, by_true, bw_true, bh_true = extract_from_label(y_true,anchor_number)
    y_true_list = tf.stack((p_obj, p_bkgrd, bx_true, by_true, bw_true, bh_true),axis=-1)

    return y_true_list

def extract_from_label(y,anchor_number):
    index = anchor_number * 6

    p_obj = y[:,:,:,index + 0]
    p_bkgrd = y[:,:,:,index + 1]
    tx = y[:,:,:,index + 2]
    ty = y[:,:,:,index + 3]
    tw = y[:,:,:,index + 4]
    th = y[:,:,:,index + 5]


    # Flatten the results of the batch
    p_obj = K.flatten(p_obj)
    p_bkgrd = K.flatten(p_bkgrd)
    tx = K.flatten(tx)
    ty = K.flatten(ty)
    tw = K.flatten(tw)
    th = K.flatten(th)

    
    return p_obj, p_bkgrd, tx, ty, tw, th

def calculate_box_data(p_obj, p_bkgrd, tx, ty, tw, th, grid_x, grid_y, grid_cell_dim_x, grid_cell_dim_y, w_a, h_a):
    
    # Calculate boxes
    bx_pred = tf.multiply((K.sigmoid(tx)+ grid_x),grid_cell_dim_x)
    by_pred = tf.multiply((K.sigmoid(ty)+ grid_y),grid_cell_dim_y)
    bw_pred = tf.multiply(w_a , K.exp(tw))
    bh_pred = tf.multiply(h_a , K.exp(th))
    
    # Calculate object and background probabilities
    p = tf.stack([p_obj, p_bkgrd],axis=-1)

    p_pred = tf.keras.activations.softmax(p)

    
    y_pred_list = tf.stack((p_pred[:,0],p_pred[:,1], bx_pred, by_pred, bw_pred, bh_pred),axis=-1)
    
    return y_pred_list

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
        
        y_true_list, y_pred_list = fpn_head(y_true, y_pred, img_dim, anchors)


        bx_pred = y_pred_list[:,2]
        by_pred = y_pred_list[:,3]
        bw_pred = y_pred_list[:,4]
        bh_pred = y_pred_list[:,5]

        
        
        p_obj_true = y_true_list[:,0]

        bx_true = y_true_list[:,2]
        by_true = y_true_list[:,3]
        bw_true = y_true_list[:,4]
        bh_true = y_true_list[:,5]

        
        # IoU's of all grid cells
        iou = iou_from_box_data(bx_true, by_true, bw_true, bh_true, bx_pred, by_pred, bw_pred, bh_pred)
        
        # Evaluate the IoU of the locations of the ground truth only:
        index = tf.where(p_obj_true)
        iou = tf.gather_nd(iou,index)
        
        average_iou = tf.math.reduce_mean(iou)
        return average_iou
    return IoU_metric_simple

def centr_dist_hyper(img_dim, anchors):
   
    def centr_dist(y_true, y_pred):
        
        y_pred = tf.cast(y_pred,tf.double)
        y_true = tf.cast(y_true,tf.double)
        
        y_true_list, y_pred_list = fpn_head(y_true, y_pred, img_dim, anchors)

        bx_pred = y_pred_list[:,2]
        by_pred = y_pred_list[:,3]


        p_obj_true = y_true_list[:,0]
        bx_true = y_true_list[:,2]
        by_true = y_true_list[:,3]

 
        
        delta_dist = tf.sqrt( tf.square(bx_pred - bx_true) + tf.square(by_pred - by_true) )
        
        # Evaluate the IoU of the locations of the ground truth only:
        index = tf.where(p_obj_true)
        delta_dist = tf.gather_nd(delta_dist,index)
        
        average_delta_dist = tf.math.reduce_mean(delta_dist)
        return average_delta_dist
    return centr_dist



def NOOBJ_metric_hyper(img_dim, anchors):
   
    def NOOBJ_metric(y_true, y_pred):
        
        y_pred = tf.cast(y_pred,tf.double)
        y_true = tf.cast(y_true,tf.double)
        
        y_true_list, y_pred_list = fpn_head(y_true, y_pred, img_dim, anchors)

        p_bkgrd_pred = y_pred_list[:,1]
        p_bkgrd_true = y_true_list[:,1]

        
        # GET THE DETECTION MASK
        index = tf.where(p_bkgrd_true)
        NOOBJ_all = tf.gather_nd(p_bkgrd_pred,index)
        NOOBJ = tf.math.reduce_mean(NOOBJ_all)

        return NOOBJ
    return NOOBJ_metric

def OBJ_metric_hyper(img_dim, anchors):
   
    def OBJ_metric(y_true, y_pred):
        
        y_pred = tf.cast(y_pred,tf.double)
        y_true = tf.cast(y_true,tf.double)
        
        y_true_list, y_pred_list = fpn_head(y_true, y_pred, img_dim, anchors)

        p_obj_pred = y_pred_list[:,0]

        p_obj_true = y_true_list[:,0]


        

        index = tf.where(p_obj_true)
        OBJ_all = tf.gather_nd(p_obj_pred,index)
        
        OBJ = tf.math.reduce_mean(OBJ_all)

        return OBJ
    return OBJ_metric


def d_w_hyper(img_dim, anchors):
   
    def d_w(y_true, y_pred):
        
        y_pred = tf.cast(y_pred,tf.double)
        y_true = tf.cast(y_true,tf.double)
        
        y_true_list, y_pred_list = fpn_head(y_true, y_pred, img_dim, anchors)

        p_obj_pred = y_pred_list[:,0]
        p_bkgrd_pred = y_pred_list[:,1]
        bx_pred = y_pred_list[:,2]
        by_pred = y_pred_list[:,3]
        bw_pred = y_pred_list[:,4]
        bh_pred = y_pred_list[:,5]

        
        
        p_obj_true = y_true_list[:,0]
        p_bkgrd_true = y_true_list[:,1]
        bx_true = y_true_list[:,2]
        by_true = y_true_list[:,3]
        bw_true = y_true_list[:,4]
        bh_true = y_true_list[:,5]

        
        delta_w = tf.sqrt( tf.square(bw_pred - bw_true))
        
        # Evaluate the IoU of the locations of the ground truth only:
        index = tf.where(p_obj_true)
        delta_w = tf.gather_nd(delta_w,index)
        
        average_delta_w = tf.math.reduce_mean(delta_w)
        return average_delta_w
    return d_w


def d_h_hyper(img_dim, anchors):
   
    def d_h(y_true, y_pred):
        
        y_pred = tf.cast(y_pred,tf.double)
        y_true = tf.cast(y_true,tf.double)
        
        y_true_list, y_pred_list = fpn_head(y_true, y_pred, img_dim, anchors)

        p_obj_pred = y_pred_list[:,0]
        p_bkgrd_pred = y_pred_list[:,1]
        bx_pred = y_pred_list[:,2]
        by_pred = y_pred_list[:,3]
        bw_pred = y_pred_list[:,4]
        bh_pred = y_pred_list[:,5]

        
        
        p_obj_true = y_true_list[:,0]
        p_bkgrd_true = y_true_list[:,1]
        bx_true = y_true_list[:,2]
        by_true = y_true_list[:,3]
        bw_true = y_true_list[:,4]
        bh_true = y_true_list[:,5]

        
        delta_h = tf.sqrt( tf.square(bh_pred - bh_true))
        
        # Evaluate the IoU of the locations of the ground truth only:
        index = tf.where(p_obj_true)
        delta_h = tf.gather_nd(delta_h,index)
        
        average_delta_h = tf.math.reduce_mean(delta_h)
        return average_delta_h
    return d_h


def apply_channel_detection(X, img_dim, anchors, fpn_model):
    
    # Use the model
    y_pred = fpn_model.predict(X)
    
    # Translate the outcome into predictions
    y_pred_list = calculate_predictions(y_pred, anchors, img_dim)

    # Take slices and reshape:
    m_pred, w_pred, h_pred, d_pred = np.shape(y_pred)
    object_predictions = np.reshape(y_pred_list[:,0],(m_pred, w_pred*h_pred))
    ROI_dims = np.reshape(y_pred_list[:,2:],(m_pred, w_pred*h_pred, 4))  

    return object_predictions, ROI_dims