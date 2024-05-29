# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 15:05:21 2022

@author: Orange
"""

import tensorflow as tf
from tensorflow.keras import backend as K

def index_for_keras_greater_than(data, threshold):
    threshold_array = tf.multiply(tf.ones_like(data), threshold) # create a tensor all ones
    mask = tf.cast(tf.greater(data, threshold_array),tf.double) # boolean tensor casted into a double tensor
    return mask

def jaccard_coef(y_true, y_pred):


    threshold = tf.constant(0.5, dtype='double')

    
    # Jaccard:
    y_true_f= K.flatten(tf.cast(y_true,tf.double))
    y_pred_f= K.flatten(tf.cast(y_pred,tf.double))
    
    y_pred_f_thres = index_for_keras_greater_than(y_pred_f, threshold)
    y_pred_f_thres= K.flatten(tf.cast(y_pred_f_thres,tf.double))
    
    intersection = K.sum(y_true_f * y_pred_f_thres)
    # return float((intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0))
    return float((intersection) / (K.sum(y_true_f) + K.sum(y_pred_f_thres) - intersection))
