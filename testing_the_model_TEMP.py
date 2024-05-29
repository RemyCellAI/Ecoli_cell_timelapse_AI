# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 15:14:05 2023

@author: Orange
"""

# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import os



# from time_lapse_processing_class_20230125 import time_lapse_processing
from time_lapse_processing_class_20231122 import time_lapse_processing

def load_file_list(path,extension):
    # list to store files
    res = []
    # Iterate directory
    for file in os.listdir(path):
        # check only text files
        if file.endswith('.' + extension):
            res.append(file)
    print(res)
    return res


# Model Paths:
# fpn_model_path = os.getcwd() + "\\fpn_model\\fpn_20221222\\"
fpn_model_path = os.getcwd() + "\\fpn_model\\fpn_20230303\\further\\"
# fpn_model_path = os.getcwd() + "\\fpn_model\\fpn_20221015\\"


# detection_model_path = os.getcwd() + "\\cell_detection\\YOLO_cell_64x256_4_anchors\\epoch_81_to_120\\"
# detection_model_path = os.getcwd() + "\\cell_detection\\YOLO_cell_64x256_4_anchors\\further_20221230_only\\"
# detection_model_path = os.getcwd() + "\\cell_detection\\YOLO_cell_64x256_4_anchors\\20230411\\model3_even_further\\"

# detection_model_path = os.getcwd() + "\\cell_detection\\training_4x16x20_20231113\\model_with_dense_layer\\run02_40epochs\\"
# detection_model_path = os.getcwd() + "\\cell_detection\\training_4x16x20_20231113\\model_without_dense_layer_7x7_5x5_3x3_kernels\\run02_40epochs\\"


# detection_model_path = os.getcwd() + "\\cell_detection\\YOLO_cell_64x256_4_anchors\\20230411\\model3_even_further\\"
# detection_model_path = os.getcwd() + "\\cell_detection\\Label_8x32x20_no_translations_incl_older\\version20230412\\run03\\"
detection_model_path = os.getcwd() + "\\cell_detection\\Label_8x32x20_no_translations_incl_older\\no_dense_layers_7x7_5x5_3x3_kernels\\run02\\"


# segmentation_model_path = os.getcwd() + "\\cell_segmentation\\20221018_model2\\"
segmentation_model_path = os.getcwd() + "\\cell_segmentation\\cell_segmentation_20230109\\epoch_21_to_40\\"
tracking_model_path = os.getcwd() + '\\cell_tracking\\mpnn_no_ta_64units_20230703\\'

# Cropping settings:
rot_angle = 90

# FNP Detection settings:
prediction_threshold_fnp = 0.95
iou_threshold_fnp = 0.1
    
# Detection settings:
prediction_threshold = 0.65
iou_threshold = 0.35

# Segmentation settings:
segmentation_threshold = 0.3
measurement_threshold = 0.05

# mu_per_pixel = 0.110
mu_per_pixel = 0.16

# Foci settings
min_distance = 2
max_value_histogram = 2000
foci_peak_threshold = 0

# Cycle settings:
frame_interval = 5
start_frame = 3 # The tiff stack and cycle extraction will begin at this frame number. This is because sometimes there are no cells in the channel in the first few frames.
global_amp_fraction_for_detection = 0.3 # 30% cell length reduction is considered a cell split
min_n_frames = 4 # Minimum number of frames for the track to be considered.
n_discon = 3 # Max number of skipped frames allowed in the tracked cells
x_threshold = 225 # Threshold for the x coordinate. Cells positioned right from this point will not be analyzed.
min_cycle_slope = 0.6 * frame_interval/60 # Minimum slope the length curve of a cycle may have to be considered a cycle. --> (length in mu) * frame_interval/60
max_dlength_step = 0.6 # Maximum length increase from one frame to another in mu. A length increase above this value will be regarded as a detection error and the cycle will be discarted.
min_cycle_frames = 5 # Minimum number of frames in a cycle for the cyle to be considered. 

# Data path:



experiment_path = [
#     'C:\\Users\\Orange\\OneDrive - University of Wollongong\\Papers\\Methods_paper\\Version_Stefan\\Rich_media_timelapse\\TL5_Nick\\'
    # 'E:\\Documents\\Methods_paper\\20210301_200300_183_RecN_SOS_response\\TIFF\\'
    # 'E:\\Experiments\\ssb_project\\new\\no_treatment\\20230901_WT\\cycle\\20230901_115823_217_WT_5min\\TIFF\\',
    # 'E:\\Experiments\\ssb_project\\new\\no_treatment\\20230520_WT\\cycle\\20230520_143051_448_no_UV_5min\\TIFF\\',
    'D:\\Experiments\\testing\\test1\\'
    ]



# for exp in range(len(experiment_path)):

exp = 0
image_path = experiment_path[exp] + '568\\TIFF_corrected\\'
ssb_image_path = experiment_path[exp] + '458\\TIFF_corrected\\'


# Create the output folder
detected_ROI_image_path = image_path + 'detected_ROI_images\\'

if not (os.path.exists(detected_ROI_image_path)): # Check if the path already exists
    os.mkdir(detected_ROI_image_path)





test = time_lapse_processing()
test.load_all_models(fpn_model_path, detection_model_path, segmentation_model_path, tracking_model_path)



""" Get the file list of all detection result files """
filenames_tiff = load_file_list(image_path,'tif')

# for i in range(len(filenames_tiff)):
    
selected_files = [0]
for i in selected_files:
    
    # if i < 9:
    #     series_number = "0" + str(i+1)
    # else:
    #     series_number = str(i+1)
    # series_number = "0" + str(i+1)

    # img_stack_filename = 'Bg_Corr_Series' + series_number + '_568.tif'
    img_stack_filename = filenames_tiff[i]
    ssb_img_stack_filename = img_stack_filename[:-8] + "_458.tif"
    
    """ TO DO get filename from the tif list instead """
    
    print('\n')
    print('\n++++++++++++++ Processing stack-file: ' + img_stack_filename + ' ++++++++++++++')
    
    prediction_threshold_fnp = 0.4
    iou_threshold_fnp = 0.1
    n_ROIs, channel_ROIs, img_example = test.load_frames_and_detect_channels(image_path, detected_ROI_image_path, img_stack_filename, start_frame, rot_angle, prediction_threshold_fnp, iou_threshold_fnp, enhance_factor=2)
    
    if n_ROIs > 0:
        ROIs_to_be_extracted = np.linspace(0,n_ROIs-1,n_ROIs).astype(int)
        
        cropped_channels_init = test.crop_channels(ROIs_to_be_extracted, channel_ROIs)
        
        object_predictions, ROI_dims = test.run_detection_model()
        
        results_original, channel_ROIs_NMS, cropped_channels = test.NMS_for_selected_ROIs(prediction_threshold, iou_threshold)
        
        cell_contours, masks, mask_coordinates = test.crop_cells_and_segment(segmentation_threshold)
        
        results_, rotated_contours,  masks_, mask_coordinates_, cell_contours_ = test.get_cell_info(measurement_threshold, mu_per_pixel)
        
        # results, adj_matrix, cell_contours, rotated_contours, masks, mask_coordinates, channel_ROIs_NMS, cropped_channels = test.track_cells()
        
        # cropped_channels_ssb = test.load_and_crop_ssb_stack(ssb_image_path, ssb_img_stack_filename, start_frame, rot_angle=90)

        # results_foci, results_n_foci, used_foci_file_names = test.extract_foci_data(min_distance, start_frame)

        test.make_gif_movie(frame_interval, bbox=True, contours=True, IDlabels=True)

        # test.make_gif_movie_with_foci(frame_interval, max_value_histogram, foci_peak_threshold, bbox=True, contours=True, IDlabels=True)
            
        test.save_results(foci=True)
        
        # all_cycles_in_series, all_non_cycles_in_series, cycle_tracks_in_series, non_cycle_tracks_in_series = test.extract_cycle_data(x_threshold, global_amp_fraction_for_detection, min_n_frames, n_discon, min_cycle_slope, max_dlength_step, min_cycle_frames)
    
        # all_cycle_foci_in_series, all_non_cycle_foci_in_series = test.extract_foci_from_cycle_data(x_threshold)


object_predictions_test1 = object_predictions

bla = object_predictions[0,9,:]
bla = np.reshape(bla, (4,8,32))











############## TESTING MODE ############################
# detection_model_path = os.getcwd() + "\\cell_detection\\YOLO_cell_64x256_4_anchors\\epoch_81_to_120\\"
# detection_model_path = os.getcwd() + "\\cell_detection\\YOLO_cell_64x256_4_anchors\\further_20221230_only\\"
# detection_model_path = os.getcwd() + "\\cell_detection\\YOLO_cell_64x256_4_anchors\\20230411\\model3_even_further\\"

# detection_model_path = os.getcwd() + "\\cell_detection\\training_4x16x20_20231113\\model_with_dense_layer\\run02_40epochs\\"
detection_model_path = os.getcwd() + "\\cell_detection\\training_4x16x20_20231113\\model_without_dense_layer_7x7_5x5_3x3_kernels\\run02_40epochs\\"

# detection_model_path = os.getcwd() + "\\cell_detection\\YOLO_cell_64x256_4_anchors_20231111\\"
# detection_model_path = os.getcwd() + "\\cell_detection\\training_8x32x20_20231113_without_older\\"



def load_YOLO_example(source_path, npy_file_name):
    with open(source_path + npy_file_name, 'rb') as f:
        img_stack = np.load(f, allow_pickle=True)
        label_stack = np.load(f, allow_pickle=True)
        file_name_stack = np.load(f, allow_pickle=True)

    return img_stack, label_stack, file_name_stack


image_path = 'C:\\Users\\Orange\\Documents\\cell_NN\\Thesis_code\\detection\\samples_20231110\\all_labels_together_without_older\\'
file_name = 'detection_training_set_64_256_8_32_4anchors_TRAIN.npy'
cropped_channels_total, label_stack, file_name_stack = load_YOLO_example(image_path, file_name)



cropped_channels = cropped_channels_total[5000:5100,:,:]
# cropped_channels = np.expand_dims(cropped_channels,0)
y_true = label_stack[5000:5100,:,:]

n_frames = len(cropped_channels)

from time_lapse_processing_class_20231112 import time_lapse_processing
from cell_detection_utils_obj_only import YOLO_cell_head, iou_from_box_data
import tensorflow as tf

test = time_lapse_processing()
test.load_all_models(fpn_model_path, detection_model_path, segmentation_model_path, tracking_model_path)

    

# img_stack_filename = 'Bg_Corr_Series' + series_number + '_568.tif'
img_stack_filename = 'testing'

print('\n')
print('\n++++++++++++++ Processing stack-file: ' + img_stack_filename + ' ++++++++++++++')

ROIs_to_be_extracted = np.array([0])

object_predictions, ROI_dims, y_pred = test.run_detection_model_testing_mode(cropped_channels, ROIs_to_be_extracted, n_frames, image_path)



def YOLO_cell_loss(y_true, y_pred):
    
    LAMBDA_COORD = tf.constant(50, dtype=tf.double)
    LAMBDA_NOOBJ = tf.constant(1, dtype=tf.double)
    LAMBDA_OBJ = tf.constant(20, dtype=tf.double)
    img_dim = [len(y_true),64,256]
    anchors = np.array(
                [
                [10,10],
                [10,30],
                [10,80],
                [10,140],
                ]
                )
    anchors = anchors.astype('double')
    
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




   
def NOOBJ_metric(y_true, y_pred):
    img_dim = [len(y_true),64,256]
    anchors = np.array(
                [
                [10,10],
                [10,30],
                [10,80],
                [10,140],
                ]
                )
    anchors = anchors.astype('double')
    
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



   
def OBJ_metric(y_true, y_pred):
    img_dim = [len(y_true),64,256]
    anchors = np.array(
                [
                [10,10],
                [10,30],
                [10,80],
                [10,140],
                ]
                )
    anchors = anchors.astype('double')
    
    y_pred = tf.cast(y_pred,tf.double)
    y_true = tf.cast(y_true,tf.double)
    
    y_true_list, y_pred_list = YOLO_cell_head(y_true, y_pred, img_dim, anchors)

    p0_pred = y_pred_list[:,0]
    p0_true = y_true_list[:,0]

    

    index = tf.where(p0_true)
    OBJ_all = tf.gather_nd(p0_pred,index)
    
    OBJ = tf.math.reduce_mean(OBJ_all)

    return OBJ


   
def d_w(y_true, y_pred):
    img_dim = [len(y_true),64,256]
    anchors = np.array(
                [
                [10,10],
                [10,30],
                [10,80],
                [10,140],
                ]
                )
    anchors = anchors.astype('double')
    
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

total_loss = YOLO_cell_loss(y_true, y_pred)

NOOBJ = NOOBJ_metric(y_true, y_pred)
OBJ = OBJ_metric(y_true, y_pred)
average_delta_w = d_w(y_true, y_pred)

print('total_loss ' + str(total_loss))
print('NOOBJ ' + str(NOOBJ))
print('OBJ ' + str(OBJ))
print('average_delta_w ' + str(average_delta_w))




from YOLO_cell_non_max_suppression_20230112_obj_only import YOLO_cell_non_max_suppression



""" APPLY NON MAX SUPPRESION """
roi = 0

# input:
# object_predictions = [frames,w*h of pred]
# class_predictions = [frames,w*h of pred]
# ROI_dims = [frames,w*h of pred,4]

object_predictions_current_ROI = object_predictions[roi,:,:]
ROI_dims_current_ROI = ROI_dims[roi,:,:,:]

""" Non Max Supression """


for frame in range(n_frames):

    # NON MAX SUPPRESION
    cell_prediction_final, cell_obj_final, cell_boxes_final = YOLO_cell_non_max_suppression(object_predictions_current_ROI[frame,:], 
                                                                                                                   ROI_dims_current_ROI[frame,:,:], 
                                                                                                                   prediction_threshold, 
                                                                                                                   iou_threshold)









###############################################################







 