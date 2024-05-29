# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 15:39:05 2023

@author: Orange
"""


import numpy as np
import os
import pickle
import tensorflow as tf
from tensorflow import keras
from keras import backend as K

import matplotlib.pyplot as plt
import time
from progress.bar import IncrementalBar
from skimage import io
from scipy import ndimage
import pandas as pd
import imageio

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
# detection_model_path = os.getcwd() + "\\cell_detection\\Label_8x32x20_no_translations_incl_older\\no_dense_layers_7x7_5x5_3x3_kernels\\run02\\"
detection_model_path = os.getcwd() + "\\cell_detection\\Label_8x32x20_with_older_with_translations\\no_dense_layers_7x7_5x5_3x3_kernels\\run07\\"


# Cropping settings:
rot_angle = 90


    
# Detection settings:
prediction_threshold = 0.65
iou_threshold = 0.35





# Cycle settings:
frame_interval = 5
start_frame = 3 # The tiff stack and cycle extraction will begin at this frame number. This is because sometimes there are no cells in the channel in the first few frames.



# # Data path:

# experiment_path = [
# #     'C:\\Users\\Orange\\OneDrive - University of Wollongong\\Papers\\Methods_paper\\Version_Stefan\\Rich_media_timelapse\\TL5_Nick\\'
#     # 'E:\\Documents\\Methods_paper\\20210301_200300_183_RecN_SOS_response\\TIFF\\'
#     # 'E:\\Experiments\\ssb_project\\new\\no_treatment\\20230901_WT\\cycle\\20230901_115823_217_WT_5min\\TIFF\\',
#     # 'E:\\Experiments\\ssb_project\\new\\no_treatment\\20230520_WT\\cycle\\20230520_143051_448_no_UV_5min\\TIFF\\',
#     'D:\\Experiments\\testing\\test1\\'
#     ]

# channel_ROIs = np.array([
#     [188,444,58,122],
#     # [187,443,116,180],
#     # [190,446,144,208],
#     [191,447,236,300],
#     # [185,441,266,330],
#     # [255,511,328,392],
#     [191,447,413,477],
#     ])
# n_ROIs = len(channel_ROIs)
# channel_ROI_dims = [64,256]


# exp = 0
# image_path = experiment_path[exp] + '568\\TIFF_corrected\\'






# """ Get the file list of all detection result files """
# filenames_tiff = load_file_list(image_path,'tif')

# i=0
# img_stack_filename = filenames_tiff[i]



""" Load train example """
def load_YOLO_example(source_path, npy_file_name):
    with open(source_path + npy_file_name, 'rb') as f:
        img_stack = np.load(f, allow_pickle=True)
        label_stack = np.load(f, allow_pickle=True)
        file_name_stack = np.load(f, allow_pickle=True)

    return img_stack, label_stack, file_name_stack


# TRAINING DATA
# mat_path = 'D:\\Code\\Thesis_code\\detection\\samples_20231110\\all_labels_together_8x32x20_with_older\\'
# file_name1 = "detection_training_set_64_256_8_32_4anchors_no_translations_no_clipping_with_older_TRAIN.npy"
mat_path = 'D:\\Code\\Thesis_code\\detection\\samples_20231110\\all_labels_together_8x32x20_with_older_with_translations\\'
file_name1 = "detection_training_set_64_256_8_32_4anchors_with_older_with_translations_VAL.npy"
              
X, Y_true, file_name_stack_train = load_YOLO_example(mat_path, file_name1)

cropped_channels = X[100:200,:,:]


n_frames = 100
n_ROIs = 1

Y_true_temp = Y_true[:,:,:,15]

np.max(Y_true_temp)


# # TRAINING DATA - matlab
# import scipy.io
# mat_path = 'D:\\Code\\Thesis_code\\detection\\samples_20230411\\'
# file_name1 = "YOLO_cell_4anchor_11-Apr-2023_64x256_8x32x20_TRAIN_20230411.mat"
# mat1 = scipy.io.loadmat(mat_path + file_name1)

# X = mat1['img_examples']
# Y_true = mat1['Y_true']
# file_names = mat1['mat_filenames']



# print(np.shape(X))
# print(np.shape(Y_true))

# cropped_channels = X[0:100,:,:]
# Y_true_temp = Y_true[:,:,:,15]

# n_frames = 100
# n_ROIs = 1






""" Load model """
# LOAD EXTRA MODEL DETAILS
with open(detection_model_path + 'data_stats.txt' ) as f:
    lines = f.readlines()
mu_detect = float(lines[0])
sigma_detect = float(lines[1])

with open(detection_model_path + 'model_parameters.txt' ) as f:
    lines = f.readlines()
    
for i in range(len(lines)):
    lines[i] = int(lines[i])
    
img_dim = [lines[0],lines[1]]
output_dims = [lines[2],lines[3],lines[4]]
anchors = np.array([[ lines[5],  lines[6]], [ lines[7],  lines[8]], [ lines[9],  lines[10]], [ lines[11], lines[12]]]).astype('float32')
n_anchors = np.shape(anchors)[0]

detection_model = tf.keras.models.load_model(detection_model_path + 'best_model.hdf5', compile=False)
                         
print('Detection model loaded with:')
print('input dimensions: ' + str(img_dim))
print('output dimensions: ' + str(output_dims))
print('Anchors: \n' + str(anchors))




# """load_img_stack"""
# rot_angle=90
# enhance_factor = 1
# """ LOAD TIF IMAGES OF THE TIME LAPSE POSITION """
# img_stack = io.imread(image_path + img_stack_filename)
# # img_stack = np.expand_dims(img_stack,0)

# """ Use the frames starting at 'start_frame' only: """
# img_stack = img_stack[start_frame:,:,:]

# """ Enhance the pixel values if needed """
# if enhance_factor != 1:
#     m,h,w = np.shape(img_stack)
#     img_stack = np.reshape(img_stack,(m*h*w,))
#     img_stack = img_stack.astype(float) * enhance_factor
#     img_stack[img_stack > (2**16 -1)] = 2**16 -1
#     img_stack = np.reshape(img_stack.astype('uint16'),(m,h,w,))

    
# """ Rotate the frames to the orientation of the detection model """
# if rot_angle == 0:
#     pass
# else:
#     img_stack = ndimage.rotate(img_stack, rot_angle, axes=(2, 1))

# n_frames = np.shape(img_stack)[0]


        


# """ crop """
# print('\n########## Cropping channels from ' + str(n_frames) + ' frames and ' + str(n_ROIs) + ' channel ROIs.')

# cropped_channels = np.zeros((n_ROIs, n_frames,channel_ROI_dims[0],channel_ROI_dims[1]), dtype='uint16')

# for i in range(n_ROIs):

#     cropped_channels[i,:,:,:] = img_stack[:,channel_ROIs[i,2]:channel_ROIs[i,3],channel_ROIs[i,0]:channel_ROIs[i,1]]
# cropped_channels = cropped_channels.astype('uint16')

# print('channel stack dimensions: ' + str(np.shape(cropped_channels)))



    
    
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
    
    
    hh = np.array([[[1,2,3],[4,5,6],[7,8,9]],[[1,2,3],[4,5,6],[7,8,9]]])
    hh_f = K.flatten(hh)
    hh_f = hh_f.numpy()


    
    return p0_pred, tx, ty, tw, th


def calculate_box_data(p0_pred, tx, ty, tw, th, grid_x, grid_y, grid_cell_dim_x, grid_cell_dim_y, w_a, h_a):
    """
    Here the object prediction is output seperately from the box prediction. In the class function for data processing they are used seperately for the NMS script.
    """
    
    test = grid_y.numpy()
    test2 = grid_cell_dim_y.numpy()
    # Calculate boxes
    bx_pred = tf.multiply((K.sigmoid(tx)+ grid_x),grid_cell_dim_x)
    by_pred = tf.multiply((K.sigmoid(ty)+ grid_y),grid_cell_dim_y)
    bw_pred = tf.multiply(w_a , K.exp(tw))
    bh_pred = tf.multiply(h_a , K.exp(th))
    
    # Calculate objectiveness and class
    p0_pred = K.sigmoid(p0_pred)

    
    box_data = tf.stack((bx_pred, by_pred, bw_pred, bh_pred),axis=-1)
    
    return p0_pred, box_data

    
def calculate_predictions(y_pred, anchors, img_dim):
    
    # GET THE GRID COORDINATES
    m = tf.shape(y_pred)[0] 
    fh = tf.shape(y_pred)[1]
    fw = tf.shape(y_pred)[2]
    
    grid_cell_dim_y = tf.cast(img_dim[0]/ fh ,dtype=tf.float32)
    grid_cell_dim_x = tf.cast(img_dim[1]/ fw ,dtype=tf.float32)

    
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




""" DETECT """
print('\n########## Detecting objects in ' + str(np.shape(cropped_channels)[0]) + ' crops.')

# SCALE THE DATA
cropped_channels_scaled = np.copy(cropped_channels).astype('float32')
n_crops_total = n_frames * n_ROIs
# img_dim --> [height, width]
h = img_dim[0]
w = img_dim[1]

cropped_channels_scaled = np.reshape(cropped_channels_scaled,n_crops_total * w * h)
cropped_channels_scaled[cropped_channels_scaled==0] = 1
cropped_channels_scaled = np.log(cropped_channels_scaled)
cropped_channels_scaled = (cropped_channels_scaled-mu_detect)/(4*sigma_detect)
cropped_channels_scaled = np.reshape(cropped_channels_scaled,(n_crops_total,h,w))
cropped_channels_scaled = np.expand_dims(cropped_channels_scaled,-1)



# Use the model
y_pred = detection_model.predict(cropped_channels_scaled)
print('predicted stack dimensions: ' + str(np.shape(y_pred)))
# Translate the outcome into predictions
object_predictions_anchor0, ROI_dims_anchor0, object_predictions_anchor1, ROI_dims_anchor1, object_predictions_anchor2, ROI_dims_anchor2, object_predictions_anchor3, ROI_dims_anchor3 = calculate_predictions(y_pred, anchors, img_dim)

# test = object_predictions_anchor0.numpy()
# test1 = ROI_dims_anchor0.numpy()


# ROI_dims --> [x_c, y_c, width, height] length of this list is number of ROIs x number of frames x y_pred_h x y_pred_w


# Reshape all in format: [number of ROI, frame, predictions of the frame] for each anchor
# and [number of ROI, frame, predictions of the frame, [x_c, y_c, width, height]] for each anchor
out_h, out_w, out_d = output_dims

object_predictions_anchor0 = np.reshape(object_predictions_anchor0,(n_ROIs, n_frames,  out_h*out_w))
ROI_dims_anchor0 = np.reshape(ROI_dims_anchor0,(n_ROIs, n_frames, out_h*out_w, 4))

object_predictions_anchor1 = np.reshape(object_predictions_anchor1,(n_ROIs, n_frames,  out_h*out_w))
ROI_dims_anchor1 = np.reshape(ROI_dims_anchor1,(n_ROIs, n_frames, out_h*out_w, 4))

object_predictions_anchor2 = np.reshape(object_predictions_anchor2,(n_ROIs, n_frames,  out_h*out_w))
ROI_dims_anchor2 = np.reshape(ROI_dims_anchor2,(n_ROIs, n_frames, out_h*out_w, 4))

object_predictions_anchor3 = np.reshape(object_predictions_anchor3,(n_ROIs, n_frames,  out_h*out_w))
ROI_dims_anchor3 = np.reshape(ROI_dims_anchor3,(n_ROIs, n_frames, out_h*out_w, 4))

# Concatenate the anchors to get an array with all results of all anchors frame after frame
object_predictions = np.concatenate((object_predictions_anchor0,object_predictions_anchor1,object_predictions_anchor2,object_predictions_anchor3),2)
ROI_dims = np.concatenate((ROI_dims_anchor0,ROI_dims_anchor1,ROI_dims_anchor2,ROI_dims_anchor3),2)


ohahaha1 = object_predictions[0,:,:]
ohahaha2 = np.reshape(ohahaha1,(100,256,4))

ohahaha3 = object_predictions[0,:,:]
ohahaha4 = np.reshape(ohahaha3,(100,256,4))

ohahaha5 = object_predictions[0,:,:]
ohahaha6 = np.reshape(ohahaha5,(100,256,4))
ohahaha7 = np.reshape(ohahaha5[20,:],(4,256))
ohahaha7 = ohahaha7.T

ohahaha7b = np.reshape(ohahaha7[:,0],(8,32))


y_true_part = Y_true[110,:,:,0]
y_true_part = np.hstack((y_true_part, Y_true[110,:,:,5] ))
y_true_part = np.hstack((y_true_part, Y_true[110,:,:,10] ))
y_true_part = np.hstack((y_true_part, Y_true[110,:,:,15] ))
y_true_part = np.reshape(y_true_part, (256,4))



y_true_part = np.stack((Y_true[100:200,:,:,0], Y_true[100:200,:,:,5], Y_true[100:200,:,:,10], Y_true[100:200,:,:,15]))
y_true_test = y_true_part[:,10,:,:]

y_true_part = np.stack((np.reshape(Y_true[120,:,:,0], (256)),
                        np.reshape(Y_true[120,:,:,5], (256)),
                        np.reshape(Y_true[120,:,:,10], (256)),
                        np.reshape(Y_true[120,:,:,15], (256))),-1)
                        

y_true_b = np.reshape(y_true_part[:,0],(8,32))




roi = 1
frame = 2
object_predictions_temp = object_predictions[roi,frame,:]
object_predictions_temp = np.reshape(object_predictions_temp,(4,8,32))
np.max(object_predictions_temp)

plt.ion()
plt.figure()
plt.imshow(cropped_channels[roi,frame,:,:])
# plt.imshow(cropped_channels[frame,:,:])

object_predictions_temp1 = object_predictions_temp.copy()



"""NMS"""
obj_prediction_final, bb_final = NMS_objects(object_predictions, ROI_dims, prediction_threshold, iou_threshold)
   



""" Make frames """
 
# Create the output folder
main_gif_path = image_path + 'gif_movies\\'

if not (os.path.exists(main_gif_path)): # Check if the path already exists
    os.mkdir(main_gif_path)

for roi in range(n_ROIs):
    
    gif_path = main_gif_path + img_stack_filename[:-4] + '_roi ' + str(roi) + '\\'
    if not (os.path.exists(gif_path)): # Check if the path already exists
        os.mkdir(gif_path)
    
    feature_list = self.results[roi]
    images = self.cropped_channels[roi]
    
    n_frames = np.unique(feature_list[:,1]).astype(int)
    
    # labels = False
    # if IDlabels:
    #     if np.shape(feature_list)[-1] < 16:
    #         print('No ID labels for ROI: ' + str(roi))
    #         labels = False
    #     else:
    #         labels = True
    
    # Status bar
    bar = IncrementalBar('Creating frames...', max = len(n_frames), suffix='%(percent)d%%')
    
    # plt.ion()
    plt.ioff()
    # Plot:
    for frame in n_frames:
        
        
        fig = plt.figure()
        fig.set_size_inches(24, 6)
        plt.imshow(images[frame,:,:], cmap = 'gray', origin='upper')
        plt.axis('off')
    
        index = feature_list[:,1] == frame
        temp = feature_list[index,:]
        index = np.argwhere(index)
        
        for cell in range(len(temp)):
            if bbox:
                x_min = temp[cell,6]
                x_max = temp[cell,7]
                y_min = temp[cell,8]
                y_max = temp[cell,9]
                
                ROI_coord_X = [x_min, x_max, x_max, x_min, x_min]
                ROI_coord_Y = [y_min, y_min, y_max, y_max, y_min]
                
                plt.plot(ROI_coord_X, ROI_coord_Y,color = 'red')
            
            if contours:
                plt.plot(self.cell_contours[roi][index[cell][0]][:,0], self.cell_contours[roi][index[cell][0]][:,1], '.',color = 'yellow')
                
        #     if IDlabels:
        #         if np.shape(feature_list)[-1] == 16:
        #             plt.text(temp[cell,4],temp[cell,5],'ID ' + str(int(temp[cell,15])), color = 'black', bbox=dict(facecolor='white', alpha=0.8), fontsize=16)

            
        # plt.xlim(0,256)
        # plt.ylim(0,64)
        
        # # Clock:
        # hrs = int((frame*frame_interval)//60)
        # mins = int((frame*frame_interval) - hrs * 60)
        # # sec = int((frame*frame_interval*60) - hrs * 60 * 60 - mins * 60)
                
        # # plt.text(10, 10, str(hrs) + ' hrs : ' + str(mins) + ' min : ' + str(sec) + ' s' , bbox=dict(facecolor='yellow', alpha=0.8), fontsize=44)
        # plt.text(10, 10, str(hrs) + ' hrs : ' + str(mins) + ' min'  , bbox=dict(facecolor='white', alpha=0.8), fontsize=44)
    
        plt.savefig(gif_path + 'img_' + str(frame) +'.png', 
                    transparent = False,  
                    facecolor = 'white'
                   )    
        plt.close()
    
        # Update the status bar
        bar.next()

        
    # End of the status bar
    bar.finish()