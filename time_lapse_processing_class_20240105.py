# -*- coding: utf-8 -*-
"""
UPDATED 20231112
'get_cell_info' : took out pop of                 
                        self.cell_contours[roi].pop(index[i][0])
                        self.masks[roi].pop(index[i][0])
and replaced it with lines that copy the contours you want to keep, rather than erasing the ones that needed to go.

cell_info: added try and except

@author: Orange
"""

""" LOAD THE LIBRARIES """


import numpy as np
import os
import pickle
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import time
from progress.bar import IncrementalBar
from skimage import io
from scipy import ndimage
import pandas as pd
import imageio



class time_lapse_processing:
    def __init__(self):
        print('Using Tensorflow version: ' )
        print(tf.__version__)

# %% 'load_all_models'
    def load_all_models(self, fpn_model_path, detection_model_path, segmentation_model_path, tracking_model_path):
        
        print('\n########## Loading models.')
        print('Tensorflow version: ' + tf.__version__) 
        self.load_fpn_model(fpn_model_path)
        self.load_detection_model(detection_model_path)
        self.load_segmentation_model(segmentation_model_path)
        self.load_tracking_model(tracking_model_path)

# %% 'load_frames_and_detect_channels'
    def load_frames_and_detect_channels(self, image_path, detected_ROI_image_path, img_stack_filename, start_frame, rot_angle, prediction_threshold_fpn, iou_threshold_fpn, enhance_factor ):
        self.image_path = image_path
        self.img_stack_filename = img_stack_filename

        self.load_img_stack(start_frame, rot_angle, enhance_factor)
        n_ROIs, channel_ROIs, img_example = self.run_fpn_model(detected_ROI_image_path, prediction_threshold_fpn, iou_threshold_fpn, growth_direction ='left_to_right')
        
        return n_ROIs, channel_ROIs, img_example
        
# %% 'load_fpn_model'
    def load_fpn_model(self, fpn_model_path):

        # LOAD EXTRA MODEL DETAILS
        with open(fpn_model_path + 'fpn_parameters.txt' ) as f:
            lines = f.readlines()
        self.mu_fpn = float(lines[0])
        self.sigma_fpn = float(lines[1])
        
        # LOAD EXTRA MODEL DETAILS
        with open(fpn_model_path + 'model_parameters.txt' ) as f:
            lines = f.readlines()
        
        for i in range(len(lines)):
            lines[i] = int(lines[i])
        
        self.fpn_dim = [lines[0],lines[1]]
        self.anchors_fpn = np.array([[lines[2],lines[3]]]).astype('float32') # defined like this to keep the option open to use multiple anchors
        
        # FPN model
        self.fpn_model = tf.keras.models.load_model(fpn_model_path + "best_model.hdf5" , compile=False)

# %% 'load_detection_model'
    def load_detection_model(self, detection_model_path):
        
        # LOAD EXTRA MODEL DETAILS
        with open(detection_model_path + 'data_stats.txt' ) as f:
            lines = f.readlines()
        self.mu_detect = float(lines[0])
        self.sigma_detect = float(lines[1])
        
        with open(detection_model_path + 'model_parameters.txt' ) as f:
            lines = f.readlines()
            
        for i in range(len(lines)):
            lines[i] = int(lines[i])
            
        self.img_dim = [lines[0],lines[1]]
        self.output_dims = [lines[2],lines[3],lines[4]]
        self.anchors = np.array([[ lines[5],  lines[6]], [ lines[7],  lines[8]], [ lines[9],  lines[10]], [ lines[11], lines[12]]]).astype('float32')

        self.detection_model = tf.keras.models.load_model(detection_model_path + 'best_model.hdf5', compile=False)
                                 
        print('Detection model loaded with:')
        print('input dimensions: ' + str(self.img_dim))
        print('output dimensions: ' + str(self.output_dims))
        print('Anchors: \n' + str(self.anchors))

# %% 'load_segmentation_model'
    def load_segmentation_model(self, segmentation_model_path):
        
        # LOAD EXTRA MODEL DETAILS
        with open(segmentation_model_path + 'data_stats.txt' ) as f:
            lines = f.readlines()
        self.mu_segm = float(lines[0])
        self.sigma_segm = float(lines[1])
        
        # LOAD EXTRA MODEL DETAILS
        with open(segmentation_model_path + 'model_parameters.txt' ) as f:
            lines = f.readlines()
        
        for i in range(len(lines)):
            lines[i] = int(lines[i])
            
        self.segm_dim = [lines[0], lines[1]]
        self.expand_pixel = lines[2]
        
        self.segmentation_model = tf.keras.models.load_model(segmentation_model_path + "best_model.hdf5", compile=False)

# %% 'load_tracking_model'
    def load_tracking_model(self, tracking_model_path):
        
        from tracking_utils_20230930 import dnn_block, cust_loss, tp, rn
        """ Load models """
        # print(tf.__version__)

        self.mpnn = tf.keras.models.load_model(tracking_model_path + "best_model_8.hdf5", compile=False, 
                                  custom_objects={'dnn_block': dnn_block,
                                                  'cust_loss':cust_loss,
                                                  'tp':tp,
                                                  'rn':rn,
                                                  })
        
        self.mpnn.compile(
            loss=cust_loss,
            optimizer=keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.),
            metrics=[tp,rn],
        )
        print('Tracking model loaded')

# %% 'load_img_stack'
    def load_img_stack(self, start_frame, rot_angle=0, enhance_factor = 1):
        """ LOAD TIF IMAGES OF THE TIME LAPSE POSITION """
        img_stack = io.imread(self.image_path + self.img_stack_filename)
        
        """ Use the frames starting at 'start_frame' only: """
        img_stack = img_stack[start_frame:,:,:]
        
        """ Enhance the pixel values if needed """
        if enhance_factor != 1:
            m,h,w = np.shape(img_stack)
            img_stack = np.reshape(img_stack,(m*h*w,))
            img_stack = img_stack.astype(float) * enhance_factor
            img_stack[img_stack > (2**16 -1)] = 2**16 -1
            img_stack = np.reshape(img_stack.astype('uint16'),(m,h,w,))

            
        """ Rotate the frames to the orientation of the detection model """
        if rot_angle == 0:
            self.img_stack = img_stack
        else:
            self.img_stack = ndimage.rotate(img_stack, rot_angle, axes=(2, 1))
        
        self.n_frames = np.shape(self.img_stack)[0]
        
# %% 'load_and_crop_ssb_stack'
    def load_and_crop_ssb_stack(self, ssb_image_path, ssb_img_stack_filename, start_frame, rot_angle=90):
        print("\n")
        print('Loading and cropping the image stack of ' + ssb_img_stack_filename)

        ssb_image_stack = io.imread(ssb_image_path + ssb_img_stack_filename)
        
        """ Use the frames starting at 'start_frame' only: """
        ssb_image_stack = ssb_image_stack[start_frame:,:,:] 
        
        n_frames = np.shape(ssb_image_stack)[0]
        
        """ Rotate the frames to the orientation of the detection model """
        if rot_angle == 0:
            ssb_image_stack = ssb_image_stack
        else:
            ssb_image_stack = ndimage.rotate(ssb_image_stack, rot_angle, axes=(2, 1))
    
        n_ROIs = np.shape(self.channel_ROIs_NMS)[0]
        cropped_channels_ssb = np.zeros((n_ROIs, n_frames, self.channel_ROI_dims[0], self.channel_ROI_dims[1]), dtype='uint16')
    
        for i in range(n_ROIs):
            cropped_channels_ssb[i,:,:,:] = ssb_image_stack[:, self.channel_ROIs_NMS[i,2]:self.channel_ROIs_NMS[i,3],self.channel_ROIs_NMS[i,0]:self.channel_ROIs_NMS[i,1]]
        
        
        self.cropped_channels_ssb = cropped_channels_ssb.astype('uint16')
        self.ssb_img_stack_filename = ssb_img_stack_filename
        self.ssb_image_path = ssb_image_path
        
        print('foci channel stack dimensions: ' + str(np.shape(cropped_channels_ssb)))
        return self.cropped_channels_ssb
    
# %% 'run_fpn_model'
    def run_fpn_model(self, detected_ROI_image_path, prediction_threshold_fnp, iou_threshold_fnp, growth_direction ='left_to_right'):
        from extract_channel_ROIs import get_channel_ROIs
        """ FEATURE PROPOSAL AND CHANNEL CROPS EXTRACTION """
        self.channel_ROI_dims = self.img_dim # --> [height, width]
        
        """ Take an example of channels filled with cells further up in the time-lapse: """
        img_example = self.img_stack[int(np.round(2 * self.n_frames /3)),:,:]
        # img_example = self.img_stack[2,:,:]
        
        # # Enhance the image:
        # img_example = img_example*2
        
        img_example_file_name = self.img_stack_filename
        
        # channel_ROIs --> [x_min, x_max, y_min, y_max]
        self.channel_ROIs = get_channel_ROIs(self.fpn_model, self.anchors_fpn, self.fpn_dim, self.mu_fpn, self.sigma_fpn, prediction_threshold_fnp, iou_threshold_fnp,  img_example, self.channel_ROI_dims, growth_direction, detected_ROI_image_path, img_example_file_name)
        self.n_ROIs = np.shape(self.channel_ROIs)[0]
        
        print('Number of channels found: ' + str(self.n_ROIs))
        return self.n_ROIs, self.channel_ROIs, img_example

# %% 'crop_channels'
    def crop_channels(self, ROIs_to_be_extracted, channel_ROIs):
        
        self.channel_ROIs = channel_ROIs
        
        print('\n########## Cropping channels from ' + str(self.n_frames) + ' frames and ' + str(self.n_ROIs) + ' channel ROIs.')
        self.ROIs_to_be_extracted = ROIs_to_be_extracted
        cropped_channels = np.zeros((len(ROIs_to_be_extracted), self.n_frames,self.channel_ROI_dims[0],self.channel_ROI_dims[1]), dtype='uint16')

        for i in range(len(ROIs_to_be_extracted)):
            j = ROIs_to_be_extracted[i]
            cropped_channels[i,:,:,:] = self.img_stack[:,self.channel_ROIs[j,2]:self.channel_ROIs[j,3],self.channel_ROIs[j,0]:self.channel_ROIs[j,1]]
        self.cropped_channels = cropped_channels.astype('uint16')
        
        print('channel stack dimensions: ' + str(np.shape(self.cropped_channels)))
        return self.cropped_channels

# %% 'run_detection_model'
    def run_detection_model(self):
        from cell_detection_utils_obj_only import apply_cell_detection
        print('\n########## Detecting objects in ' + str(np.shape(self.cropped_channels)[0]) + ' crops.')
        
        # SCALE THE DATA
        cropped_channels_scaled = np.copy(self.cropped_channels).astype('float32')
        n_crops_total = self.n_frames * len(self.ROIs_to_be_extracted)
        # img_dim --> [height, width]
        h = self.img_dim[0]
        w = self.img_dim[1]

        cropped_channels_scaled = np.reshape(cropped_channels_scaled,n_crops_total * w * h)
        cropped_channels_scaled[cropped_channels_scaled==0] = 1
        cropped_channels_scaled = np.log(cropped_channels_scaled)
        cropped_channels_scaled = (cropped_channels_scaled-self.mu_detect)/(4*self.sigma_detect)
        cropped_channels_scaled = np.reshape(cropped_channels_scaled,(n_crops_total,h,w))
        cropped_channels_scaled = np.expand_dims(cropped_channels_scaled,-1)
        cropped_channels_scaled = cropped_channels_scaled.astype('float32')

        object_predictions_anchor0, ROI_dims_anchor0, object_predictions_anchor1, ROI_dims_anchor1, object_predictions_anchor2, ROI_dims_anchor2, object_predictions_anchor3, ROI_dims_anchor3 = apply_cell_detection(cropped_channels_scaled, self.anchors, self.img_dim, self.detection_model)
        # ROI_dims --> [x_c, y_c, width, height] length of this list is number of ROIs x number of frames
        
        
        # Reshape all in format: [number of ROI, frame, predictions of the frame] for each anchor
        # and [number of ROI, frame, predictions of the frame, [x_c, y_c, width, height]] for each anchor
        out_h, out_w, out_d = self.output_dims
        object_predictions_anchor0 = np.reshape(object_predictions_anchor0,(len(self.ROIs_to_be_extracted), self.n_frames,  out_h*out_w))
        ROI_dims_anchor0 = np.reshape(ROI_dims_anchor0,(len(self.ROIs_to_be_extracted), self.n_frames, out_h*out_w, 4))
        object_predictions_anchor1 = np.reshape(object_predictions_anchor1,(len(self.ROIs_to_be_extracted), self.n_frames,  out_h*out_w))
        ROI_dims_anchor1 = np.reshape(ROI_dims_anchor1,(len(self.ROIs_to_be_extracted), self.n_frames, out_h*out_w, 4))
        object_predictions_anchor2 = np.reshape(object_predictions_anchor2,(len(self.ROIs_to_be_extracted), self.n_frames,  out_h*out_w))
        ROI_dims_anchor2 = np.reshape(ROI_dims_anchor2,(len(self.ROIs_to_be_extracted), self.n_frames, out_h*out_w, 4))
        object_predictions_anchor3 = np.reshape(object_predictions_anchor3,(len(self.ROIs_to_be_extracted), self.n_frames,  out_h*out_w))
        ROI_dims_anchor3 = np.reshape(ROI_dims_anchor3,(len(self.ROIs_to_be_extracted), self.n_frames, out_h*out_w, 4))
        
        # Concatenate the anchors to get an array with all results of all anchors frame after frame
        self.object_predictions = np.concatenate((object_predictions_anchor0,object_predictions_anchor1,object_predictions_anchor2,object_predictions_anchor3),2)
        self.ROI_dims = np.concatenate((ROI_dims_anchor0,ROI_dims_anchor1,ROI_dims_anchor2,ROI_dims_anchor3),2)
        
        return self.object_predictions, self.ROI_dims
    
    
    

# %% 'NMS_for_selected_ROIs'
    def NMS_for_selected_ROIs(self, prediction_threshold, iou_threshold):

        from detection_non_max_suppression_20231223 import NMS_objects
        
        """
        OUTPUT: results = [roi_number, frame_number, n_cells, cell_number, x_cent, y_cent, x_min, x_max, y_min, y_max]
        
        """
        
        print("\n")
        print("########## Applying Non Max Supression")
        
        self.results = []
        ROIs_kept_bool = np.ndarray((0,1))
        """ APPLY NON MAX SUPPRESION """
        for roi in self.ROIs_to_be_extracted:
            
            # input:
            # object_predictions = [frames,w*h of pred]
            # ROI_dims = [frames,w*h of pred,4]
            
            object_predictions_current_ROI = self.object_predictions[roi,:,:]
            ROI_dims_current_ROI = self.ROI_dims[roi,:,:,:]
            
            """ Non Max Supression """
 
            roi_results = np.ndarray((0,10))
            
            for frame in range(self.n_frames):

                # NON MAX SUPPRESION
                obj_prediction_final, bb_final = NMS_objects(object_predictions_current_ROI[frame,:],
                                                                ROI_dims_current_ROI[frame,:,:], 
                                                                prediction_threshold, 
                                                                iou_threshold)
   

                # From cell NMS cell_boxes_final--> [x_cent, y_cent, x_min, x_max, y_min, y_max]
                
                """ STORE THE RESULTS OF THIS IMAGE """
                # First sort the cells from left to right
                # x_cent = cell_boxes_final[:,0]
                # index = np.argsort(x_cent).astype('uint8')
                # cell_boxes_final = cell_boxes_final[index,:]
                
                x_cent = bb_final[:,0]
                index = np.argsort(x_cent).astype('uint8')
                cell_boxes_final = bb_final[index,:]
                
                
                # Check if the box is still inside the image:
                    
                # self.channel_ROI_dims --> [height, width]
                # cell_boxes_this_ROI --> per frame [x_cent, y_cent, x_min, x_max, y_min, y_max]
                # xmin
                index = cell_boxes_final[:,2] < 0
                cell_boxes_final[index,2] = 0
                # xmax
                index = cell_boxes_final[:,3] > self.channel_ROI_dims[1]-1
                cell_boxes_final[index,3] = self.channel_ROI_dims[1]-1
                # ymin
                index = cell_boxes_final[:,4] < 0
                cell_boxes_final[index,4] = 0
                # ymax
                index = cell_boxes_final[:,5] > self.channel_ROI_dims[0]-1
                cell_boxes_final[index,5] = self.channel_ROI_dims[0]-1
                
                # Store their bounding box coordinates:
                # cell_boxes_final-->  per frame: [x_cent, y_cent, x_min, x_max, y_min, y_max]
                
                
                """ Create a list of frame number and bb coordinates needed for the segmentation step: """
                # Number of cells detected in this frame:
                n_cells_in_frame = int(np.shape(obj_prediction_final)[0])
                
                # NOTE THAT IF THERE ARE NO CELLS DETECTED IN THIS FRAME, n_cells_in_frame IS ZEROS AND THE
                # NEXT 5 LINES WILL PRODUCE EMPTY ARRAYS. THIS MEANS THAT NOTHING WILL BE CONCATENATED IN temp.
                # THIS MEANS THAT FRAMES WITHOUT ANY DETECTED CELLS WILL NOT BE STORED IN THE RESULT FILE!
                
                roi_number = np.ones((n_cells_in_frame,1)) * roi
                frame_number = np.ones((n_cells_in_frame,1)) * frame
                n_cells = np.ones((n_cells_in_frame,1)) * n_cells_in_frame
                cell_number = np.arange(n_cells_in_frame)
                cell_number = np.expand_dims(cell_number,-1)
                
                
                temp = np.concatenate((roi_number, frame_number, n_cells, cell_number, cell_boxes_final),-1)
                roi_results = np.vstack((roi_results, temp))
                
                

                # output:
                # [roi, frame, n_cells_in_frame, cell number, x_cent, y_cent, x_min, x_max, y_min, y_max]
            
            if len(roi_results) > (4 * self.n_frames): # NOTE: This is a random chosen number of minimal cells required for the channel to be saved.
                """ Store the results of this ROI in the results list: """
                self.results.append(roi_results)
                ROIs_kept_bool = np.vstack((ROIs_kept_bool, 1))

            else:
                print('No cells detected in ROI ' + str(roi) + '. This ROI will not be saved.')
                ROIs_kept_bool = np.vstack((ROIs_kept_bool, 0))
            print("ROI " + str(roi) + " done")
        
        """
        NOTE: Update cropped channels and the channel_ROIs list which will be used for the foci image crops:
        """
        ROIs_kept_bool = np.squeeze(ROIs_kept_bool.astype(bool))
        
        self.channel_ROIs_NMS = self.channel_ROIs[ROIs_kept_bool,:]
        self.cropped_channels = self.cropped_channels[ROIs_kept_bool,:,:,:]
        
        if (np.size(ROIs_kept_bool) == 1) and ROIs_kept_bool:
            self.channel_ROIs_NMS = np.squeeze(self.channel_ROIs_NMS,0)
            self.cropped_channels = np.squeeze(self.cropped_channels,0)
        
        # Correct the ROI numbering in the results file in case a ROI has been removed:
        new_n_rois = len(self.results)
        for roi in range(new_n_rois):
            self.results[roi][:,0] = roi
            
        return self.results, self.channel_ROIs_NMS, self.cropped_channels

# %% 'crop_cells_and_segment'
    def crop_cells_and_segment(self, segmentation_threshold):
        from get_contour import get_contour
        """ Crop all the cells from the original ROI image stack & resize and stack all the crops:
        
        Using: results = [roi_number, frame_number, n_cells, cell_number, x_cent, y_cent, x_min, x_max, y_min, y_max]
        
        Each row in results is a detected cell.
        These are coordinates wrt channel_ROI_dims.
        Each cell is cropped and its dimensions are stored in cell_crops_dims --> [x_cent, y_cent, height, width].
        The crop is resized to segm_dim and stored in an array for all cells in this ROI.
        Then the segmentation model is ran at once.
        The resulting masks are resized to the original crop dimensions with cell_crops_dims.
        These resulting masks are used to extract the contour coordinates. 
        The contour coordinates are stored per frame (in the order as found in results).
        
        OURPUT: 
            cell_contours: List of X and Y coordinates wrt to the channel roi for each detection. Shape --> [roi, # detections (all frames), [X, Y]]
            masks: List of masks. Shape ---> [roi, # detections (all frames), binary image with the dimensions of the crop]
            mask_coordinates: The mask position wrt to the channel ROI. Shape --> [roi, # detections (all frames), [x_min, x_max, y_min, y_max]]
        
        """
        
        print('\n########## Segementing cells in bounding boxes.')
        
        if len(self.results) == 0:
            print('No ROIs found in the results. Cell segmentation will be skipped.')
            
        self.cell_contours = []
        self.masks = []
        self.mask_coordinates = []
            
        for roi in range(len(self.results)):
            # Find the max number of bounding boxes detected in the this ROI:
            total_n_cells_detected = len(self.results[roi])
            
            # Allocate space for the cell crops and their original dimensions:
            # segm_dim --> [height, width]
            cell_crops = np.ndarray((total_n_cells_detected, self.segm_dim[0], self.segm_dim[1],1)).astype('float32')
            cell_crops_dims = np.ndarray((total_n_cells_detected,4)).astype('float32')
            mask_coord = np.ndarray((total_n_cells_detected,4)).astype('float32')
            
            print('Segmenting ' + str(total_n_cells_detected) + ' cells in channel ' + str(roi))
            
            for i in range(total_n_cells_detected):
                
                # Get the box dimensions of the current cell and expand with a few pixels:
                # self.results[roi][7] --> [x_cent, y_cent, x_min, x_max, y_min, y_max]
                x_min = self.results[roi][i,6] - self.expand_pixel
                x_max = self.results[roi][i,7] + self.expand_pixel
                y_min = self.results[roi][i,8] - self.expand_pixel
                y_max = self.results[roi][i,9] + self.expand_pixel
        
                # Assert that the crop is still inside the image:
                # self.channel_ROI_dims --> [height, width]
                if x_min < 0:
                    x_min = 0
                if x_max > self.channel_ROI_dims[1]-1:
                    x_max = self.channel_ROI_dims[1]-1
                if y_min < 0:
                    y_min = 0
                if y_max > self.channel_ROI_dims[0]-1:
                    y_max = self.channel_ROI_dims[0]-1
                
                x_min = int(x_min)
                x_max = int(x_max)
                y_min = int(y_min)
                y_max = int(y_max)
                frame = int(self.results[roi][i,1])
                
                cell_crop_temp = self.cropped_channels[roi, frame, y_min:y_max,x_min:x_max]

                # Store the original crop dimensions:
                # cell_crops_dims --> [x_cent, y_cent, height, width]
                cell_crops_dims[i,0] = self.results[roi][i,4]
                cell_crops_dims[i,1] = self.results[roi][i,5]
                cell_crops_dims[i,2:] = np.shape(cell_crop_temp)
                
                # Resize the crop to the segmentation model dimensions:
                cell_crop_temp = tf.constant(np.expand_dims(cell_crop_temp,axis=-1))
                cell_crop_temp = tf.image.resize(cell_crop_temp, self.segm_dim)
                cell_crops[i,:,:] = cell_crop_temp
                
                # Store the mask position wrt to the channel ROI:
                mask_coord[i,0] = x_min
                mask_coord[i,1] = x_max
                mask_coord[i,2] = y_min
                mask_coord[i,3] = y_max
                
            """ perform segmentation on the cells of this ROI """
            # Scale the pixel values with the mu and sigma from the training set
            cell_crops = np.reshape(cell_crops,total_n_cells_detected * self.segm_dim[0] * self.segm_dim[1])
            cell_crops[cell_crops==0] = 1 #1000
            cell_crops = np.log(cell_crops)
            cell_crops = (cell_crops-self.mu_segm)/(self.sigma_segm)
            cell_crops = np.reshape(cell_crops,(total_n_cells_detected , self.segm_dim[0] , self.segm_dim[1]))

            # Use the UNET model to predict the semantic segmentation
            cell_segmentations = self.segmentation_model.predict(cell_crops)

            """ Resize the segmentations to the original crop dimensions """
            cell_contours_roi = []
            masks_roi = []

            for i in range(total_n_cells_detected):

                # Resize the prediction
                # cell_crops_dims --> [x_cent, y_cent, height, width]
                pred_temp = tf.image.resize(cell_segmentations[i,:,:,:], [cell_crops_dims[i,2].astype('int32'),cell_crops_dims[i,3].astype('int32')]).numpy()


                # Make it a binary mask
                pred_temp[pred_temp > segmentation_threshold] = 1
                pred_temp[pred_temp <= segmentation_threshold] = 0  

                # Get the contour and offset these coordinates
                # cell_crops_dims --> [x_cent, y_cent, height, width] --> coordinates wrt the channel ROI
                # contour_coordinates --> [X,Y]
                contour_coordinates = get_contour(pred_temp[:,:,0], flip_rotate_mask = True)
                contour_coordinates[:,0] = contour_coordinates[:,0] + cell_crops_dims[i,0] - cell_crops_dims[i,3]/2
                contour_coordinates[:,1] = contour_coordinates[:,1] + cell_crops_dims[i,1] - cell_crops_dims[i,2]/2 
                
                """ Store the contour at the corresponding frame"""
                # self.results[roi][6][i] --> frame number of this cell
                cell_contours_roi.append(contour_coordinates)
                masks_roi.append(pred_temp)
            
            """ Add the contours to the ROI results"""
            self.cell_contours.append(cell_contours_roi) # per frame the contour coordinates of each cell in order min to max of x_cent.
            """ Add the contours to the ROI results"""
            self.masks.append(masks_roi) # per frame the mask of each cell in order min to max of x_cent.
            self.mask_coordinates.append(mask_coord)
        return self.cell_contours, self.masks, self.mask_coordinates
    
# %% 'get_cell_info'
    def get_cell_info(self, measurement_threshold, mu_per_pixel):
        from cell_info_20221221 import cell_info, correct_cell_number
        print('\n########## Measuring all cell lengths.')
        
        if len(self.results) == 0:
            print('No ROIs found in the results. Cell measurement will be skipped.')
        
        self.rotated_contours = []
        
        for roi in range(len(self.results)):
            # Create a matrix for all the cell data:
            cell_data = np.zeros((len(self.cell_contours[roi]), 5))
            
            """
            cell_data: first dimension --> number of detection
            cell_data: second dimension  --> [x_cent, y_cent, width, length, angle]
            
            This is then concatenated to the results.
            
            OUTPUT:
                results = [roi_number, frame_number, n_cells, cell_number, x_cent, y_cent, x_min, x_max, y_min, y_max, x_cent, y_cent, width, length, angle]
            """
            
            # Start a list for each frame to store the rotated contours (cells flattened)
            rotated_contours_roi = []
            for cell in range(len(cell_data)):

                # Measure the cell length and width:
                """
                cell_measurement = [yc, xc, cell height, cell length]
                data_rot = cell contour rotated to lay it flat
                angle = angle of the cell (used to rotate the cell contour)
                
                NOTE: If during the ellipse fit there was a determinant equal to zero, all output will be zeros.
                """
                cell_measurement, data_rot, angle = cell_info(self.cell_contours[roi][cell][:,1], self.cell_contours[roi][cell][:,0], measurement_threshold)
                
                if cell_measurement[2] > 0: # If there was a problem fitting an elipse, 'cell_info' will return a length of zero.
                    # Pixel to microns
                    cell_measurement[2] = cell_measurement[2] * mu_per_pixel
                    cell_measurement[3] = cell_measurement[3] * mu_per_pixel
                    
                    cell_data[cell,:4] = cell_measurement
                    cell_data[cell,4] = angle 
                
                    # Add the rotated cell contour to the results:
                    rotated_contours_roi.append(data_rot)
                else:
                    print('Could not fit an ellipse for cell ' + str(cell) + ' in ROI ' + str(roi))
            
            """ Take out the cell data of cells that could not be measured: """
            # Take out all entries with a zero as length. These are cells that could not be fit:
            
            # Keep the entries that contain cell data:
            if np.sum(cell_data[:,2] == 0) > 0:
                index = cell_data[:,2] > 0
                cell_data = cell_data[index,:]
                self.results[roi] = self.results[roi][index,:]
                self.mask_coordinates[roi] = self.mask_coordinates[roi][index,:]
                
                # # Invert index to pop the entries that do not contain cell data:
                # index = np.argwhere(np.invert(index))
                # for i in range(len(index)):
                #     self.cell_contours[roi].pop(index[i][0])
                #     self.masks[roi].pop(index[i][0])
                
                index = np.argwhere(index)
                cell_contours_temp = []
                masks_temp = []
                for ind in range(len(index)):
                    i = index[ind][0]
                    cell_contours_temp.append(self.cell_contours[roi][i])
                    masks_temp.append(self.masks[roi][i])
                self.cell_contours[roi] = cell_contours_temp
                self.masks[roi] = masks_temp
            
            self.results[roi] = np.concatenate((self.results[roi], cell_data), -1)
            
            # Correct the number of cells and cell numbers now that cells have been removed from the list:
            self.results = correct_cell_number(self.results, roi)


            # Add the rotated cell contour of this ROI:
            self.rotated_contours.append(rotated_contours_roi)
            print("ROI " + str(roi) + " done")
        return self.results, self.rotated_contours, self.masks, self.mask_coordinates, self.cell_contours

# %% 'track_cells'
    def track_cells(self, track_x_threshold):
        
        print("\n")
        print("########## Tracking Cells")
        """
        Track all cell detections through the frames in each roi. Concatenate the tracking IDs to results.
        
        OUTPUT:
                results = [roi_number, frame_number, n_cells, cell_number, x_cent, y_cent, x_min, x_max, y_min, y_max, x_cent, y_cent, width, length, angle, cell ID]
                adj_matrix = The adjacency matrix for each roi
        """
            
        self.adj_matrix = []
        results_tracked = self.results.copy()
        
        ROIs_kept_bool = np.ndarray((0,1))
        if len(results_tracked) > 0:
            
            from tracking_utils_20230930 import take_out_last_cells, take_detection_crops, split_real_data_with_crops, convert_data_to_tensors, MPNNDataset, stitch_results, filter_predictions, create_adj_mat, construct_ID_list
    
            img_h, img_w = np.shape(self.cropped_channels[0][0])
            n_frames_in_smp = 15
            overlap = 14
            frame_depth = 4
            no_gt_in_smp = True
            n_smp_to_average = 10 # Number of frames used to average the last cell position 
            offset = 35 # Offset to the left (in pixels) wrt the average the last cell position to define the cut off threshold.
    
            """ Erase the detections on the right side of the channels: (asuming these are accumulated cells at the channel entrance) """
            results_tracked, self.masks, self.mask_coordinates, self.cell_contours = take_out_last_cells(results_tracked, self.masks, self.mask_coordinates, self.cell_contours, n_smp_to_average, offset)
            
            """ Predict the tracking for each roi: """
            for roi in range(len(results_tracked)):
                print("\n")
                print('Processing ROI: ' + str(roi) + ' of ' + self.img_stack_filename)
    
                feature_list_roi = results_tracked[roi]
                
                print('roi ' + str(roi) + '  shape feature_list_roi : ' + str(np.shape(feature_list_roi)))
                
                ## If after taking out the 'last' cells, there are too few cells left, discard this roi:
                if len(feature_list_roi) > (2 * self.n_frames): # NOTE: This is a random chosen number of minimal cells required for the channel to be saved.
                    
                    
                    
                    # """ First check whether there are frames missing: (frames with no detections at all) Skip this roi if so. """
                    # frames = np.unique(feature_list_roi[:,1])
                    # if len(frames) == (np.max(frames)+1): 
                    
                    cropped_channels_roi = self.cropped_channels[roi]
                    adjacency_matrix_gt = []
                    
                    """ Create the sub samples to be processed: """
                    detection_crops = take_detection_crops(feature_list_roi, cropped_channels_roi)
                    
                    # Normalize
                    rows, cols = np.shape(detection_crops)
                    detection_crops = np.reshape(detection_crops,(rows*cols))
                    detection_crops = np.log(detection_crops + 1)
                    detection_crops = detection_crops / np.log(2**16)
                    detection_crops = detection_crops - np.mean(detection_crops)
                    detection_crops = np.reshape(detection_crops,(rows,cols))
                    
                    # print('Max value of crops: ' + str(np.max(detection_crops)))
                    # print('Min value of crops: ' + str(np.min(detection_crops)))
                    
                    sub_node_features, sub_edge_features, sub_edge_indices, sub_adjacency_matrix, sub_adjacency_matrix_gt, sub_gt, sub_gt_one_hot, sub_gt_binary, frame_and_cell_log, sub_edge_indices_original_indices = split_real_data_with_crops(feature_list_roi, detection_crops, img_w, img_h, n_frames_in_smp, overlap, frame_depth, adjacency_matrix_gt)
                    print('Number of sub-samples: ' + str(len(sub_node_features)))
                    
                    """ Predict the edges """
                    n_smp = len(sub_node_features)
                    print('Loading ' + str(len(sub_node_features)) + ' sub-samples in tensors...')
                    data = [sub_node_features, sub_edge_features, sub_edge_indices, sub_gt]
                    x_data, _ = convert_data_to_tensors(data)
                    data_set = MPNNDataset(x_data, _ , batch_size=n_smp)
                    
                    print('Predicting ' + str(len(sub_node_features)) + ' sub-samples...')
                    y_pred = self.mpnn.predict(data_set)
                    
                    
                    """ Create the complete gr list and stitch all the sub samples together: """
                    print('Stitching ' + str(len(sub_node_features)) + ' sub-predictions together...')
                    complete_adj = create_adj_mat(feature_list_roi, frame_depth)
                    complete_index_list = np.argwhere(complete_adj)
                    
                    all_y_pred_final_average, all_gt_final =  stitch_results(complete_index_list, sub_edge_indices_original_indices, y_pred, frame_depth, no_gt_in_smp)
                        
                    
                    """ Construct an adjacency matrix with the predicted edges: """
                    
                    # Actual predicted values:
                    adj_pred_prob = np.zeros((len(feature_list_roi),len(feature_list_roi)))
                    for idx in range(len(complete_index_list)):
                        adj_pred_prob[complete_index_list[idx,0], complete_index_list[idx,1]] = all_y_pred_final_average[idx]
                    
                    # Filter
                    threshold = 0.15
                    adj = filter_predictions(feature_list_roi, adj_pred_prob, threshold)
                            
                            
                    """ Create an ID list only using the adjacency matrix: """
                    print('Creating the ID list based on the predictions...')
                    ID_list = construct_ID_list(adj, feature_list_roi)
                    
                    feature_list_roi = np.hstack((feature_list_roi, ID_list))
                    results_tracked[roi] = feature_list_roi
                    self.adj_matrix.append(adj)
                    
                    ROIs_kept_bool = np.vstack((ROIs_kept_bool, 1))
                    
                    print('Done')
                    
                else:
                    print('There are too few cells left in roi ' + str(roi) + '. This ROI will be removed.')
                    
                    
                    ROIs_kept_bool = np.vstack((ROIs_kept_bool, 0))

            
            """
            NOTE: Update cropped channels and the channel_ROIs list which will be used for the foci image crops:
            """
            ROIs_kept_bool = np.squeeze(ROIs_kept_bool.astype(bool))
            
            self.channel_ROIs_NMS = self.channel_ROIs_NMS[ROIs_kept_bool,:]
            self.cropped_channels = self.cropped_channels[ROIs_kept_bool,:,:,:]
            
            ROIs_pop_bool = np.array(np.invert(ROIs_kept_bool))
            s=np.shape(ROIs_pop_bool)
            
            
            if len(s) == 0: # We are indexing arrays. When the array has only one input, len() does not work anymore. So we need to expand the dimensions:
                ROIs_pop_bool = np.expand_dims(ROIs_pop_bool,0)
            
            
            index = np.argwhere(ROIs_pop_bool)
            index = np.flip(index) # Reverse the order so that the removal of roi's from the lists will go from end to beginning in case there are multiple roi's to be removed.
            for i in range(len(index)):
                results_tracked.pop(index[i][0])
                self.cell_contours.pop(index[i][0])
                self.rotated_contours.pop(index[i][0])
                self.masks.pop(index[i][0])
                self.mask_coordinates.pop(index[i][0])
            
            if (np.size(ROIs_kept_bool) == 1) and ROIs_kept_bool:
                self.channel_ROIs_NMS = np.squeeze(self.channel_ROIs_NMS,0)
                self.cropped_channels = np.squeeze(self.cropped_channels,0)
            
            # else:
            #     self.adj_matrix.append([])
            #     print('ROI ' + str(roi) + ' has missing frames and will be skipped for tracking.')
            
            self.results = results_tracked
        else:
            print('No ROIs found in the results. Tracking will be skipped.')
            
        return self.results, self.adj_matrix, self.cell_contours, self.rotated_contours, self.masks, self.mask_coordinates, self.channel_ROIs_NMS, self.cropped_channels

# %% 'extract_foci_data'
    def extract_foci_data(self, min_distance, start_frame):
        
        """
        
        OUTPUT: 
            results_foci = [roi_number, frame_number, n_cells, cell number in this frame, x_cent, y_cent, x_min, x_max, y_min, y_max, x_cent, y_cent, width, length, angle, cell ID, n_foci in this cell, foci details from 'peak fitter' of ImageJ]
                    dim = [36 x total number of foci detected in the channel in the whole time-lapse experiment]
            results_n_foci = [n_foci in the cell], dim = [1 x length of reults matrix] 
        """
        from expand_mask import expand_mask
        print("\n")
        print("########## Extracting foci")
        print('image file:.............' + self.img_stack_filename)
        print('foci image file.........' + self.ssb_img_stack_filename)
        print('foci file...............' + self.ssb_img_stack_filename[:-4] + '.csv')
        
        
        """ Load all of the ssb data from the imageJ csv file """
        file_name_csv = self.ssb_image_path + self.ssb_img_stack_filename[:-4] + '.csv'
        ssb_data = pd.read_csv(file_name_csv)
        ssb_data = ssb_data.to_numpy()
        
        """ Use the frames starting at 'start_frame' only: """
        index = ssb_data[:,-1] >= (start_frame + 1)
        ssb_data = ssb_data[index,:]
        ssb_data[:,-1] = ssb_data[:,-1] - ssb_data[0,-1] + 1
        
        self.results_foci = []
        self.results_n_foci = []
                
        for roi in range(len(self.results)):
            

            results_foci_roi = np.zeros((0, np.shape(self.results[roi])[-1] + 23))
            results_n_foci_roi = np.zeros((len(self.results[roi]),1))
            for cell in range(len(self.results[roi])):
                
                ssb_data_current_cell = np.zeros((0,26))
                
                current_frame = int(self.results[roi][cell,1])
                
                mask = np.squeeze(self.masks[roi][cell])
                """ Expand the mask with a chosen number of pixels """
                mask = expand_mask(mask,pixels=1)
                
                """ Convert the mask to a coordinate list and translate them wrt the 512x512 (original) image coordinates """ 
                segmentation_coordinates = np.argwhere(mask)
                
                # Coordinates of the bbox wrt the channel ROI
                x_min = self.mask_coordinates[roi][cell,0]
                y_min = self.mask_coordinates[roi][cell,2]
                
                # The mask coordinates are translated with the coordinates wrt to the ROI AND the coordinates of the ROI wrt to the original 512x512 image
                segmentation_coordinates[:,0] = segmentation_coordinates[:,0] + y_min + self.channel_ROIs_NMS[roi,2] + 1
                segmentation_coordinates[:,1] = segmentation_coordinates[:,1] + x_min + self.channel_ROIs_NMS[roi,0] + 1

                """ Extract the foci with the coordinates of this current cell mask """
                
                """ Use these coordinates to extract the ssb data from the list """
                frame = self.results[roi][cell,1]
                index = ssb_data[:,-1] == frame + 1
                ssb_data_current_frame = ssb_data[index,:]
                ## Round all values. The coordinates need to be integers:
                ssb_data_current_frame = np.round(ssb_data_current_frame).astype(int)
                
                ssb_data_current_cell = np.ndarray((0,20))
                ssb_data_current_cell_temp = np.ndarray((0,20))
                cell_intensity = np.zeros((1,2))
                for c in range(len(segmentation_coordinates)):
                    index = ( ssb_data_current_frame[:,4] == segmentation_coordinates[c,0] ) * ( ssb_data_current_frame[:,3] == segmentation_coordinates[c,1] )
                    if np.sum(index)>0:
                        ssb_data_current_cell_temp = np.vstack((ssb_data_current_cell_temp,ssb_data_current_frame[index,:]))
                    
                    """ Extract the foci channel pixel values of this cell to get the total and average intensity: """
                    # These coordinates are wrt the ROI only:
                    y = int(segmentation_coordinates[c,0] - self.channel_ROIs_NMS[roi,2])
                    x = int(segmentation_coordinates[c,1] - self.channel_ROIs_NMS[roi,0])
                    
                    # print('roi' + str(roi))
                    # print('current_frame' + str(current_frame))
                    # print('x' + str(x) + ' y' + str(y))
                    
                    cell_intensity[0,0] += self.cropped_channels_ssb[roi,current_frame,y,x]
                
                cell_intensity[0,1] = cell_intensity[0,0]/len(segmentation_coordinates)
                
                """ If there is only one foci, store it in the list directly:.
                    If there are more, check if the foci was fit twice, or if foci were fit too close to each other. In that case take the one with the max peak height.
                    If there are no foci, do nothing. The empty ndarray will be appended.
                """
                if len(ssb_data_current_cell_temp) > 0:
                    if len(ssb_data_current_cell_temp) == 1:
                        ssb_data_current_cell = np.vstack((ssb_data_current_cell,ssb_data_current_cell_temp[0,:]))
                    
                    elif len(ssb_data_current_cell_temp) > 1: # """ If there are more, check if there are foci too close to each other (= same foci) """
    
                        while len(ssb_data_current_cell_temp) > 0:
                            x = ssb_data_current_cell_temp[0,3]
                            y = ssb_data_current_cell_temp[0,4]
                            distance = np.sqrt( (x-ssb_data_current_cell_temp[:,3])**2 + (y-ssb_data_current_cell_temp[:,4])**2)
                            
                            # Extract all foci within a radius
                            index = distance < min_distance
                            
                            if np.sum(index) > 1:
                                # Of these foci find the one with the MAX peak value and store only this one:
                                index2 = ssb_data_current_cell_temp[index,2] == np.max(ssb_data_current_cell_temp[index,2])
                                index = np.argwhere(index)
                                index2 = index[index2]
                                
                                ssb_data_current_cell = np.vstack((ssb_data_current_cell,ssb_data_current_cell_temp[int(index2[0]),:])) # index2 is indexed with 0 to ensure only one value is selected. This is in case peak fitter detected the same foci twice and gave it the same peak value. This means that the max peak value is found twice.
                                
                                # Remove the foci treated in this run and repeat until there are none left.
                                ssb_data_current_cell_temp = np.delete(ssb_data_current_cell_temp,index,0)
                            else: # If only one foci is in this radius, than it is the foci being treated and there are no other foci within the radius. Store this foci:
                                ssb_data_current_cell = np.vstack((ssb_data_current_cell,ssb_data_current_cell_temp[0,:]))
                                
                                # Remove this foci treated in this run and repeat until there are none left.
                                ssb_data_current_cell_temp = np.delete(ssb_data_current_cell_temp,0,0)
                           
                    
                    """ Store the ssb results of this series, roi, cell, and frame """
                    n_cell_foci = len(ssb_data_current_cell)
                    
                    results_n_foci_roi[cell] = n_cell_foci
                    
                    # Translate the foci coordinates wrt to the channel roi:
                    ssb_data_current_cell[:,3] = ssb_data_current_cell[:,3] - self.channel_ROIs_NMS[roi,0]
                    ssb_data_current_cell[:,4] = ssb_data_current_cell[:,4] - self.channel_ROIs_NMS[roi,2]
                    ssb_data_current_cell.astype(int)
                    
                    temp = self.results[roi][cell,:]
                    temp = np.hstack((temp, n_cell_foci)) # <--- Add 'number of foci in this cell' to the end of the 'results' row.
                    temp = np.tile(temp,(n_cell_foci,1))
                    temp = np.concatenate((temp, ssb_data_current_cell),-1)
                    
                    # Add the cell intensity:
                    temp_intensity = np.tile(cell_intensity,(n_cell_foci,1))
                    temp = np.concatenate((temp, temp_intensity),-1)
                    
                    results_foci_roi = np.vstack((results_foci_roi, temp))

                else:
                    """ If there are no foci in this cell, in this frame, store the cell data followed by a row of zeros: """
                    n_cell_foci = 0
                    results_n_foci_roi[cell] = n_cell_foci
                    
                    temp = self.results[roi][cell,:]
                    temp = np.hstack((temp, n_cell_foci)) # <--- Add 'number of foci in this cell' to the end of the 'results' row.
                    temp = np.expand_dims(temp,0)
                    temp = np.concatenate((temp, np.zeros((1,20))),-1) 
                                        
                    # Add the cell intensity:
                    temp_intensity = cell_intensity
                    temp = np.concatenate((temp, temp_intensity),-1)
                    
                    results_foci_roi = np.vstack((results_foci_roi, temp))
                
            self.results_foci.append(results_foci_roi)
            self.results_n_foci.append(results_n_foci_roi)
        
        """ Store the file names used in this analysis: """
        self.used_foci_file_names = np.ndarray((3,1)).astype(str)
        self.used_foci_file_names[0] = self.img_stack_filename
        self.used_foci_file_names[1] = self.ssb_img_stack_filename
        self.used_foci_file_names[2] = file_name_csv
        print('Done')
        return self.results_foci, self.results_n_foci, self.used_foci_file_names

# %% 'extract_cycle_data'
    def extract_cycle_data(self, x_threshold, global_amp_fraction_for_detection, min_n_frames, n_discon, min_cycle_slope, max_dlength_step, min_cycle_frames, saving = False):
        
        print("\n")
        print("########## Extracting Cycle and Non-Cycle data")
        
        from detect_cycle_def_20230805 import detect_cycle
        
        all_cycles_in_series = []
        all_non_cycles_in_series = []
        cycle_tracks_in_series = []
        non_cycle_tracks_in_series = []
        tracks_first_frames = []
        
        if len(self.results) > 0:
            """ Create the Result plot directory: """
            channel_result_plots_path = self.image_path + 'cycle_plots\\'
            if not (os.path.exists(channel_result_plots_path)): # Check if the path already exists
                os.mkdir(channel_result_plots_path)
    
            
            for roi in range(len(self.results)):
        
                cell_data = self.results[roi]
            
                all_cycles, all_non_cycles, cycle_tracks, non_cycle_tracks, tracks_first_frame = detect_cycle(cell_data, roi, channel_result_plots_path, self.img_stack_filename, global_amp_fraction_for_detection, min_n_frames, n_discon, min_cycle_slope, max_dlength_step, min_cycle_frames, x_threshold)
    
                all_cycles_in_series.append(all_cycles)
                all_non_cycles_in_series.append(all_non_cycles)
                cycle_tracks_in_series.append(np.array(cycle_tracks))
                non_cycle_tracks_in_series.append(np.array(non_cycle_tracks))
                tracks_first_frames.append(tracks_first_frame)
        
            if saving == True:
                """ Save the result list """
                name = channel_result_plots_path + 'cycles_and_non_cycles_of_' + self.img_stack_filename[:-4] + '.npy'
                fptr = open(name, "wb")  # open file in write binary mode
                pickle.dump(all_cycles_in_series, fptr)  # dump list data into file 
                pickle.dump(all_non_cycles_in_series, fptr)  # dump list data into file 
                pickle.dump(cycle_tracks_in_series, fptr)  # dump list data into file 
                pickle.dump(non_cycle_tracks_in_series, fptr)  # dump list data into file 
                pickle.dump(tracks_first_frames, fptr)  # dump list data into file 
            
                fptr.close()  # close file pointer
                
                print('\n')
                print('Cycle data saved in: ')
                print(name)
                print('Done')
            
            self.channel_result_plots_path = channel_result_plots_path
            
        else:
            print('No ROIs found in the results. Cycle extraction will be skipped.')
            
        self.all_cycles_in_series = all_cycles_in_series
        self.all_non_cycles_in_series = all_non_cycles_in_series
        self.cycle_tracks_in_series = cycle_tracks_in_series
        self.non_cycle_tracks_in_series = non_cycle_tracks_in_series
        self.tracks_first_frames = tracks_first_frames
        


            
        return self.all_cycles_in_series, self.all_non_cycles_in_series, self.cycle_tracks_in_series, self.non_cycle_tracks_in_series

# %% 'extract_foci_from_cycle_data'
    def extract_foci_from_cycle_data(self, x_threshold):
        
        print("\n")
        print("########## Extracting Foci from Cycle data")
        
        all_cycle_foci_in_series = []
        all_non_cycle_foci_in_series = []
    
        if len(self.all_cycles_in_series) > 0:
    # try:
            for roi in range(len(self.all_cycles_in_series)):
                foci_data = self.results_foci[roi]
                all_cycle_foci = []
                all_non_cycle_foci = []
                
                
                # ## Remove the frames prior to the defined starting frame: 
                # # Foci data:
                # start_frame_index = np.argwhere(foci_data[:,1] == start_frame) # Assuming that the start_frame is in this dataset...
                # start_frame_index = np.squeeze(start_frame_index)
                # foci_data = foci_data[start_frame_index[0]:,:] # Only use the data starting from the start frame.
                
                # Remove the detections too close to the left border of the FOV:
                # Foci data:
                index = foci_data[:,10] <= x_threshold
                foci_data = foci_data[index, :]
                
                ## Use these results to extract the corresponding foci data:
    
                for cycle in range(len(self.all_cycles_in_series[roi])):
                    tr = self.cycle_tracks_in_series[roi][cycle]
                    index = foci_data[:,15] == tr
                    temp = foci_data[index,:]
                    start_index = np.squeeze(np.argwhere(temp[:,1] == self.all_cycles_in_series[roi][cycle][0,0])[0])
                    end_index = np.squeeze(np.argwhere(temp[:,1] == self.all_cycles_in_series[roi][cycle][-1,0])[-1]+1)
                    all_cycle_foci.append(temp[start_index:end_index,:])
    
                for cycle in range(len(self.all_non_cycles_in_series[roi])):
                    tr = self.non_cycle_tracks_in_series[roi][cycle]
                    index = foci_data[:,15] == tr
                    temp = foci_data[index,:]
                    all_non_cycle_foci.append(temp)
                
                all_cycle_foci_in_series.append(all_cycle_foci)
                all_non_cycle_foci_in_series.append(all_non_cycle_foci)
                
                print("ROI " + str(roi) + " done")
            
            """ Save the result list """
    
            name = self.channel_result_plots_path + 'cycles_and_non_cycles_and_foci_of_' + self.img_stack_filename[:-4] + '.npy'
            fptr = open(name, "wb")  # open file in write binary mode
            pickle.dump(self.all_cycles_in_series, fptr)  # dump list data into file 
            pickle.dump(self.all_non_cycles_in_series, fptr)  # dump list data into file 
            pickle.dump(all_cycle_foci_in_series, fptr)  # dump list data into file 
            pickle.dump(all_non_cycle_foci_in_series, fptr)  # dump list data into file 
            pickle.dump(self.cycle_tracks_in_series, fptr)  # dump list data into file 
            pickle.dump(self.non_cycle_tracks_in_series, fptr)  # dump list data into file 
            pickle.dump(self.tracks_first_frames, fptr)  
        
            fptr.close()  # close file pointer
            
            print('Cycle and foci data saved in: ' + name)
        # except:
        #     print('No foci data found.')
            
            print('Done')
        
        else:
            print('No ROIs found in the results. Foci in cycle extraction will be skipped.')
            
        return all_cycle_foci_in_series, all_non_cycle_foci_in_series

# %% 'save_results_with_tracking'
    def save_results(self, foci=False):
        print('\n########## Saving to file... ')
        if len(self.results) > 0:
            
            result_files_path = self.image_path + 'result_files\\'
            if not (os.path.exists(result_files_path)): # Check if the path already exists
                os.mkdir(result_files_path)
            
            if foci:
                save_file_name = 'Results_incl_tracking_incl_foci_of_' + self.img_stack_filename[0:-4] + '.npy'
            else:
                save_file_name = 'Results_incl_tracking_of_' + self.img_stack_filename[0:-4] + '.npy'
            
            try:
                print('\n########## Saving the results with tracking to: ')
                print('\nDir:.........' + self.image_path)
                print('\nFilename:....'  + save_file_name)
                
                """ Save the result list """
    
                fptr = open(result_files_path + save_file_name, "wb") 
                
                pickle.dump(self.img_stack_filename, fptr)
                pickle.dump(self.cropped_channels, fptr)
                pickle.dump(self.channel_ROIs, fptr)
                pickle.dump(self.channel_ROIs_NMS, fptr)
                pickle.dump(self.results, fptr)
                pickle.dump(self.masks, fptr)
                pickle.dump(self.mask_coordinates, fptr)
                pickle.dump(self.cell_contours, fptr)
                pickle.dump(self.rotated_contours, fptr)
                pickle.dump(self.adj_matrix, fptr)
                
                if foci:
                    try:
                        pickle.dump(self.results_foci, fptr)
                        pickle.dump(self.results_n_foci, fptr)
                        pickle.dump(self.used_foci_file_names, fptr)
                        pickle.dump(self.cropped_channels_ssb, fptr)
                    except:
                        print('No foci data found in file ' + self.img_stack_filename)
        
                fptr.close()  # close file pointer
            except:
                print('\n No tracking data found. File not saved.')
        else:
            print('No ROIs found in the results. This series will not be saved.')
            
# %% 'load_result_file'
    def load_result_file(self, result_files_path, result_file_name, image_path, ssb_image_path, foci=False):
        print("\n")
        print("############## Loading Result File")
        print('Opening result file with tracking:.....' + result_file_name)
        
            
        with open(result_files_path + result_file_name, 'rb') as f:
            self.img_stack_filename = np.load(f, allow_pickle=True)
            self.cropped_channels = np.load(f, allow_pickle=True)
            self.channel_ROIs = np.load(f, allow_pickle=True)
            self.channel_ROIs_NMS = np.load(f, allow_pickle=True)
            self.results = np.load(f, allow_pickle=True)
            self.masks = np.load(f, allow_pickle=True)
            self.mask_coordinates = np.load(f, allow_pickle=True)
            self.cell_contours = np.load(f, allow_pickle=True)
            self.rotated_contours = np.load(f, allow_pickle=True)
            self.adj_matrix = np.load(f, allow_pickle=True)
            
            if foci:
                try:
                    self.results_foci = np.load(f, allow_pickle=True)
                    self.results_n_foci = np.load(f, allow_pickle=True)
                    self.used_foci_file_names = np.load(f, allow_pickle=True)
                    self.cropped_channels_ssb = np.load(f, allow_pickle=True)
                    
                    self.ssb_img_stack_filename = self.used_foci_file_names[1][0]
                    self.ssb_image_path = ssb_image_path
                except:
                    print('No foci data found in file ' + result_file_name)
            
        self.n_frames = len(self.results[0][0])
        self.channel_ROI_dims = np.shape(self.cropped_channels[0][0])
        self.image_path = image_path
        
        print('of image stack:........................' + self.img_stack_filename)
        print('Number of ROIs:........................' + str(len(self.results)))
        
        if foci:
            return self.results, self.results_foci, self.cropped_channels, self.cropped_channels_ssb, self.masks, self.mask_coordinates, self.cell_contours, self.img_stack_filename, self.ssb_img_stack_filename
        else:
            return self.results, self.cropped_channels, self.masks, self.mask_coordinates, self.cell_contours, self.img_stack_filename

# %% 'make_gif_movie'
    def make_gif_movie(self, frame_interval, bbox=True, contours=True, IDlabels=True):
        print("\n")
        
        # Create the output folder
        main_gif_path = self.image_path + 'gif_movies\\'
        
        if not (os.path.exists(main_gif_path)): # Check if the path already exists
            os.mkdir(main_gif_path)
        
        for roi in range(len(self.results)):
            
            gif_path = main_gif_path + self.img_stack_filename[:-4] + '_roi ' + str(roi) + '\\'
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
            # print('\nCreating gif...')
        
            # frames = []
            # for t in n_frames:
            #     image = imageio.imread(gif_path + 'img_' + str(t) +'.png')
            #     frames.append(image)
                
                
                
                
            # imageio.mimsave(gif_path + self.img_stack_filename[:-4] + '_roi ' + str(roi) + '.gif', # output gif
            #                 frames,          # array of input frames
            #                 fps = 2)         # optional: frames per second

# %% 'make_gif_movie_with_foci'

    def make_gif_movie_with_foci(self, frame_interval, max_value_histogram, foci_peak_threshold, bbox=True, contours=True, IDlabels=True):
        
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        from matplotlib.colors import LinearSegmentedColormap
        
        print("\n")
        
        """ Define custom color map (same normalization for all frames) """
        colors = ['black', 'blue', 'white']
        foci_cmap = LinearSegmentedColormap.from_list('name', colors)
        norm = plt.Normalize(0, max_value_histogram)
    
        # Create the output folder
        main_gif_path = self.image_path + 'gif_movies\\'
        
        if not (os.path.exists(main_gif_path)): # Check if the path already exists
            os.mkdir(main_gif_path)
        
        for roi in range(len(self.results)):
            
            gif_path = main_gif_path + self.img_stack_filename[:-4] + '_roi ' + str(roi) + '\\'
            if not (os.path.exists(gif_path)): # Check if the path already exists
                os.mkdir(gif_path)
            
            feature_list = self.results[roi]
            foci_feature_list = self.results_foci[roi]
            images = self.cropped_channels[roi]
            foci_images = self.cropped_channels_ssb[roi]
            
            n_frames = np.unique(feature_list[:,1]).astype(int)
            

            # labels = False
            # if IDlabels:
            #     if np.shape(feature_list)[-1] == 36:
            #         print('No ID labels for ROI: ' + str(roi))
            #         labels = False
            #     else:
            #         labels = True
            
            # Status bar
            bar = IncrementalBar(' Creating frames...', max = len(n_frames), suffix='%(percent)d%%')
            
            # plt.ion()
            plt.ioff()
            # Plot:
            for frame in n_frames:
                
                fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(24,12))
                
                # fig = plt.figure()
                # fig.set_size_inches(24, 24)
                
                # plt.subplot(2,1,1)
                ax1.imshow(images[frame,:,:], cmap = 'gray', origin='upper')
                ax1.axis('off')
                ax1.set_title('Cell channel',fontsize=24)
                
                # plt.subplot(2,1,2)
                im = ax2.imshow(foci_images[frame,:,:], cmap=foci_cmap, norm=norm, origin='upper', interpolation='none')
                ax2.axis('off')
                ax2.set_title('Foci channel',fontsize=24)

                
                divider = make_axes_locatable(ax2)
                cax = divider.append_axes('right', size="3%", pad=0)

                fig.colorbar(im, orientation='vertical', shrink = 0.6, cax = cax)

            
                index = feature_list[:,1] == frame
                temp = feature_list[index,:]
                index = np.argwhere(index)
                
                index_foci = foci_feature_list[:,1] == frame
                temp_foci = foci_feature_list[index_foci]
                
                for cell in range(len(temp)):
                    if bbox:
                        x_min = temp[cell,6]
                        x_max = temp[cell,7]
                        y_min = temp[cell,8]
                        y_max = temp[cell,9]
                        
                        ROI_coord_X = [x_min, x_max, x_max, x_min, x_min]
                        ROI_coord_Y = [y_min, y_min, y_max, y_max, y_min]
                        
                        # plt.subplot(2,1,1)
                        ax1.plot(ROI_coord_X, ROI_coord_Y,color = 'red')
                    
                    if contours:
                        # plt.subplot(2,1,1)
                        ax1.plot(self.cell_contours[roi][index[cell][0]][:,0], self.cell_contours[roi][index[cell][0]][:,1], '.',color = 'yellow')
                        
                        # plt.subplot(2,1,2)
                        ax2.plot(self.cell_contours[roi][index[cell][0]][:,0], self.cell_contours[roi][index[cell][0]][:,1], '.',color = 'yellow')
                        
                    # if IDlabels:
                    #     # ax1.subplot(2,1,1)
                    #     ax1.text(temp[cell,4],temp[cell,5],'ID ' + str(int(temp[cell,15])), color = 'black', bbox=dict(facecolor='white', alpha=0.8), fontsize=16)

                for foci in range(len(temp_foci)):
                    """ 
                        NOTE: 'results_foci' is a list containing all detections, including the detections WITHOUT foci. 
                        For this the 'foci_peak_threshold' needs to be AT LEAST ZERO. Detections without foci have a ZERO 
                        as a foci value. These will then be skipped.
                    """
                    if temp_foci[foci,19] > foci_peak_threshold: 
                        y = temp_foci[foci,21]
                        x = temp_foci[foci,20]
                        xmin = x-2
                        xmax = x+2
                        ymin = y-2
                        ymax = y+2
                        
                        ROI_coord_X = [xmin, xmax, xmax, xmin, xmin]
                        ROI_coord_Y = [ymin, ymin, ymax, ymax, ymin]
                        # plt.subplot(2,1,2)
                        ax2.plot(ROI_coord_X, ROI_coord_Y, 'r')
                        ax2.text(xmin, ymax+3, str(int(temp_foci[foci,19])) , bbox=dict(facecolor='red', alpha=0.8), fontsize=12)
                        
                
                # Clock:
                hrs = int((frame*frame_interval)//60)
                mins = int((frame*frame_interval) - hrs * 60)
                # sec = int((frame*frame_interval*60) - hrs * 60 * 60 - mins * 60)
                        
                # #plt.text(10, 10, str(hrs) + ' hrs : ' + str(mins) + ' min : ' + str(sec) + ' s' , bbox=dict(facecolor='yellow', alpha=0.8), fontsize=44)
                # #plt.subplot(2,1,1)
                ax1.text(10, 10, str(hrs) + ' hrs : ' + str(mins) + ' min'  , bbox=dict(facecolor='white', alpha=0.8), fontsize=44)
                ax1.set_xlim(0,256)
                ax1.set_ylim(0,64)
                
                # plt.subplot(2,1,2)
                # ax2.set_xlim(0,256)
                # ax2.set_ylim(0,64)
                
                plt.savefig(gif_path + 'img_' + str(frame) +'.png', 
                            transparent = False,  
                            facecolor = 'white'
                            )    
                plt.close()
                # Update the status bar
                bar.next()
                
            # End of the status bar
            bar.finish()
            # print('n\Creating gif...')
        
            # frames = []
            # for t in n_frames:
            #     image = imageio.imread(gif_path + 'img_' + str(t) +'.png')
            #     frames.append(image)
            
            # imageio.mimsave(gif_path + self.img_stack_filename[:-4] + '_roi ' + str(roi) + '_with_foci.gif', # output gif
            #                 frames,          # array of input frames
            #                 duration=500)
            #                 # fps = 2)         # optional: frames per second


    # def load_result_file(self, image_path, result_file_name):
    #     print('Opening result file: ' + result_file_name)
        
            
    #     with open(image_path + result_file_name, 'rb') as f:
    #         results = np.load(f, allow_pickle=True)
    #         channel_ROIs = np.load(f, allow_pickle=True)
            
    #     self.n_frames = len(results[0][0])
    #     self.img_stack_filename = results[0][1]
    #     self.results = results
    #     self.channel_ROIs = channel_ROIs
    #     self.image_path = image_path
        
    #     print('of image stack: ' + self.img_stack_filename)
    #     print('containing ' + str(len(results)) + ' ROIs')
    #     return results





# %% 'check_class'
    def check_class(self):
        """
        This def returns the variables as they are loaded in the class.

        """
        return self.results, self.results_foci, self.cropped_channels, self.cropped_channels_ssb, self.cell_contours




# if __name__ == '__main__':
#     test = time_lapse_processing()
#     test.load_all_models(fpn_model_path, detection_model_path, segmentation_model_path, tracking_model_path)
