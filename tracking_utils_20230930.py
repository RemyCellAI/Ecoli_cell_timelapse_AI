# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 18:42:29 2023

@author: Orange
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
import os
import imageio
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers




""" Convert the train and validation data into ragged tf tensors: """

def convert_data_to_tensors(data):

    node_features = data[0]
    edge_features = data[1]
    edge_indices = data[2]
    y = data[3]

    # Convert lists to ragged tensors for tf.data.Dataset later on
    node_features = tf.ragged.constant(node_features, dtype=tf.float32)
    edge_features = tf.ragged.constant(edge_features, dtype=tf.float32)
    edge_indices = tf.ragged.constant(edge_indices, dtype=tf.int64)
    y = tf.ragged.constant(y, dtype=tf.float32)

    return (node_features, edge_features, edge_indices) , y



def prepare_batch(x_batch, y_batch):
    """Merges (sub)graphs of batch into a single global (disconnected) graph
    """

    node_features, edge_features, edge_indices = x_batch

    # Obtain number of atoms and bonds for each graph (molecule)
    num_nodes = node_features.row_lengths()
    num_edges = edge_features.row_lengths()

    # Obtain partition indices (molecule_indicator), which will be used to
    # gather (sub)graphs from global graph in model later on
    smp_indices = tf.range(len(num_nodes))
    # smp_indicator = tf.repeat(smp_indices, num_nodes)

    # Merge (sub)graphs into a global (disconnected) graph. Adding 'increment' to
    # 'pair_indices' (and merging ragged tensors) actualizes the global graph
    gather_indices = tf.repeat(smp_indices[:-1], num_edges[1:])
    increment = tf.cumsum(num_nodes[:-1])
    increment = tf.pad(tf.gather(increment, gather_indices), [(num_edges[0], 0)])
    edge_indices = edge_indices.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
    edge_indices = edge_indices + increment[:, tf.newaxis]
    node_features = node_features.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
    edge_features = edge_features.merge_dims(outer_axis=0, inner_axis=1).to_tensor()


#    y_batch = y_batch.merge_dims(outer_axis=0, inner_axis=1).to_tensor()

    return (node_features, edge_features, edge_indices), y_batch

def MPNNDataset(X, y, batch_size=32, shuffle=False):
    dataset = tf.data.Dataset.from_tensor_slices((X, (y)))
    if shuffle:
        dataset = dataset.shuffle(1024)
    return dataset.batch(batch_size).map(prepare_batch, -1).prefetch(-1)



def construct_tracking_dataset_one_sample_with_crops(features, img_w, img_h, frame_depth):
    """
    INPUT:
        features : [roi, frame, number of detections, detection number in frame, bb_xc, bb_yc, bb_w, bb_h, cell_xc, cell_yc, cell width, cell length, cell angle]

    OUTPUT:
    nodes_feat = [x, y, w, h, time]
    edge_feat = [dx, dy, dbb_w, dbb_h, dtime]
    edge_indices
    """
    
    # Build the adjacency matrix:
    # adjacency_mat = np.ones((len(all_detections),len(all_detections)))
    # n_frames = len(np.unique(all_detections[:,0]))
    # for frame in range(n_frames):
    #     index = np.argwhere(all_detections[:,0] == frame)
    #     start = index[0][0]
    #     end = index[-1][0]+1
    #     adjacency_mat[start:end, start:end] = 0
    
    """ Produce the adjacency matrices used for training (all edge combinations with different time stamps) """
    adjacency_mat_frame_depth = create_adj_mat(features, frame_depth)
    
    """ Build the node feature list """
    xc_yc = features[:,4:6]
    t = np.expand_dims(features[:,1],-1)
    
    # # Based on the bounding boxes:
    # w = np.expand_dims(features[:,8]-features[:,7],1)
    # h = np.expand_dims(features[:,10]-features[:,9],1)
    
    # Based on the cell dimensions:
    w = np.expand_dims(features[:,12],1) # This is in mu, not pixels! Later these values will be divided by other lenghs, such that the mu per pixel is irrelevant.
    h = np.expand_dims(features[:,13],1)
    
    
    node_feat = np.concatenate((xc_yc, w, h, t),1)
    node_feat[:,0] = node_feat[:,0] / img_w
    node_feat[:,1] = node_feat[:,1] / img_h
    node_feat[:,2] = node_feat[:,2] / img_w
    node_feat[:,3] = node_feat[:,3] / img_h
    node_feat[:,4] = (node_feat[:,4] - node_feat[0,4]) / np.max(node_feat[:,4] - node_feat[0,4]) # Offset the time with the time stamp of the first frame.
    
    """ Get all possible links: """
    edge_index = np.argwhere(adjacency_mat_frame_depth)
    nodes_temp = np.ndarray((0,5))
    linked_node_feat = np.ndarray((0,5))
    frame1_temp = np.ndarray((0,1))
    frame2_temp = np.ndarray((0,1))
    for i in range(len(node_feat)):
        index = adjacency_mat_frame_depth[i,:] == 1
        
        n = np.sum(index)
        temp_i = np.tile(node_feat[i,:],(n,1)) # repeat the feature i n times to be combined with the linked features
        temp_j = node_feat[index,:]
        nodes_temp = np.vstack((nodes_temp,temp_i))
        linked_node_feat = np.vstack((linked_node_feat,temp_j))
        
        temp_i = np.tile(features[i,1],(n,1)) # repeat the feature i n times to be combined with the linked features
        temp_j = np.expand_dims(features[index,1],-1)
        frame1_temp = np.vstack((frame1_temp,temp_i))
        frame2_temp = np.vstack((frame2_temp,temp_j))
        
    frames_of_indices = np.hstack((frame1_temp,frame2_temp))
    
    # # Done like in the paper:
    # dx = 2*(linked_node_feat[:,0] - nodes_temp[:,0]) / (linked_node_feat[:,3] + nodes_temp[:,3])
    # dy = 2*(linked_node_feat[:,1] - nodes_temp[:,1]) / (linked_node_feat[:,3] + nodes_temp[:,3])
    
    # Own version:
    dx = linked_node_feat[:,0] - nodes_temp[:,0]
    dy = linked_node_feat[:,1] - nodes_temp[:,1]
    
    dw = np.log(nodes_temp[:,2]/linked_node_feat[:,2])
    dh = np.log(nodes_temp[:,3]/linked_node_feat[:,3])
    dt = linked_node_feat[:,4] - nodes_temp[:,4]
    edge_feat = np.vstack((dx,dy,dw,dh,dt))
    
    
    return edge_feat.T, edge_index, adjacency_mat_frame_depth, frames_of_indices




def create_adj_mat(detections, frame_depth):
    adjacency_mat_frame_depth = np.zeros((len(detections),len(detections)))
    unique_frames = np.unique(detections[:,1])

    for f in range(len(unique_frames)):
        frame = int(unique_frames[f])
        index_current_frame = np.argwhere(detections[:,1] == frame)
        if (f >= frame_depth) and ((f+frame_depth) <= np.max(unique_frames)):
            frames_past = unique_frames[(f-frame_depth):f]
            frames_fut = unique_frames[f+1:(f+frame_depth)+1]
            
        elif (f < frame_depth): 
            frames_past = unique_frames[0:f]
            frames_fut = unique_frames[f+1:(f+frame_depth)+1]
        elif ((f+frame_depth) > np.max(unique_frames)): 
            frames_past = unique_frames[(f-frame_depth):f]
            frames_fut = unique_frames[f+1:]
        
        frames = np.hstack((frames_past, frames_fut)).astype(int)
        for linked_frame in frames:
            index = detections[:,1] == linked_frame
            for row in index_current_frame:
                adjacency_mat_frame_depth[row, index] = 1
    return adjacency_mat_frame_depth


def create_gt_from_adj_mat(adj_gt, eidx):
    gt_edges = np.argwhere(adj_gt)
    
    # Find the splits:
    adj_gt_sum = np.sum(adj_gt,1)
    split_index_row = np.argwhere(adj_gt_sum==2)
    split_index = np.zeros((len(split_index_row),2))
    for row in range(len(split_index_row)):
        split_index_col = np.argwhere(adj_gt[split_index_row[row],:][0])
        split_index[row,0] = split_index_row[row]
        split_index[row,1] = split_index_col[1]
        
    gt = np.zeros((len(eidx),1))
    for i in range(len(gt_edges)):
        index = (gt_edges[i,0]==eidx[:,0]) * (gt_edges[i,1]==eidx[:,1])
        if np.sum(index)==1:
            # Check if this is the offspring of a split:
            if  np.sum((gt_edges[i,0]==split_index[:,0]) * (gt_edges[i,1]==split_index[:,1])):
                gt[index] = 2
            else:
                gt[index] = 1
                
    gt_one_hot = np.zeros((len(gt),3))
    for i in range(len(gt)):
        gt_one_hot[i,int(gt[i])] = 1

    gt_binary = gt.copy()
    gt_binary[gt>0] = 1

    return gt.astype(int), gt_one_hot.astype(int), gt_binary.astype(int)


def random_color_generator():
    color = np.random.randint(0, 255, size=3)
    return tuple(color.astype(int))

def split_real_data_with_crops(features, detection_crops, img_w, img_h, n_frames_in_smp, overlap, frame_depth, adjacency_matrix_gt=[]):
    
    """
    This def takes the generated samples and creates overlapping sub samples with them.
    The length of these sub samples is 'n_frames_in_smp' frames, and the overlap is 'overlap' frames.
    
    """
    sub_node_features = []
    sub_edge_features = []
    sub_edge_indices = []
    sub_edge_indices_original_indices = []
    sub_adjacency_matrix= []
    sub_adjacency_matrix_gt = []
    sub_gt = []
    sub_gt_one_hot = []
    sub_gt_binary = []
    frame_and_cell_log = np.ndarray((0,5))
    
    unique_frames = np.unique(features[:,1]).astype(int) # The start and end frame of the sub_smp will be read from the unique frames. This is in case there are missing frames, and you can't simply add a number to the previous starting or ending frame number.
    n_frames = len(unique_frames)
    n_sub_smps = int(np.ceil( ((n_frames-n_frames_in_smp) / (n_frames_in_smp-overlap) + 1 )))

    for sub_smp in range(n_sub_smps):

        if sub_smp == 0:
            start_frame = unique_frames[0]  # There is no overlap for the first n_frames_in_smp frames.
            end_frame = unique_frames[0 + n_frames_in_smp -1]
            
        elif sub_smp == (n_sub_smps-1): # The last sub_smp will simply be the last n_frames_in_smp frames of the time lapse.
            start_frame = unique_frames[n_frames - n_frames_in_smp]
            end_frame = unique_frames[n_frames-1]
            
        else:
            start_frame = unique_frames[0 + sub_smp*(n_frames_in_smp - overlap)]
            end_frame = unique_frames[sub_smp*(n_frames_in_smp - overlap) + n_frames_in_smp -1]
            
        
        idx_start = np.argwhere(features[:,1]==start_frame)
        idx_end = np.argwhere(features[:,1]==end_frame)
        if (len(idx_start)>0) and (len(idx_end)>0):
            idx_start = int(idx_start[0])
            idx_end = int(idx_end[-1])+1
            ft = features[idx_start:idx_end,:]
            ft_crops = detection_crops[idx_start:idx_end,:]
            eft, eidx, adj, frames_of_indices = construct_tracking_dataset_one_sample_with_crops(ft, img_w, img_h, frame_depth)
            
            # Needed to stitch every sample together after running them through the model:
            eidx_original_indices = np.hstack((eidx + idx_start, frames_of_indices))

            frame_of_index = np.zeros((len(eidx,)))
            detection = np.unique(eidx[:,0])
            for det in detection:
                index = eidx[:,0] == det
                frame_of_index[index] = ft[det,1]
            
            unique_frame_of_index = np.unique(frame_of_index)
            last_indices = np.argwhere(frame_of_index == (unique_frame_of_index[0 + n_frames_in_smp - overlap]))
            last_index = last_indices[0][0]

            
            # adj_gt = adjacency_matrix_gt[idx_start:idx_end, idx_start:idx_end]
            
            if len(adjacency_matrix_gt)>0: # If there is a gt adjacency matrix given:
                adj_gt = adjacency_matrix_gt[idx_start:idx_end, idx_start:idx_end]
                # Create the ground truth:
                gt, gt_one_hot, gt_binary = create_gt_from_adj_mat(adj_gt, eidx)
            else:
                adj_gt = []
                gt = []
                gt_one_hot = []
                gt_binary = []
                # print("No gt adjacency matrix given.")
            
            # Log the frame and detection numbers of the original data:
            frame_and_cell_log = np.vstack((frame_and_cell_log, [start_frame, end_frame, idx_start, idx_end, last_index]))
            sub_node_features.append(ft_crops)
            sub_edge_features.append(eft)
            sub_edge_indices.append(eidx)
            sub_edge_indices_original_indices.append(eidx_original_indices)
            sub_adjacency_matrix.append(adj)
            sub_adjacency_matrix_gt.append(adj_gt)
            sub_gt.append(gt)
            sub_gt_one_hot.append(gt_one_hot)
            sub_gt_binary.append(gt_binary)
    return sub_node_features, sub_edge_features, sub_edge_indices, sub_adjacency_matrix, sub_adjacency_matrix_gt, sub_gt, sub_gt_one_hot, sub_gt_binary, frame_and_cell_log, sub_edge_indices_original_indices
    
    
def take_detection_crops(look_up_table, images):
    detection_crops = np.zeros((len(look_up_table),16*24))
        
    for feat in range(len(look_up_table)):
        frame = int(look_up_table[feat,1])
        frame_image = images[frame,:,:]
        
        h1 = int(look_up_table[feat,8])
        h2 = int(look_up_table[feat,9])+1
        w1 = int(look_up_table[feat,6])
        w2 = int(look_up_table[feat,7])+1
        
        # dw = look_up_table[:,8]-look_up_table[:,7]
        # dh = look_up_table[:,10]-look_up_table[:,9]
        
        # plt.figure()
        # plt.hist(dw)
        
        # plt.figure()
        # plt.hist(dh)
        
        crop = frame_image[h1:h2,w1:w2]
        crop_resize = tf.image.resize(np.expand_dims(crop,-1), (16,24)).numpy()
        
        detection_crops[feat,:] = np.reshape(crop_resize,(16*24))
        
        # test = np.reshape(detection_crops,(len(look_up_table),16,32))
        # plt.ion()
        # plt.figure()
        # plt.subplot(3,1,1)
        # plt.imshow(frame_image)
        # plt.subplot(3,1,2)
        # plt.imshow(crop)
        # plt.subplot(3,1,3)
        # plt.imshow(test[-1,:,:])
    return detection_crops
        

def take_out_last_cells(detections, masks, masks_coordinates, cell_contours, n_smp_to_average, offset):
    
    from cell_info_20221221 import correct_cell_number
    cell_contours_tracked = []
    masks_tracked = []
    masks_coordinates_tracked = []
    
    for roi in range(len(detections)):
        
        """ Define the position of cells at the channel entrance: """
        frames = np.unique(detections[roi][:,1]).astype(int)
        
        # First check if there are more than 4 cell present at the end of the time lapse. If not, cutting off 40 pixels will leave zero cell detections. In case of <= 2 cell, skip the cutting off step:
        index = detections[roi][:,1] == (np.max(frames) - n_smp_to_average)
        if np.sum(index) > 4:
                
            last_positions = 0
            j = 0
            for i in range(n_smp_to_average):
                index = detections[roi][:,1] == (np.max(frames) - n_smp_to_average + i)
                temp = detections[roi][index,4]
                if len(temp)>0:
                    last_positions += temp[-1] # take the last centroid in this frame
                    j += 1
            x_thres = int((last_positions/j) - offset)
            index = detections[roi][:,4] < x_thres
            detections[roi] = detections[roi][index,:]
            
            # Extract those cell contours belonging to the selected tracking detections:
            contours_temp = []
            masks_temp = []
            
            masks_coordinates_tracked.append(masks_coordinates[roi][index,:])
            
            ind = np.argwhere(index)
            for i in range(len(ind)):
                contours_temp.append(cell_contours[roi][ind[i][0]])
                masks_temp.append(masks[roi][ind[i][0]])
            cell_contours_tracked.append(contours_temp)
            masks_tracked.append(masks_temp)
            
            # Correct the number of cells and cell numbers now that cells have been removed from the list:
            detections = correct_cell_number(detections, roi)
                    
                    
        else:
            cell_contours_tracked.append(cell_contours[roi])
            masks_tracked.append(masks[roi])
            masks_coordinates_tracked.append(masks_coordinates[roi])
    
    return detections, masks_tracked, masks_coordinates_tracked, cell_contours_tracked



def stitch_results(complete_index_list, sub_edge_indices_original_indices, y_pred, frame_depth, no_gt_in_smp):
    
    n_smp = len(sub_edge_indices_original_indices)
    all_indices = np.zeros((len(complete_index_list),n_smp*2))
    all_y_pred_final = -np.ones((len(complete_index_list),n_smp))
    all_gt_final = -np.ones((len(complete_index_list),))
    
    n_idx_previous = 0
    for smp in range(n_smp):
            # The predictions are the sub_smp predictions stacked in one list. Extract the current sub_smp:
            n_idx = len(sub_edge_indices_original_indices[smp])
            y_pred_current_smp = np.squeeze(y_pred[n_idx_previous:n_idx_previous+n_idx])
            
            """
            Find the unique indices in the first column of the current smp sub_edge_index. 
            Treat each unique index as a block. The first entry of this block has an index in 
            the second column which it is connected to. The combination of the first pair of 
            these indices is used to find the corresponding index in the complete index list. 
            This block is then copied in the final index matric using the corresponding index. 
            The same thing is done with the y_predictions of this block. Each block in this 
            sub sample is done. THen the next sample is treated etc.
            """
            
            dets_curr = np.unique(sub_edge_indices_original_indices[smp][:,0])
            for det in dets_curr:
                det_index = sub_edge_indices_original_indices[smp][:,0]==det
                # np.sum(det_index)
                target_0_idx = sub_edge_indices_original_indices[smp][np.argwhere(det_index)[0],1] 
                
                # The index in the complete_index_list where this first index-pair is found:
                list_idx = np.argwhere((complete_index_list[:,0]== det)*(complete_index_list[:,1]== target_0_idx))[0][0]
                
                # This index is then used to copy the 'det' block of indices of this sub sample. The same is done for the y_preditions:
                all_indices[list_idx : list_idx+np.sum(det_index),(smp*2):(smp*2)+2] = sub_edge_indices_original_indices[smp][det_index ,0:2]
                all_y_pred_final[list_idx : list_idx+np.sum(det_index),smp] = y_pred_current_smp[det_index]
                
                # if not no_gt_in_smp:
                #     all_gt_final[list_idx : list_idx+np.sum(det_index),] = np.squeeze(sub_gt[smp][det_index])
            n_idx_previous += n_idx
    
    # Now each row in 'all_y_pred_final' is averaged to get the average prediction of each detection in the whole time-lapse:
    all_y_pred_final_average = np.zeros((len(all_y_pred_final),))
    for row in range(len(all_y_pred_final)):
        all_y_pred_final_average[row] = np.mean(all_y_pred_final[row,all_y_pred_final[row,:]>-1]) 
    return all_y_pred_final_average, all_gt_final
    


def filter_predictions(feature_list_roi, adj_pred_prob, threshold):
    adj_pred_prob_filtered_columns = np.zeros((len(feature_list_roi),len(feature_list_roi)))
    adj_pred_prob_filtered_rows = np.zeros((len(feature_list_roi),len(feature_list_roi)))
    for col in range(len(adj_pred_prob)):
        max_prob = np.nanmax(adj_pred_prob[:,col])
        index = adj_pred_prob[:,col] == max_prob
        row = np.argwhere(index)
        if len(row)>1:
            row = row[1][0]# If there are more, take the second one
        else:
            row = row[0][0]
        
        if adj_pred_prob[row,col] > threshold:
            adj_pred_prob_filtered_columns[row,col] = adj_pred_prob[row,col]

    for row in range(len(adj_pred_prob_filtered_columns)):
        index = adj_pred_prob_filtered_columns[row,:] > 0
        cols = np.argwhere(index)
        if len(cols)>2:
            probs = adj_pred_prob_filtered_columns[row,index]
            max_order = np.sort(np.arange(len(probs)))
            cols = cols[max_order]
            cols = cols[:2] # Make sure you take the two indices with the higfhest probabilities.

        for col in cols:
            if adj_pred_prob[row,col] > threshold:
                # adj_pred_prob_filtered_rows[row,col] = adj_pred_prob_filtered_columns[row,col]
                adj_pred_prob_filtered_rows[row,col] = 1
    return adj_pred_prob_filtered_rows



def construct_ID_list(adj_pred, feature_list_roi):
    rows, cols = np.shape(adj_pred)
    ID_list = np.zeros(rows)
    
    # Find the # of cells in the first frame:
    n_cells_in_first_frame = np.sum(feature_list_roi[:,1]==0)
    
    # Start with IDs of these cells:
    ID_list[:n_cells_in_first_frame] = np.arange(n_cells_in_first_frame) + 1
    max_ID = n_cells_in_first_frame +1
    for i in range(rows):
        cell_slice = np.argwhere(adj_pred[i,:])
        if len(cell_slice) > 0: # Note that if there are no links, the ID will just be 0
            
            if ID_list[i]==0:
                # This means a new detection suddenly appeared
                ID_list[cell_slice[0]] = max_ID
                ID_list[i] = max_ID
                max_ID += 1
                    
            if len(cell_slice) == 1:
                ID_list[cell_slice[0]] = ID_list[i]
            else:
                # Original cell:
                ID_list[cell_slice[0]] = ID_list[i]
                
                # New cell:
                # Assign a new ID to this new cell:
                cell_ID_original = str(int(ID_list[i]))
                increment = 1
                flag = 1
                while flag:
                    cell_ID = cell_ID_original + str(increment)
                    cell_ID = int(cell_ID)
                    if np.sum(ID_list==cell_ID)==0:
                        flag = 0
                    else:
                        increment += 1
                ID_list[cell_slice[1]] = cell_ID
    ID_list = np.expand_dims(ID_list,-1)
    return ID_list




class dnn_block(keras.layers.Layer):

    def __init__(self, hidden_units, output_dim,**kwargs):
        super(dnn_block, self).__init__(**kwargs)


        self.hidden_units = hidden_units
        self.output_dim = output_dim

        self.dblock1 = layers.Dense(hidden_units,
                                    kernel_initializer=tf.keras.initializers.VarianceScaling(
                                                                    scale=1.0,
                                                                    mode='fan_in',
                                                                    distribution='truncated_normal',
                                                                    seed=42
                                                                ))
        self.dblock2 = layers.Dense(hidden_units,
                                    kernel_initializer=tf.keras.initializers.VarianceScaling(
                                                                    scale=1.0,
                                                                    mode='fan_in',
                                                                    distribution='truncated_normal',
                                                                    seed=42
                                                                ))
        self.dblock3 = layers.Dense(output_dim,
                                    kernel_initializer=tf.keras.initializers.VarianceScaling(
                                                                    scale=1.0,
                                                                    mode='fan_in',
                                                                    distribution='truncated_normal',
                                                                    seed=42
                                                                ))
        self.ln1 = tf.keras.layers.LayerNormalization()
        self.ln2 = tf.keras.layers.LayerNormalization()
        self.ln3 = tf.keras.layers.LayerNormalization()


    def call(self, inputs):
        #dropout_frac = 0.05
        x = self.dblock1(inputs)
        x = tf.keras.layers.LeakyReLU()(x)
        #x = tf.keras.layers.Dropout(dropout_frac)(x)
        x = self.ln1(x)

        x = self.dblock2(x)
        x = tf.keras.layers.LeakyReLU()(x)
        #x = tf.keras.layers.Dropout(dropout_frac)(x)
        x = self.ln2(x)

        x = self.dblock3(x)
        x = tf.keras.layers.LeakyReLU()(x)
        #x = tf.keras.layers.Dropout(dropout_frac)(x)
        x = self.ln3(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_units": self.hidden_units,
            "output_dim": self.output_dim,
        })
        return config

    # There's actually no need to define `from_config` here, since returning
    # `cls(**config)` is the default behavior.
    @classmethod
    def from_config(cls, config):
        return cls(**config)


def tp(y_true, y_pred):
    detection_mask = tf.identity(y_true)
    tp_acc = tf.math.divide_no_nan(tf.reduce_sum( tf.multiply(detection_mask , y_pred)), tf.reduce_sum(detection_mask))
    return tp_acc

def rn(y_true, y_pred):
    one_ = tf.constant(1, dtype=tf.float32)
    detection_mask_inv = tf.math.abs(tf.multiply(tf.math.negative(one_) , (tf.identity(y_true) - one_)))
    # tn_acc = tf.math.divide_no_nan(tf.reduce_sum( tf.multiply(detection_mask_inv , y_pred)), tf.reduce_sum(detection_mask_inv))
    # return tf.math.subtract(one_,tn_acc)
    
    tn_acc = tf.reduce_sum( tf.multiply(detection_mask_inv , y_pred))
    return tn_acc
# Custom los function with dynamic weights for 0s and 1s



def cust_loss(y_true, y_pred):

    y_true = tf.cast(y_true,dtype=tf.float32)
    y_pred = tf.cast(y_pred,dtype=tf.float32)


    detection_mask = tf.identity(y_true)
    one_ = tf.constant(1, dtype=tf.float32)
    detection_mask_inv = tf.math.abs(tf.multiply(tf.math.negative(one_) , (tf.identity(y_true) - one_)))

    sum_y = tf.reduce_sum( y_true )
    sum_y = tf.cast(sum_y, dtype=tf.float32)
    size_y = tf.size( y_true)
    size_y = tf.cast(size_y, dtype=tf.float32)

    frac = tf.math.divide_no_nan(tf.math.subtract(size_y,sum_y), sum_y)
    LAMBDA_0 = tf.constant(1, dtype=tf.float32)
    LAMBDA_1 = frac


    # OBJECTIVENESS LOSS
    loss0 = tf.multiply(detection_mask_inv , tf.sqrt(tf.square(y_pred - y_true)))

    loss0 = tf.multiply(LAMBDA_0 , loss0 )

    loss1 = tf.multiply(detection_mask , tf.sqrt(tf.square(y_pred - y_true)))

    loss1 = tf.multiply(LAMBDA_1 , loss1 )

    loss = tf.math.add(loss0 , loss1)

    return tf.reduce_sum(loss)

