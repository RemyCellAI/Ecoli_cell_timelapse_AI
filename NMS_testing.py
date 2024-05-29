
import numpy as np
import matplotlib.pyplot as plt
import os


def NMS_objects(object_predictions, ROI_dims, prediction_threshold, iou_threshold):
    # Calculate the min max values of the boxes
    xmin = np.zeros((len(ROI_dims),1))
    xmax = np.zeros((len(ROI_dims),1))
    ymin = np.zeros((len(ROI_dims),1))
    ymax = np.zeros((len(ROI_dims),1))
    
    
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

def create_label_from_bb(bboxes, n_boxes_h, n_boxes_w, anchor_dim, aspect_ratio_anchor, grid_dim):

    YOLO_label = np.zeros((n_boxes_h, n_boxes_w, 5 * len(anchor_dim)))
    
    ## Create the label
    for bb in range(len(bboxes)):
    
        x_cent = bboxes[bb,0]
        y_cent = bboxes[bb,1]
        x_min = bboxes[bb,2]
        x_max = bboxes[bb,3]
        y_min = bboxes[bb,4]
        y_max = bboxes[bb,5]
        
        aspect_ratio_index, label_index_start, label_index_end = find_aspect_ratio_index(x_max, x_min, y_max, y_min, aspect_ratio_anchor)
        boxes, conv_idx = get_parameters(x_cent, y_cent, x_max, x_min, y_max, y_min, grid_dim)
        YOLO_label[conv_idx[1], conv_idx[0], label_index_start : label_index_end] = np.hstack((1, boxes)) # objectiveness + coordinates and dimensions + class
    
    return YOLO_label


def find_aspect_ratio_index(x_max, x_min, y_max, y_min, aspect_ratio_anchor):

    bw = x_max - x_min
    bh = y_max - y_min
    aspect_ratio_cell = bw / bh
    aspect_ratio_delta = np.abs(aspect_ratio_cell - aspect_ratio_anchor)
    aspect_ratio_index = np.argwhere(aspect_ratio_delta == np.min(aspect_ratio_delta))[0][0]
    label_index_start = int( aspect_ratio_index * 5 )
    label_index_end = int(( aspect_ratio_index + 1) * 5 )
    
    return aspect_ratio_index, label_index_start, label_index_end

def get_parameters(x_cent, y_cent, x_max, x_min, y_max, y_min, grid_dim):

    corner_coord_x = int(np.floor( x_cent / grid_dim ))
    corner_coord_y = int(np.floor( y_cent / grid_dim ))
    
    bx = x_cent
    by = y_cent
    bw = x_max - x_min
    bh = y_max - y_min
    
    # In the training data the absolute dimenions are given:
    boxes = [bx, by, bw, bh]
    conv_idx = np.array([corner_coord_x, corner_coord_y]).astype(int)
    
    return boxes, conv_idx

def load_YOLO_example(source_path, npy_file_name):
    with open(source_path + npy_file_name, 'rb') as f:
        img_stack = np.load(f, allow_pickle=True)
        label_stack = np.load(f, allow_pickle=True)
        file_name_stack = np.load(f, allow_pickle=True)

    return img_stack, label_stack, file_name_stack


def plot_label(img, YOLO_label, file_name):
    """
    This definition tests the YOLO label by using the YOLO label and the image to show the detections and the YOLO grid cells. 
    """
    
    
    plt.figure()
    plt.imshow(img, cmap='bone')
    plt.title(file_name)
    
    for anc in range(len(anchor_dim)):
    
        grid_locations = np.argwhere(YOLO_label[:,:,anc*5])
        
        
        for bb in range(len(grid_locations)):
            # Objectiveness location
            x = grid_locations[bb,0]
            y = grid_locations[bb,1]
            
            x_cent = YOLO_label[x,y,anc*5 + 1]
            y_cent = YOLO_label[x,y,anc*5 + 2]
            x_min = x_cent - (YOLO_label[x,y,anc*5 + 3] / 2)
            x_max = x_cent + (YOLO_label[x,y,anc*5 + 3] / 2)
            y_min = y_cent - (YOLO_label[x,y,anc*5 + 4] / 2)
            y_max = y_cent + (YOLO_label[x,y,anc*5 + 4] / 2)
            
            corner_coord_x = np.floor( x_cent / grid_dim ) * grid_dim
            corner_coord_y = np.floor( y_cent / grid_dim ) * grid_dim
            
            obj_x_min = corner_coord_x
            obj_x_max = corner_coord_x + grid_dim
            obj_y_min = corner_coord_y
            obj_y_max = corner_coord_y + grid_dim
        
            ROI_coord_X = [obj_x_min , obj_x_max , obj_x_max , obj_x_min , obj_x_min]
            ROI_coord_Y = [obj_y_min , obj_y_min , obj_y_max , obj_y_max , obj_y_min]
            plt.plot(ROI_coord_X,ROI_coord_Y, 'b');
            
            ROI_coord_X = [x_min , x_max , x_max , x_min , x_min]
            ROI_coord_Y = [y_min , y_min , y_max , y_max , y_min]
            plt.plot(ROI_coord_X,ROI_coord_Y, 'r');
            
            plt.plot(x_cent, y_cent, 'y*');
            
            

def plot_bb(img, smp, bb_final):
    
    
    
    plt.figure()
    plt.imshow(img, cmap = 'gray', origin='upper')
    plt.axis('off')
    plt.title('from_boxes_smp_' + str(smp))
    
    for bb in range(len(bb_final)):
    
            x_min = bb_final[bb,2]
            x_max = bb_final[bb,3]
            y_min = bb_final[bb,4]
            y_max = bb_final[bb,5]
            
            ROI_coord_X = [x_min, x_max, x_max, x_min, x_min]
            ROI_coord_Y = [y_min, y_min, y_max, y_max, y_min]
            
    
            color = 'red'
                
            plt.plot(ROI_coord_X, ROI_coord_Y,color = color)
            plt.text(x_min, y_max, str(bb) , bbox=dict(facecolor='red', alpha=0.8), fontsize=12)
    
    

from time_lapse_processing_class_20231122 import time_lapse_processing

# Model Paths:
fpn_model_path = os.getcwd() + "\\fpn_model\\fpn_20230303\\further\\"
detection_model_path = os.getcwd() + "\\cell_detection\\training_8x32x20_20231121_without_translation\\run04\\"
segmentation_model_path = os.getcwd() + "\\cell_segmentation\\cell_segmentation_20230109\\epoch_21_to_40\\"
tracking_model_path = os.getcwd() + '\\cell_tracking\\mpnn_no_ta_64units_20230703\\'


image_path = 'C:\\Users\\Orange\\Documents\\cell_NN\\Thesis_code\\detection\\samples_20231110\\all_labels_together_8x32x20\\'

file_name = 'detection_training_set_64_256_8_32_4anchors_no_translations_VAL.npy'
cropped_channels_total, label_stack, file_name_stack = load_YOLO_example(image_path, file_name)


cropped_channels = cropped_channels_total[:,:,:]


n_frames = len(cropped_channels)
roi = 0

grid_dim = 8
anchor_dim = np.array(
                [
                [10,10],
                [10,30],
                [10,80],
                [10,140],
                ]
            )

aspect_ratio_anchor = anchor_dim[:,1] / anchor_dim[:,0]

img_w = 256
img_h = 64
n_boxes_w = int(img_w / grid_dim)
n_boxes_h = int(img_h / grid_dim)



test = time_lapse_processing()
test.load_all_models(fpn_model_path, detection_model_path, segmentation_model_path, tracking_model_path)


img_stack_filename = 'testing'
ROIs_to_be_extracted = np.array([0])

object_predictions_total, ROI_dims_total, y_pred = test.run_detection_model_testing_mode(cropped_channels, ROIs_to_be_extracted, n_frames, image_path)


prediction_threshold = .3
iou_threshold =.3





true_positives = np.zeros((n_frames,3,4))
false_positives = np.zeros((n_frames,3,4))

true_positives_all = np.zeros((n_frames,3))
false_positives_all = np.zeros((n_frames,3))

smps = np.arange(20) + 20
for smp in smps:
    
# for smp in range(n_frames):
    object_predictions = object_predictions_total[roi, smp,:]
    ROI_dims = ROI_dims_total[roi, smp,:,:]
    
    # object_predictions_reshaped = np.reshape(object_predictions,[8,32,4])
    
    obj_prediction_final, bb_final = NMS_objects(object_predictions, ROI_dims, prediction_threshold, iou_threshold)
    
    bboxes = bb_final
    label_pred = create_label_from_bb(bboxes, n_boxes_h, n_boxes_w, anchor_dim, aspect_ratio_anchor, grid_dim)
    
    
    
    img = cropped_channels[smp,:,:]
    # plot_bb(img, smp, bb_final)
    plot_label(img, label_pred, 'from_label_smp_' + str(smp))
    
    
    
    label_true = label_stack[smp]
    
    for anch in range(len(anchor_dim)):
        slice_number = anch * 5
        
        l_h, l_w, l_d = np.shape(label_true)
        y_true = np.reshape( label_true[:,:, slice_number] , l_h * l_w) 
        y_pred = np.reshape( label_pred[:,:, slice_number] , l_h * l_w) 
        
        object_index = y_true == 1

        true_positives[smp, 0, anch] = np.sum(y_true)
        true_positives[smp, 1, anch] = np.sum(y_pred[object_index])
        if true_positives[smp, 0, anch] > 0:
            true_positives[smp, 2, anch] = true_positives[smp, 1, anch] / true_positives[smp, 0, anch]
        
        non_object_index = np.invert(object_index)
        
        false_positives[smp, 0, anch] = np.sum(non_object_index)
        false_positives[smp, 1, anch] = np.sum(y_pred[non_object_index])
        if false_positives[smp, 0, anch] > 0:
            false_positives[smp, 2, anch] = false_positives[smp, 1, anch] / false_positives[smp, 0, anch]
        
        
        
    y_pred = np.zeros(1,)
    y_true = np.zeros(1,)
    for anch in range(len(anchor_dim)):
        slice_number = anch * 5
        
        l_h, l_w, l_d = np.shape(label_true)
        y_true_temp = np.reshape( label_true[:,:, slice_number] , l_h * l_w) 
        y_pred_temp = np.reshape( label_pred[:,:, slice_number] , l_h * l_w) 
        
        y_pred = np.hstack((y_pred, y_pred_temp))
        y_true = np.hstack((y_true, y_true_temp))
        
    object_index = y_true == 1

    true_positives_all[smp, 0] = np.sum(y_true)
    true_positives_all[smp, 1] = np.sum(y_pred[object_index])
    if true_positives_all[smp, 0] > 0:
        true_positives_all[smp, 2] = true_positives_all[smp, 1] / true_positives_all[smp, 0]
    
    non_object_index = np.invert(object_index)
    
    false_positives_all[smp, 0] = np.sum(non_object_index)
    false_positives_all[smp, 1] = np.sum(y_pred[non_object_index])
    if false_positives_all[smp, 0] > 0:
        false_positives_all[smp, 2] = false_positives_all[smp, 1] / false_positives_all[smp, 0]
    


all_positives = np.sum(true_positives[:, 1, :])
all_negatives = np.sum(false_positives[:, 1, :])

bins = 100
plt.figure()
plt.title('True Positives ' + str(all_positives) + ' detections  obj_thres = ' + str(prediction_threshold) +  '  iou_thres = ' + str(iou_threshold))
for anch in range(len(anchor_dim)):
    index =  true_positives[:, 0, anch] > 0
    plt.hist(true_positives[index, 2, anch], bins, label = str(anch))
plt.legend()
plt.xlim([0,1])
plt.ylim([0,80])
    
plt.figure()
plt.title('False Positives ' + str(all_negatives) + ' detections  obj_thres = ' + str(prediction_threshold) +  '  iou_thres = ' + str(iou_threshold))

for anch in range(len(anchor_dim)):
    index =  false_positives[:, 0, anch] > 0
    plt.hist(false_positives[index, 2, anch], bins, label = str(anch))
plt.legend()
plt.xlim([0,1])
plt.ylim([0,80])






all_positives = np.sum(true_positives_all[:, 1])
all_negatives = np.sum(false_positives_all[:, 1])

bins = 100
plt.figure()
plt.title('True Positives ' + str(all_positives) + ' detections  obj_thres = ' + str(prediction_threshold) +  '  iou_thres = ' + str(iou_threshold))
index =  true_positives_all[:, 0] > 0
plt.hist(true_positives_all[index, 2], bins)
plt.xlim([0,1])
plt.ylim([0,80])
    
plt.figure()
plt.title('False Positives ' + str(all_negatives) + ' detections  obj_thres = ' + str(prediction_threshold) +  '  iou_thres = ' + str(iou_threshold))
index =  false_positives_all[:, 0] > 0
plt.hist(false_positives_all[index, 2], bins)
plt.xlim([0,1])
plt.ylim([0,80])


