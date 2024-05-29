# -*- coding: utf-8 -*-


import numpy as np
import os



# from time_lapse_processing_class_20230125 import time_lapse_processing
from time_lapse_processing_class_20240105 import time_lapse_processing

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
fpn_model_path = os.getcwd() + "\\fpn_model\\fpn_20230303\\further\\"
detection_model_path = os.getcwd() + "\\cell_detection\\Label_8x32x20_with_older_with_translations\\no_dense_layers_7x7_5x5_3x3_kernels\\run11_extra_smp_20240103\\"
segmentation_model_path = os.getcwd() + "\\cell_segmentation\\cell_segmentation_20230109\\epoch_21_to_40\\"
tracking_model_path = os.getcwd() + '\\cell_tracking\\mpnn_no_ta_64units_20230703\\'

# Cropping settings:
rot_angle = 90

# FPN Detection settings:
prediction_threshold_fpn = 0.95
iou_threshold_fpn = 0
    
# Detection settings:
prediction_threshold = 0.8
iou_threshold = 0.20

# Segmentation settings:
segmentation_threshold = 0.3
measurement_threshold = 0.05

# Tracking settings
track_x_threshold = 225

mu_per_pixel = 0.110
# mu_per_pixel = 0.16

# Foci settings
min_distance = 2
max_value_histogram = 2000
foci_peak_threshold = 0


# Data path:

exp_root_path = 'C:\\Users\\Orange\\Desktop\\New folder\\'

# Time lapse experiment path:
frame_interval = 5
experiment_path = [

    # 'WT\\20230520_WT\\cycle\\20230520_143051_448_no_UV_5min\\TIFF\\',
    
    ]




# Cycle detection settings:
start_frame = 0 # The tiff stack and cycle extraction will begin at this frame number. This is because sometimes there are no cells in the channel in the first few frames.
global_amp_fraction_for_detection = 0.3 # 30% cell length reduction is considered a cell split
min_n_frames = 4 # Minimum number of frames for the track to be considered.
n_discon = 3 # Max number of skipped frames allowed in the tracked cells
x_threshold = 225 # Threshold for the x coordinate. Cells positioned right from this point will not be analyzed.
min_cycle_slope = 0.6 * frame_interval/60 # Minimum slope the length curve of a cycle may have to be considered a cycle. --> (length in mu) * frame_interval/60
max_dlength_step = 0.6 # Maximum length increase from one frame to another in mu. A length increase above this value will be regarded as a detection error and the cycle will be discarted.
min_cycle_frames = 5 # Minimum number of frames in a cycle for the cyle to be considered. 



for exp in range(len(experiment_path)):
    
        
    image_path = exp_root_path + experiment_path[exp] + '568\\TIFF_corrected\\'
    ssb_image_path = exp_root_path + experiment_path[exp] + '458\\TIFF_corrected\\'

    
    # Create the output folder
    detected_ROI_image_path = image_path + 'detected_ROI_images\\'
    
    if not (os.path.exists(detected_ROI_image_path)): # Check if the path already exists
        os.mkdir(detected_ROI_image_path)
    
    
    
    
    
    test = time_lapse_processing()
    test.load_all_models(fpn_model_path, detection_model_path, segmentation_model_path, tracking_model_path)
    
    
    
    """ Get the file list of all detection result files """
    filenames_tiff = load_file_list(image_path,'tif')

    for i in range(len(filenames_tiff)):

        img_stack_filename = filenames_tiff[i]
        ssb_img_stack_filename = img_stack_filename[:-8] + "_458.tif"
        
        """ TO DO get filename from the tif list instead """
        
        print('\n')
        print('\n++++++++++++++ Processing stack-file: ' + img_stack_filename + ' ++++++++++++++')
        

        n_ROIs, channel_ROIs, img_example = test.load_frames_and_detect_channels(image_path, detected_ROI_image_path, img_stack_filename, start_frame, rot_angle, prediction_threshold_fpn, iou_threshold_fpn, enhance_factor=1)
        
        if n_ROIs > 0:
            ROIs_to_be_extracted = np.linspace(0,n_ROIs-1,n_ROIs).astype(int)
            
            cropped_channels_init = test.crop_channels(ROIs_to_be_extracted, channel_ROIs)
            
            object_predictions, ROI_dims = test.run_detection_model()
            
            results_original, channel_ROIs_NMS, cropped_channels = test.NMS_for_selected_ROIs(prediction_threshold, iou_threshold)
            
            cell_contours, masks, mask_coordinates = test.crop_cells_and_segment(segmentation_threshold)
            
            results_, rotated_contours,  masks_, mask_coordinates_, cell_contours_ = test.get_cell_info(measurement_threshold, mu_per_pixel)
            
            results, adj_matrix, cell_contours, rotated_contours, masks, mask_coordinates, channel_ROIs_NMS, cropped_channels = test.track_cells(track_x_threshold)
            
            cropped_channels_ssb = test.load_and_crop_ssb_stack(ssb_image_path, ssb_img_stack_filename, start_frame, rot_angle=90)
    
            results_foci, results_n_foci, used_foci_file_names = test.extract_foci_data(min_distance, start_frame)
                
            test.save_results(foci=True)
            
            all_cycles_in_series, all_non_cycles_in_series, cycle_tracks_in_series, non_cycle_tracks_in_series = test.extract_cycle_data(x_threshold, global_amp_fraction_for_detection, min_n_frames, n_discon, min_cycle_slope, max_dlength_step, min_cycle_frames)
        
            all_cycle_foci_in_series, all_non_cycle_foci_in_series = test.extract_foci_from_cycle_data(x_threshold)
    
            # test.make_gif_movie(frame_interval, bbox=True, contours=True, IDlabels=True)
    
            # test.make_gif_movie_wit h_foci(frame_interval, max_value_histogram, foci_peak_threshold, bbox=True, contours=True, IDlabels=True)
