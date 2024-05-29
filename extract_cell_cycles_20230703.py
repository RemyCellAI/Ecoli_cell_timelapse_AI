# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 14:09:02 2023

@author: Orange
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

from detect_cycle_def_20230801 import detect_cycle 

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



""" Real data """

experiment_paths = [
                    'C:\\Users\\Orange\\Documents\\Experiments\\ssb_project\\20230218_WT\\568\\',
                    ]


for exp in range(len(experiment_paths)):
result_files = load_file_list(experiment_paths,extension='npy')

""" Create the Result plot directory: """
channel_result_plots_path = image_path + 'cycle_plots\\'
if not (os.path.exists(channel_result_plots_path)): # Check if the path already exists
    os.mkdir(channel_result_plots_path)


for series in range(result_files):
    
    filename = result_files[series]
    
    series = 0
    with open(image_path + filename, 'rb') as f:
    
        img_stack_filename = np.load(f, allow_pickle=True)
        cropped_channels = np.load(f, allow_pickle=True)
        channel_ROIs = np.load(f, allow_pickle=True)
        results_tracked = np.load(f, allow_pickle=True)
        masks = np.load(f, allow_pickle=True)
        mask_coordinates = np.load(f, allow_pickle=True)
        cell_contours = np.load(f, allow_pickle=True)
        rotated_contours = np.load(f, allow_pickle=True)
        adj_matrix = np.load(f, allow_pickle=True)
        results_foci = np.load(f, allow_pickle=True)
        results_n_foci = np.load(f, allow_pickle=True)
        used_foci_file_names = np.load(f, allow_pickle=True)
        
    
    all_cycles_in_series = []
    all_non_cycles_in_series = []
    all_cycle_foci_in_series = []
    all_non_cycle_foci_in_series = []
    cycle_tracks_in_series = []
    non_cycle_tracks_in_series = []
    
    for roi in range(len(results_tracked)):

        cell_data = results_tracked[roi]
        foci_data = results_foci[roi]
        start_frame = 5
    
        roi = 0
        
        all_cycles, all_non_cycles, all_cycle_foci, all_non_cycle_foci, cycle_tracks, non_cycle_tracks = detect_cycle(cell_data, foci_data, roi, channel_result_plots_path, filename, start_frame)

        all_cycles_in_series.append(all_cycles)
        all_non_cycles_in_series.append(all_non_cycles)
        all_cycle_foci_in_series.append(all_cycle_foci)
        all_non_cycle_foci_in_series.append(all_non_cycle_foci)
        cycle_tracks_in_series.append(cycle_tracks)
        non_cycle_tracks_in_series.append(non_cycle_tracks)



    """ Save the result list """
    name = channel_result_plots_path + 'Cell_cycles_and_non_cycles_of_' + experiment + '_' + filename[:-4] + '.npy'
    fptr = open(name, "wb")  # open file in write binary mode
    pickle.dump(all_cycles_in_series, fptr)  # dump list data into file 
    pickle.dump(all_non_cycles_in_series, fptr)  # dump list data into file 
    pickle.dump(all_cycle_foci_in_series, fptr)  # dump list data into file 
    pickle.dump(all_non_cycle_foci_in_series, fptr)  # dump list data into file 
    pickle.dump(cycle_tracks_in_series, fptr)  # dump list data into file 
    pickle.dump(non_cycle_tracks_in_series, fptr)  # dump list data into file 

    fptr.close()  # close file pointer
    
    print('Cycle data saved in: ' + name)







# min_frames = 10
# if np.shape(results_tracked[roi])[-1]==16:
#     traces = np.unique(results_tracked[roi][:,15])
#     traces = traces[traces > 0] # track 0 is a possible detection error
#     n_traces = len(traces)
#     i = 1
#     plt.ion()
#     # plt.figure()
#     # for track in traces:
#     for t in range(12):
#         track = int(traces[t])
#         index = results_tracked[roi][:,15]==track
#         frames = results_tracked[roi][index,1]
        
#         if frames[-1]-frames[0] > min_frames:
            
#             cell_data = results_tracked[roi][index,:]
            
#             detect_cycle(cell_data, roi, track, time_interval, channel_result_plots_path, filename)
              
            
#             # cell_lengths = results_tracked[roi][index,12]
#             # plt.subplot(6,2,i)
#             # plt.plot(frames, cell_lengths)
#             # plt.title(int(track))
#             # i += 1
            
            
        