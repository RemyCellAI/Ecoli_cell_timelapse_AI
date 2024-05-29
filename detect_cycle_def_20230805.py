# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt


def find_peaks(cell_lengths, global_amp_fraction_for_detection):
    amp = np.max(cell_lengths) - np.min(cell_lengths)
    amp_thres = -global_amp_fraction_for_detection * amp
    d_length = np.diff(cell_lengths)
    peaks = np.argwhere(d_length <= (amp_thres))
    n_peaks = len(peaks)
    peaks = np.squeeze(peaks)
    peaks = np.array(peaks)
    peaks = peaks.astype(int)
    
    # plt.figure()
    # plt.plot(d_length)
    # plt.plot(d_length,'k.')
    # plt.plot([0,len(d_length)], [amp_thres, amp_thres], 'r')
    # plt.plot(peaks, np.ones((n_peaks))*amp_thres,'k*')
    return peaks, n_peaks

def get_cycles(cell_lengths, frames, peaks, n_peaks, all_cycles, cycle_tracks, tr, min_n_frames, n_discon, min_cycle_slope, max_dlength_step, min_cycle_frames, from_start = True):
    plot_cycles = []
    peaks = peaks + 1
    
    # Offspring cells start from the beginning of a cycle, so the data up to the first peak is a valid cycle:
    if from_start:
        n_peaks += 1
        temp = np.zeros((n_peaks))
        temp[1:] = peaks
        peaks = temp.astype(int)
        peaks = np.squeeze(peaks)
    else:
        peaks = np.squeeze(peaks)
        
    # Extract the cycles:
    for peak in range(n_peaks-1):
        frame =  frames[peaks[peak] : peaks[int(peak+1)]]
        length = cell_lengths[peaks[peak] : peaks[int(peak+1)]]

        if len(frame) >= min_n_frames:        
            if np.max(np.diff(frame)) <= n_discon: # Assert that there are no large discontinuities in the track
                
                if len(frame) >= min_cycle_frames: # The number of frames ust be equal or higher than the threshold for the cycle to be considered.
                    
                    if np.abs(np.max(np.diff(length))) <= max_dlength_step: # Check if there are jumps in length (positive or negative) from one frame to the other that could indicate a detection error. Use the threshold.
                        
                        # Calculate the slope of the cycle:
                        cycle_fit = np.polyfit(frame, length, 1)
                        
                        if cycle_fit[0] >= min_cycle_slope: # Only keep the cycle if the slope is greater than the threshold:
                            
                            all_cycles.append(np.vstack((frame,length)).T)
                            plot_cycles.append(np.vstack((frame,length)).T)
                            cycle_tracks = np.vstack((cycle_tracks, tr))
    return all_cycles, plot_cycles, cycle_tracks

# cell_data = results[0]

def detect_cycle(cell_data, roi, channel_result_plots_path, filename, global_amp_fraction_for_detection, min_n_frames, n_discon, min_cycle_slope, max_dlength_step, min_cycle_frames, x_threshold):
    """
    cell_data = results = [roi_number, frame_number, n_cells, cell_number, x_cent, y_cent, x_min, x_max, y_min, y_max, x_cent, y_cent, width, length, angle]
    
    """


    all_cycles = []
    all_non_cycles = []
    cycle_tracks = np.ndarray((0,1)).astype(int)
    non_cycle_tracks = np.ndarray((0,1)).astype(int)


    # ## Remove the frames prior to the defined starting frame: 
    # # Cell data:
    # start_frame_index = np.argwhere(cell_data[:,1] == start_frame) # Assuming that the start_frame is in this dataset...
    # start_frame_index = np.squeeze(start_frame_index)
    # cell_data = cell_data[start_frame_index[0]:,:] # Only use the data starting from the start frame.
    # # cell_data[:,1] = cell_data[:,1] - start_frame # Correct the peak positions with the defined starting frame.

    # Remove the detections too close to the left border of the FOV:
    # Cell data:
    index = cell_data[:,10] <= x_threshold
    cell_data = cell_data[index, :]

    
    # Find the unique tracks and the track numbers present in the starting frame:
    tracks = np.unique(cell_data[:,15])
    tracks = tracks[tracks>0] # Remove track 0 because it is an indicator of false detections.
    index = cell_data[:,1] == cell_data[0,1] # The first frame is the frame with which cell_data starts. Find all first cells.
    tracks_first_frame = np.unique(cell_data[index,15])
    
    # Start a plot for all tracks:
    n_sub_plots = int(np.ceil(len(tracks)/3))
    plt.ioff()
    plt.figure()
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.suptitle('File: ' + filename[0:-4] + ' ROI: ' + str(roi))
    n_fig = 1
    

    for tr in tracks:
        tr = int(tr)
        index = cell_data[:,15] == tr
        
        if np.sum(index) >= min_n_frames:
            print("Extracting track " + str(int(tr)) + " of ROI " + str(roi) + ' in ' + filename)

            temp = cell_data[index,:]
            frames = np.unique(temp[:,1])
            cell_lengths = temp[:,12]
            
            if len(frames) == len(cell_lengths): # If these are not equal, there have been a tracking error with the same ID appearing twice in a frame.
                
                # Add to sub-plot:
                plt.subplot(n_sub_plots, 3, int(n_fig))
                plt.plot(frames,cell_lengths,'k')
                plt.plot(frames,cell_lengths,'k.')
                plt.legend(['Track ' + str(int(tr))])
                
                # If the cell length plots belong to cells present in the first frame, don't use the data up to the first peak. If it is not, the first data is a valid cycle:
                if np.sum(tracks_first_frame == tr):
                    from_start = False
                else:
                    from_start = True
                    
                    
                # plt.figure()
                # plt.plot(frames,cell_lengths)
                # plt.plot(frames,cell_lengths,'k.')
                
                peaks, n_peaks = find_peaks(cell_lengths, global_amp_fraction_for_detection)
                peaks = peaks.astype(int)
                
                if n_peaks == 2:
                    all_cycles, plot_cycles, cycle_tracks  = get_cycles(cell_lengths, frames, peaks, n_peaks, all_cycles, cycle_tracks, tr, min_n_frames, n_discon, min_cycle_slope, max_dlength_step, min_cycle_frames, from_start)
                    if len(plot_cycles) == 0:
                        plt.text(np.mean(frames), np.mean(cell_lengths), 'NO VALID CYCLES', fontsize=16)
                        
                elif  n_peaks == 1 and from_start: # One full cycle is detected from the beginning
                    all_cycles, plot_cycles, cycle_tracks  = get_cycles(cell_lengths, frames, peaks, n_peaks, all_cycles, cycle_tracks, tr, min_n_frames, n_discon, min_cycle_slope, max_dlength_step, min_cycle_frames, from_start)
                    if len(plot_cycles) == 0:
                        plt.text(np.mean(frames), np.mean(cell_lengths), 'NO VALID CYCLES', fontsize=16)
                        
                elif n_peaks >= 2: # If there are more than 2 peaks, its better to detect the peaks locally with a sliding window:
                    # window_size = int(len(frames)/(len(peaks)-1))
                    window_size = int(len(frames)/2)
                    runs = len(frames)-window_size + 1
                    all_sub_peaks = np.ndarray((0))
                    
                    for run in range(runs):
                        start = 0 + run
                        end  = window_size + run
                        window = cell_lengths[start:end]
                        
                        sub_peaks, _ = find_peaks(window, global_amp_fraction_for_detection)
                        sub_peaks = np.array(sub_peaks + start)
                        all_sub_peaks = np.hstack((all_sub_peaks, sub_peaks))
    
                    peaks = np.unique(all_sub_peaks).astype(int)
                    peaks = peaks.astype(int)
                    
                    all_cycles, plot_cycles, cycle_tracks = get_cycles(cell_lengths, frames, peaks, len(peaks), all_cycles, cycle_tracks, tr, min_n_frames, n_discon, min_cycle_slope, max_dlength_step, min_cycle_frames, from_start)
                    if len(plot_cycles) == 0:
                        plt.plot(frames,cell_lengths,'r')
                        plt.plot(frames,cell_lengths,'r.')
                        plt.text((frames[0]+frames[-1])/2, np.mean(cell_lengths), 'NO VALID CYCLES', fontsize=16)
                        
                elif n_peaks == 0:
                    plot_cycles = []
                    if np.max(np.diff(frames)) <= n_discon: # Assert that there are no large discontinuities in the track
                        temp = np.vstack((frames,cell_lengths)).T
                        all_non_cycles.append(temp)
                        non_cycle_tracks = np.vstack((non_cycle_tracks, tr))
                    
                        plt.plot(frames,cell_lengths,'b')
                        plt.plot(frames,cell_lengths,'b.')
                        plt.text((frames[0]+frames[-1])/2, np.mean(cell_lengths), 'SAVED AS NO CYCLE', fontsize=16)
                    else:
                        plt.plot(frames,cell_lengths,'r')
                        plt.plot(frames,cell_lengths,'r.')
                        plt.text((frames[0]+frames[-1])/2, np.mean(cell_lengths), 'TOO MANY MISSSING FRAMES', fontsize=16)
                    
                elif n_peaks == 1:
                    plot_cycles = []
                    plt.plot(frames,cell_lengths,'r')
                    plt.plot(frames,cell_lengths,'r.')
                    plt.text((frames[0]+frames[-1])/2, np.mean(cell_lengths), 'NOT SAVED', fontsize=16)
                    
                
                # Add to plot:
                for i in range(len(plot_cycles)):
                    plt.plot(plot_cycles[i][:,0],plot_cycles[i][:,1],'g')
                    plt.plot(plot_cycles[i][:,0],plot_cycles[i][:,1],'g.')
                n_fig += 1
        
    if np.size(cycle_tracks) > 1:
        cycle_tracks = np.squeeze(cycle_tracks)
    if np.size(non_cycle_tracks) > 1:
        non_cycle_tracks = np.squeeze(non_cycle_tracks)
    


    """ Plot and save for visual inspection """
    plt.savefig(channel_result_plots_path + filename[0:-4] + '_ROI0' + str(roi) + '.jpg', dpi=100)
    plt.close()
        
    return all_cycles, all_non_cycles, cycle_tracks, non_cycle_tracks, tracks_first_frame
        




