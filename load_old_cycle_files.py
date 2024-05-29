# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 19:20:03 2023

@author: Orange
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

file_path = 'C:\\Users\\Orange\\Documents\\Experiments\\ssb_project\\TEST\\cycle_plots\\old\\'
file_name = 'Cell_cycles_of_20230218_SSB_WT_Bg_Corr_Series001_568.npy'
file_name_ssb = 'All_selected_cell_cycles_of_20230218_SSB_WT.npy'

fptr = open(file_path + file_name, "rb")  # open file in read binary mode
all_periods_detected_in_series = pickle.load(fptr)
all_growth_rate_detected_in_series = pickle.load(fptr)
all_min_cell_length_in_series = pickle.load(fptr)
all_max_cell_length_in_series = pickle.load(fptr)
all_cell_lengths_in_period_in_series = pickle.load(fptr)
all_frames_of_period_in_series = pickle.load(fptr)
fptr.close()


fptr = open(file_path + file_name_ssb, "rb")  # open file in read binary mode
all_cycle_data_compact = pickle.load(fptr)
all_cycles_period_growth_minmax = pickle.load(fptr)
all_cycles = pickle.load(fptr)
all_n_foci_in_cycle = pickle.load(fptr)
channel_counter = pickle.load(fptr)
fptr.close()

