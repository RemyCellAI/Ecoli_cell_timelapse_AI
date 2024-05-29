# Ecoli_cell_timelapse_AI
This class detects, segments, and tracks Ecoli cells in microfluidic channels. Python, Tensorflow

The timelapse processing is called in the 'main' file, where you can define the folders containing the tiff stacks and (optionally) the csv files containing the detected fluorescent proteins in the cells. Note that you can comment out the functions of the class you don't need. For example, if you don't have fluorescent proteins to be analyzed.

'main_time_lapse_20240105.py'

The class is:

'time_lapse_processing_class_20240105.py'

This class calls all the required definitions in the utility files. The required library versions can be found in the 'lib_versions.txt' file.
