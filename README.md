# Ecoli_cell_timelapse_AI
This script detects, segments, and tracks E. coli cells in microfluidic channels. Python, Tensorflow

The timelapse processing is called in the 'main' file, where you can define the folders containing the tiff stacks and (optionally) the csv files containing the detected fluorescent proteins in the cells. Note that you can comment out the definitions in the main definition you don't need. For example, if you don't have fluorescent proteins to be analyzed.

The main file:
'time_lapse_processing_MAIN.py'

This file calls main definition, which runs through each task using the corresponding definitions in the utility files. The required library versions can be found in the 'lib_versions.txt' file.

The program treats one field of view at a time. It uses fluorescent images of E. coli cells in the 568 nm tiff file to detect and segment the cells in the microchannels. Later the 458 nm tiff images are loaded to extract the SSB foci data using the detection and segmentation results, and the foci locations in the csv file. This csv file is created by processing the 458 nm tiff file in a public ImageJ library. (https://github.com/SingleMolecule/smb-plugins/tree/master/jar)

1) The main definition starts with loading the image data from the 568 nm tiff file (containing the fluorescently marked E. coli cells).
2) The microchannels in the field of view are detected with the feature proposal ML model. These microchannels are cropped to 64x256 pixel crops in all frames. A stack of 64x256 microchannel crops (of all frames) is called an ROI.
3) Each ROI is processed by the object detection algorithm. The output is the bounding box probabilities, coordinates and dimensions.
4) Each ROI is processed by the non-max-suppression algorithm which uses the output of the object detection to filter out redundant detections of the cells.
5) Each ROI is processed by the cell segmentation algorithm using the filtered bounding box coordinates and dimensions. There are 4 segmentation models, each for a different category of bounding box length. The bounding box dimensions are used to determine what segmentation model will be used, and are used to crop the cell in the 64x256 image with the crop dimensions corresponding to the chosen segmentation model.
6) The segmented cells are measured by rotating them until the maximum length is reached. This is considered the cell laying perfectly flat.
7) The bounding box information is used by the tracking algorithm to create tracking sub-samples.
8) The tracking sub-samples are processed by the tracking model. The results of the sub-samples are combined to derive the average edge predictions. The predictions are thresholded and the result is used to construct cell lineages.
9) The 458 nm tiff file is loaded and cropped (64x256) using the results of the 568 nm images. This is for visual output later on.
10) The 458 nm foci data in the csv file is loaded.
11) Using the segmentation results, the foci data of each cell is extracted from the csv file.

After segmentation (step 5) a "result" list is created. Each sub-list is an array with all the information of the cells in an ROI. Each row of the array represents 1 cell. After each step processed information of the cell is added to this result list. The result list is central for all the algorithms including the tracking algorithm. The columns of a result array are:

tiff series (field of view tiff file)

ROI

frame

total number of cells in the ROI

cell number

bounding box centroid coordinate x

bounding box centroid coordinate y

bounding box min x coordinate

bounding box max x coordinate

bounding box min y coordinate

bounding box max y coordinate

cell centroid coordinate x

cell centroid coordinate y

cell width (in mu)

cell height (in mu)

cell angle (deg)

cell tracking ID



After the extraction of the SSB foci from the 458 nm csv file, a version of this results list is made in which each row of the array represents a foci. Therefore, if there are 3 foci in a cell, the info of that cell is copied from the results list 3 times, and the foci data is added. 
