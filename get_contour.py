# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 14:36:59 2022

@author: Orange
"""
import numpy as np

def get_contour(mask, flip_rotate_mask = True):
    
    
    if flip_rotate_mask:
        #mask = flip(rot90(mask)
        mask = np.flip(mask, axis=1)
        mask = np.rot90(mask, k=1, axes=(0, 1))
        [w, h] = np.shape(mask)
    else:
        [h, w] = np.shape(mask)
    
    X = []
    Y = []

    for i in range(w-1):
        for j in range(h-1):
            if mask[i,j] == 1: # treat the cell pixels only
                if (i == 1) or (i == w) or (j == 1) or (j == h): # a pixel equal to 1 at the border, is a contour pixel
                    X.append(i)
                    Y.append(j)

                else: # check the surrounding of the pixel
                    check = np.sum([mask[ i-1,j ], mask[ i+1,j ], mask[ i,j-1 ], mask[ i,j+1 ], mask[ i-1,j+1 ], mask[ i-1,j-1 ], mask[ i+1,j+1 ], mask[ i+1,j-1 ]])
                    if check < 8:
                        X.append(i)
                        Y.append(j)
    
    m = len(X)
    contour_coordinates = np.ndarray((m,2),dtype='float32')
    for i in range(m):
        contour_coordinates[i,0] = X[i]
        contour_coordinates[i,1] = Y[i]
    
    return contour_coordinates