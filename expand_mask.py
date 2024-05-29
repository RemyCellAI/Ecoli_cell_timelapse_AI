# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 13:51:57 2023

@author: Orange
"""

import numpy as np


def expand_mask(mask,pixels):
    m_h, m_w = np.shape(mask)
    mask_expand = np.zeros((m_h,m_w))
    for i in range(pixels):
        for h in range(m_h):
            for w in range(m_w):
                if mask[h,w] == 1:
                    mask_expand[h,w] = 1
                    
                    if h+1 < m_h:
                        mask_expand[h+1,w] = 1
                    if h-1 >= 0:
                        mask_expand[h-1,w] = 1
                    
                    if (h+1 < m_h) and (w+1 < m_w):
                        mask_expand[h+1,w+1] = 1
                    if (h-1 >= 0) and (w+1 < m_w):
                        mask_expand[h-1,w+1] = 1
                    if (h+1 < m_h) and (w-1 <= 0):
                        mask_expand[h+1,w-1] = 1
                    if (h-1 <= 0) and (w-1 <= 0):
                        mask_expand[h-1,w-1] = 1
                    
                    if w+1 < m_w:
                        mask_expand[h,w+1] = 1
                    if w-1 >= 0:
                        mask_expand[h,w-1] = 1
                    
    return mask_expand


