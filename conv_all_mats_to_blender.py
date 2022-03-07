#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 20:57:36 2021

@author: karl
"""

import glob
import numpy as np
import scipy.io as sio

fileList = glob.glob('/media/karl/DATA/pupilShadowMesh/*.mat')

for file in fileList:
    
    
    
    mat = sio.loadmat(file,struct_as_record=False,squeeze_me=True)
    
    this_dict = {}
    
    for keys in mat.keys():
        
        this_dict[keys] = mat[keys]
        
    
    this_str = file.split('/')[-1].split('.')[0]
    np.save('/media/karl/DATA/pupilShadowMeshBlender/'+this_str+'.npy',this_dict)