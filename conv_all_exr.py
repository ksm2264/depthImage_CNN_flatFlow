#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 15:12:28 2021

@author: karl
"""

import glob
import OpenEXR as oxr
import Imath, array
import numpy as np
import scipy.io as sio
import os

fileList = glob.glob('/media/karl/DATA/retinalImageDepth/*/*.exr')

for idx,file in enumerate(fileList):
    out_str = file.split('.')[:-1][0]
    
    if not os.path.exists(out_str+'.mat'):
        
        print(idx/len(fileList))
        
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        golden = oxr.InputFile(file)
        dw = golden.header()['dataWindow']
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
        redstr = golden.channel('R', pt)
        red = np.fromstring(redstr, dtype = np.float32)
        red.shape = (size[1],size[0])
        
        
        
        sio.savemat(out_str+'.mat',{'map':red})