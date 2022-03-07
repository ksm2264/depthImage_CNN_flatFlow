#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 17:58:32 2021

@author: karl
"""

import numpy as np
import matplotlib.pyplot as plt

means=[]
medians=[]

model_type = ['JAC','JAW','mag_diff']

plt.close('all')

model_type = model_type[:2]

for m_str in model_type:
    this_means = []
    this_medians = []
    plt.figure(1)
    for idx in range(15):
        
        this_dict = np.load('results/depth/{mdl_type}/{idx}.npy'.format(idx=idx,mdl_type=m_str),allow_pickle=True).item()
        this_means.append(this_dict['mean'])
        this_medians.append(this_dict['median'])
    
    
    plt.hist(this_means)
#    plt.vlines(np.median(this_means),plt.ylim()[0],plt.ylim()[1])
    means.append(np.median(this_means))
    
    plt.figure(2)
    plt.hist(this_medians)
#    plt.vlines(np.median(this_medians),plt.ylim()[0],plt.ylim()[1])
    medians.append(np.median(this_medians))

    
plt.figure(1)
plt.legend(model_type)
plt.title('Means')

plt.figure(2)
plt.legend(model_type)
plt.title('Medians')

for idx,model_str in enumerate(model_type):
    
    print(model_str+' mean: '+str(means[idx]))
    print(model_str+' median: '+str(medians[idx]))