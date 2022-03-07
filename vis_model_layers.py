#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 21:19:01 2021

@author: karl
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

model_type = ['flow']

plt.close('all')

means,medians = [],[]

for m_str in model_type:
    this_means = []
    this_medians = []
    plt.figure(1)
    for idx in range(32):
        
        this_dict = np.load('results/{mdl_type}/{idx}.npy'.format(idx=idx,mdl_type=m_str),allow_pickle=True).item()
        this_means.append(this_dict['mean'])
        this_medians.append(this_dict['median'])
    
    
    
    
best_model_idx = np.argmax(this_medians)

best_model = tf.keras.models.load_model('models_{this_mode}/{ee}'.format(this_mode='flow',ee=best_model_idx))


filters,biases = best_model.layers[3].get_weights()