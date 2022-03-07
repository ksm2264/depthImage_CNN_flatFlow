#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 18:00:28 2021

@author: karl
"""

import numpy as np
import glob
import matplotlib.pyplot as plt
import scipy.io as sio

result_list = glob.glob('results/depth/*/0.npy');

all_means = []
all_medians = []


mat = sio.loadmat('leg_lengths_npy.mat',struct_as_record=False,squeeze_me=True)
ll = mat['leg_length']
#ll = list(ll)
#ll = ll[:5]+ll[6:]

ll_order = np.array((4,0,9,6,5,2,3,7,8,1))

ll = np.array(ll)[ll_order]

ll = list(ll)
ll = ll[:4]+ll[5:]

sorted_idx = np.argsort(ll)


#all_medians = all_medians[sorted_idx]

for result_npy in result_list:
    
    this_result_dict = np.load(result_npy,allow_pickle=True).item()
    
    this_mean = this_result_dict['mean']
    this_median = this_result_dict['median']
    
    this_subj = result_npy.split('/')[-2].split('.')[0]
    
    if this_subj!='s6':
        
        all_means.append(this_mean)
        all_medians.append(this_median)
plt.close('all')        
fr = plt.gca()
for ii in range(9):
    plt.plot(ii,all_medians[int(sorted_idx[ii])],'.',markersize=14,label=None)
    
plt.plot([-0.5,9],[0.5,0.5],linestyle='--',color='black',label='Chance')
plt.xlim([-0.5,9])
fr.axes.xaxis.set_ticklabels([])
plt.ylabel('Median AUC')
plt.rc('font',size=14)
leg = plt.legend()

means = []
medians = []

for idx,result_npy in enumerate(result_list):
    
    this_result_dict = np.load(result_npy,allow_pickle=True).item()
    
    this_mean = this_result_dict['mean']
    this_median = this_result_dict['median']
    
    means.append(this_mean)
    medians.append(this_median)
    
means = np.stack(means)
medians=np.stack(medians)
sio.savemat('/home/karl/thesis/projects/foothold_finding/CNN_results.mat',{'means':means,'medians':medians})