
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 13:34:05 2021

@author: karl
"""
from keras.utils import Sequence
import numpy as np   
import scipy.io as sio
import glob
import cv2
from PIL import Image
import imutils

imutils.rotate_bound

#def getEvens(matList):

def getOdds(fileList):

    oddsList = []
    for filename in fileList:
        
        if int(filename.split('/')[-2].split('_')[1])%2==1:
            oddsList.append(filename)
            
    
    return oddsList


def getEvens(fileList):

    evensList = []
    for filename in fileList:
        
        if int(filename.split('/')[-2].split('_')[1])%2==0:
            evensList.append(filename)
            
    
    return evensList

testList = glob.glob('/media/karl/DATA/retinalCNN_data/*/*.mat')
testList = getEvens(testList)

def my_readfunction(filename):
    
    mat = sio.loadmat(filename,squeeze_me=True,struct_as_record=False)['depth']
    mat = mat-mat[50,50]
    mat[np.isnan(mat)]=0
    rgb = sio.loadmat(filename,squeeze_me=True,struct_as_record=False)['rgb']/255
    mat = np.concatenate((mat[:,:,None],rgb),axis=2)
    
    return mat

def my_readfunction_out(filename):
    
    mat = sio.loadmat(filename,squeeze_me=True,struct_as_record=False)['this_foot_map']
    mat[mat==0]=1e-32
    mat = np.array(mat)
    mat = mat[:,:,None]
    
    return mat


def get_foot_coords(filename):
    
    mat = sio.loadmat(filename,squeeze_me=True,struct_as_record=False)
    
    return mat['step_ii'],mat['step_jj']

def my_readfunction_rgb(filename):
    
    mat = sio.loadmat(filename,squeeze_me=True,struct_as_record=False)['rgb']
    
    return mat

def getInputStack_saliency_rgb(fileList):
    

    x = np.stack([my_readfunction(filename) for filename in fileList])
    y = np.stack([my_readfunction_out(filename) for filename in fileList])
    z = np.stack([my_readfunction_rgb(filename) for filename in fileList])
    
    foot_coords = [get_foot_coords(filename) for filename in fileList]
    
    return x,y,z,foot_coords


#%%
from nn_utils import getCNN_saliency
import tensorflow as tf
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from scipy.signal import convolve2d
import os

this_means = []
this_medians = []
tf.compat.v1.disable_eager_execution()
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

#sub_test_list = [testList[idx] for idx in list(np.random.choice(len(testList),1000))]
sub_test_list = testList

x_test,y_test,rgb_test,foot_ii_jj = getInputStack_saliency_rgb(sub_test_list)
    



for ee in range(1):
    if not os.path.exists('results/both/{ee}.npy'.format(ee=ee)):
        model = tf.keras.models.load_model('models_both/{ee}'.format(ee=ee))
        test = model.predict(x_test)
        quants = []  
        overlayed_img = []
        
        
        
        for idx,pred_map in enumerate(list(test)):
            
            
            this_img = rgb_test[idx]
            this_map = cv2.applyColorMap(np.round((1-pred_map/np.max(pred_map))*255).astype(np.uint8),cv2.COLORMAP_JET)
            
            out_img = cv2.addWeighted(this_map,0.3,this_img,0.7,0)#  
            
            this_ii = foot_ii_jj[idx][0]
            this_jj = foot_ii_jj[idx][1]
        
            try:
                for edx in range(this_ii.shape[0]):
                        
                    this_quant = np.sum(pred_map[this_ii[edx],this_jj[edx]]>pred_map.ravel())/(np.prod(pred_map.shape[:2]))
                    quants.append(this_quant)
                    
        #            
                    out_img = cv2.circle(out_img,(this_jj[edx],this_ii[edx]),4,(0,255,0),2)
                    
    #            this_quant = np.sum(pred_map.ravel()[max_dex]>pred_map.ravel())/(np.prod(pred_map.shape[:2]))
    #            
            except:
                pass
#            quants.append(this_quant)
#            
            
            
            
            this_image_str = testList[idx].split('/')
            this_image_str[4] = 'retinalCNN_outputs'
            last_bit = this_image_str[-1]
            last_bit = last_bit.split('.')
            last_bit[-1] = 'png'
            this_image_str[-1] = '.'.join(last_bit)
            
            this_image_str = '/'.join(this_image_str)
            Image.fromarray(out_img).save(this_image_str)
            
#            cv2.imwrite(,out_img)
#            overlayed_img.append(out_img)
            
        print(str(ee)+'_'+str(np.mean(quants)))
        print(str(ee)+'_'+str(np.median(quants)))
        
    #    this_means.append(np.mean(quants))
    #    this_medians.append(np.median(quants))
    
    
        np.save('results/both/{ee}.npy'.format(ee=ee),{'mean':np.mean(quants),'median':np.median(quants)})
    #model.save('63')
    
    #all_quants = []
    # 
    #for idx,this_map in enumerate(list(test)):
    #    
    #    this_coords = z_test[idx]
    #    
    #    quants=[]
    #    
    #    for coords in this_coords:
    #
    #        this_quant = np.sum(this_map[coords[0],coords[1]]>this_map.ravel())/(np.prod(this_map.shape[:2]))
    #        quants.append(this_quant)
    #        
    #    all_quants.append(quants)
    #        
    #print(np.median(all_quants,axis=0))