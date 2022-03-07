#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 12:57:47 2021

@author: karl
"""

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
import os


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
        
        if int(filename.split('/')[-1].split('_')[1])%2==0:
            evensList.append(filename)
            
    
    return evensList
        
trainList = glob.glob('/media/karl/DATA/retinalCNN_data/*/*.mat')
#trainListStraight=glob.glob('../turns_cnn/footLocSaliency_single/J*.mat')

trainList = getOdds(trainList)
#trainListCurved = getOdds(trainListCurved)
#trainListStraight = getOdds(trainListStraight)

#trainList = trainListCurved+trainListStraight
#trainClasses =np.concatenate((np.ones(len(trainListCurved)),np.zeros(len(trainListStraight))),axis=0)

#trainList = 4*trainList
#trainClasses = np.array(4*list(trainClasses))

shuffIdx = np.arange(0,len(trainList))
np.random.shuffle(shuffIdx)

trainList = [trainList[idx] for idx in list(shuffIdx)]
#trainClasses = [trainClasses[idx] for idx in list(shuffIdx)]

testList = glob.glob('../turns_cnn/footLocSaliency_single/J*.mat')
#testListStraight=glob.glob('../turns_cnn/footLocSaliency_single/J*.mat')

testList = glob.glob('/media/karl/DATA/retinalCNN_data/*/*.mat')
#testListStraight = getEvens(testListStraight)

#testList = testListCurved+testListStraight
#testClasses =np.concatenate((np.ones(len(testListCurved)),np.zeros(len(testListStraight))),axis=0)

#trainList = 8*trainList
#trainClasses = np.array(4*list(trainClasses))
#
#xx,yy = np.meshgrid(np.arange(101),np.arange(101))
#
#xx = xx-50
#yy = yy-50
#
#dist = np.sqrt(xx*xx+yy*yy)
#dist_mask = dist>50

def my_readfunction(filename):
    
    mat = sio.loadmat(filename,squeeze_me=True,struct_as_record=False)['depth']
    mat = mat-mat[50,50]
    mat[np.isnan(mat)]=0
    
#    rgb = sio.loadmat(filename,squeeze_me=True,struct_as_record=False)['rgb']/255
#    mat = cv2.resize(mat,(int(mat.shape[1]/2),int(mat.shape[0]/2)))
#    mat = Image.fromarray(mat)
#    mat=mat.rotate(np.random.rand()*360)
#    mat = np.array(mat)
#    mat = np.abs(mat)
#    mat = np.concatenate((mat[:,:,None],rgb),axis=2)
    
#    mat = mat[:,:,None]
    
    return mat[:,:,None]

def my_readfunction_mag_diff(filename):
    
    mat = sio.loadmat(filename,squeeze_me=True,struct_as_record=False)['mag_diff']
    #mat = mat-mat[50,50]
    mat[np.isnan(mat)]=0
    
#    rgb = sio.loadmat(filename,squeeze_me=True,struct_as_record=False)['rgb']/255
#    mat = cv2.resize(mat,(int(mat.shape[1]/2),int(mat.shape[0]/2)))
#    mat = Image.fromarray(mat)
#    mat=mat.rotate(np.random.rand()*360)
#    mat = np.array(mat)
#    mat = np.abs(mat)
#    mat = np.concatenate((mat[:,:,None],rgb),axis=2)
    
#    mat = mat[:,:,None]
    
    return mat[:,:,None]

def my_readfunction_flow(filename):
    
    mat_vx = sio.loadmat(filename,squeeze_me=True,struct_as_record=False)['vx']
    #mat = mat-mat[50,50]
    mat_vx[np.isnan(mat_vx)]=0
    
    mat_vy = sio.loadmat(filename,squeeze_me=True,struct_as_record=False)['vy']
    #mat = mat-mat[50,50]
    mat_vy[np.isnan(mat_vx)]=0
    
#    rgb = sio.loadmat(filename,squeeze_me=True,struct_as_record=False)['rgb']/255
#    mat = cv2.resize(mat,(int(mat.shape[1]/2),int(mat.shape[0]/2)))
#    mat = Image.fromarray(mat)
#    mat=mat.rotate(np.random.rand()*360)
#    mat = np.array(mat)
#    mat = np.abs(mat)
    mat = np.concatenate((mat_vx[:,:,None],mat_vy[:,:,None]),axis=2)
    
#    mat = mat[:,:,None]
    
    return mat

def my_readfunction_out(filename):
    
    mat = sio.loadmat(filename,squeeze_me=True,struct_as_record=False)['this_foot_map']
    mat[mat==0]=1e-32
#    mat = mat/np.sum(mat.ravel())
#    mat = np.exp(mat)
#    mat = mat/np.sum(mat.ravel())
#    mat = cv2.resize(mat,(int(mat.shape[1]/2),int(mat.shape[0]/2)))
#    mat = Image.fromarray(mat)
#    mat=mat.rotate(np.random.rand()*360)
    mat = np.array(mat)
#    mat = np.abs(mat)
    mat = mat[:,:,None]
    
    return mat

def my_readfunction_rgb(filename):
    
    mat = sio.loadmat(filename,squeeze_me=True,struct_as_record=False)['rgb']
#    mat = mat-np.mean(mat.ravel())
    
    return mat

class Mygenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        # read your data here using the batch lists, batch_x and batch_y
        x = [my_readfunction(filename) for filename in batch_x] 
        y = [my_readfunction_out(filename) for filename in batch_y] 
        
        
        return applyRandRot(x,y)
    
    
class Mygenerator_flow(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        # read your data here using the batch lists, batch_x and batch_y
        x = [my_readfunction_flow(filename) for filename in batch_x] 
        y = [my_readfunction_out(filename) for filename in batch_y] 
        
        
        return applyRandRot(x,y)
    
    
class Mygenerator_mag_diff(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        # read your data here using the batch lists, batch_x and batch_y
        x = [my_readfunction_mag_diff(filename) for filename in batch_x] 
        y = [my_readfunction_out(filename) for filename in batch_y] 
        
        
        return applyRandRot(x,y)

def applyRandRot(x,y):
    
    out_x,out_y=[],[]
    for pair in zip(x,y):
        
        this_theta = np.random.rand()*360
        
        input_image = imutils.rotate(pair[0],this_theta)
        if len(input_image.shape)==2:
            input_image = input_image[:,:,None]
        
        out_x.append(input_image)
        out_y.append(imutils.rotate(pair[1],this_theta))
        
    return np.stack(out_x),np.stack(out_y)

def my_readfunction_ij(filename):
    
    ii = sio.loadmat(filename,squeeze_me=True,struct_as_record=False)['ii']
    jj = sio.loadmat(filename,squeeze_me=True,struct_as_record=False)['jj']
    
    
    return np.concatenate((ii[:,None],jj[:,None]),axis=1)


def getInputStack_saliency(fileList):
    

    x = np.stack([my_readfunction(filename) for filename in fileList])
    y = np.stack([my_readfunction_out(filename) for filename in fileList])
#    z = np.stack([my_readfunction_ij(filename) for filename in fileList])
    
    return x,y


def getInputStack_saliency_rgb(fileList):
    

    x = np.stack([my_readfunction(filename) for filename in fileList])
    y = np.stack([my_readfunction_out(filename) for filename in fileList])
    z = np.stack([my_readfunction_rgb(filename) for filename in fileList])
    
    return x,y,z

def mat2matrotate(mat_in,useAng=False,myAng=0):
    
    mat_in = Image.fromarray(mat_in[:,:,0])
    
    if not useAng:
        myAng = np.random.rand()*360
    mat_in=mat_in.rotate(myAng)
        
    
        
    mat_in = np.array(mat_in)
    
    
    return mat_in[:,:,None],myAng

def applyRandCrop_rgb(x_in,y_in,z_in):
    
    x_out = []
    y_out = []
    z_out = []
    
    for idx in range(x_in.shape[0]):
        
        
        x_left = np.max((1,np.round(np.random.rand()*99))).astype(int)
        y_lower = np.max((1,np.round(np.random.rand()*99))).astype(int)
        
        x_right = x_left+101
        y_upper = y_lower+101
    
        this_x_out = x_in[idx,y_lower:y_upper,x_left:x_right]
    
        this_x_out = this_x_out-this_x_out[50,50]
    
        x_out.append(this_x_out)
        y_out.append(y_in[idx,y_lower:y_upper,x_left:x_right])
        z_out.append(z_in[idx,y_lower:y_upper,x_left:x_right])
        
#        x_out[-1][dist_mask]=0
#        x_out[-1],myAng = mat2matrotate(x_out[-1])
#        
#        y_out[-1][dist_mask]=0
#        y_out[-1],_ = mat2matrotate(y_out[-1],useAng=True,myAng=myAng)
        
    
    return np.stack(x_out),np.stack(y_out),np.stack(z_out)


def applyRandCrop(x_in,y_in):
    
    x_out = []
    y_out = []
    
    for idx in range(x_in.shape[0]):
        
        
        x_left = np.max((1,np.round(np.random.rand()*99))).astype(int)
        y_lower = np.max((1,np.round(np.random.rand()*99))).astype(int)
        
        x_right = x_left+101
        y_upper = y_lower+101
    
        this_x_out = x_in[idx,y_lower:y_upper,x_left:x_right]
    
        this_x_out = this_x_out-this_x_out[50,50]

        x_out.append(this_x_out)
        y_out.append(y_in[idx,y_lower:y_upper,x_left:x_right])
    
#        x_out[-1][dist_mask]=0
#        x_out[-1],myAng = mat2matrotate(x_out[-1])
#        
#        y_out[-1][dist_mask]=0
#        y_out[-1],_ = mat2matrotate(y_out[-1],useAng=True,myAng=myAng)
        
    
    return np.stack(x_out),np.stack(y_out)

    
def applyFixedCrop(x_in,y_in):
    
    
    return x_in[:,50:-50,50:-50,:],y_in[:,50:-50,50:-50,:]

#%%
from nn_utils import getCNN_saliency
import tensorflow as tf
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split


tf.compat.v1.disable_eager_execution()
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

for mode in ['flow','mag_diff']:
    
    for ii in range(100):
        
        if not os.path.exists('models_{this_mode}/{ii}'.format(this_mode=mode,ii=ii)):
            
            if mode=='flow':
                model = getCNN_saliency(2)
            else:
                model = getCNN_saliency(1)
                
            
            #my_opt = tf.keras.optimizers.Nadam(learning_rate=1e-3)
            opt= tf.keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
            
            
            
            model.compile(loss='kl_divergence',optimizer=opt,metrics=['KLDivergence'])
            
            reduceLR_cb = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,patience=1,min_lr = 1e-16)
            ES_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                  min_delta=0, patience=3, verbose=0, mode='auto', baseline=None, restore_best_weights=True)
            
            #weights = class_weight.compute_class_weight('balanced',np.unique(trainClasses),trainClasses)
            
            #weights = {i : weights[i] for i in range(2)}
            #
            weights=None
            #x_in,y_in = getInputStack_saliency(trainList)
            
            #x_in,y_in = applyRandCrop(x_in,y_in)
            
            this_trainList,valList = train_test_split(trainList,test_size=0.1)
            
            batch_size = 20
            
            if mode=='flow':
                train_gen = Mygenerator_flow(this_trainList,this_trainList,batch_size)
                val_gen = Mygenerator_flow(valList,valList,10)
            else:
                train_gen = Mygenerator_mag_diff(this_trainList,this_trainList,batch_size)
                val_gen = Mygenerator_mag_diff(valList,valList,10)
            
            #batch_size = 30
            s_p_e = len(this_trainList)/batch_size
            
            #history = model.fit(x_in,y_in,validation_split=0.1,batch_size=batch_size,validation_steps=5,steps_per_epoch=s_p_e,epochs=100,
            #                    verbose=1,callbacks=[reduceLR_cb],class_weight=weights)
            history = model.fit(train_gen,epochs=100,steps_per_epoch=s_p_e,validation_data=val_gen,validation_steps=5,
                                verbose=1,callbacks=[reduceLR_cb, ES_cb],class_weight=weights)
            
            #test_gen = Mygenerator(testList,testClasses,1)
            
            model.save('models_{this_mode}/{ii}'.format(this_mode=mode,ii=ii))


#%%
#
#import matplotlib.pyplot as plt
#
#x_test,y_test,rgb_test = getInputStack_saliency_rgb(testList)
#
#x_test,y_test,rgb_test = applyRandCrop_rgb(x_test,y_test,rgb_test)
#
##z_test = np.round(z_test/2).astype(int)
##z_test[:,:,0]= np.clip(z_test[:,:,0],0,x_in.shape[1]-1)
##z_test[:,:,1]= np.clip(z_test[:,:,1],0,x_in.shape[2]-1)
###y_test = np.array(testClasses)
###
###results=model.evaluate(x_test,y_test)
##
#test = model.predict(x_test)
#quants = []
#
#overlayed_img = []
#
#for idx,pred_map in enumerate(list(test)):
#    
#    max_dex = np.argmax(y_test[idx].ravel())
#    
#    this_quant = np.sum(pred_map.ravel()[max_dex]>pred_map.ravel())/(np.prod(pred_map.shape[:2]))
#    
#    quants.append(this_quant)
#    
#    
#    this_img = rgb_test[idx]
#    this_map = cv2.applyColorMap(np.round((1-pred_map/np.max(pred_map))*255).astype(np.uint8),cv2.COLORMAP_JET)
#    
#    out_img = cv2.addWeighted(this_map,0.3,this_img,0.7,0)
#    
#    y_cen,x_cen,_ = np.unravel_index(max_dex,pred_map.shape)
#    
#    out_img = cv2.circle(out_img,(x_cen,y_cen),10,(0,255,0),2)
#    
#    overlayed_img.append(out_img)
#    
#print(np.median(quants))
#print(np.mean(quants))


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