import numpy as np
import bpy
import mathutils
import os

import sys
import math

depth=False

argv = sys.argv

argv = argv[argv.index("--")+1:]

subj = argv[0].split('/')[-1].split('_')[0]
walkNum = argv[0].split('/')[-1].split('_')[1]
walkNum = int(walkNum)-1

bpy.types.RenderSettings
#for subj in  ['s3','s4','s5','s7','s8','s9','s10']:
#    
#    num_walks = len(glob.glob('/media/karl/DATA/pupilShadowMeshBlender/'+subj+'_*'))
#for walkNum in range(num_walks):
scene=bpy.context.scene

bpy.data.cameras['Camera'].lens_unit='FOV'
bpy.data.cameras['Camera'].angle=math.pi/2

scene.render.resolution_x=250
scene.render.resolution_y=250

basePath = '/media/karl/DATA/'

loadPath = basePath+'pupilShadowMeshBlender/'+subj+'_'+str(walkNum+1)+'_pupilShadowMesh.npy'

loadStruct = np.load(loadPath,allow_pickle=True).item()

orig2alignMat = loadStruct['orig2alignedMat']
cens = loadStruct['cens']
eyeVec = loadStruct['eyeVec']
y_flip = loadStruct['y_flip']
z_flip = loadStruct['z_flip']

if y_flip==1:
    orig2alignMat[1,:] = -orig2alignMat[1,:]

if z_flip==1:
    orig2alignMat[2,:] = -orig2alignMat[2,:]


#bpy.ops.object.select_all(action='DESELECT')
obj = bpy.ops.import_scene.obj(filepath=basePath+'allMeshes/'+subj+'_'+str(walkNum)+'_out/texturedMesh.obj')
obj = [obj for obj in bpy.data.objects if 'texturedMesh' in obj.name][0]



set_mat = np.array(obj.matrix_world)
set_mat[:3,:3] = np.transpose(orig2alignMat)
obj.matrix_world=mathutils.Matrix(set_mat)

for idx in range(len(cens)):

#bpy.data.objects['Camera'].location = cens[idx]
    print(subj+'_'+str(walkNum+1)+' '+str(idx/len(cens)))     
    eyeDir = eyeVec[idx]
    eyeDir = eyeDir/np.linalg.norm(eyeDir)

    eyeRight = np.cross(eyeDir,np.array((0,1,0)))
    eyeRight = eyeRight/np.linalg.norm(eyeRight)

    eyeUp = np.cross(eyeRight,eyeDir)
    eyeUp = eyeUp/np.linalg.norm(eyeUp)

    eyeMatrix = np.concatenate((eyeRight[None,:],eyeUp[None,:],eyeDir[None,:]),axis=0)
    eyeMatrix = np.transpose(eyeMatrix)

    new_mat = np.zeros((4,4))

    #eyeMatrix[2,:] = -eyeMatrix[2,:]
    #eyeMatrix[1,:] = -eyeMatrix[1,:]

    eyeMatrix[:,2] = -eyeMatrix[:,2]
    #eyeMatrix[:,1] = -eyeMatrix[:,1]

    new_mat[:3,:3] = eyeMatrix
    new_mat[3,3] = 1

    bpy.data.objects['Camera'].matrix_world = mathutils.Matrix(new_mat)
    bpy.data.objects['Camera'].location = cens[idx]
    
    if depth:
        if not os.path.exists(basePath+'/retinalImageDepth/' +subj+'_'+str(walkNum+1)+'/'+ str(idx) +'.exr'):
            scene.render.image_settings.file_format = 'OPEN_EXR'
            scene.render.image_settings.use_zbuffer=True
            scene.render.image_settings.color_mode='BW'
            scene.render.filepath = basePath+'/retinalImageDepth/' +subj+'_'+str(walkNum+1)+'/'+ str(idx) +'.exr'
            bpy.ops.render.render(write_still = 1)
    else:
        if not os.path.exists(basePath+'/retinalImageRGB/' +subj+'_'+str(walkNum+1)+'/'+ str(idx) +'.png'):
            scene.render.image_settings.file_format = 'PNG'
            scene.render.filepath = basePath+'/retinalImageRGB/' +subj+'_'+str(walkNum+1)+'/'+ str(idx) +'.png'
            bpy.ops.render.render(write_still = 1)
            
objs = bpy.data.objects
objs.remove(obj, do_unlink=True)
