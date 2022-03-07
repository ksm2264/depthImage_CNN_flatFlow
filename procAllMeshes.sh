#!/bin/bash

for ii in /media/karl/DATA/pupilShadowMeshBlender/s*;do blender --background getRetinalInputs.blend --python snapshotScript.py -- ${ii};done
