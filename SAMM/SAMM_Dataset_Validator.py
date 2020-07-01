import numpy
import os

segmentName='UpperFace'
sizeH=32
sizeV=32
sizeD=30

angerpath = '../../../Datasets/SAMM_categorical/Anger/'
sadnesspath = '../../../Datasets/SAMM_categorical/Sadness/'
happinesspath = '../../../Datasets/SAMM_categorical/Happiness/'
disgustpath = '../../../Datasets/SAMM_categorical/Disgust/'
fearpath = '../../../Datasets/SAMM_categorical/Fear/'
surprisepath = '../../../Datasets/SAMM_categorical/Surprise/'
contemptpath = '../../../Datasets/SAMM_categorical/Contempt/'
otherpath = '../../../Datasets/SAMM_categorical/Other/'

paths=[angerpath,  happinesspath,surprisepath,contemptpath,otherpath]

segment_traininglabels = numpy.load('numpy_training_datasets/{0}_labels_{1}x{2}x{3}.npy'.format(segmentName,sizeH, sizeV,sizeD))
print(segment_traininglabels)
cat=[0]*len(paths)
for item in segment_traininglabels:
    for c in range(len(cat)):
        if item[c]==1:
            cat[c]+=1

print(cat)



cat=[0]*len(paths)
dir=0
for typepath in (paths):
    directorylisting = os.listdir(typepath)
    for video in directorylisting:
        cat[dir]+=1
    dir+=1
print(cat)
print(sum(cat))