import numpy
import os

segmentName='UpperFace'
sizeH=32
sizeV=32
sizeD=9

negativepath = '../../../Datasets/CAS(ME)2_categorical/Negative/'
positivepath = '../../../Datasets/CAS(ME)2_categorical/Positive/'
surprisepath = '../../../Datasets/CAS(ME)2_categorical/Surprise/'
othersepath = '../../../Datasets/CAS(ME)2_categorical/others/'
paths=[negativepath, positivepath, surprisepath]

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