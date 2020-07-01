import numpy
import os

segmentName='UpperFace'
sizeH=32
sizeV=32
sizeD=100

segment_traininglabels = numpy.load('numpy_training_datasets/{0}_labels_{1}x{2}x{3}.npy'.format(segmentName,sizeH, sizeV,sizeD))
cat = [0] * 3
for item in segment_traininglabels:
    for c in range(len(cat)):
        if item[c] == 1:
            cat[c] += 1

print(cat)

negativepath = '../../../Datasets/SIMC_E_categorical/Negative/'
positivepath = '../../../Datasets/SIMC_E_categorical/Positive/'
surprisepath = '../../../Datasets/SIMC_E_categorical/Surprise/'
cat=[0]*3
dir=0
for typepath in (negativepath, positivepath, surprisepath):
    directorylisting = os.listdir(typepath)
    for video in directorylisting:
        cat[dir]+=1
    dir+=1
print(cat)