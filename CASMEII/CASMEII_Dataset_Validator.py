import numpy
import os

segmentName='Eyes'
sizeH=32
sizeV=32
sizeD=24


segment_traininglabels = numpy.load('numpy_training_datasets/{0}_labels_{1}x{2}x{3}.npy'.format(segmentName,sizeH, sizeV,sizeD))
print(segment_traininglabels)
cat=[0]*5
for item in segment_traininglabels:
    for c in range(len(cat)):
        if item[c]==1:
            cat[c]+=1

print(cat)




cat=[0]*7
dir=0
for typepath in os.listdir('../../../Datasets/CASMEII_categorical/'):
    directorylisting = os.listdir('../../../Datasets/CASMEII_categorical/'+typepath)
    for video in directorylisting:
        cat[dir]+=1
    dir+=1
print(cat)
print(sum(cat))