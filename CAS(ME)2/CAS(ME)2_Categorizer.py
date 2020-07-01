import os
import numpy as np
import pandas as pd
import shutil



path='../../../Datasets/CAS(ME)2/selectedpic/selectedpic/'

catdatafile = pd.read_excel('../../../Datasets/CAS(ME)2/cat.xlsx')
catdata = np.array(catdatafile)

namedatafile = pd.read_excel('../../../Datasets/CAS(ME)2/name.xlsx')
namedata = np.array(namedatafile)

# print(catdata)
# print(namedata)


targetpath= '../../../Datasets/CAS(ME)2_categorical/'

directorylisting = os.listdir(path)


count=0
for subject in directorylisting:
    # print(subject)
    subjectdirectorylisting=os.listdir(path+subject)
    for video in subjectdirectorylisting:
        videopath = path+subject +'/'+ video
        found=False
        for vidid in range(len(catdata)):
            if catdata[vidid][1]==str(video) and namedata[vidid][0]==subject and catdata[vidid][3]=="micro-expression":
                print(video,catdata[vidid][2],catdata[vidid])
                # print(str(targetpath)+str(catdata[vidid][2])+"/"+str(namedata[vidid][0])+'_'+str(video))
                shutil.copytree(videopath, str(targetpath)+str(catdata[vidid][2])+"/"+str(namedata[vidid][0])+'_'+str(video))
                if found==True:
                    print("multiple",video,catdata[vidid])
                found=True
                count+=1
        if found==False:
            print("not found",video)
        # print(str(subjectdirectorylisting)+str(video))
        continue

print(count)