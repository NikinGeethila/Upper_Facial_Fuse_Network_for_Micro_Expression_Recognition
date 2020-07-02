import os
import numpy as np
import pandas as pd
import shutil



path='D:/University/Detecting Forced Emotions through Micro-Expression Recognition using Neural Networks/Datasets/CASMEII/CASME2_RAW_selected/CASME2_RAW_selected/'

catdatafile = pd.read_excel('../../../Datasets/CASMEII/cat.xlsx')
catdata = np.array(catdatafile)


# print(catdata)
# print(namedata)


targetpath= '../../../Datasets/CASMEII_categorical/'

directorylisting = os.listdir(path)


count=0
for subject in directorylisting:
    # print(subject)
    subjectdirectorylisting=os.listdir(path+subject)
    for video in subjectdirectorylisting:
        videopath = path+subject +'/'+ video
        found=False
        for vidid in range(len(catdata)):
            if catdata[vidid][1]==str(video) and ("sub"+str(catdata[vidid][0])==subject or "sub0"+str(catdata[vidid][0])==subject):
                print(video,catdata[vidid][2],catdata[vidid])
                # print(str(targetpath)+str(catdata[vidid][2])+"/"+str(catdata[vidid][0])+'_'+str(video))
                viddirectorylisting = os.listdir(videopath)
                shutil.copytree(videopath, str(targetpath)+str(catdata[vidid][2])+"/"+str(catdata[vidid][0])+'_'+str(video))
                if found==True:
                    print("multiple",video,catdata[vidid])
                found=True
                count+=1
        if found==False:
            print("not found",video)
        # print(str(subjectdirectorylisting)+str(video))
        continue

print(count)