import os
import numpy as np
import pandas as pd
import shutil



path='../../../Datasets/SAMM/SAMM/'

catdatafile = pd.read_excel('../../../Datasets/SAMM/cat.xlsx')
catdata = np.array(catdatafile)

# print(catdata)


targetpath= '../../../Datasets/SAMM_categorical/'

directorylisting = os.listdir(path)
for subject in directorylisting:
    # print(subject)
    subjectdirectorylisting=os.listdir(path+subject)
    for video in subjectdirectorylisting:
        videopath = path+subject +'/'+ video
        found=False
        for vid in catdata:
            if vid[0]==str(video):
                print(video,vid[1],vid)
                shutil.copytree(videopath, str(targetpath)+str(vid[1])+"/"+str(video))
                if found==True:
                    print("multiple",video,vid)
                found=True
        if found==False:
            print("not found",video)
        # print(str(subjectdirectorylisting)+str(video))
        continue