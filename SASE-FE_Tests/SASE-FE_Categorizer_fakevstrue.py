import os
import shutil
import cv2


path="../../sase_fe_database-001/FakeTrue_DB/"

subjectlisting = os.listdir(path)


targetpath="../../SASE-FE_Categorical_truevsfake_reduced/"

if os.path.exists(targetpath ):
    shutil.rmtree(targetpath )
os.mkdir(targetpath , mode=0o777)
truetargetpath=targetpath+'true/'
if os.path.exists(truetargetpath ):
    shutil.rmtree(truetargetpath )
os.mkdir(truetargetpath , mode=0o777)
faketargetpath=targetpath+'fake/'
if os.path.exists(faketargetpath ):
    shutil.rmtree(faketargetpath )
os.mkdir(faketargetpath , mode=0o777)
angerpath='anger'
surprisepath='surprise'
disgustpath='disgust'
sadpath='sad'
happypath='happy'
contemptpath='contempt'

paths=[angerpath,surprisepath,disgustpath,sadpath,happypath,contemptpath]
for p in paths:
    if os.path.exists(truetargetpath+p):
        shutil.rmtree(truetargetpath+p)
    os.mkdir(truetargetpath+p, mode=0o777)
for p in paths:
    if os.path.exists(faketargetpath+p):
        shutil.rmtree(faketargetpath+p)
    os.mkdir(faketargetpath+p, mode=0o777)

for subject in subjectlisting:
    subjectpath=path+subject
    videolisting= os.listdir(subjectpath)
    for video in videolisting:
        print(subjectpath,video)
        v_name=video[:-4]
        if len(v_name)>3 and v_name[2:].lower()!="sur":
            name="F_"+subject
            v_name = v_name[4:]
            ForT=faketargetpath
        else:
            name = "T_" + subject
            v_name = v_name[2:]
            ForT = truetargetpath
        if v_name.lower() == "sur":
            newpath =ForT+ surprisepath
        elif v_name.lower()=="a":
            newpath=ForT+angerpath
        elif v_name.lower()=="d":
            newpath=ForT+disgustpath
        elif v_name.lower()=="s":
            newpath=ForT+sadpath
        elif v_name.lower()=="h":
            newpath=ForT+happypath
        elif v_name.lower()=="c":
            newpath=ForT+contemptpath

        newpath=newpath+'/'+name
        if os.path.exists(newpath):
            shutil.rmtree(newpath)
        os.mkdir(newpath, mode=0o777)
        vidcap = cv2.VideoCapture(subjectpath+'/'+video)
        vidcap.set(cv2.CAP_PROP_POS_MSEC, 1000)
        success, image = vidcap.read()
        count = 0
        while success:
            cv2.imwrite(newpath+'/'+"frame%d.jpg" % count, image)  # save frame as JPEG file
            success, image = vidcap.read()
            # print('Read a new frame: ', success)
            count += 1
