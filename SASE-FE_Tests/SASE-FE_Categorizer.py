import os
import shutil
import cv2


path="../../sase_fe_database-001/FakeTrue_DB/"

subjectlisting = os.listdir(path)


targetpath="../../SASE-FE_Categorical/"

if os.path.exists(targetpath ):
    shutil.rmtree(targetpath )
os.mkdir(targetpath , mode=0o777)
angerpath=targetpath+'anger'
surprisepath=targetpath+'surprise'
disgustpath=targetpath+'disgust'
sadpath=targetpath+'sad'
happypath=targetpath+'happy'
contemptpath=targetpath+'contempt'

paths=[angerpath,surprisepath,disgustpath,sadpath,happypath,contemptpath]
for p in paths:
    if os.path.exists(p):
        shutil.rmtree(p)
    os.mkdir(p, mode=0o777)

for subject in subjectlisting:
    subjectpath=path+subject
    videolisting= os.listdir(subjectpath)
    for video in videolisting:
        v_name=video[:-4]
        if len(v_name)>3 and v_name[2:].lower()!="sur":
            name="F_"+subject
            v_name = v_name[4:]
        else:
            name = "T_" + subject
            v_name = v_name[2:]
        if v_name.lower() == "sur":
            newpath = surprisepath
        elif v_name.lower()=="a":
            newpath=angerpath
        elif v_name.lower()=="d":
            newpath=disgustpath
        elif v_name.lower()=="s":
            newpath=sadpath
        elif v_name.lower()=="h":
            newpath=happypath
        elif v_name.lower()=="c":
            newpath=contemptpath

        newpath=newpath+'/'+name
        if os.path.exists(newpath):
            shutil.rmtree(newpath)
        os.mkdir(newpath, mode=0o777)
        vidcap = cv2.VideoCapture(subjectpath+'/'+video)
        success, image = vidcap.read()
        count = 0
        while success:
            cv2.imwrite(newpath+'/'+"frame%d.jpg" % count, image)  # save frame as JPEG file
            success, image = vidcap.read()
            print('Read a new frame: ', success)
            count += 1
