import os



path='../../../Datasets/CASMEII_categorical/'



directorylisting = os.listdir(path)
img_count=[]
for cat in directorylisting:
    # print(subject)`
    subjectdirectorylisting=os.listdir(path+cat)
    for video in subjectdirectorylisting:
        videopath = path+cat +'/'+ video
        # print(videopath)
        imgs=os.listdir(videopath)
        count=0
        for img in imgs:
            count+=1
        print(count)
        img_count.append(count)
        # print(str(subjectdirectorylisting)+str(video))
print(img_count)
print(sum(img_count)/len(img_count))
print(max(img_count))
print(min(img_count))
print(len(img_count))