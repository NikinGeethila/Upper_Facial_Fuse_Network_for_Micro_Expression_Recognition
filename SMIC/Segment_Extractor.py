import os
import cv2
import dlib
import numpy
from keras.utils import np_utils
from keras import backend as K
import matplotlib.pyplot as plt

K.set_image_dim_ordering('th')

# DLib Face Detection path setup
predictor_path = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)
detector = dlib.get_frontal_face_detector()

class TooManyFaces(Exception):
    pass

class NoFaces(Exception):
    pass

def get_landmark(img):
    rects = detector(img, 1)
    if len(rects) > 1:
        pass
    if len(rects) == 0:
        pass
    ans = numpy.matrix([[p.x, p.y] for p in predictor(img, rects[0]).parts()])
    return ans

def annotate_landmarks(img, landmarks, font_scale = 0.4):
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(img, str(idx), pos, fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale=font_scale, color=(0, 0, 255))
        cv2.circle(img, pos, 3, color=(0, 255, 255))
    return img

negativepath = '../../../Datasets/SIMC_E_categorical/Negative/'
positivepath = '../../../Datasets/SIMC_E_categorical/Positive/'
surprisepath = '../../../Datasets/SIMC_E_categorical/Surprise/'

segmentName='UpperFace'

segment_training_list = []
counting=0
for typepath in (negativepath,positivepath,surprisepath):
    directorylisting = os.listdir(typepath)
    print(typepath)

    for video in directorylisting:
        videopath = typepath + video
        segment_frames = []
        framelisting = os.listdir(videopath)
        framerange = [x for x in range(18)]
        for frame in framerange:
               imagepath = videopath + "/" + framelisting[frame]
               image = cv2.imread(imagepath)
               landmarks = get_landmark(image)
               if counting<1:
                   img=annotate_landmarks(image, landmarks)
                   imgplot = plt.imshow(img)
                   plt.show()
               numpylandmarks = numpy.asarray(landmarks)
               up = min(numpylandmarks[18][1], numpylandmarks[19][1], numpylandmarks[23][1], numpylandmarks[24][1]) - 20
               down = max(numpylandmarks[31][1], numpylandmarks[32][1], numpylandmarks[33][1], numpylandmarks[34][1],
                          numpylandmarks[35][1]) + 5
               left = min(numpylandmarks[17][0], numpylandmarks[18][0], numpylandmarks[36][0])
               right = max(numpylandmarks[26][0], numpylandmarks[25][0], numpylandmarks[45][0])
               segment_image = image[up:down, left:right]
               if counting<1:
                   img=annotate_landmarks(segment_image, landmarks)
                   imgplot = plt.imshow(img)
                   plt.show()
                   counting+=1
               segment_image = cv2.resize(segment_image, (32, 32), interpolation = cv2.INTER_AREA)
               segment_image = cv2.cvtColor(segment_image, cv2.COLOR_BGR2GRAY)

               segment_frames.append(segment_image)

        segment_frames = numpy.asarray(segment_frames)
        segment_videoarray = numpy.rollaxis(numpy.rollaxis(segment_frames, 2, 0), 2, 0)
        segment_training_list.append(segment_videoarray)

print(len(segment_videoarray))
segment_training_list = numpy.asarray(segment_training_list)

segment_trainingsamples = len(segment_training_list)

segment_traininglabels = numpy.zeros((segment_trainingsamples, ), dtype = int)


for typepath in (negativepath,positivepath,surprisepath):
    directorylisting = os.listdir(typepath)
    print(typepath)
    for video in range(len(directorylisting)):
        if typepath==negativepath:
            segment_traininglabels[video]=0
        if typepath==positivepath:
            segment_traininglabels[video]=1
        if typepath==surprisepath:
            segment_traininglabels[video]=2


segment_traininglabels = np_utils.to_categorical(segment_traininglabels, 3)

segment_training_data = [segment_training_list, segment_traininglabels]
(segment_trainingframes, segment_traininglabels) = (segment_training_data[0], segment_training_data[1])
segment_training_set = numpy.zeros((segment_trainingsamples, 1,32, 32, 18))
for h in range(segment_trainingsamples):
    segment_training_set[h][0][:][:][:] = segment_trainingframes[h,:,:,:]

segment_training_set = segment_training_set.astype('float32')
segment_training_set -= numpy.mean(segment_training_set)
segment_training_set /= numpy.max(segment_training_set)


numpy.save('numpy_training_datasets/{0}_images.npy'.format(segmentName), segment_training_set)
numpy.save('numpy_training_datasets/{0}_labels.npy'.format(segmentName), segment_traininglabels)