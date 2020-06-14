import os
import cv2
from keras import regularizers
import dlib
import numpy
import imageio
from sklearn.metrics import confusion_matrix
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.utils import multi_gpu_model
from keras.optimizers import SGD, RMSprop
from keras.layers import Concatenate, Input, concatenate, add, multiply, maximum
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils, generic_utils
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras import backend as K
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
K.set_image_dim_ordering('th')
'''
# DLib Face Detection
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

left_eye_training_list = []
right_eye_training_list = []
nose_training_list = []

for typepath in (negativepath,positivepath,surprisepath):
    directorylisting = os.listdir(typepath)
    print(typepath)
    countimg = 0
    for video in directorylisting:
        videopath = typepath + video
        left_eye_frames = []
        right_eye_frames = []
        nose_frames = []
        framelisting = os.listdir(videopath)
        framerange = [x for x in range(18)]
        for frame in framerange:
               imagepath = videopath + "/" + framelisting[frame]
               image = cv2.imread(imagepath)
               landmarks = get_landmark(image)
               if countimg<1:
                   img=annotate_landmarks(image, landmarks)
                   imgplot = plt.imshow(img)
                   plt.show()

               numpylandmarks = numpy.asarray(landmarks)
               up=min(numpylandmarks[17][1],numpylandmarks[18][1],numpylandmarks[19][1],numpylandmarks[20][1],numpylandmarks[21][1])-20
               down = max(numpylandmarks[36][1], numpylandmarks[39][1], numpylandmarks[40][1], numpylandmarks[41][1]) +10
               left = min(numpylandmarks[17][0], numpylandmarks[18][0], numpylandmarks[36][0])
               right = max(numpylandmarks[39][0], numpylandmarks[21][0])+10
               left_eye_image = image[numpylandmarks[19][1]:numpylandmarks[1][1], numpylandmarks[1][0]:numpylandmarks[15][0]]

               left_eye_image = cv2.resize(left_eye_image, (32, 32), interpolation = cv2.INTER_AREA)
               left_eye_image = cv2.cvtColor(left_eye_image, cv2.COLOR_BGR2GRAY)
               if countimg<1:
                   img=annotate_landmarks(left_eye_image, landmarks)
                   imgplot = plt.imshow(img)
                   plt.show()
               up = min(numpylandmarks[22][1], numpylandmarks[23][1], numpylandmarks[24][1], numpylandmarks[25][1],
                        numpylandmarks[26][1]) - 20
               down = max(numpylandmarks[42][1], numpylandmarks[47][1], numpylandmarks[46][1], numpylandmarks[45][1]) + 10
               right = max(numpylandmarks[26][0], numpylandmarks[25][0], numpylandmarks[45][0])
               left = min(numpylandmarks[22][0], numpylandmarks[42][0])-10
               right_eye_image = image[up:down, left:right]

               right_eye_image = cv2.resize(right_eye_image, (32, 32), interpolation=cv2.INTER_AREA)
               right_eye_image = cv2.cvtColor(right_eye_image, cv2.COLOR_BGR2GRAY)

               if countimg<1:
                   img=annotate_landmarks(right_eye_image, landmarks)
                   imgplot = plt.imshow(img)
                   plt.show()
               up = numpylandmarks[27][1] - 5
               down = max(numpylandmarks[31][1], numpylandmarks[32][1], numpylandmarks[33][1], numpylandmarks[34][1], numpylandmarks[35][1]) + 5
               left = numpylandmarks[31][0]
               right = numpylandmarks[35][0]

               nose_image = image[up:down, left:right]

               nose_image = cv2.resize(nose_image, (32, 32), interpolation = cv2.INTER_AREA)
               nose_image = cv2.cvtColor(nose_image, cv2.COLOR_BGR2GRAY)
               if countimg<1:
                   img=annotate_landmarks(nose_image, landmarks)
                   imgplot = plt.imshow(img)
                   plt.show()
                   countimg+=1
               left_eye_frames.append(left_eye_image)
               right_eye_frames.append(right_eye_image)
               nose_frames.append(nose_image)
        left_eye_frames = numpy.asarray(left_eye_frames)
        right_eye_frames = numpy.asarray(right_eye_frames)
        nose_frames = numpy.asarray(nose_frames)
        left_eye_videoarray = numpy.rollaxis(numpy.rollaxis(left_eye_frames, 2, 0), 2, 0)
        right_eye_videoarray = numpy.rollaxis(numpy.rollaxis(right_eye_frames, 2, 0), 2, 0)
        nose_videoarray = numpy.rollaxis(numpy.rollaxis(nose_frames, 2, 0), 2, 0)
        left_eye_training_list.append(left_eye_videoarray)
        right_eye_training_list.append(right_eye_videoarray)
        nose_training_list.append(nose_videoarray)
        if typepath==surprisepath:
            left_eye_training_list.append(left_eye_videoarray)
            right_eye_training_list.append(right_eye_videoarray)
            nose_training_list.append(nose_videoarray)


left_eye_training_list = numpy.asarray(left_eye_training_list)
right_eye_training_list = numpy.asarray(right_eye_training_list)
nose_training_list = numpy.asarray(nose_training_list)

left_eye_trainingsamples = len(left_eye_training_list)
right_eye_trainingsamples = len(right_eye_training_list)
nose_trainingsamples = len(nose_training_list)

left_eye_traininglabels = numpy.zeros((left_eye_trainingsamples, ), dtype = int)
right_eye_traininglabels = numpy.zeros((right_eye_trainingsamples, ), dtype = int)
nose_traininglabels = numpy.zeros((nose_trainingsamples, ), dtype = int)

left_eye_traininglabels[0:66] = 0
left_eye_traininglabels[66:113] = 1
left_eye_traininglabels[113:156] = 2

right_eye_traininglabels[0:66] = 0
right_eye_traininglabels[66:113] = 1
right_eye_traininglabels[113:156] = 2

nose_traininglabels[0:66] = 0
nose_traininglabels[66:113] = 1
nose_traininglabels[113:156] = 2

left_eye_traininglabels = np_utils.to_categorical(left_eye_traininglabels, 3)
right_eye_traininglabels = np_utils.to_categorical(right_eye_traininglabels, 3)
nose_traininglabels = np_utils.to_categorical(nose_traininglabels, 3)

left_eye_training_data = [left_eye_training_list, left_eye_traininglabels]
right_eye_training_data = [right_eye_training_list, right_eye_traininglabels]
nose_training_training_data = [nose_training_list, nose_traininglabels]

(left_eye_training_frames, left_eye_training_labels) = (left_eye_training_data[0], left_eye_training_data[1])
(right_eye_training_frames, right_eye_training_labels) = (right_eye_training_data[0], right_eye_training_data[1])
(nose_training_frames, nose_training_labels) = (nose_training_training_data[0], nose_training_training_data[1])

left_eye_training_set = numpy.zeros((left_eye_trainingsamples, 1, 32, 32, 18))
right_eye_training_set = numpy.zeros((right_eye_trainingsamples, 1, 32, 32, 18))
nose_training_set = numpy.zeros((nose_trainingsamples, 1, 32, 32, 18))

for h in range(left_eye_trainingsamples):
    left_eye_training_set[h][0][:][:][:] = left_eye_training_frames[h,:,:,:]
for h in range(right_eye_trainingsamples):
    right_eye_training_set[h][0][:][:][:] = right_eye_training_frames[h,:,:,:]
for h in range(nose_trainingsamples):
    nose_training_set[h][0][:][:][:] = nose_training_frames[h, :, :, :]

left_eye_training_set = left_eye_training_set.astype('float32')
left_eye_training_set -= numpy.mean(left_eye_training_set)
left_eye_training_set /= numpy.max(left_eye_training_set)

right_eye_training_set = right_eye_training_set.astype('float32')
right_eye_training_set -= numpy.mean(right_eye_training_set)
right_eye_training_set /= numpy.max(right_eye_training_set)

nose_training_set = nose_training_set.astype('float32')
nose_training_set -= numpy.mean(nose_training_set)
nose_training_set /= numpy.max(nose_training_set)




numpy.save('numpy_training_datasets/late_microexpfusenetlefteyeimages.npy', left_eye_training_set)
numpy.save('numpy_training_datasets/late_microexpfusenetrighteyeimages.npy', right_eye_training_set)
numpy.save('numpy_training_datasets/late_microexpfusenetnoseimages.npy', nose_training_set)

numpy.save('numpy_training_datasets/late_microexpfusenetlefteyelabels.npy', left_eye_training_labels)
numpy.save('numpy_training_datasets/late_microexpfusenetrighteyelabels.npy', right_eye_training_labels)
numpy.save('numpy_training_datasets/late_microexpfusenetnoselabels.npy', nose_traininglabels)

'''
# Load training images and labels that are stored in numpy array

left_eye_training_set = numpy.load('numpy_training_datasets/late_microexpfusenetlefteyeimages.npy')
right_eye_training_set = numpy.load('numpy_training_datasets/late_microexpfusenetrighteyeimages.npy')
nose_training_set = numpy.load('numpy_training_datasets/late_microexpfusenetnoseimages.npy')


left_eye_training_labels = numpy.load('numpy_training_datasets/late_microexpfusenetlefteyelabels.npy')
right_eye_training_labels = numpy.load('numpy_training_datasets/late_microexpfusenetrighteyelabels.npy')
nose_training_labels = numpy.load('numpy_training_datasets/late_microexpfusenetnoselabels.npy')

print(len(left_eye_training_labels))
print(len(right_eye_training_labels))
print(len(nose_training_labels))
# Late MicroExpFuseNet Model
left_eye_input = Input(shape = (1, 32, 32, 18))
left_eye_conv = Convolution3D(32, (3, 3, 15))(left_eye_input)
ract_1 = Activation('relu')(left_eye_conv)
maxpool_1 = MaxPooling3D(pool_size=(3, 3, 3))(ract_1)
ract_2 = Activation('relu')(maxpool_1)
dropout_1 = Dropout(0.5)(ract_2)
flatten_1 = Flatten()(dropout_1)
dense_1 = Dense(1024, )(flatten_1)
#dropout_2 = Dropout(0.5)(dense_1)
# dense_2= Dense(128, )(dropout_2)
# dropout_3 = Dropout(0.5)(dense_2)

right_eye_input = Input(shape = (1, 32, 32, 18))
right_eye_conv = Convolution3D(32, (3, 3, 15))(right_eye_input)
ract_3 = Activation('relu')(right_eye_conv)
maxpool_2 = MaxPooling3D(pool_size=(3, 3, 3))(ract_3)
ract_4 = Activation('relu')(maxpool_2)
dropout_4 = Dropout(0.5)(ract_4)
flatten_2 = Flatten()(dropout_4)
dense_3 = Dense(1024, )(flatten_2)
#dropout_5 = Dropout(0.5)(dense_3)
# dense_4= Dense(128, )(dropout_5)
# dropout_6= Dropout(0.5)(dense_4)

nose_input = Input(shape = (1, 32, 32, 18))
nose_conv = Convolution3D(32, (3, 3, 15))(nose_input)
ract_5 = Activation('relu')(nose_conv)
maxpool_3 = MaxPooling3D(pool_size=(3, 3, 3))(ract_5)
ract_6 = Activation('relu')(maxpool_3)
dropout_7 = Dropout(0.5)(ract_6)
flatten_3 = Flatten()(dropout_7)
dense_5= Dense(1024, )(flatten_3)
#dropout_8 = Dropout(0.5)(dense_5)
# dense_6 = Dense(128, )(dropout_8)
# dropout_9 = Dropout(0.5)(dense_6)

concat = Concatenate(axis = 1)([dense_1, dense_3,dense_5])
dropout_10 = Dropout(0.5)(concat)
dense_7 = Dense(3, )(dropout_10)
activation = Activation('softmax')(dense_7)

model = Model(inputs = [left_eye_input,right_eye_input,nose_input], outputs = activation)
model.compile(loss = 'categorical_crossentropy', optimizer = 'SGD', metrics = ['accuracy'])

filepath="weights_late_microexpfusenet/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.summary()

# Load pre-trained weights
"""
model.load_weights('weights_late_microexpfusenet/weights-improvement-22-0.83.hdf5')
"""

# Spliting the dataset into training and validation sets
left_eye_train_images, left_eye_validation_images, left_eye_train_labels, left_eye_validation_labels =  train_test_split(left_eye_training_set, left_eye_training_labels, test_size=0.2, shuffle=True)
right_eye_train_images, right_eye_validation_images, right_eye_train_labels, right_eye_validation_labels =  train_test_split(right_eye_training_set, right_eye_training_labels, test_size=0.2, shuffle=True)
nose_train_images, nose_validation_images, nose_train_labels, nose_validation_labels =  train_test_split(nose_training_set, nose_training_labels, test_size=0.2, shuffle=True)
print(len(left_eye_train_images))
print(len(left_eye_train_labels))
print(len(left_eye_validation_images))
print(len(left_eye_validation_labels))
print(len(left_eye_training_set))
print(len(left_eye_training_labels))
print(152,121)


# Save validation set in a numpy array
numpy.save('numpy_validation_datasets/late_microexpfusenet_left_eye_val_images.npy', left_eye_validation_images)
numpy.save('numpy_validation_datasets/late_microexpfusenet_right_eye_val_images.npy', right_eye_validation_images)
numpy.save('numpy_validation_datasets/late_microexpfusenet_nose_val_images.npy', nose_validation_images)

numpy.save('numpy_validation_datasets/late_microexpfusenet_left_eye_val_labels.npy', left_eye_validation_labels)
numpy.save('numpy_validation_datasets/late_microexpfusenet_right_eye_val_labels.npy', right_eye_validation_labels)
numpy.save('numpy_validation_datasets/late_microexpfusenet_nose_val_labels.npy', nose_validation_labels)

# Training the model
history = model.fit([left_eye_train_images,right_eye_train_images,nose_train_images], left_eye_train_labels, validation_data = ([left_eye_training_set,right_eye_training_set,nose_training_set], left_eye_training_labels), callbacks=callbacks_list, batch_size = 16, nb_epoch = 100, shuffle=True)

# Loading Load validation set from numpy array

elimg = numpy.load('numpy_validation_datasets/late_microexpfusenet_left_eye_val_images.npy')
erimg = numpy.load('numpy_validation_datasets/late_microexpfusenet_right_eye_val_images.npy')
nimg = numpy.load('numpy_validation_datasets/late_microexpfusenet_nose_val_images.npy')
labels = numpy.load('numpy_validation_datasets/late_microexpfusenet_left_eye_val_labels.npy')


# Finding Confusion Matrix using pretrained weights

predictions = model.predict([elimg,erimg,nimg])
predictions_labels = numpy.argmax(predictions, axis=1)
validation_labels = numpy.argmax(labels, axis=1)
cfm = confusion_matrix(validation_labels, predictions_labels)
print (cfm)

