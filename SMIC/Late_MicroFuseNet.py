import numpy
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.layers import Concatenate, Input, concatenate, add, multiply, maximum
from keras.layers import LeakyReLU,PReLU
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras import backend as K
import timeit


def evaluate(segment_train_images, segment_validation_images, segment_train_labels, segment_validation_labels,test_index ):
    # Fusion Model
    left_eye_input = Input(shape=(1, 32, 32, 18))
    left_eye_conv = Convolution3D(32, (3, 3, 15))(left_eye_input)
    ract_1 = PReLU(alpha_initializer="zeros")(left_eye_conv)
    dropout_11 = Dropout(0.5)(ract_1)
    maxpool_1 = MaxPooling3D(pool_size=(3, 3, 3))(dropout_11)
    ract_2 = PReLU(alpha_initializer="zeros")(maxpool_1)
    dropout_1 = Dropout(0.5)(ract_2)
    flatten_1 = Flatten()(dropout_1)
    dense_1 = Dense(1024, )(flatten_1)
    dropout_2 = Dropout(0.5)(dense_1)
    dense_2 = Dense(128, )(dropout_2)
    dropout_3 = Dropout(0.5)(dense_2)

    right_eye_input = Input(shape=(1, 32, 32, 18))
    right_eye_conv = Convolution3D(32, (3, 3, 15))(right_eye_input)
    ract_3 = PReLU(alpha_initializer="zeros")(right_eye_conv)
    dropout_12 = Dropout(0.5)(ract_3)
    maxpool_2 = MaxPooling3D(pool_size=(3, 3, 3))(dropout_12)
    ract_4 = PReLU(alpha_initializer="zeros")(maxpool_2)
    dropout_4 = Dropout(0.5)(ract_4)
    flatten_2 = Flatten()(dropout_4)
    dense_3 = Dense(1024, )(flatten_2)
    dropout_5 = Dropout(0.5)(dense_3)
    dense_4 = Dense(128, )(dropout_5)
    dropout_6 = Dropout(0.5)(dense_4)

    nose_input = Input(shape=(1, 32, 32, 18))
    nose_conv = Convolution3D(32, (3, 3, 15))(nose_input)
    ract_5 = PReLU(alpha_initializer="zeros")(nose_conv)
    dropout_13 = Dropout(0.5)(ract_5)
    maxpool_3 = MaxPooling3D(pool_size=(3, 3, 3))(dropout_13)
    ract_6 = PReLU(alpha_initializer="zeros")(maxpool_3)
    dropout_7 = Dropout(0.5)(ract_6)
    flatten_3 = Flatten()(dropout_7)
    dense_5 = Dense(1024, )(flatten_3)
    dropout_8 = Dropout(0.5)(dense_5)
    dense_6 = Dense(128, )(dropout_8)
    dropout_9 = Dropout(0.5)(dense_6)

    concat = Concatenate(axis=1)([dropout_3, dropout_6, dropout_9])
    dense_7 = Dense(3, )(concat)
    activation = Activation('softmax')(dense_7)

    model = Model(inputs=[left_eye_input, right_eye_input, nose_input], outputs=activation)
    model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])

    filepath = "weights_late_microexpfusenet/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    model.summary()

    # Training the model
    history = model.fit([left_eye_train_images,right_eye_train_images,nose_train_images], left_eye_train_labels, validation_data = ([left_eye_validation_images,right_eye_validation_images,nose_validation_images], left_eye_validation_labels), callbacks=callbacks_list, batch_size = 16, nb_epoch = 100, shuffle=True)

    # Finding Confusion Matrix using pretrained weights

    predictions = model.predict([elimg,erimg,nimg])
    predictions_labels = numpy.argmax(predictions, axis=1)
    validation_labels = numpy.argmax(labels, axis=1)
    cfm = confusion_matrix(validation_labels, predictions_labels)
    print (cfm)
    print("accuracy: ",accuracy_score(validation_labels, predictions_labels))

    return accuracy_score(validation_labels, predictions_labels)


K.set_image_dim_ordering('th')
SegmentNameOne='LeftEye'
SegmentNameTwo='RightEye'
SegmentNameThree='Nose'

# Load training images and labels that are stored in numpy array

left_eye_training_set = numpy.load('numpy_training_datasets/{0}_images.npy'.format(SegmentNameOne))
right_eye_training_set = numpy.load('numpy_training_datasets/{0}_images.npy'.format(SegmentNameTwo))
nose_training_set = numpy.load('numpy_training_datasets/{0}_images.npy'.format(SegmentNameThree))

left_eye_training_labels = numpy.load('numpy_training_datasets/{0}_labels.npy'.format(SegmentNameOne))
right_eye_training_labels = numpy.load('numpy_training_datasets/{0}_labels.npy'.format(SegmentNameTwo))
nose_training_labels = numpy.load('numpy_training_datasets/{0}_labels.npy'.format(SegmentNameThree))



'''
#-----------------------------------------------------------------------------------------------------------------
#LOOCV
loo = LeaveOneOut()
loo.get_n_splits(segment_training_set)
tot=0
count=0
for train_index, test_index in loo.split(segment_training_set):

    print(segment_traininglabels[train_index])
    print(segment_traininglabels[test_index])

    val_acc = evaluate(segment_training_set[train_index], segment_training_set[test_index],segment_traininglabels[train_index], segment_traininglabels[test_index] ,test_index)
    tot+=val_acc
    count+=1
    print("------------------------------------------------------------------------")
    print("validation acc:",val_acc)
    print("------------------------------------------------------------------------")
print(tot/count)

'''




#-----------------------------------------------------------------------------------------------------------------
#Test train split


# Spliting the dataset into training and validation sets
left_eye_train_images, left_eye_validation_images, left_eye_train_labels, left_eye_validation_labels =  train_test_split(left_eye_training_set, left_eye_training_labels, test_size=0.2, random_state=42)
right_eye_train_images, right_eye_validation_images, right_eye_train_labels, right_eye_validation_labels =  train_test_split(right_eye_training_set, right_eye_training_labels, test_size=0.2, random_state=42)
nose_train_images, nose_validation_images, nose_train_labels, nose_validation_labels =  train_test_split(nose_training_set, nose_training_labels, test_size=0.2, random_state=42)



# Save validation set in a numpy array
numpy.save('numpy_validation_datasets/{0}_images.npy'.format(SegmentNameOne), left_eye_validation_images)
numpy.save('numpy_validation_datasets/{0}_images.npy'.format(SegmentNameTwo), right_eye_validation_images)
numpy.save('numpy_validation_datasets/{0}_images.npy'.format(SegmentNameThree), nose_validation_images)

numpy.save('numpy_validation_datasets/{0}_labels.npy'.format(SegmentNameOne), left_eye_validation_labels)
numpy.save('numpy_validation_datasets/{0}_labels.npy'.format(SegmentNameTwo), right_eye_validation_labels)
numpy.save('numpy_validation_datasets/{0}_labels.npy'.format(SegmentNameThree), nose_validation_labels)



# Loading Load validation set from numpy array

# elimg = numpy.load('numpy_validation_datasets/{0}_images.npy'.format(SegmentNameOne))
# erimg = numpy.load('numpy_validation_datasets/{0}_images.npy'.format(SegmentNameTwo))
# nimg = numpy.load('numpy_validation_datasets/{0}_images.npy'.format(SegmentNameThree))
# labels = numpy.load('numpy_validation_datasets/{0}_labels.npy'.format(SegmentNameOne))

evaluate(segment_train_images, segment_validation_images,segment_train_labels, segment_validation_labels ,0)


