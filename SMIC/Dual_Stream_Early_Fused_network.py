import numpy
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.layers import Concatenate, Input, concatenate, add, multiply, maximum
from keras.layers import LeakyReLU,PReLU
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split,LeaveOneOut
from keras import backend as K
import timeit


def evaluate(SegmentOne_train_images,SegmentTwo_train_images, SegmentOne_validation_images,SegmentTwo_validation_images,SegmentOne_train_labels,SegmentOne_validation_labels ,test_index ):
    # Fusion Model
    SegmentOne_input = Input(shape=(1, sizeH, sizeV, 18))
    SegmentOne_conv = Convolution3D(32, (3, 3, 15))(SegmentOne_input)
    ract_1 = PReLU(alpha_initializer="zeros")(SegmentOne_conv)
    dropout_11 = Dropout(0.5)(ract_1)



    SegmentTwo_input = Input(shape=(1, sizeH, sizeV, 18))
    SegmentTwo_conv = Convolution3D(32, (3, 3, 15))(SegmentTwo_input)
    ract_3 = PReLU(alpha_initializer="zeros")(SegmentTwo_conv)
    dropout_12 = Dropout(0.5)(ract_3)



    concat = Concatenate(axis=1)([dropout_11, dropout_12])
    maxpool_2 = MaxPooling3D(pool_size=(3, 3, 3))(concat)
    ract_4 = PReLU(alpha_initializer="zeros")(maxpool_2)
    dropout_4 = Dropout(0.5)(ract_4)
    flatten_2 = Flatten()(dropout_4)
    dense_3 = Dense(1024, )(flatten_2)
    dropout_5 = Dropout(0.5)(dense_3)
    dense_4 = Dense(128, )(dropout_5)
    dropout_6 = Dropout(0.5)(dense_4)
    dense_7 = Dense(3, )(dropout_6)
    activation = Activation('softmax')(dense_7)

    model = Model(inputs=[SegmentOne_input, SegmentTwo_input], outputs=activation)
    model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])

    filepath = "weights_late_microexpfusenet/early-weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    model.summary()

    # Training the model
    history = model.fit([SegmentOne_train_images,SegmentTwo_train_images], SegmentOne_train_labels, validation_data = ([SegmentOne_validation_images,SegmentTwo_validation_images], SegmentOne_validation_labels), callbacks=callbacks_list, batch_size = 16, nb_epoch = 100, shuffle=True)

    # Finding Confusion Matrix using pretrained weights

    predictions = model.predict([SegmentOne_validation_images,SegmentTwo_validation_images])
    predictions_labels = numpy.argmax(predictions, axis=1)
    validation_labels = numpy.argmax(SegmentOne_validation_labels, axis=1)
    cfm = confusion_matrix(validation_labels, predictions_labels)
    print (cfm)
    print("accuracy: ",accuracy_score(validation_labels, predictions_labels))

    return accuracy_score(validation_labels, predictions_labels)


K.set_image_dim_ordering('th')
SegmentNameOne='Eyes'
SegmentNameTwo='Nose'
sizeH=32
sizeV=32

# Load training images and labels that are stored in numpy array

SegmentOne_training_set = numpy.load('numpy_training_datasets/{0}_images_{1}x{2}.npy'.format(SegmentNameOne,sizeH, sizeV))
SegmentTwo_training_set = numpy.load('numpy_training_datasets/{0}_images_{1}x{2}.npy'.format(SegmentNameTwo,sizeH, sizeV))

SegmentOne_training_labels = numpy.load('numpy_training_datasets/{0}_labels_{1}x{2}.npy'.format(SegmentNameOne,sizeH, sizeV))
SegmentTwo_training_labels = numpy.load('numpy_training_datasets/{0}_labels_{1}x{2}.npy'.format(SegmentNameTwo,sizeH, sizeV))




#-----------------------------------------------------------------------------------------------------------------
#LOOCV
loo = LeaveOneOut()
loo.get_n_splits(SegmentOne_training_set)
tot=0
count=0
for train_index, test_index in loo.split(SegmentOne_training_set):

    # print(segment_traininglabels[train_index])
    # print(segment_traininglabels[test_index])

    val_acc = evaluate(SegmentOne_training_set[train_index],SegmentTwo_training_set[train_index],
     SegmentOne_training_set[test_index],SegmentTwo_training_set[test_index]
    ,SegmentOne_training_labels[train_index], SegmentOne_training_labels[test_index] ,test_index)
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
SegmentOne_train_images, SegmentOne_validation_images, SegmentOne_train_labels, SegmentOne_validation_labels =  train_test_split(SegmentOne_training_set, SegmentOne_training_labels, test_size=0.2, random_state=42)
SegmentTwo_train_images, SegmentTwo_validation_images, SegmentTwo_train_labels, SegmentTwo_validation_labels =  train_test_split(SegmentTwo_training_set, SegmentTwo_training_labels, test_size=0.2, random_state=42)



# Save validation set in a numpy array
numpy.save('numpy_validation_datasets/{0}_images_{1}x{2}.npy'.format(SegmentNameOne,sizeH, sizeV), SegmentOne_validation_images)
numpy.save('numpy_validation_datasets/{0}_images_{1}x{2}.npy'.format(SegmentNameTwo,sizeH, sizeV), SegmentTwo_validation_images)

numpy.save('numpy_validation_datasets/{0}_labels_{1}x{2}.npy'.format(SegmentNameOne,sizeH, sizeV), SegmentOne_validation_labels)
numpy.save('numpy_validation_datasets/{0}_labels_{1}x{2}.npy'.format(SegmentNameTwo,sizeH, sizeV), SegmentTwo_validation_labels)



# Loading Load validation set from numpy array

# SegmentOne_validation_images = numpy.load('numpy_validation_datasets/{0}_images_{1}x{2}.npy'.format(segmentName,sizeH, sizeV))
# SegmentTwo_validation_images = numpy.load('numpy_validation_datasets/{0}_images_{1}x{2}.npy'.format(segmentName,sizeH, sizeV))
# SegmentOne_validation_labels = numpy.load('numpy_validation_datasets/{0}_labels_{1}x{2}.npy'.format(segmentName,sizeH, sizeV))

evaluate(SegmentOne_train_images,SegmentTwo_train_images, SegmentOne_validation_images,SegmentTwo_validation_images,SegmentOne_train_labels,SegmentOne_validation_labels ,0)
'''

