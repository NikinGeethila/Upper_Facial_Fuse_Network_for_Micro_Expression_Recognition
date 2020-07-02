import numpy
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.layers import Concatenate, Input, concatenate, add, multiply, maximum
from keras.layers import LeakyReLU,PReLU
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau,Callback
from sklearn.model_selection import train_test_split,LeaveOneOut
from keras import backend as K
from keras.optimizers import Adam,SGD
import timeit

class myCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('val_acc') >= 1.0):
            print("\nReached %2.2f%% accuracy, so stopping training!!" %(1.0*100))
            self.model.stop_training = True

def evaluate(SegmentOne_train_images,SegmentTwo_train_images,SegmentThree_train_images, SegmentOne_validation_images,SegmentTwo_validation_images, SegmentThree_validation_images,SegmentOne_train_labels,SegmentOne_validation_labels ,test_index ):
    # Fusion Model
    SegmentOne_input = Input(shape=(1, sizeH, sizeV, sizeD))
    SegmentOne_conv = Convolution3D(32, (20, 20, 9), strides=(10, 10, 3), padding='Same')(SegmentOne_input)
    ract_1 = PReLU()(SegmentOne_conv)


    SegmentTwo_input = Input(shape=(1, sizeH, sizeV, sizeD))
    SegmentTwo_conv = Convolution3D(32, (20, 20, 9), strides=(10, 10, 3), padding='Same')(SegmentTwo_input)
    ract_2 = PReLU()(SegmentTwo_conv)


    SegmentThree_input = Input(shape=(1, sizeH, sizeV, sizeD))
    SegmentThree_conv = Convolution3D(32, (20, 20, 9), strides=(10, 10, 3), padding='Same')(SegmentThree_input)
    ract_3 = PReLU()(SegmentThree_conv)


    concat = Concatenate(axis=1)([ract_1, ract_2, ract_3])
    Combined_conv = Convolution3D(32, (3, 3, 3), strides=1, padding='Same')(concat)
    ract_3 = PReLU()(Combined_conv)
    flatten_1 = Flatten()(ract_3)
    dense_1 = Dense(3, init='normal')(flatten_1)
    drop1 = Dropout(0.5)(dense_1)
    activation = Activation('softmax')(drop1)
    opt = SGD(lr=0.01)

    model = Model(inputs=[SegmentOne_input, SegmentTwo_input, SegmentThree_input], outputs=activation)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    filepath = "weights_CAS(ME)2/Late-Triple-weights-improvement" + str(test_index) + "-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    EarlyStop = EarlyStopping(monitor='val_acc', min_delta=0, patience=50, restore_best_weights=True, verbose=1,
                              mode='max')
    reduce = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=30, cooldown=10, verbose=1, min_delta=0,
                               mode='max', min_lr=0.0005)
    callbacks_list = [EarlyStop, reduce, myCallback()]

    model.summary()

    # Training the model
    history = model.fit([SegmentOne_train_images,SegmentTwo_train_images,SegmentThree_train_images], SegmentOne_train_labels, validation_data = ([SegmentOne_validation_images,SegmentTwo_validation_images,SegmentThree_validation_images], SegmentOne_validation_labels), callbacks=callbacks_list, batch_size = 16, nb_epoch = 100, shuffle=True)

    # Finding Confusion Matrix using pretrained weights

    predictions = model.predict([SegmentOne_validation_images,SegmentTwo_validation_images,SegmentThree_validation_images])
    predictions_labels = numpy.argmax(predictions, axis=1)
    validation_labels = numpy.argmax(SegmentOne_validation_labels, axis=1)
    cfm = confusion_matrix(validation_labels, predictions_labels)
    print (cfm)
    print("accuracy: ",accuracy_score(validation_labels, predictions_labels))

    return accuracy_score(validation_labels, predictions_labels)


K.set_image_dim_ordering('th')
SegmentNameOne='LeftEye'
SegmentNameTwo='RightEye'
SegmentNameThree='Nose'
sizeH=32
sizeV=32
sizeD=30
# Load training images and labels that are stored in numpy array

SegmentOne_training_set = numpy.load('numpy_training_datasets/{0}_images_{1}x{2}x{3}.npy'.format(SegmentNameOne,sizeH, sizeV,sizeD))
SegmentTwo_training_set = numpy.load('numpy_training_datasets/{0}_images_{1}x{2}x{3}.npy'.format(SegmentNameTwo,sizeH, sizeV,sizeD))
SegmentThree_training_set = numpy.load('numpy_training_datasets/{0}_images_{1}x{2}x{3}.npy'.format(SegmentNameThree,sizeH, sizeV,sizeD))

SegmentOne_training_labels = numpy.load('numpy_training_datasets/{0}_labels_{1}x{2}x{3}.npy'.format(SegmentNameOne,sizeH, sizeV,sizeD))
SegmentTwo_training_labels = numpy.load('numpy_training_datasets/{0}_labels_{1}x{2}x{3}.npy'.format(SegmentNameTwo,sizeH, sizeV,sizeD))
SegmentThree_training_labels = numpy.load('numpy_training_datasets/{0}_labels_{1}x{2}x{3}.npy'.format(SegmentNameThree,sizeH, sizeV,sizeD))




#-----------------------------------------------------------------------------------------------------------------
#LOOCV
loo = LeaveOneOut()
loo.get_n_splits(SegmentOne_training_set)
tot=0
count=0
for train_index, test_index in loo.split(SegmentOne_training_set):

    print("RUN: ",test_index)


    val_acc = evaluate(SegmentOne_training_set[train_index],SegmentTwo_training_set[train_index],SegmentThree_training_set[train_index],
     SegmentOne_training_set[test_index],SegmentTwo_training_set[test_index],SegmentThree_training_set[test_index]
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
SegmentThree_train_images, SegmentThree_validation_images, SegmentThree_train_labels, SegmentThree_validation_labels =  train_test_split(SegmentThree_training_set, SegmentThree_training_labels, test_size=0.2, random_state=42)



# Save validation set in a numpy array
numpy.save('numpy_validation_datasets/{0}_images_{1}x{2}x{3}.npy'.format(SegmentNameOne,sizeH, sizeV,sizeD), SegmentOne_validation_images)
numpy.save('numpy_validation_datasets/{0}_images_{1}x{2}x{3}.npy'.format(SegmentNameTwo,sizeH, sizeV,sizeD), SegmentTwo_validation_images)
numpy.save('numpy_validation_datasets/{0}_images_{1}x{2}x{3}.npy'.format(SegmentNameThree,sizeH, sizeV,sizeD), SegmentThree_validation_images)

numpy.save('numpy_validation_datasets/{0}_labels_{1}x{2}x{3}.npy'.format(SegmentNameOne,sizeH, sizeV,sizeD), SegmentOne_validation_labels)
numpy.save('numpy_validation_datasets/{0}_labels_{1}x{2}x{3}.npy'.format(SegmentNameTwo,sizeH, sizeV,sizeD), SegmentTwo_validation_labels)
numpy.save('numpy_validation_datasets/{0}_labels_{1}x{2}x{3}.npy'.format(SegmentNameThree,sizeH, sizeV,sizeD), SegmentThree_validation_labels)



# Loading Load validation set from numpy array

# SegmentOne_validation_images = numpy.load('numpy_validation_datasets/{0}_images_{1}x{2}x{3}.npy'.format(SegmentNameOne,sizeH, sizeV,sizeD))
# SegmentTwo_validation_images = numpy.load('numpy_validation_datasets/{0}_images_{1}x{2}x{3}.npy'.format(SegmentNameTwo,sizeH, sizeV,sizeD))
# SegmentThree_validation_images = numpy.load('numpy_validation_datasets/{0}_images_{1}x{2}x{3}.npy'.format(SegmentNameThree,sizeH, sizeV,sizeD))
# SegmentOne_validation_labels = numpy.load('numpy_validation_datasets/{0}_labels_{1}x{2}x{3}.npy'.format(SegmentNameOne,sizeH, sizeV,sizeD))

evaluate(SegmentOne_train_images,SegmentTwo_train_images,SegmentThree_train_images, SegmentOne_validation_images,SegmentTwo_validation_images, SegmentThree_validation_images,SegmentOne_train_labels,SegmentOne_validation_labels ,0)

'''