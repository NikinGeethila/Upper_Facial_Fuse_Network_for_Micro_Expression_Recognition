import numpy
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,f1_score
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.layers import Concatenate, Input, concatenate, add, multiply, maximum
from keras.layers import LeakyReLU,PReLU
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau,Callback
from sklearn.model_selection import train_test_split,LeaveOneOut,KFold
from keras import backend as K
from keras.optimizers import Adam,SGD
import timeit
import os

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
    SegmentOne_conv_Two = Convolution3D(32, (3, 3, 3), strides=1, padding='Same')(ract_1)
    ract_2 = PReLU()(SegmentOne_conv_Two)
    flatten_1 = Flatten()(ract_2)

    SegmentTwo_input = Input(shape=(1, sizeH, sizeV, sizeD))
    SegmentTwo_conv = Convolution3D(32, (20, 20, 9), strides=(10, 10, 3), padding='Same')(SegmentTwo_input)
    ract_3 = PReLU()(SegmentTwo_conv)
    SegmentTwo_conv_Two = Convolution3D(32, (3, 3, 3), strides=1, padding='Same')(ract_3)
    ract_4 = PReLU()(SegmentTwo_conv_Two)
    flatten_2 = Flatten()(ract_4)

    SegmentThree_input = Input(shape=(1, sizeH, sizeV, sizeD))
    SegmentThree_conv = Convolution3D(32, (20, 20, 9), strides=(10, 10, 3), padding='Same')(SegmentThree_input)
    ract_5 = PReLU()(SegmentThree_conv)
    SegmentThree_conv_Two = Convolution3D(32, (3, 3, 3), strides=1, padding='Same')(ract_5)
    ract_6 = PReLU()(SegmentThree_conv_Two)
    flatten_3 = Flatten()(ract_6)

    concat = Concatenate(axis=1)([flatten_1, flatten_2, flatten_3])
    dense_1 = Dense(5, init='normal' )(concat)
    drop1 = Dropout(0.5)(dense_1)
    activation = Activation('softmax')(drop1)
    opt = SGD(lr=0.01)

    model = Model(inputs=[SegmentOne_input, SegmentTwo_input, SegmentThree_input], outputs=activation)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    filepath = "weights_CASMEII/Late-Triple-weights-improvement" + str(test_index) + "-{epoch:02d}-{val_acc:.2f}.hdf5"
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

    return accuracy_score(validation_labels, predictions_labels), validation_labels, predictions_labels


# -----------------------------------------------------------------------------------------------------------------
# LOOCV
def loocv():
    loo = LeaveOneOut()
    loo.get_n_splits(SegmentOne_training_set)
    tot = 0
    count = 0
    val_labels = []
    pred_labels = []
    accs = []
    accs2 = []
    for train_index, test_index in loo.split(SegmentOne_training_set):
        print("RUN: ", test_index)

        val_acc, val_label, pred_label = evaluate(SegmentOne_training_set[train_index],
                                                  SegmentTwo_training_set[train_index],
                                                  SegmentThree_training_set[train_index],
                                                  SegmentOne_training_set[test_index],
                                                  SegmentTwo_training_set[test_index],
                                                  SegmentThree_training_set[test_index]
                                                  , SegmentOne_training_labels[train_index],
                                                  SegmentOne_training_labels[test_index], test_index)
        tot += val_acc
        val_labels.extend(val_label)
        pred_labels.extend(pred_label)
        accs.append(val_acc)
        accs2.append(SegmentOne_training_labels[test_index])
        count += 1
        print("------------------------------------------------------------------------")
        print("validation acc:", val_acc)
        print("------------------------------------------------------------------------")

    cfm = confusion_matrix(val_labels, pred_labels)
    # tp_and_fn = sum(cfm.sum(1))
    # tp_and_fp = sum(cfm.sum(0))
    # tp = sum(cfm.diagonal())
    print("cfm: \n", cfm)
    # print("tp_and_fn: ",tp_and_fn)
    # print("tp_and_fp: ",tp_and_fp)
    # print("tp: ",tp)
    #
    # precision = tp / tp_and_fp
    # recall = tp / tp_and_fn
    # print("precision: ",precision)
    # print("recall: ",recall)
    # print("F1-score: ",f1_score(val_labels,pred_labels,average="macro"))
    print("F1-score: ", f1_score(val_labels, pred_labels, average="weighted"))
    return val_labels, pred_labels


# -----------------------------------------------------------------------------------------------------------------
# Test train split

def split():
    # Spliting the dataset into training and validation sets
    SegmentOne_train_images, SegmentOne_validation_images, SegmentOne_train_labels, SegmentOne_validation_labels = train_test_split(
        SegmentOne_training_set, SegmentOne_training_labels, test_size=0.2, random_state=42)
    SegmentTwo_train_images, SegmentTwo_validation_images, SegmentTwo_train_labels, SegmentTwo_validation_labels = train_test_split(
        SegmentTwo_training_set, SegmentTwo_training_labels, test_size=0.2, random_state=42)

    # Save validation set in a numpy array
    numpy.save('numpy_validation_datasets/{0}_images_{1}x{2}x{3}.npy'.format(SegmentNameOne, sizeH, sizeV, sizeD),
               SegmentOne_validation_images)
    numpy.save('numpy_validation_datasets/{0}_images_{1}x{2}x{3}.npy'.format(SegmentNameTwo, sizeH, sizeV, sizeD),
               SegmentTwo_validation_images)

    numpy.save('numpy_validation_datasets/{0}_labels_{1}x{2}x{3}.npy'.format(SegmentNameOne, sizeH, sizeV, sizeD),
               SegmentOne_validation_labels)
    numpy.save('numpy_validation_datasets/{0}_labels_{1}x{2}x{3}.npy'.format(SegmentNameTwo, sizeH, sizeV, sizeD),
               SegmentTwo_validation_labels)

    # Loading Load validation set from numpy array

    # SegmentOne_validation_images = numpy.load('numpy_validation_datasets/{0}_images_{1}x{2}x{3}.npy'.format(SegmentNameOne,sizeH, sizeV,sizeD))
    # SegmentTwo_validation_images = numpy.load('numpy_validation_datasets/{0}_images_{1}x{2}x{3}.npy'.format(SegmentNameTwo,sizeH, sizeV,sizeD))

    evaluate(SegmentOne_train_images, SegmentTwo_train_images, SegmentOne_validation_images,
             SegmentTwo_validation_images, SegmentOne_train_labels, SegmentOne_validation_labels, 0)


def kfold():
    kf = KFold(n_splits=10, random_state=42, shuffle=True)
    # kf.get_n_splits(segment_training_set)
    tot = 0
    count = 0
    accs = []
    accs2 = []

    val_labels = []
    pred_labels = []
    for train_index, test_index in kf.split(SegmentOne_training_set):
        # print(segment_traininglabels[train_index])
        # print(segment_traininglabels[test_index])
        print(test_index)
        val_acc, val_label, pred_label = evaluate(SegmentOne_training_set[train_index],
                                                  SegmentTwo_training_set[train_index],
                                                  SegmentThree_training_set[train_index],
                                                  SegmentOne_training_set[test_index],
                                                  SegmentTwo_training_set[test_index],
                                                  SegmentThree_training_set[test_index]
                                                  , SegmentOne_training_labels[train_index],
                                                  SegmentOne_training_labels[test_index], test_index)
        tot += val_acc
        val_labels.extend(val_label)
        pred_labels.extend(pred_label)
        accs.append(val_acc)
        accs2.append(SegmentOne_training_labels[test_index])
        count += 1
        print("------------------------------------------------------------------------")
        print("validation acc:", val_acc)
        print("------------------------------------------------------------------------")
    print("accuracy: ", accuracy_score(val_labels, pred_labels))
    cfm = confusion_matrix(val_labels, pred_labels)
    # tp_and_fn = sum(cfm.sum(1))
    # tp_and_fp = sum(cfm.sum(0))
    # tp = sum(cfm.diagonal())
    print("cfm: \n", cfm)
    # print("tp_and_fn: ",tp_and_fn)
    # print("tp_and_fp: ",tp_and_fp)
    # print("tp: ",tp)
    #
    # precision = tp / tp_and_fp
    # recall = tp / tp_and_fn
    # print("precision: ",precision)
    # print("recall: ",recall)
    # print("F1-score: ",f1_score(val_labels,pred_labels,average="macro"))
    print("F1-score: ", f1_score(val_labels, pred_labels, average="weighted"))
    return val_labels, pred_labels


####################################
# edit params
K.set_image_dim_ordering('th')

SegmentNameOne = 'LeftEye'
SegmentNameTwo = 'RightEye'
SegmentNameThree = 'Nose'
sizeH = 32
sizeV = 32
sizeD = 30
testtype = 'kfold'
####################################


K.set_image_dim_ordering('th')

# Load training images and labels that are stored in numpy array

SegmentOne_training_set = numpy.load(
    'numpy_training_datasets/{0}_images_{1}x{2}x{3}.npy'.format(SegmentNameOne, sizeH, sizeV, sizeD))
SegmentTwo_training_set = numpy.load(
    'numpy_training_datasets/{0}_images_{1}x{2}x{3}.npy'.format(SegmentNameTwo, sizeH, sizeV, sizeD))
SegmentThree_training_set = numpy.load(
    'numpy_training_datasets/{0}_images_{1}x{2}x{3}.npy'.format(SegmentNameThree, sizeH, sizeV, sizeD))

SegmentOne_training_labels = numpy.load(
    'numpy_training_datasets/{0}_labels_{1}x{2}x{3}.npy'.format(SegmentNameOne, sizeH, sizeV, sizeD))
SegmentTwo_training_labels = numpy.load(
    'numpy_training_datasets/{0}_labels_{1}x{2}x{3}.npy'.format(SegmentNameTwo, sizeH, sizeV, sizeD))
SegmentThree_training_labels = numpy.load(
    'numpy_training_datasets/{0}_labels_{1}x{2}x{3}.npy'.format(SegmentNameThree, sizeH, sizeV, sizeD))

if testtype == "kfold":
    val_labels, pred_labels = kfold()
elif testtype == "loocv":
    val_labels, pred_labels = loocv()
elif testtype == "split":
    val_labels, pred_labels = split()
else:
    print("error")

# ---------------------------------------------------------------------------------------------------
# write to results

results = open("../TempResults.txt", 'a')
results.write("---------------------------\n")
full_path = os.path.realpath(__file__)
results.write(
    str(__file__) + " {0}_{1}_{2}_{3}_{4}x{5}x{6}\n".format(testtype, SegmentNameOne, SegmentNameTwo, SegmentNameThree,
                                                            sizeH, sizeV, sizeD))
results.write("---------------------------\n")
results.write("accuracy: " + str(accuracy_score(val_labels, pred_labels)) + "\n")
results.write("F1-score: " + str(f1_score(val_labels, pred_labels, average="weighted")) + "\n")