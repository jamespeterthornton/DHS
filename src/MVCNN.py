# import libraries
from __future__ import print_function
from __future__ import division

import numpy as np 
import pandas as pd
import os
import re

import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, Concatenate
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, GlobalAveragePooling2D, Input, Convolution2D
from keras.models import Model
from keras import backend as K
from keras.layers import Activation
from keras.layers import BatchNormalization, GlobalMaxPooling2D
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import ModelCheckpoint
from keras.applications.vgg16 import VGG16
from keras.backend.tensorflow_backend import set_session

import random
from timeit import default_timer as timer

import tsahelper.tsahelper as tsa
import scipy
#---------------------------------------------------------------------------------------
# Constants
#
# INPUT_FOLDER:                 The folder that contains the source data
#
# PREPROCESSED_DATA_FOLDER:     The folder that contains preprocessed .npy files 
# 
# STAGE1_LABELS:                The CSV file containing the labels by subject
#
# THREAT_ZONE:                  Threat Zone to train on (actual number not 0 based)
#
# BATCH_SIZE:                   Number of Subjects per batch
#
# EXAMPLES_PER_SUBJECT          Number of examples generated per subject
#
# FILE_LIST:                    A list of the preprocessed .npy files to batch
# 
# TRAIN_TEST_SPLIT_RATIO:       Ratio to split the FILE_LIST between train and test
#
# TRAIN_SET_FILE_LIST:          The list of .npy files to be used for training
#
# TEST_SET_FILE_LIST:           The list of .npy files to be used for testing
#
# IMAGE_DIM:                    The height and width of the images in pixels
#
# LEARNING_RATE                 Learning rate for the neural network
#
# N_TRAIN_STEPS                 The number of train steps (epochs) to run
#
# TRAIN_PATH                    Place to store the tensorboard logs
#
# MODEL_PATH                    Path where model files are stored
#
# MODEL_NAME                    Name of the model files
#
#----------------------------------------------------------------------------------------
INPUT_FOLDER = 'stage1_aps'
PREPROCESSED_DATA_FOLDER = 'tsa_datasets/preprocessed/'
STAGE1_LABELS = 'labels/stage1_labels.csv'
SUBJECT_LABELS_PATH = "labels/stage1_labels.csv"
THREAT_ZONE = 11
BATCH_SIZE = 16
EXAMPLES_PER_SUBJECT = 182

FILE_LIST = []
TRAIN_TEST_SPLIT_RATIO = 0.2
TRAIN_SET_FILE_LIST = []
TEST_SET_FILE_LIST = []

IMAGE_DIM = 250
LEARNING_RATE = 1e-3
N_TRAIN_STEPS = 1
TRAIN_PATH = 'tsa_logs/train/'
MODEL_PATH = 'tsa_logs/model/'
MODEL_NAME = ('tsa-{}-lr-{}-{}-{}-tz-{}'.format('alexnet-v0.1', LEARNING_RATE, IMAGE_DIM, 
                                                IMAGE_DIM, THREAT_ZONE )) 

MVCNN_PATH = "weights/MVCNN.h5"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config = config))

def get_train_test_file_list():
    global SUBJECT_LIST
    global TRAIN_SUBJECT_LIST
    global TEST_SUBJECT_LIST
    
    df = pd.read_csv(STAGE1_LABELS)
    df['Subject'], df['Zone'] = df['Id'].str.split('_',1).str
    SUBJECT_LIST = df['Subject'].unique()
    train_test_split = len(SUBJECT_LIST) - max(int(len(SUBJECT_LIST)*TRAIN_TEST_SPLIT_RATIO),1)
    TRAIN_SUBJECT_LIST = SUBJECT_LIST[:train_test_split]
    TEST_SUBJECT_LIST = SUBJECT_LIST[train_test_split:]
    print('Train/Test Split -> {} file(s) of {} used for testing'.format( 
         len(SUBJECT_LIST) - train_test_split, len(SUBJECT_LIST)))

def get_subject_labels():
    infile = SUBJECT_LABELS_PATH
    df = pd.read_csv(infile)
    # Separate the zone and subject id into a df
    df['Subject'], df['Zone'] = df['Id'].str.split('_',1).str
    #TODO: convert zone to correct int here
    df = df[['Subject', 'Zone', 'Probability']]
    return df

# Note that a batch size of 1 will actually return 16 examples because of implementation ease
def dae_generator(subjects, batch_size):
    # intialize tracking and saving items
    threat_zone_examples = []
    start_time = timer()
    labels = get_subject_labels()
    print_shape = True
    while True:
        for i in range(0, len(subjects), batch_size):
            y_batch = []
            x_batch = []
            for subject in subjects[i:i+batch_size]:
                images = tsa.read_data(INPUT_FOLDER + '/' + subject + '.aps')
                # transpose so that the slice is the first dimension shape(16, 660, 512)
                images = images.transpose()
                images = np.reshape(images, (16, 660, 512, 1))
                for image in images:
                    y_batch.append(image)
                y_batch = np.array(y_batch)
                noise_factor = 0.2
                noisy_images = images + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=images.shape)
                for image in noisy_images:
                    x_batch.append(image)
                x_batch = np.array(x_batch)
            yield np.array(y_batch), np.array(y_batch)

def generator(subjects, batch_size):
    # intialize tracking and saving items
    threat_zone_examples = []
    start_time = timer()
    labels = get_subject_labels()
    print_shape = True
    while True:
        for i in range(0, len(subjects), batch_size):
            y_batch = []
            features = {}
            for j in range(0, 16):
                features[str(j)] = []
            for subject in subjects[i:i+batch_size]:
                images = tsa.read_data(INPUT_FOLDER + '/' + subject + '.aps')
                # transpose so that the slice is the first dimension shape(16, 620, 512)
                images = images.transpose()
                for j in range(0, 16):
                    fake_rgb = np.array([images[j], images[j], images[j]])
                    image = fake_rgb.transpose()
                    """if print_shape: 
                        print ("Shape of re-transposed image:")
                        print (image.shape)
                        print_shape = False
                    """    
                    #resized_image = images[j] #scipy.ndimage.zoom(images[j], (0.5, 0.5))
                    #features[str(j)].append(np.reshape(resized_image, (660, 512, 1)))# (330, 256, 1)))
                    features[str(j)].append(image)# (330, 256, 1)))
                     
                # get label
                y = np.zeros((17))
                threat_list = labels.loc[labels['Subject'] == subject.split(".")[0]]
                threat_iter = threat_list.iterrows()
                while True:
                    threat = next(threat_iter, None)
                    if threat is None:
                        break
                    threat = threat[1]
                    if threat['Probability'] is 1:
                        zone = threat['Zone']
                        zone = int(zone[4:])
                        y[zone-1] = 1
                """
                y = np.array(tsa.get_subject_zone_label(THREAT_ZONE, 
                                 tsa.get_subject_labels(STAGE1_LABELS, subject)))
                np.reshape(y, (2, 1))
                """
                y_batch.append(y)
 
            for j in range(0, 16):
                features[str(j)] = np.array(features[str(j)])
                #features2.append(np.array(features[j]))
 
            yield features, np.array(y_batch)


def big_im_generator(subjects, batch_size):

    for_vgg = False

    # intialize tracking and saving items
    threat_zone_examples = []
    start_time = timer()
    labels = get_subject_labels()
    print_shape = True
    while True:
        for i in range(0, len(subjects), batch_size):
            y_batch = []
            x_batch = []
            for subject in subjects[i:i+batch_size]:
                images = tsa.read_data(INPUT_FOLDER + '/' + subject + '.aps')
                # transpose so that the slice is the first dimension shape(16, 620, 512)
                images = images.transpose()
                x_images = []
                if for_vgg:
                    for j in range(0, 16):
                        image_to_use = scipy.ndimage.zoom(images[j], (0.75, 0.75))
                        fake_rgb = np.array([image_to_use, image_to_use, image_to_use])
                        image = fake_rgb.transpose()
                        x_images.append(image)
                else:
                    for j in range(0, 16):
                        image_to_use = scipy.ndimage.zoom(images[j], (0.5, 0.5))
                        image_to_use = image_to_use.reshape(len(image_to_use), len(image_to_use[0]), 1)
                        x_images.append(image_to_use)
                x = np.hstack(np.array(x_images))
                #x = x.reshape((495, 6144, 1))
                # get label
                y = np.zeros((17))
                threat_list = labels.loc[labels['Subject'] == subject.split(".")[0]]
                threat_iter = threat_list.iterrows()
                while True:
                    threat = next(threat_iter, None)
                    if threat is None:
                        break
                    threat = threat[1]
                    if threat['Probability'] is 1:
                        zone = threat['Zone']
                        zone = int(zone[4:])
                        y[zone-1] = 1
                y_batch.append(y)
                x_batch.append(x) 
            yield np.array(x_batch), np.array(y_batch)

def get_relevant_subjects(subjects, zone):
    threats = []
    non_threats = []
    relevant_subjects = []
    features = []
    labels = []
    for subject in subjects:
        # get label
        label = np.array(tsa.get_subject_zone_label(THREAT_ZONE, 
		     tsa.get_subject_labels(STAGE1_LABELS, subject)))
        if np.array_equal(label, [0, 1]):
            threats.append(subject)
        else:
            non_threats.append(subject)
    
    for i in range(0, len(threats)):
        relevant_subjects.append(threats[i])
        relevant_subjects.append(non_threats[i])
    
    np.random.shuffle(relevant_subjects)
    
    return relevant_subjects

def make_view_pool(view_features, name):
    vp = K.expand_dims(view_features[0], 0) # eg. [100] -> [1, 100]
    for v in view_features[1:]:
        v = K.expand_dims(v, 0)
        vp = Concatenate(axis=0)([vp, v]) #tf.concat([vp, v], 0)
    #print 'vp before reducing:', vp.get_shape().as_list()
    #vp = tf.reduce_max(vp, [0], name=name)
    return vp  


def MVCNN(weights_path=None):

    inputs = []
    view_pool = []

    """
    new_input = Input((660, 512, 1)) #make new input
    #new_input = Input((330, 256, 1)) #make new input
    x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_normal')(new_input)
    x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_normal')(x)
    x = MaxPooling2D((2,2), strides=(2,2))(x)
    x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_normal')(x)
    x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_normal')(x)
    x = MaxPooling2D((2,2), strides=(2,2))(x)
    x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_normal')(x)
    x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_normal')(x)
    x = MaxPooling2D((2,2), strides=(2,2))(x)
    x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_normal')(x)
    x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_normal')(x)
    x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_normal')(x)
    x = MaxPooling2D((2,2), strides=(2,2))(x)
    x = Flatten()(x)
    cnn1 = Model(inputs = new_input, outputs = x)
    """
    for i in range(0, 16):
        new_input = Input((660, 512, 1), name=str(i)) #make new input
        x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_normal')(new_input)
        x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_normal')(x)
        x = MaxPooling2D((2,2), strides=(2,2))(x)
        x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_normal')(x)
        x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_normal')(x)
        x = MaxPooling2D((2,2), strides=(2,2))(x)
        x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_normal')(x)
        x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_normal')(x)
        x = MaxPooling2D((2,2), strides=(2,2))(x)
        x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_normal')(x)
        x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_normal')(x)
        x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_normal')(x)
        x = MaxPooling2D((2,2), strides=(2,2))(x)
        x = Flatten()(x)
        inputs.append(new_input)
        view_pool.append(x)
        """
        cnn1 = Model(inputs = new_input, outputs = x)
        new_input = Input((660, 512, 1), name=str(i)) #make new input
        #new_input = Input((330, 256, 1), name=str(i)) #make new input
        new_model = cnn1(new_input)
        inputs.append(new_input)
        view_pool.append(new_model)
        """
    vp = Concatenate(axis=0)(view_pool) #tf.concat([vp, v], 0)
    model = Dense(512, activation='relu', kernel_initializer='glorot_normal')(vp)
    #model = Dropout(0.2)(model)
    model = Dense(512, activation='relu', kernel_initializer='glorot_normal')(model)
    #model = Dropout(0.2)(model)
    #model = Flatten()(model)
    """model = Dense(2048, activation='relu')(vp)
    model = Dense(1024, activation='relu')(model)
    model = Dense(512, activation='relu')(model)
    model = Dense(256, activation='relu')(model) """
    model = Dense(17, activation='sigmoid', kernel_initializer='glorot_normal')(model)
    
    full_model = Model(inputs=inputs, outputs=model)

    full_model.compile(loss=keras.losses.categorical_crossentropy,
            optimizer= keras.optimizers.AdamAccum(lr=0.01, accumulator=32.0), metrics=['acc'])

    full_model.summary()

    return full_model


def VGG_big_im(weights_path=None):
    #base_model = VGG16(weights='imagenet', include_top=False, input_shape=(384, 7920, 3))
    #x = base_model.layers[10].output
    #x = base_model.get_layer('block5_pool').output
    inputs = Input((330, 4096, 1))
    x = Conv2D(32, (2, 2), activation='relu')(inputs)
    x = BatchNormalization(axis=1)(x)
 
    x = Conv2D(32, (2, 2), activation='relu')(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization(axis=1)(x)
    x = MaxPooling2D((4,4), strides=(2,2))(x)
    x = Conv2D(32, (2, 2), activation='relu')(x)
    x = BatchNormalization(axis=1)(x)
    x = Conv2D(32, (2, 2), activation='relu')(x)
    x = Dropout(0.2)(x)
    x = MaxPooling2D((4,4), strides=(2,2))(x)
    x = BatchNormalization(axis=1)(x)
    x = Conv2D(32, (2, 2), activation='relu')(x)
    x = BatchNormalization(axis=1)(x)
    x = Conv2D(32, (2, 2), activation='relu')(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization(axis=1)(x)
    x = Conv2D(32, (2, 2), activation='relu')(x)
    x = Dropout(0.2)(x)
    x = MaxPooling2D((4,4), strides=(2,2))(x)
    x = Conv2D(32, (2, 2), activation='relu')(x)
    x = BatchNormalization(axis=1)(x)
    x = Conv2D(32, (2, 2), activation='relu')(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization(axis=1)(x)
    x = Conv2D(32, (2, 2), activation='relu')(x)
    x = Dropout(0.2)(x)
    x = MaxPooling2D((4,4), strides=(2,2))(x)
    x = Conv2D(32, (2, 2), activation='relu')(x)
    x = BatchNormalization(axis=1)(x)
    x = Conv2D(32, (2, 2), activation='relu')(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization(axis=1)(x)
    x = Conv2D(32, (2, 2), activation='relu')(x)
    x = Dropout(0.2)(x)
    x = MaxPooling2D((4,4), strides=(2,2))(x)
    """x = Conv2D(128, (2, 2), activation='relu')(x)
    x = Conv2D(128, (2, 2), activation='relu')(x)
    x = Conv2D(128, (2, 2), activation='relu')(x)
    x = MaxPooling2D((2,2), strides=(2,2))(x)
    x = Conv2D(256, (2, 2), activation='relu')(x)
    x = Conv2D(256, (2, 2), activation='relu')(x)
    x = Conv2D(256, (2, 2), activation='relu')(x)
    x = MaxPooling2D((2,2), strides=(2,2))(x)"""
    x = Flatten()(x)
    x = BatchNormalization(axis=1)(x)
    #x = GlobalMaxPooling2D()(x)
    x = Dense(128, activation='relu', name='fc2')(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    #x = Dropout(0.2)(x)
    x = Dense(256, activation='relu', name='fc3')(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    
    x = Dense(17, activation='relu', name='prediction')(x)


    #for layer in base_model.layers:
    #    layer.trainable = False

    model = Model(inputs=inputs, outputs=x)

    sgd = keras.optimizers.SGD(lr=1e-3, decay=1e-6, momentum=0.7, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    #base_model.summary()

    return model

def VGG_16(weights_path=None):

    inputs = []
    view_pool = []

    #new_input = Input((512, 660, 3)) #make new input

    #Instantiate shared CNN1 Layers
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(512, 660, 3))
    x = base_model.layers[10].output
    #conv1 = Conv2D(32, (3, 3), activation='relu')(x)
    #pool1 = MaxPooling2D((2,2), strides=(2,2))(conv1)
    #conv2 = Conv2D(32, (3, 3), activation='relu')(conv1)
    #pool2 = MaxPooling2D((2,2), strides=(2,2))(conv2)
    #conv3 = Conv2D(32, (3, 3), activation='relu')(conv2)
    #conv4 = Conv2D(32, (3, 3), activation='relu')(conv3)
    #conv5 = Conv2D(32, (3, 3), activation='relu')(conv4)
    #pool3 = MaxPooling2D((2,2), strides=(2,2))(conv5)
    x = Flatten()(x)
    cnn1 = Model(inputs = base_model.input, outputs = x)# = pool3)

    for i in range(0, 16):
        new_input = Input((512, 660, 3), name=str(i)) #make new input
        new_model = cnn1(new_input)
        inputs.append(new_input)
        view_pool.append(new_model)

    #vp = make_view_pool(view_pool, "vp")

    for layer in base_model.layers:
        layer.trainable = False

    vp = Concatenate(axis=0)(view_pool) #tf.concat([vp, v], 0)
    model = Dense(512, activation='relu')(vp)
    model = Dropout(0.2)(model)
    model = Dense(512, activation='relu')(model)
    model = Dropout(0.2)(model)
    model = Flatten()(model)
    model = Dense(17, activation='sigmoid')(model)
    
    full_model = Model(inputs=inputs, outputs=model)

    full_model.compile(loss=keras.losses.categorical_crossentropy,
            optimizer= keras.optimizers.Adam(lr=0.001), metrics=['acc'])

    base_model.summary()
    print("----")

    full_model.summary()

    return full_model

def DAE():
    #Encode
    inputs = Input((660, 512, 1))
    x = Convolution2D(32, 3, 3, activation='relu')(inputs)
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(32, 3, 3, activation='relu')(x)
    x = ZeroPadding2D((1,1))(x)
    x = MaxPooling2D((2,2), strides=(2,2))(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = ZeroPadding2D((1,1))(x)
    x = MaxPooling2D((2,2), strides=(2,2))(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = ZeroPadding2D((1,1))(x)
    x = MaxPooling2D((2,2), strides=(2,2))(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = ZeroPadding2D((1,1))(x)
    x = MaxPooling2D((2,2), strides=(2,2))(x)
    #Center
    x = Convolution2D(64, 3, 3, activation='relu')(x)
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(64, 3, 3, activation='relu')(x)
    x = ZeroPadding2D((1,1))(x)
    #Decode
    x = UpSampling2D((2,2))(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = ZeroPadding2D((1,1))(x)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = ZeroPadding2D((1,1))(x)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = ZeroPadding2D((2,1))(x)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = ZeroPadding2D((2,2))(x)
    x = Conv2D(1, (3, 3), activation='relu')(x)
    
    model = Model(inputs=inputs, outputs=x)
    model.compile(loss=keras.losses.binary_crossentropy,
            optimizer= keras.optimizers.Adam(lr=0.001), metrics=['acc'])
    model.summary()
    
    return model

def train_binary_net():

    batch_size = 1

    get_train_test_file_list()
    test_subjects =TEST_SUBJECT_LIST# get_relevant_subjects(TEST_SUBJECT_LIST, THREAT_ZONE)
    train_subjects =TRAIN_SUBJECT_LIST# get_relevant_subjects(TRAIN_SUBJECT_LIST, THREAT_ZONE)
  
    test_gen = generator(test_subjects, batch_size)
    train_gen = generator(train_subjects, batch_size)
    
    #test_gen = generator(TEST_SUBJECT_LIST, batch_size)
    #train_gen = generator(TRAIN_SUBJECT_LIST, batch_size)
    print("train_gen info:") 
    print(np.array(next(train_gen)[1]).shape)
    print(next(train_gen)[1])
    print ("data lengths:")
    print (len(train_subjects))
    print (len(test_subjects))


    train_steps = np.ceil(float(len(train_subjects)) / float(batch_size))
    test_steps = np.ceil(float(len(test_subjects)) / float(batch_size))

    LR = 0.01

    model = VGG_16()#MVCNN()
    #model.load_weights(MVCNN_PATH)

    print ("LR is " + str(LR))
    model.compile(loss=keras.losses.binary_crossentropy,
            optimizer= keras.optimizers.AdamAccum(lr=LR, accumulator=32.0), metrics=['acc'])
    mvcnn_checkpoint = ModelCheckpoint(MVCNN_PATH, monitor='loss', verbose=1, save_best_only=True, mode='min')
    model.fit_generator(generator=train_gen, validation_data=test_gen, steps_per_epoch = train_steps, validation_steps = test_steps, 
        epochs = 1000, verbose=2, callbacks=[mvcnn_checkpoint])

    """for i in range(0, 15):
    
        im, label = next(train_gen)
    
        print("imshape, label:")
        print (im.shape)
        print (label)
    
    print ("lengths:")
    print(len(test_subjects))
    print(len(train_subjects))
    # get train and test batches
    get_train_test_file_list()
    features, labels = get_dataset(TRAIN_SET_FILE_LIST, PREPROCESSED_DATA_FOLDER)
    val_features, val_labels = get_dataset(TEST_SET_FILE_LIST, PREPROCESSED_DATA_FOLDER)
    features = features.reshape(-1, IMAGE_DIM, IMAGE_DIM, 3)
    labels = labels.reshape(-1, 2)
    val_features = val_features.reshape(-1, IMAGE_DIM, IMAGE_DIM, 3)
    val_labels = val_labels.reshape(-1, 2)
    print (features[0].shape)    


    # instantiate model
    #model = alexnet(IMAGE_DIM, IMAGE_DIM, LEARNING_RATE)
    log = model.fit(x=features, y=labels, batch_size=1, epochs=10000, verbose=2, validation_data=(val_features, val_labels))
    """
#train_binary_net()

def test_dae():

    DAE_PATH = "weights/DAE.h5"

    #bs actually *16, see dae_gen
    batch_size = 1
    get_train_test_file_list()
    train_gen = dae_generator(TRAIN_SUBJECT_LIST, batch_size)
    test_gen = dae_generator(TEST_SUBJECT_LIST, batch_size)
    train_steps = np.ceil(float(len(TRAIN_SUBJECT_LIST)) / float(batch_size))
    test_steps = np.ceil(float(len(TEST_SUBJECT_LIST)) / float(batch_size))
    model = DAE()

    model.load_weights(DAE_PATH)
    model.compile(loss=keras.losses.binary_crossentropy,
            optimizer= keras.optimizers.Adam(lr=0.1), metrics=['acc'])

    dae_checkpoint = ModelCheckpoint(DAE_PATH, monitor='loss', verbose=1, save_best_only=True, mode='min')

    model.fit_generator(generator=train_gen, validation_data=test_gen, steps_per_epoch = train_steps, validation_steps = test_steps, 
        epochs = 1000, verbose=2, callbacks=[dae_checkpoint])

#test_dae()

def test_big_cnn():
    batch_size = 2
    model = VGG_big_im()
    get_train_test_file_list()
    #TODO Shuffle the train subjects!
    dae_checkpoint = ModelCheckpoint("best_big_cnn.h5", monitor='loss', verbose=1, save_best_only=True, mode='min')

    train_gen = big_im_generator(TRAIN_SUBJECT_LIST, batch_size)
    test_gen = big_im_generator(TRAIN_SUBJECT_LIST, batch_size)
    train_steps = np.ceil(float(len(TRAIN_SUBJECT_LIST)) / float(batch_size))
    test_steps = np.ceil(float(len(TEST_SUBJECT_LIST)) / float(batch_size))
    
    model.fit_generator(generator=train_gen, validation_data=test_gen, steps_per_epoch = train_steps, validation_steps = test_steps, 
        epochs = 1000, verbose=2, callbacks=[dae_checkpoint])

test_big_cnn()

def make_submission():
    model = MVCNN()

    sub_file = open("labels/stage1_submission_1.csv", "w")

    sub_file.write("Id,Probability\n")

    model.load_weights(MVCNN_PATH)
    example_sub = "labels/stage1_sample_submission.csv"
    df = pd.read_csv(example_sub)
    # Separate the zone and subject id into a df
    df['Subject'], df['Zone'] = df['Id'].str.split('_',1).str
    #TODO: convert zone to correct int here
    df = df[['Subject', 'Zone', 'Probability']]
    subjects = df['Subject'].unique()

    get_train_test_file_list()
    """
    print("hmm we tryin")
    gen = generator(TRAIN_SUBJECT_LIST, 1)
    x, y = next(gen)
    print (x['0'].shape)

    model.predict(x)
    print ("ok, here")
    """

    prog = 0

    for subject in subjects:
        images = tsa.read_data(INPUT_FOLDER + '/' + subject + '.aps')
        images = images.transpose()
        images_to_predict = {}
        image_array = []
        for i in range(0, len(images)):
            images_to_predict[str(i)] = []
            images_to_predict[str(i)].append(np.reshape(images[i], (660, 512, 1)))
            images_to_predict[str(i)] = np.array(images_to_predict[str(i)])
            image_array.append(np.array(images_to_predict[str(i)]))
        def pred_gen():
            while True:
                yield images_to_predict

        prediction = model.predict_generator(pred_gen(),steps=1)
        prediction = prediction[0]
        print (prediction)
        for i in range(0, len(prediction)):
            sub_file.write(str(subject) + "_" + "Zone" + str(i+1) + ","+str(prediction[i])+"\n")     
        print(str(prog))
        prog = prog+1


    sub_file.close()

#make_submission() 
