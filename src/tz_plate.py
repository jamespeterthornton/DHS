'''
Brief replication instructions -
1. Extract threat plates and lb threat plates using preprocess plates and preprocess lb plates
2. Train Ensembles using the train plate nets method
3. Extract network train predictions using evaluate plates method
4. Call predict plates, which will fit an ensemble classifier on the train predictions and make test predictions
'''

# import libraries
from __future__ import print_function
from __future__ import division

import numpy as np 
import pandas as pd
import os
import re

import tensorflow as tf

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, Concatenate
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, GlobalAveragePooling2D, Input, Convolution2D
from keras.models import Model
from keras import backend as K
from keras.layers import Activation, LeakyReLU
from keras.layers import BatchNormalization, GlobalMaxPooling2D
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.backend.tensorflow_backend import set_session
from keras.utils.np_utils import to_categorical

from resnet import custom_resnet

import random
from timeit import default_timer as timer

import time
import pickle
from logging_callback import LoggingCallback
from zone_crops import get_zone_fs


import tsahelper.tsahelper as tsa
import scipy
import scipy.stats as stats
import sklearn.metrics
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier, BaggingRegressor, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression

#from imageio
import cv2
from PIL import Image

INPUT_FOLDER =  'stage2_aps'
PREPROCESSED_DATA_FOLDER = 'tsa_datasets/preprocessed/'
STAGE1_LABELS = 'labels/stage1_labels.csv'
STAGE1_SAMPLE_SUB = 'labels/stage1_sample_submission.csv'
STAGE2_SAMPLE_SUB = 'labels/stage2_sample_submission.csv'
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


def get_lb_file_list():
    global LB_SUBJECT_LIST
    df = pd.read_csv(STAGE2_SAMPLE_SUB)
    #df = pd.read_csv(STAGE1_SAMPLE_SUB)
    df['Subject'], df['Zone'] = df['Id'].str.split('_',1).str
    LB_SUBJECT_LIST = df['Subject'].unique()
    print('Sample sub loaded -> {} file(s) used for sub'.format( 
         len(LB_SUBJECT_LIST)))

def get_subject_labels():
    infile = SUBJECT_LABELS_PATH
    return get_subject_labels_for_infile(infile)

def get_subject_labels_for_infile(infile):
    df = pd.read_csv(infile)
    # Separate the zone and subject id into a df
    df['Subject'], df['Zone'] = df['Id'].str.split('_',1).str
    #TODO: convert zone to correct int here
    df = df[['Subject', 'Zone', 'Probability']]
    return df

def get_threat_prob(t_zone, labels, subject):
    threat_list = labels.loc[labels['Subject'] == subject.split(".")[0]]
    threat_iter = threat_list.iterrows()
    threat_prob = 0
    while True:
        threat = next(threat_iter, None)
        if threat is None:
            break
        threat = threat[1]
        if threat['Probability'] is 1:
            zone = threat['Zone']
            zone = int(zone[4:])
            if zone == t_zone:
                threat_prob = 1
    return threat_prob

def make_five_folds(accurate_dirs=True):
    tzs = [{'tz': 13, 'use_fs': True}]
    labels = get_subject_labels()
    #tzs = [{'tz': 9, 'use_fs': True}, {'tz': 15, 'use_fs': True}, {'tz': 3, 'use_fs': True}, {'tz': 11, 'use_fs': True}, {'tz': 6, 'use_fs': True}, {'tz': 5, 'use_fs': True}, {'tz': 4, 'use_fs': True}]
    for tz_dict in tzs:
        tz = tz_dict['tz']
        use_fs = tz_dict['use_fs']
        top_path = 'plates/' + str(tz) + '/'
        if use_fs == True:
            top_path = 'plates/' + str(tz) + '/full_size/'
        files_list = []
        threats_list = []
        non_threats_list = []
        if accurate_dirs:
            for dir_path in ['train/threats/', 'test/threats/']:
                dir_list = os.listdir(top_path + dir_path)
                threats_list = threats_list + dir_list
                for filename in dir_list:
                    files_list.append(top_path+dir_path+filename)
            for dir_path in ['train/non_threats/', 'test/non_threats/']:
                dir_list = os.listdir(top_path + dir_path)
                non_threats_list = non_threats_list + dir_list
                for filename in dir_list:
                    files_list.append(top_path+dir_path+filename)
        else:
            for dir_path in ['train/threats/', 'test/threats/', 'train/non_threats/', 'test/non_threats/']:
                dir_list = os.listdir(top_path + dir_path)
                for filename in dir_list:
                    if filename not in threats_list and filename not in non_threats_list:
                        if "flip" in filename:
                            #TODO - Don't hardcode this
                            tz = 14
                        else:
                            tz = 13
                        threat_prob = get_threat_prob(tz, labels, filename.split("_")[0])
                        if threat_prob > 0.5:
                            threats_list.append(filename)
                        else:
                            non_threats_list.append(filename)
                        files_list.append(top_path+dir_path+filename)
            
        np.random.shuffle(files_list)
        length = len(files_list)
        fold_len = 200 # int(length/5)
        final_fold = []
        for subject_i in range(4*fold_len, 5*fold_len):#len(files_list)):
            final_fold.append(files_list[subject_i])
        for fold_i in range(0, 5):
            fold_path = top_path+"300_fold_" + str(fold_i)
            print("huh.." + fold_path)
            os.mkdir(fold_path)
            files_in_fold = []
            for subject_i in range(fold_i*fold_len, (fold_i+1)*fold_len):
                files_in_fold.append(files_list[subject_i])
            exclusion_list = files_in_fold + final_fold[:100]
            print(str(len(exclusion_list)))
            for dir_path in ['train', 'test']:
                for threat_status in ['threats', 'non_threats']:
                    path = fold_path + "/" + dir_path + "/"
                    try: 
                        os.mkdir(path)
                    except:
                        #Should happen regularly
                        print("Do nothing")
                    path = path + threat_status + "/"
                    os.mkdir(path)
                    relevant_files = []
                    if dir_path == 'test':
                        if files_in_fold == final_fold:
                            relevant_files = files_in_fold
                        else:
                            relevant_files = files_in_fold + final_fold[:100]
                    else: 
                        relevant_files = [x for x in files_list if x not in exclusion_list]
                    relevant_list = non_threats_list 
                    if threat_status == 'threats':
                        relevant_list = threats_list
                    for rel_file in relevant_files:
                        parts = rel_file.split('/')
                        sub_file = parts[len(parts)-1]
                        if sub_file in relevant_list:
                            sub_name = sub_file.split('.')[0]
                            os.link(rel_file, path + sub_name + ".png")

#make_five_folds(accurate_dirs=False)


def make_lb_1_links():
    infile = "labels/stage1_solution.csv"
    top_plates_dir = "s1_solution_plates"
    top_source_dir = "lb_plates"
    labels = get_subject_labels_for_infile(infile)
    df = pd.read_csv(infile)
    df['Subject'], df['Zone'] = df['Id'].str.split('_',1).str
    subjects = df['Subject'].unique()

    for tz_pair in [(8, 10)]:# [(4, 2), (9, None), (15, 16), (5, 17), (3, 1), (11, 12), (6, 7), (8, 10), (13, 14)]:
        tz = tz_pair[0]
        try:
            os.mkdir(top_plates_dir + "/" + str(tz))
            os.mkdir(top_plates_dir + "/" + str(tz) + "/threats")
            os.mkdir(top_plates_dir + "/" + str(tz) + "/non_threats")
        except:
            print("Dirs already made")
        plates_dir = top_plates_dir + "/" + str(tz)
        for tz in tz_pair:
            if tz:
                for subject in subjects:
                    threat_prob = get_threat_prob(tz, labels, subject) 
                    original_path = top_source_dir + "/" + str(tz_pair[0]) + "/full_size/ims/" + subject + "_Zone" + str(tz) + ".jpg"
                    if threat_prob > 0.5:
                        os.link(original_path, top_plates_dir + "/" + str(tz_pair[0]) + "/threats/" + subject + "_Zone" + str(tz) + ".jpg")
                    else:
                        os.link(original_path, top_plates_dir + "/" + str(tz_pair[0]) + "/non_threats/" + subject + "_Zone" + str(tz) + ".jpg")
    
#make_lb_1_links()
 
def switch_chest_back(image):
    temp_im = image
    for i in range(0, 4):
        image[i] = temp_im[i+8]
    for i in range(4, 8):
        image[i+8] = temp_im[i]
    return image

def crop_and_resize_2D(data, x_resize_ratio = 1.0):
    max_y = 0
    min_y = len(data)
    max_x = 0
    min_x = len(data[0])

    for y in range(0, len(data)):
        for x in range(0, len(data[y])):
            #print (data[y][x])
            if data[y][x] > 0.00015:
                #print ("made it in!")
                if x > max_x: max_x = x
                if x < min_x: min_x = x
                if y > max_y: max_y = y
                if y < min_y: min_y = y

    cropped_data = data[min_y:max_y, min_x:max_x]                   

    x_ratio = len(data[0]) / (max_x - min_x) * x_resize_ratio
    y_ratio = len(data) / (max_y - min_y)
    return scipy.ndimage.zoom(cropped_data, (y_ratio, x_ratio))

def flip_and_switch(image):
    a = [None] * 16

    temp_im = image
    a[0] = np.fliplr(temp_im[0])
    a[8] = np.fliplr(temp_im[8])
    for i in range(1, 8):
        temp = np.fliplr(image[i])
        a[i] = np.fliplr(temp_im[16-i])
        a[16-i] = temp
    return a

#Update - TZ (top_dir), whether to switch and flip, get_zone, whether to resize, threat prob / path, dirs, sf, sub list
def preprocess_plates(tz, flip_tz):
    print ("preprocess plates for tz: " + str(tz) + " and: " + str(flip_tz))
    TOP_DIR = "plates/" + str(tz) + "/full_size/"
    TRAIN_DIR_NAME = "train/"
    TEST_DIR_NAME = "test/"
    THREAT_DIR_NAME = "threats/"
    NON_THREAT_DIR_NAME = "non_threats/"
    FLIP_NAME = "flip_"
    DIRS = [TOP_DIR+TRAIN_DIR_NAME+THREAT_DIR_NAME, TOP_DIR+TRAIN_DIR_NAME+NON_THREAT_DIR_NAME, 
        TOP_DIR+TEST_DIR_NAME+THREAT_DIR_NAME, TOP_DIR+TEST_DIR_NAME+NON_THREAT_DIR_NAME]
    get_train_test_file_list()
    labels = get_subject_labels()
    #list existing files so we can skip those
    files_list = []
    for directory in DIRS:
        files_list = files_list + os.listdir(directory)
    already_prepped_subs = []
    for sub_file in files_list:
        already_prepped_subs.append(sub_file.split(".")[0])
    for already_prepped in already_prepped_subs[:5]:
        print ("ap: " + already_prepped)

    for should_flip in [False, True]:
        print("advance sf")
        for sub_list in [TEST_SUBJECT_LIST, TRAIN_SUBJECT_LIST]:
            print("advance sub list")
            folder = "train"
            if sub_list is TEST_SUBJECT_LIST:
                folder = "test"
            for subject in sub_list:
                sub_name = subject
                if should_flip:
                    sub_name = FLIP_NAME + subject
                print ("name: " + str(sub_name))
                if sub_name in already_prepped_subs:
                    print("already has... " + str(sub_name))
                else:
                    #Try - Keep running even if one file is troublesome (corrupted or missing data)
                    try:
                        images = tsa.read_data(INPUT_FOLDER + '/' + subject + '.aps')
                        images = images.transpose()
                        if tz == 5:
                            images_to_use = switch_chest_back(images) if should_flip else images 
                        else:
                            images_to_use = flip_and_switch(images) if should_flip else images
                        cropped_ims = []
                        for i in range(0, len(images_to_use)):
                            if i==3 or i==5 or i==11 or i==13:
                                cropped_ims.append(crop_and_resize_2D(images_to_use[i], x_resize_ratio=0.75))
                            elif i==4 or i==12:
                                cropped_ims.append(crop_and_resize_2D(images_to_use[i], x_resize_ratio=0.5))
                            else:
                                cropped_ims.append(crop_and_resize_2D(images_to_use[i]))
                        pre_stack = get_zone_fs(tz, cropped_ims)
                        stack = np.hstack(pre_stack)
                        path = TOP_DIR + folder + "/"
                        threat_prob_0 =  get_threat_prob(tz, labels, subject)
                        threat_prob_1 =  get_threat_prob(flip_tz, labels, subject)
                        """threat_prob_9 =  get_threat_prob(9, labels, subject)
                        if threat_prob_9 == True:
                            path = path + THREAT_DIR_NAME
                        else:
                            path = path + NON_THREAT_DIR_NAME"""
                        if threat_prob_0 == 1 and threat_prob_1 == 1:
                            path = path + THREAT_DIR_NAME 
                        elif should_flip and threat_prob_1 == 1:
                            path = path + THREAT_DIR_NAME
                        elif not should_flip and threat_prob_0 == 1:
                            path = path + THREAT_DIR_NAME
                        else:
                            path = path + NON_THREAT_DIR_NAME
                        scipy.misc.imsave(path + sub_name + ".jpg", stack)   
                    except:
                        #time.sleep(5)
                        print ("Failed!!!")    
 
#preprocess_plates(13, 14)
#preprocess_plates(8, 10)
#preprocess_plates(9, None)

#preprocess_plates(5, 17)
#preprocess_plates(6, 7)
#preprocess_plates(15, 16)

#preprocess_plates(11, 12)
#preprocess_plates(3, 1)
#preprocess_plates(4, 2)

def preprocess_lb_plates_for_tz(tz, flip_tz):
    print("prep lb for zone - " + str(tz))
    TOP_DIR = "stage_2_plates/" + str(tz) + "/full_size/ims/"
    #TOP_DIR = "lb_plates/" + str(tz) + "/full_size/ims/"
    get_lb_file_list()
    #list existing files so we can skip those
    files_list = os.listdir(TOP_DIR)
    already_prepped_subs = []
    for sub_file in files_list:
        already_prepped_subs.append(sub_file.split(".")[0])
    if flip_tz:
        flip_list = [False, True]
    else:
        flip_list = [False]

    for should_flip in flip_list:

        print("advance sf")
        for subject in LB_SUBJECT_LIST:
            sub_name = subject
            if should_flip:
                sub_name = sub_name + "_Zone" + str(flip_tz)
            else:
                sub_name = sub_name + "_Zone" + str(tz)

            print ("name: " + str(sub_name))
            if sub_name not in already_prepped_subs:
                #Try - Keep running even if one file is troublesome (corrupted or missing data)
                try:
                    images = tsa.read_data(INPUT_FOLDER + '/' + subject + '.aps')
                    images = images.transpose()
                    if tz == 9:
                        images_to_use = images
                    elif tz == 5:
                        images_to_use = switch_chest_back(images) if should_flip else images  
                    else:
                        images_to_use = flip_and_switch(images) if should_flip else images
                    cropped_ims = []
                    for i in range(0, len(images_to_use)):
                        if tz in []:
                        #if tz in [9, 15, 3, 13]:
                            cropped_ims.append(crop_and_resize_2D(images_to_use[i]))
                        else:
                            if i==3 or i==5 or i==11 or i==13:
                                cropped_ims.append(crop_and_resize_2D(images_to_use[i], x_resize_ratio=0.75))
                            elif i==4 or i==12:
                                cropped_ims.append(crop_and_resize_2D(images_to_use[i], x_resize_ratio=0.5))
                            else:
                                cropped_ims.append(crop_and_resize_2D(images_to_use[i]))
                    pre_stack = get_zone_fs(tz, cropped_ims)
                    stack = np.hstack(pre_stack)
                    path = TOP_DIR + sub_name + ".jpg"
                    scipy.misc.imsave(path, stack)   
                except:
                    print ("Failed!!!")    
    print ("next!")

#preprocess_lb_plates_for_tz(4, 2)
"""preprocess_lb_plates_for_tz(3, 1)
preprocess_lb_plates_for_tz(11, 12)
preprocess_lb_plates_for_tz(6, 7)
preprocess_lb_plates_for_tz(5, 17)
preprocess_lb_plates_for_tz(9, None)
preprocess_lb_plates_for_tz(15, 16)"""
#preprocess_lb_plates_for_tz(13, 14)
#preprocess_lb_plates_for_tz(8, 10)

def convert_to_grayscale(img):
    # scale pixel values to grayscale
    base_range = np.amax(img) - np.amin(img)
    rescaled_range = 255 - 0
    img_rescaled = (((img - np.amin(img)) * rescaled_range) / base_range)

    return np.uint8(img_rescaled)

#Preprocessing option I tested but found unhelpful
def spread_spectrum(img):
    #img = stats.threshold(img, threshmin=12, newval=0)
    img = np.clip(img, 12, None)

    # see http://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img= clahe.apply(img)
    
    return img

def resnet50(input_size):
    resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(input_size[0], input_size[1], 3))
    x = resnet_model.output
    x = Flatten()(x)
    x = Dense(1024, activation='linear', name='fc1')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.1)(x)
    x = Dense(1, activation='sigmoid', name='prediction')(x)
    model = Model(inputs=resnet_model.inputs, outputs=x)
    sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.7, nesterov=True)
    model.compile(loss='binary_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    return model

def train_plate_net(tz, fold):
    print('train plate net for tz: ' + str(tz))
    get_train_test_file_list()
    batch_size = 4

    INPUT_SHAPES = {15: (197, 2115), 3: (220, 2200), 9: (220, 2000), 13: (197, 2225), 11: (197, 2009), 
        8: (197, 1651), 6: (197, 1818), 4: (197, 2000), 5: (197, 1337)}

    INPUT_SIZE = INPUT_SHAPES[tz]

    train_gen = ImageDataGenerator(width_shift_range=0.0156, height_shift_range=0.2, zoom_range=0.05)
    test_gen = ImageDataGenerator()

    model_name = "resnet50_1024_fs_w_d_" + fold + ".h5"

    checkpoint1_path = "weights/plate_models/" + str(tz) +"/" + model_name
    momentum_path = checkpoint1_path + "_momentum.pickle"
    model = resnet50(INPUT_SIZE)

    cnn_checkpoint = ModelCheckpoint(checkpoint1_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    es = EarlyStopping('val_loss', patience=12, mode="min")
    lc = LoggingCallback('resnet_1_tz_' + str(tz), momentum_path, net_name=str(tz)+model_name)

    sgd = keras.optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.7, nesterov=True)
    model.compile(loss='binary_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    model.fit_generator(generator=train_gen.flow_from_directory("plates/" + str(tz) + "/full_size/" + fold + "/train", class_mode="binary", batch_size=batch_size, target_size=INPUT_SIZE), validation_data=test_gen.flow_from_directory("plates/" + str(tz) + "/full_size/" + fold + "/test", class_mode="binary", batch_size=batch_size, target_size=INPUT_SIZE), steps_per_epoch = 1500, validation_steps = 458, epochs = 1, verbose=2, callbacks=[cnn_checkpoint, lc])
    
    print ("increasing LR")
    sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.7, nesterov=True)
    model.compile(loss='binary_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    try:
        model.fit_generator(generator=train_gen.flow_from_directory("plates/" + str(tz) + "/full_size/" + fold + "/train", class_mode="binary", batch_size=batch_size, target_size=INPUT_SIZE, follow_links=True), validation_data=test_gen.flow_from_directory("plates/" + str(tz) + "/full_size/" + fold + "/test", class_mode="binary", batch_size=batch_size, target_size=INPUT_SIZE, follow_links=True), steps_per_epoch = 1500, validation_steps = 458, epochs = 40, verbose=2, callbacks=[cnn_checkpoint, es, lc])
    except:
        print("onwards!") 
    model.load_weights(checkpoint1_path)

    #One preliminary epoch at a really low LR
    print ("low low LR")
    es2 = EarlyStopping('val_loss', patience=6, mode="min")
    sgd = keras.optimizers.SGD(lr=0.00001, decay=1e-6, momentum=0.7, nesterov=True)
    model.compile(loss='binary_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    checkpoint2 = ModelCheckpoint("weights/plate_models/" + str(tz) +"/resnet_finetune_" + fold + ".h5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    
    model.fit_generator(generator=train_gen.flow_from_directory("plates/" + str(tz) + "/full_size/" + fold + "/train", class_mode="binary", batch_size=batch_size, target_size=INPUT_SIZE, follow_links=True), validation_data=test_gen.flow_from_directory("plates/" + str(tz) + "/full_size/" + fold + "/test", class_mode="binary", batch_size=batch_size, target_size=INPUT_SIZE, follow_links=True), steps_per_epoch = 1500, validation_steps = 458, epochs = 1, verbose=2, callbacks=[checkpoint2, es2, lc])
    print ("back to low LR")
    es2 = EarlyStopping('val_loss', patience=6, mode="min")
    sgd = keras.optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.7, nesterov=True)
    model.compile(loss='binary_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    checkpoint2 = ModelCheckpoint("weights/plate_models/" + str(tz) +"/resnet_finetune_" + fold + ".h5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    
    model.fit_generator(generator=train_gen.flow_from_directory("plates/" + str(tz) + "/full_size/" + fold + "/train", class_mode="binary", batch_size=batch_size, target_size=INPUT_SIZE, follow_links=True), validation_data=test_gen.flow_from_directory("plates/" + str(tz) + "/full_size/" + fold + "/test", class_mode="binary", batch_size=batch_size, target_size=INPUT_SIZE, follow_links=True), steps_per_epoch = 1500, validation_steps = 458, epochs = 40, verbose=2, callbacks=[checkpoint2, es2, lc])

def train_plate_nets():
    print("train plate nets")
    for tz in [9, 15, 5, 3, 11, 6, 4, 8, 13]:
        for fold in ['300_fold_0', '300_fold_1', '300_fold_2', '300_fold_3']: 
            try: 
                train_plate_net(tz, fold)
            except:
                print("Failed!")
                time.sleep(15)
        K.clear_session()
        time.sleep(30)

#train_plate_nets()


INPUT_SHAPES = {15: (139, 2115), 3: (139, 1707), 9: (139, 1502), 13: (139, 2225), 11: (139, 2009), 8: (139, 1651), 6: (139, 1818),
        4: (139, 1524), 5: (139, 1337)}

def get_fold_models_dict():
    tz_models_dict = {
        15: [],
        3: [],
        9: [],
        13: [],
        11: [],
        8: [],
        6: [],
        4: [],
        5: []
    }
    
    SHAPE_13_1 = (197, 2225)
    tz_13_model_1 = resnet50(SHAPE_13_1)
    tz_13_model_1.load_weights("weights/plate_models/13/best_resnet50_dropout_2.h5")
    tz_13_dict_1 = {"model": tz_13_model_1, "shape": SHAPE_13_1, "use_fs" : True}
    tz_models_dict[13].append(tz_13_dict_1)

    #Second net from trained ensemble
    SHAPE_6_1 = (197, 1818)
    tz_6_model_1 = resnet50(SHAPE_6_1)
    tz_6_model_1.load_weights("weights/plate_models/6/resnet_finetune_300_fold_1.h5")
    tz_6_dict_1 = {"model": tz_6_model_1, "shape": SHAPE_6_1, "use_fs" : True}
    tz_models_dict[6].append(tz_6_dict_1)   
    
    SHAPE_11_1 = (197, 2009)
    tz_11_model_1 = resnet50(SHAPE_11_1)
    tz_11_model_1.load_weights("weights/plate_models/11/resnet_finetune.h5")
    tz_11_dict_1 = {"model": tz_11_model_1, "shape": SHAPE_11_1, "use_fs" : True}
    tz_models_dict[11].append(tz_11_dict_1)   
    
    INPUT_SHAPES = {15: (197, 2115), 3: (220, 2200), 9: (220, 2000), 13: (197, 2225), 11: (197, 2009), 
        8: (197, 1651), 6: (197, 1818), 4: (197, 2000), 5: (197, 1337)}

    #N.B. These dictionaries should be updated with the best models to come out of training
    first_net_folds = {13:[], 5:[],6:[], 8:['300_fold_1'],9:['300_fold_0', '300_fold_1', '300_fold_3'], 4: ['300_fold_0', '300_fold_1', '300_fold_2', '300_fold_3'], 15: ['300_fold_0', '300_fold_1', '300_fold_3'], 3: ['300_fold_0', '300_fold_1', '300_fold_2'], 11: ['300_fold_0', '300_fold_3']}

    #Next line also specifies paths - which networks to load for the ensemble
    for tz in []:# [13, 15, 9, 4, 3, 8, 5]:        #5, 15]:#15, 5, 9, 3, 11, 6, 4]: #[15, 5]:#, 9]: # [15, 3, 9, 11, 6, 5]:
        tz_fold_list = ['300_fold_0', '300_fold_1', '300_fold_2', '300_fold_3']
        for fold in tz_fold_list:
            model_name = "weights/plate_models/" + str(tz) + "/resnet_finetune_" + fold + ".h5"
            if fold in first_net_folds[tz]:
                model_name = "weights/plate_models/" + str(tz) + "/resnet50_1024_fs_w_d_" + fold + ".h5"
            model_shape = INPUT_SHAPES[tz]
            model = resnet50(model_shape)
            model.load_weights(model_name)
            model_dict = {"model":model, "shape":model_shape, "use_fs":True}
            tz_models_dict[tz].append(model_dict)
    return tz_models_dict

def predict_plates():
    print ("predicting...")

    #logistic_reg = train_ensemble(5)
    ensemble_list = [3, 8, 5, 15]
    average_list = [9, 4]

    models_dict = get_fold_models_dict()
    sub_file = open("labels/s2_submission.csv", "w")
    sub_file.write("Id,Probability\n")
    pred_gen = ImageDataGenerator()

    for tz in [13, 15, 9, 11, 6, 3, 8, 4, 5]:
        ensemble_tuple = train_ensemble(tz) if tz in ensemble_list else None
        if ensemble_tuple:
            regressor = ensemble_tuple[0]
            can_predict_proba = ensemble_tuple[1]
        print("Tz: " + str(tz) + " with " + str(len(models_dict[tz])) + " models")
        predictions = []
        input_shape = models_dict[tz][0]['shape']
        use_fs = models_dict[tz][0]['use_fs']
        #dir_path = "lb_plates/" + str(tz) + "/ims"
        dir_path = "stage_2_plates/" + str(tz) + "/ims"
        if use_fs == True:
            #dir_path = "lb_plates/" + str(tz) + "/full_size/ims"
            dir_path = "stage_2_plates/" + str(tz) + "/full_size/ims"
        plate_files = os.listdir(dir_path)
        j = 0
        for plate_file in plate_files:
            if j  % 20 == 0:
                print (str(j))
            plate = np.asarray(Image.open(dir_path + "/" + plate_file).convert("RGB"))
            plate = scipy.ndimage.zoom(plate, (input_shape[0]/plate.shape[0], input_shape[1]/plate.shape[1], 1))
            plate_name = plate_file.split(".")[0]
            plate_array = []
            plate_array.append(plate)
            x = np.array(plate_array)
            all_predictions = []
            for i in range(0, len(models_dict[tz])):
                model_dict = models_dict[tz][i]
                model = model_dict['model']
                predictions = model.predict(x)
                all_predictions.append(predictions[0][0])
            if tz in ensemble_list:
                if can_predict_proba:
                    average_prediction = regressor.predict_proba([all_predictions])[0][1]
                else:
                    average_prediction = regressor.predict([all_predictions])[0]
            elif tz in average_list:
                average_prediction = np.mean([all_predictions])
            else:
                average_prediction = np.mean(predictions) 
            average_prediction = np.clip(average_prediction, 0.015, 0.985)
            sub_file.write(plate_name + ", " + str(average_prediction) + "\n")
            j = j  + 1

def evaluate_plates():
    VOTING_MODE = "voting"
    AVERAGE_MODE = "average"
    mode = AVERAGE_MODE

    train_or_test = 'lb'
    save_predictions = True

    print("evaluate")
    pred_gen = ImageDataGenerator()
    models_dict = get_fold_models_dict()
    print('got model')
    #Path - next line specifies which tz to extract for
    for tz in [13, 5, 15, 9, 3, 11, 6, 4, 8]:
        print("Tz: " + str(tz) + " with " + str(len(models_dict[tz])) + " models")
        y_true = []
        y_pred = []
        model_predictions = []
        input_shape = models_dict[tz][0]['shape']
        fold_4_dir = "plates/" + str(tz) + "/full_size/300_fold_4/"
        plates_dir = "plates/" + str(tz) + "/full_size/300_fold_4/" + train_or_test
        if train_or_test == 'lb' or train_or_test == 'lb_orig':
            plates_dir = "s1_solution_plates/" + str(tz)
        for threat_status in ['threats', 'non_threats']:
            full_plates_path = plates_dir + "/" + threat_status
            plate_files = os.listdir(full_plates_path)
            j = 0
            for plate_file in plate_files:
                if j % 20 == 0:
                    print (str(j))
                plate = np.asarray(Image.open(full_plates_path + "/" + plate_file).convert("RGB"))
                plate = scipy.ndimage.zoom(plate, (input_shape[0]/plate.shape[0], input_shape[1]/plate.shape[1], 1))
                plate_name = plate_file.split(".")[0]
                plate_array = []
                plate_array.append(plate)
                x = np.array(plate_array)
                all_predictions = []
                for i in range(0, len(models_dict[tz])):
                    model_dict = models_dict[tz][i]
                    predictions = model_dict['model'].predict(x)
                    all_predictions.append(predictions[0][0])
                model_predictions.append(all_predictions)
                if mode == VOTING_MODE:
                    vote_count = 0
                    for prediction in all_predictions:
                        if prediction > 0.5:
                            vote_count = vote_count + 1
                    voting_prediction = 0.05 
                    if vote_count == 2:
                        voting_prediction = 0.4
                    elif vote_count > 2:
                        voting_prediction = 0.95
                    y_pred.append(voting_prediction)
                elif mode == AVERAGE_MODE:
                    average_prediction = np.mean(all_predictions)
                    average_prediction = np.clip(average_prediction, 0.000001, 0.999999)
                    y_pred.append(average_prediction)
                if threat_status == 'threats':
                    y_true.append(1)
                else:
                    y_true.append(0)
                j = j + 1
        if save_predictions:
            print("saving predictions...")
            with open(fold_4_dir + "y_pred_" + train_or_test + ".pickle", "wb") as handle:
                pickle.dump(model_predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(fold_4_dir + "y_true_" + train_or_test + ".pickle", "wb") as handle:
                pickle.dump(y_true, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("log loss: " + str(sklearn.metrics.log_loss(y_true, y_pred))) 

evaluate_plates()

def train_ensemble(tz):
    fold = '300_fold_4/'
    fold_4_dir = "plates/" + str(tz) + "/full_size/" + fold
    with open(fold_4_dir + "y_pred_test.pickle", "rb") as handle:
        y_pred_test = pickle.load(handle)
    with open(fold_4_dir + "y_true_test.pickle", "rb") as handle:
        y_true_test = pickle.load(handle)
    with open(fold_4_dir + "y_pred_train.pickle", "rb") as handle:
        y_pred_train = pickle.load(handle)
    with open(fold_4_dir + "y_true_train.pickle", "rb") as handle:
        y_true_train = pickle.load(handle)

    '''
    Code for using the predictions of only some of the networks supplied for a given tz
 
    if tz == 15:
        new_y_pred_test = []
        for i in range(0, len(y_pred_test)):
            new_y_pred_test.append([y_pred_test[i][1], y_pred_test[i][2], y_pred_test[i][3], y_pred_test[i][4]])
        y_pred_test = new_y_pred_test
        new_y_pred_train = []
        for i in range(0, len(y_pred_train)):
            new_y_pred_train.append([y_pred_train[i][1], y_pred_train[i][2], y_pred_train[i][3], y_pred_train[i][4]])
        y_pred_train = new_y_pred_train
    '''
 
    average_list = []
    for i in range(0, len(y_pred_test)):
        #print(np.mean(y_pred_test[i]))
        average_list.append(np.clip(np.mean(y_pred_test[i]), 0.00001, 0.99999)) 
    net_lists = []
    for x in range(0, len(y_pred_test[0])):
        net_lists.append([])

    threshold_list = []
    for i in range(0, len(y_pred_test)):
        for j in range(0, len(y_pred_test[i])):
            net_lists[j].append(np.clip(y_pred_test[i][j], 0.00001, 0.99999))
        #print(np.mean(y_pred_test[i]))
        if any(j>=0.75 for j in y_pred_test[i]):
            threshold_list.append(1)
        else:
            threshold_list.append(0)

    # The following section is a bit repretitive, but the purpose of this function was to rapidly test out a 
    # variety of ensemble techniques. Clean up would be more about removing the failed techniques, but I'm leaving
    # the code to show what I tried.
    
    #average_list = np.clip(average_list, 0.05, 0.95)
    bag = BaggingRegressor()
    bag.fit(y_pred_train, y_true_train)
    bag_pred = bag.predict(y_pred_test)
    best_bag = sklearn.metrics.log_loss(y_true_test, bag_pred)
    for i in range(0, 20): 
        new_bag = BaggingRegressor()
        new_bag.fit(y_pred_train, y_true_train)
        new_bag_pred = new_bag.predict(y_pred_test)
        new_bag_score = sklearn.metrics.log_loss(y_true_test, new_bag_pred)
        if new_bag_score < best_bag:
            best_bag = new_bag_score
            bag = new_bag
            bag_pred = new_bag_pred
    
    ada = AdaBoostClassifier()
    ada.fit(y_pred_train, y_true_train)
    ada_pred = ada.predict_proba(y_pred_test)
    best_ada = sklearn.metrics.log_loss(y_true_test, ada_pred)
    for i in range(0, 20): 
        new_ada = AdaBoostClassifier()
        new_ada.fit(y_pred_train, y_true_train)
        new_ada_pred = ada.predict_proba(y_pred_test)
        new_ada_score = sklearn.metrics.log_loss(y_true_test, new_ada_pred)
        if new_ada_score < best_ada:
            best_ada = new_ada_score
            ada = new_ada
            ada_pred = new_ada_pred

    logistic = LogisticRegression()
    logistic.fit(y_pred_train, y_true_train)
    logistic_pred = logistic.predict_proba(y_pred_test)

    clf = RandomForestClassifier(criterion='entropy', max_depth=2, random_state=1)
    clf.fit(y_pred_train, y_true_train)
    clf_pred = clf.predict_proba(y_pred_test)
    best_clf = sklearn.metrics.log_loss(y_true_test, clf_pred)
    for i in range(0, 20): 
        new_clf = RandomForestClassifier(criterion='entropy', max_depth=2, random_state=1)
        new_clf.fit(y_pred_train, y_true_train)
        new_clf_pred = clf.predict_proba(y_pred_test)
        new_clf_score = sklearn.metrics.log_loss(y_true_test, clf_pred)
        if new_clf_score < best_clf:
            best_clf = new_clf_score
            clf = new_clf
            clf_pred = new_clf_pred
  
    clf2 = RandomForestRegressor(random_state=1)
    clf2.fit(y_pred_train, y_true_train)
    clf2_pred = clf.predict(y_pred_test)

    print("tz: " + str(tz))  
    print(logistic.score(y_pred_test, y_true_test)) 
    print("logistic log loss: " + str(sklearn.metrics.log_loss(y_true_test, logistic_pred))) 
    #print("voting log loss: " + str(sklearn.metrics.log_loss(y_true_test, vote_pred))) 
    print("forest log loss: " + str(sklearn.metrics.log_loss(y_true_test, clf_pred))) 
    print("forest reg log loss: " + str(sklearn.metrics.log_loss(y_true_test, clf2_pred))) 
    print("average log loss: " + str(sklearn.metrics.log_loss(y_true_test, average_list))) 
    print("threshold log loss: " + str(sklearn.metrics.log_loss(y_true_test, threshold_list))) 
    print("bag log loss: " + str(sklearn.metrics.log_loss(y_true_test, bag_pred))) 
    print("ada log loss: " + str(sklearn.metrics.log_loss(y_true_test, ada_pred))) 

    for net_list in net_lists:
        print("net list loss: " + str(sklearn.metrics.log_loss(y_true_test, net_list)))

    #Second element specifies whether predict_proba is an available method
    if tz == 5:
        return (bag, False)
    elif tz == 11:
        return (clf, True)
    elif tz == 15:
        return (clf, True)
    elif tz == 3:
        return (ada, True)
    elif tz == 8:
        return (ada, True)
    else:
        return (clf, True)

#for tz in [13]:# [3, 8, 5]:#, 8]: #5, #13
#    train_ensemble(tz)

#predict_plates()

