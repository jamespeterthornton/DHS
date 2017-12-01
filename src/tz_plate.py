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

import random
from timeit import default_timer as timer

import time

from logging_callback import LoggingCallback

import tsahelper.tsahelper as tsa
import scipy
import scipy.stats as stats
#from imageio
import cv2
from PIL import Image

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
STAGE1_SAMPLE_SUB = 'labels/stage1_sample_submission.csv'
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
    df = pd.read_csv(STAGE1_SAMPLE_SUB)
    df['Subject'], df['Zone'] = df['Id'].str.split('_',1).str
    LB_SUBJECT_LIST = df['Subject'].unique()
    print('Sample sub loaded -> {} file(s) used for sub'.format( 
         len(LB_SUBJECT_LIST)))

def get_subject_labels():
    infile = SUBJECT_LABELS_PATH
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

def get_zone15(img):
    crops = []
    for slice_num in range(0, len(img)):
        if slice_num == 0:
            crops.append(np.array(img[slice_num])[ 0:60, 0:128])
        elif slice_num == 1:
            crops.append(np.array(img[slice_num])[0:60, 0:128])
        elif slice_num == 2:
            crops.append(np.array(img[slice_num])[0:60, 0:128])
        elif slice_num == 3:
            crops.append(np.array(img[slice_num])[0:60, 0:160])
        elif slice_num == 5:
            crops.append(np.array(img[slice_num])[ 0:60,160:256])
        elif slice_num == 6:
            crops.append(np.array(img[slice_num])[ 0:60, 140:256])
        elif slice_num == 7:
            crops.append(np.array(img[slice_num])[ 0:60, 128:256])
        elif slice_num == 8:
            crops.append(np.array(img[slice_num])[ 0:60, 130:256])
        elif slice_num == 9:
            crops.append(np.array(img[slice_num])[ 0:60, 120:256])
        elif slice_num == 10:
            crops.append(np.array(img[slice_num])[ 0:60, 100:256])
        elif slice_num == 11:
            crops.append(np.array(img[slice_num])[ 0:60, 75:256])
        elif slice_num == 12:
            crops.append(np.array(img[slice_num])[ 0:60, 0:256])
        elif slice_num == 13:
            crops.append(np.array(img[slice_num])[ 0:60, 0:120])
        elif slice_num == 14:
            crops.append(np.array(img[slice_num])[ 0:60, 0:128])
        elif slice_num == 15:
            crops.append(np.array(img[slice_num])[ 0:60, 0:128])
    return crops   

def get_zone3(img):
    crops = []
    for slice_num in range(0, len(img)):
        if slice_num == 0:
            crops.append(np.array(img[slice_num])[200:315,150:256])
        elif slice_num == 1:
            crops.append(np.array(img[slice_num])[200:315,150:256])
        elif slice_num == 2:
            crops.append(np.array(img[slice_num])[200:315,150:256])
        elif slice_num == 3:
            crops.append(np.array(img[slice_num])[200:315,150:256])
        elif slice_num == 4:
            crops.append(np.array(img[slice_num])[200:315,55:205])
        elif slice_num == 5:
            crops.append(np.array(img[slice_num])[200:315,0:200])
        elif slice_num == 6:
            crops.append(np.array(img[slice_num])[200:315,0:155])
        elif slice_num == 7:
            crops.append(np.array(img[slice_num])[200:315,0:130])
        elif slice_num == 8:
            crops.append(np.array(img[slice_num])[200:315,0:110])
        elif slice_num == 9:
            crops.append(np.array(img[slice_num])[200:315,0:110])
        elif slice_num == 10:
            crops.append(np.array(img[slice_num])[200:315,0:110])
        elif slice_num == 13:
            crops.append(np.array(img[slice_num])[200:315,150:256])
        elif slice_num == 14:
            crops.append(np.array(img[slice_num])[200:315,150:256])
        elif slice_num == 15:
            crops.append(np.array(img[slice_num])[200:315,150:256])
    return crops

def get_zone9(img):
    crops = []
    for slice_num in range(0, len(img)):
        if slice_num == 0:
            crops.append(np.array(img[slice_num])[80:190,80:176])
        elif slice_num == 1:
            crops.append(np.array(img[slice_num])[80:190,70:176])
        elif slice_num == 2:
            crops.append(np.array(img[slice_num])[80:190,45:176])
        elif slice_num == 3:
            crops.append(np.array(img[slice_num])[80:190,15:176])
        elif slice_num == 6:
            crops.append(np.array(img[slice_num])[80:190,80:256])
        elif slice_num == 7:
            crops.append(np.array(img[slice_num])[80:190,80:186])
        elif slice_num == 8:
            crops.append(np.array(img[slice_num])[80:190,80:176])
        elif slice_num == 9:
            crops.append(np.array(img[slice_num])[80:190,40:176])
        elif slice_num == 10:
            crops.append(np.array(img[slice_num])[80:190,0:156])
        elif slice_num == 13:
            crops.append(np.array(img[slice_num])[80:190,80:216])
        elif slice_num == 14:
            crops.append(np.array(img[slice_num])[80:190,80:186])
        elif slice_num == 15:
            crops.append(np.array(img[slice_num])[80:190,80:176])
    return crops

def get_zone13(img):
    crops = []
    for slice_num in range(0, len(img)):
        if slice_num == 0:
            crops.append(np.array(img[slice_num])[15:100,0:128])
        elif slice_num == 1:
            crops.append(np.array(img[slice_num])[15:100,0:128])
        elif slice_num == 2:
            crops.append(np.array(img[slice_num])[15:100,0:148])
        elif slice_num == 3:
            crops.append(np.array(img[slice_num])[15:100,0:190])
        elif slice_num == 5:
            crops.append(np.array(img[slice_num])[15:100,160:256])
        elif slice_num == 6:
            crops.append(np.array(img[slice_num])[15:100,140:256])
        elif slice_num == 7:
            crops.append(np.array(img[slice_num])[15:100,128:256])
        elif slice_num == 8:
            crops.append(np.array(img[slice_num])[15:100,130:256])
        elif slice_num == 9:
            crops.append(np.array(img[slice_num])[15:100,120:256])
        elif slice_num == 10:
            crops.append(np.array(img[slice_num])[15:100,80:256])
        elif slice_num == 11:
            crops.append(np.array(img[slice_num])[15:100,35:256])
        elif slice_num == 12:
            crops.append(np.array(img[slice_num])[15:100,0:256])
        elif slice_num == 13:
            crops.append(np.array(img[slice_num])[15:100,0:120])
        elif slice_num == 14:
            crops.append(np.array(img[slice_num])[15:100,0:128])
        elif slice_num == 15:
            crops.append(np.array(img[slice_num])[15:100,0:128])
    return crops

def get_zone11(img):
    crops = []
    for slice_num in range(0, len(img)):
        if slice_num == 0:
            crops.append(np.array(img[slice_num])[40:150,0:128])
        elif slice_num == 1:
            crops.append(np.array(img[slice_num])[40:150,0:128])
        elif slice_num == 2:
            crops.append(np.array(img[slice_num])[40:150,0:148])
        elif slice_num == 3:
            crops.append(np.array(img[slice_num])[40:150,0:150])
        elif slice_num == 5:
            crops.append(np.array(img[slice_num])[40:150,100:192])
        elif slice_num == 6:
            crops.append(np.array(img[slice_num])[40:150,140:256])
        elif slice_num == 7:
            crops.append(np.array(img[slice_num])[40:150,128:256])
        elif slice_num == 8:
            crops.append(np.array(img[slice_num])[40:150,130:256])
        elif slice_num == 9:
            crops.append(np.array(img[slice_num])[40:150,120:256])
        elif slice_num == 10:
            crops.append(np.array(img[slice_num])[40:150,80:256])
        elif slice_num == 11:
            crops.append(np.array(img[slice_num])[40:150,35:192])
        elif slice_num == 12:
            crops.append(np.array(img[slice_num])[40:150,0:128])
        elif slice_num == 13:
            crops.append(np.array(img[slice_num])[40:150,0:120])
        elif slice_num == 14:
            crops.append(np.array(img[slice_num])[40:150,0:138])
        elif slice_num == 15:
            crops.append(np.array(img[slice_num])[40:150,0:138])
    return crops

def get_zone8(img):
    crops = []
    for slice_num in range(0, len(img)):
        if slice_num == 0:
            crops.append(np.array(img[slice_num])[85:180,0:128])
        elif slice_num == 1:
            crops.append(np.array(img[slice_num])[85:180,0:128])
        elif slice_num == 2:
            crops.append(np.array(img[slice_num])[85:180,0:148])
        elif slice_num == 7:
            crops.append(np.array(img[slice_num])[85:180,128:256])
        elif slice_num == 8:
            crops.append(np.array(img[slice_num])[85:180,130:256])
        elif slice_num == 9:
            crops.append(np.array(img[slice_num])[85:180,120:256])
        elif slice_num == 10:
            crops.append(np.array(img[slice_num])[85:180,80:256])
        elif slice_num == 11:
            crops.append(np.array(img[slice_num])[85:180,35:192])
        elif slice_num == 12:
            crops.append(np.array(img[slice_num])[85:180,0:128])
        elif slice_num == 13:
            crops.append(np.array(img[slice_num])[85:180,0:120])
        elif slice_num == 14:
            crops.append(np.array(img[slice_num])[85:180,0:138])
        elif slice_num == 15:
            crops.append(np.array(img[slice_num])[85:180,0:138])
    return crops

def get_zone6(img):
    crops = []
    for slice_num in range(0, len(img)):
        if slice_num == 0:
            crops.append(np.array(img[slice_num])[130:230,0:128])
        elif slice_num == 1:
            crops.append(np.array(img[slice_num])[130:230,0:128])
        elif slice_num == 5:
            crops.append(np.array(img[slice_num])[130:230,100:192])
        elif slice_num == 6:
            crops.append(np.array(img[slice_num])[130:230,140:256])
        elif slice_num == 7:
            crops.append(np.array(img[slice_num])[130:230,128:256])
        elif slice_num == 8:
            crops.append(np.array(img[slice_num])[130:230,130:256])
        elif slice_num == 9:
            crops.append(np.array(img[slice_num])[130:230,105:256])
        elif slice_num == 10:
            crops.append(np.array(img[slice_num])[130:230,65:256])
        elif slice_num == 11:
            crops.append(np.array(img[slice_num])[130:230,0:192])
        elif slice_num == 12:
            crops.append(np.array(img[slice_num])[130:230,0:128])
        elif slice_num == 13:
            crops.append(np.array(img[slice_num])[130:230,0:150])
        elif slice_num == 14:
            crops.append(np.array(img[slice_num])[130:230,0:150])
        elif slice_num == 15:
            crops.append(np.array(img[slice_num])[130:230,0:138])
    return crops

def get_zone4(img):
    crops = []
    for slice_num in range(0, len(img)):
        if slice_num == 0:
            crops.append(np.array(img[slice_num])[245:330,150:256])
        elif slice_num == 1:
            crops.append(np.array(img[slice_num])[245:330,150:256])
        elif slice_num == 2:
            crops.append(np.array(img[slice_num])[245:330,135:256])
        elif slice_num == 3:
            crops.append(np.array(img[slice_num])[245:330,20:192])
        elif slice_num == 4:
            crops.append(np.array(img[slice_num])[245:330,0:64])
        elif slice_num == 5:
            crops.append(np.array(img[slice_num])[245:330,0:64])
        elif slice_num == 6:
            crops.append(np.array(img[slice_num])[245:330,0:155])
        elif slice_num == 7:
            crops.append(np.array(img[slice_num])[245:330,0:130])
        elif slice_num == 8:
            crops.append(np.array(img[slice_num])[245:330,0:110])
        elif slice_num == 9:
            crops.append(np.array(img[slice_num])[245:330,0:110])
        elif slice_num == 10:
            crops.append(np.array(img[slice_num])[245:330,0:110])
        elif slice_num == 13:
            crops.append(np.array(img[slice_num])[245:330,128:256])
        elif slice_num == 14:
            crops.append(np.array(img[slice_num])[245:330,150:256])
        elif slice_num == 15:
            crops.append(np.array(img[slice_num])[245:330,150:256])
    return crops

def get_zone5(img):
    crops = []
    for slice_num in range(0, len(img)):
        if slice_num == 0:
            crops.append(np.array(img[slice_num])[185:260,46:210])
        elif slice_num == 1:
            crops.append(np.array(img[slice_num])[185:260,36:220])
        elif slice_num == 2:
            crops.append(np.array(img[slice_num])[185:260,0:210])
        elif slice_num == 3:
            crops.append(np.array(img[slice_num])[185:260,0:192])
        elif slice_num == 13:
            crops.append(np.array(img[slice_num])[185:260,25:192])
        elif slice_num == 14:
            crops.append(np.array(img[slice_num])[185:260,46:256])
        elif slice_num == 15:
            crops.append(np.array(img[slice_num])[185:260,46:256])
    return crops   

def get_zone(tz, cropped_ims):
    if tz == 15:
        return get_zone15(cropped_ims)
    elif tz == 3:
        return get_zone3(cropped_ims)
    elif tz == 9:
        return get_zone9(cropped_ims)
    elif tz == 13:
        return get_zone13(cropped_ims)
    elif tz == 11:
        return get_zone11(cropped_ims)
    elif tz == 8:
        return get_zone8(cropped_ims)
    elif tz == 6:
        return get_zone6(cropped_ims)
    elif tz == 4:
        return get_zone4(cropped_ims)
    elif tz == 5:
        return get_zone5(cropped_ims)
    else:
        print ("Get zone FAILURE!")
        return None
 
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
 
def preprocess_plates():
    TOP_DIR = "plates/9/full_size/"
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
                if sub_name not in already_prepped_subs:
                    #Try - Keep running even if one file is troublesome (corrupted or missing data)
                    try:
                        images = tsa.read_data(INPUT_FOLDER + '/' + subject + '.aps')
                        images = images.transpose()
                        #images_to_use = switch_chest_back(images) if should_flip else images 
                        images_to_use = flip_and_switch(images) if should_flip else images
                        cropped_ims = []
                        for i in range(0, len(images_to_use)):
                            if i==3 or i==5 or i==11 or i==13:
                                cropped_ims.append(crop_and_resize_2D(images_to_use[i], x_resize_ratio=0.75))
                            elif i==4 or i==12:
                                cropped_ims.append(crop_and_resize_2D(images_to_use[i], x_resize_ratio=0.5))
                            else:
                                cropped_ims.append(crop_and_resize_2D(images_to_use[i]))
                        #for i in range(0, len(cropped_ims)):
                        #    cropped_ims[i] = cv2.resize(cropped_ims[i], (0,0), fx=0.5, fy=0.5) 
                        #ims = scipy.ndimage.zoom(cropped_ims, (1, 0.5, 0.5))
                        pre_stack = get_zone9(cropped_ims)
                        stack = np.hstack(pre_stack)
                        path = TOP_DIR + folder + "/"
                        #threat_prob_5 =  get_threat_prob(5, labels, subject)
                        #threat_prob_17 =  get_threat_prob(17, labels, subject)
                        #threat_prob_1 =  get_threat_prob(1, labels, subject)
                        #threat_prob_3 =  get_threat_prob(3, labels, subject)
                        threat_prob_9 =  get_threat_prob(9, labels, subject)
                        if threat_prob_9 == True:
                            path = path + THREAT_DIR_NAME
                        else:
                            path = path + NON_THREAT_DIR_NAME
                        """if threat_prob_5 == 1 and threat_prob_17 == 1:
                            path = path + THREAT_DIR_NAME 
                        elif should_flip and threat_prob_17 == 1:
                            path = path + THREAT_DIR_NAME
                        elif not should_flip and threat_prob_5 == 1:
                            path = path + THREAT_DIR_NAME
                        else:
                            path = path + NON_THREAT_DIR_NAME"""
                        scipy.misc.imsave(path + sub_name + ".jpg", stack)   
                    except:
                        print ("Failed!!!")    

#preprocess_plates()

def preprocess_lb_plates_for_tz(tz, flip_tz):
    TOP_DIR = "lb_plates/" + str(tz) + "/ims/"
    get_lb_file_list()
    #list existing files so we can skip those
    files_list = os.listdir(TOP_DIR)
    already_prepped_subs = []
    for sub_file in files_list:
        already_prepped_subs.append(sub_file.split(".")[0])
    for already_prepped in already_prepped_subs[:5]:
        print ("ap: " + already_prepped)
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
                if True:#try:
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
                        if tz in [15, 3, 9, 13]:
                            cropped_ims.append(crop_and_resize_2D(images_to_use[i]))
                        else:
                            if i==3 or i==5 or i==11 or i==13:
                                cropped_ims.append(crop_and_resize_2D(images_to_use[i], x_resize_ratio=0.75))
                            elif i==4 or i==12:
                                cropped_ims.append(crop_and_resize_2D(images_to_use[i], x_resize_ratio=0.5))
                            else:
                                cropped_ims.append(crop_and_resize_2D(images_to_use[i]))
                    for i in range(0, len(cropped_ims)):
                        cropped_ims[i] = cv2.resize(cropped_ims[i], (0,0), fx=0.5, fy=0.5) 
                    pre_stack = get_zone(tz, cropped_ims)
                    stack = np.hstack(pre_stack)
                    path = TOP_DIR + sub_name + ".jpg"
                    scipy.misc.imsave(path, stack)   
                #except:
                #    print ("Failed!!!")    
    print ("next!")
"""preprocess_lb_plates_for_tz(15, 16)
preprocess_lb_plates_for_tz(9, None)
preprocess_lb_plates_for_tz(3, 1)
preprocess_lb_plates_for_tz(13, 14)
preprocess_lb_plates_for_tz(11, 12)
preprocess_lb_plates_for_tz(8, 10)
preprocess_lb_plates_for_tz(6, 7)
preprocess_lb_plates_for_tz(4, 2)
preprocess_lb_plates_for_tz(5, 17)"""

def convert_to_grayscale(img):
    # scale pixel values to grayscale
    base_range = np.amax(img) - np.amin(img)
    rescaled_range = 255 - 0
    img_rescaled = (((img - np.amin(img)) * rescaled_range) / base_range)

    return np.uint8(img_rescaled)

def spread_spectrum(img):
    #img = stats.threshold(img, threshmin=12, newval=0)
    img = np.clip(img, 12, None)

    # see http://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img= clahe.apply(img)
    
    return img

def make_bigger_plates():
    ORIGINAL_DIR = "plates/4/"
    CHANNELS_DIR = "plates/1/channels/"
    VERTICAL_DIR = "plates/1/vertical/"
    PREPROC_DIR = "plates/4/preproc/"
    SUB_DIRS = ["test/threats/", "test/non_threats/", "train/threats/", "train/non_threats/"]
    
    for directory in SUB_DIRS:
        print ("next_dir")
        orig_path = ORIGINAL_DIR + directory
        chan_path = CHANNELS_DIR + directory
        preproc_path = PREPROC_DIR + directory
        vert_path = VERTICAL_DIR + directory
        files = os.listdir(orig_path)
        for sub_file in files:
            if True: #try:
                """is_flip = False
                sub_name = sub_file.split(".")[0]
                flip_split = sub_name.split("_")
                if len(flip_split) > 1:
                    sub_name = flip_split[1]
                    is_flip = True
                orig_image = imageio.imread(ORIGINAL_DIR + directory + sub_file)
                #print(orig_image[:20][:20][:])
                flip_name = sub_name
                if is_flip == False:
                    flip_name = "flip_" + sub_name
                other_dir = "non_threats/" if directory.split("/")[1] == "threats" else "threats/"
                try:
                    flip_image = imageio.imread(ORIGINAL_DIR + directory + flip_name + ".jpg")
                except:
                    flip_image = imageio.imread(ORIGINAL_DIR + directory.split("/")[0] + "/" + other_dir + flip_name + ".jpg")
                #vertical_concat = np.vstack([orig_image, flip_image])
                #scipy.misc.imsave(vert_path + sub_file, vertical_concat)
                flip_image = np.array(flip_image)
                orig_image = np.array(orig_image)

                orig_image = convert_to_grayscale(orig_image)"""
                orig_image = Image.open(orig_path + sub_file).convert('I')
                orig_image = convert_to_grayscale(orig_image)
                orig_image = spread_spectrum(orig_image)
                scipy.misc.imsave(preproc_path + sub_file, orig_image)


                #channels_image = 2*orig_image/3 + flip_image/3

                """trans_flip_image = flip_image.transpose()
                trans_orig_image = orig_image.transpose()
                trans_channels_image = np.array([trans_orig_image[0], trans_orig_image[1], trans_flip_image[0]])
                channels_image = trans_channels_image.transpose()"""
                #scipy.misc.imsave(chan_path + sub_file, channels_image)

#make_bigger_plates()
#print("plates made")

def plate_generator(subjects, batch_size):

    zone = 15
    for_vgg = True
    # intialize tracking and saving items
    threat_zone_examples = []
    start_time = timer()
    labels = get_subject_labels()
    print_shape = True

    threat_subjects = []
    non_threat_subjects = []

    for subject in subjects:
        threat_prob = get_threat_prob(zone, labels, subject)
        if threat_prob == 1:
            threat_subjects.append(subject)
        else:
            non_threat_subjects.append(subject)

    np.random.shuffle(non_threat_subjects)
    prepped_subjects = []
    for i in range(0, len(threat_subjects)):
        prepped_subjects.append(threat_subjects[i])
        prepped_subjects.append(non_threat_subjects[i])
    np.random.shuffle(prepped_subjects)
    #Use full dataset instead
    prepped_subjects = subjects
    while True:
        for i in range(0, len(prepped_subjects), batch_size):
            y_batch = []
            x_batch = []
            for subject in prepped_subjects[i:i+batch_size]:
                images = tsa.read_data(INPUT_FOLDER + '/' + subject + '.aps')
                # transpose so that the slice is the first dimension shape(16, 660, 512)
                images = images.transpose()
                images = scipy.ndimage.zoom(images, (1, 0.5, 0.5))
                crops = get_zone15(images)
                x = np.hstack(crops)
                if for_vgg:
                    fake_rgb = np.array([x, x, x])
                    x = fake_rgb.transpose()
                else:
                    x = np.reshape(x.transpose(), (1065, 60, 1))
                # get label
                threat_prob = get_threat_prob(zone, labels, subject)
                #y = [0, 0]
                #y[threat_prob] = 1
   
                y_batch.append([threat_prob])
                x_batch.append(x)
            yield (np.array(x_batch), np.array(y_batch))

def test_plate_gen():
    get_train_test_file_list()
    plate_gen = plate_generator(TRAIN_SUBJECT_LIST, 1)
    test_plate_gen = plate_generator(TEST_SUBJECT_LIST, 1)
    x, y = next(plate_gen)
    print (y)
    next(test_plate_gen)

#test_plate_gen()

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

def resnet50(input_size):
    resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(input_size[0], input_size[1], 3))
    x = resnet_model.output
    x = Flatten()(x)
    x = Dense(1024, activation='linear', name='fc1')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.2)(x)
    #x = BatchNormalization()(x)
    x = Dense(1, activation='sigmoid', name='prediction')(x)
    model = Model(inputs=resnet_model.inputs, outputs=x)
    sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.7, nesterov=True)
    model.compile(loss='binary_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    return model

def res_plate(input_size):
    #base_model = VGG16(weights='imagenet', include_top=False, input_shape=(input_size[0], input_size[1], 3))
    #vgg = base_model.get_layer('block2_pool').output
    resnet_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(input_size[0], input_size[1], 3))#(34, 528, 128))
    x = resnet_model.output#(vgg)
    x = Flatten()(x)
    x = Dense(1024, activation='relu', name='fc1')(x)
    #x = Dropout(0.15)(x)
    x = Dense(1, activation='sigmoid', name='prediction')(x)
    model = Model(inputs=resnet_model.inputs, outputs=x)
    #model.summary()
    #base_model.summary()
    sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.7, nesterov=True)
    model.compile(loss='binary_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    return model

def VGG_plate(input_size, weights_path=None):
    print ("VGG!")
    #inputs  = Input((1707, 115, 3))
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(input_size[0], input_size[1], 3))
    x = base_model.get_layer('block2_pool').output
    """x = Conv2D(256, (2, 2), activation='relu')(inputs)
    x = BatchNormalization(axis=1)(x)"""
    """
    x = Conv2D(64, (3, 3), activation='relu')(inputs)
    x = BatchNormalization(axis=1)(x)
    x = ZeroPadding2D((0, 1))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = BatchNormalization(axis=1)(x)

    x = ZeroPadding2D((1, 2))(x)
    x = MaxPooling2D((2,2), strides=(2,2))(x)

    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = BatchNormalization(axis=1)(x)
    x = ZeroPadding2D((0, 1))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = BatchNormalization(axis=1)(x)
    
    x = ZeroPadding2D((1, 2))(x)
    x = MaxPooling2D((3,3), strides=(2,2))(x)
    """
    x = Conv2D(256, (3, 3), activation='relu')(x)
    x = BatchNormalization(axis=1)(x)
    x = ZeroPadding2D((1, 0))(x)
    x = Conv2D(256, (3, 3), activation='relu')(x)
    x = BatchNormalization(axis=1)(x)
    x = ZeroPadding2D((1, 0))(x)
    x = Conv2D(256, (3, 3), activation='relu')(x)
    x = BatchNormalization(axis=1)(x)
   
    x = ZeroPadding2D((2, 1))(x)
    x = MaxPooling2D((3,3), strides=(2,2))(x)

    x = Conv2D(512, (3, 3), activation='relu')(x)
    x = BatchNormalization(axis=1)(x)
    x = ZeroPadding2D((1, 0))(x)
    x = Conv2D(512, (3, 3), activation='relu')(x)
    x = BatchNormalization(axis=1)(x)
    x = ZeroPadding2D((1, 0))(x)
    x = Conv2D(512, (3, 3), activation='relu')(x)
    x = BatchNormalization(axis=1)(x)
    x = ZeroPadding2D((1, 0))(x)
   
    x = ZeroPadding2D((2, 1))(x)
    x = MaxPooling2D((3,3), strides=(2,2))(x)

    x = Conv2D(512, (3, 3), activation='relu')(x)
    x = BatchNormalization(axis=1)(x)
    x = ZeroPadding2D((1, 0))(x)
    x = Conv2D(512, (3, 3), activation='relu')(x)
    x = BatchNormalization(axis=1)(x)
    x = ZeroPadding2D((1, 0))(x)
    x = Conv2D(512, (3, 3), activation='relu')(x)
    x = BatchNormalization(axis=1)(x)
    x = ZeroPadding2D((1, 0))(x)
   
    x = ZeroPadding2D((2, 1))(x)
    x = MaxPooling2D((3,3), strides=(2,2))(x)

    x = Flatten()(x)

    x = Dense(256, activation='relu', name='fc2')(x)
    x = BatchNormalization()(x)
    """x = Dense(512, activation='relu', name='fc2')(x)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='relu', name='fc2')(x)
    x = Dropout(0.2)(x)"""
    x = Dense(1, activation='sigmoid', name='prediction')(x)

    #for layer in base_model.layers:
    #    layer.trainable = False

    model = Model(inputs=base_model.inputs, outputs=x)

    #base_model.summary()

    model.summary()

    sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.7, nesterov=True)
    model.compile(loss='binary_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    #base_model.summary()

    return model

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
    # get train and test batches get_train_test_file_list()
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

def train_plate_net(tz):
    get_train_test_file_list()
    batch_size = 4
    #train_gen = plate_generator(TRAIN_SUBJECT_LIST, batch_size)
    #test_gen = plate_generator(TEST_SUBJECT_LIST, batch_size)
    #train_steps = np.ceil(float(len(TRAIN_SUBJECT_LIST)) / float(batch_size))
    #test_steps = np.ceil(float(len(TEST_SUBJECT_LIST)) / float(batch_size))
    #train_steps = np.ceil(float(189) / float(batch_size))
    #test_steps = np.ceil(float(45) / float(batch_size))
    INPUT_SHAPES = {15: (197, 2115), 3: (197, 1707), 9: (197, 1502), 13: (197, 2225), 11: (197, 2009), 
        8: (197, 1651), 6: (197, 1818), 4: (197, 1524), 5: (197, 1337)}
    INPUT_SIZE = INPUT_SHAPES[tz]

    train_gen = ImageDataGenerator(width_shift_range=0.0156, height_shift_range=0.2, zoom_range=0.05)

    #zone 4
    #train_gen = ImageDataGenerator(width_shift_range=0.03, height_shift_range=0.3, zoom_range=0.1)
    test_gen = ImageDataGenerator()

    checkpoint1_path = "weights/plate_models/" + str(tz) +"/best_resnet50_dropout_3.h5"

    model = resnet50(INPUT_SIZE)
    #model.load_weights("weights/plate_models/" + str(tz) +"/best_resnet50_aug_bs_4.h5")

    cnn_checkpoint = ModelCheckpoint(checkpoint1_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    es = EarlyStopping('val_loss', patience=14, mode="min")
    lc = LoggingCallback('resnet_1_tz_' + str(tz), net_name=str(tz)+"_resnet50_high_lr_bs_4")

    print ("low init LR")
    sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.7, nesterov=True)
    model.compile(loss='binary_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])


    #model.fit_generator(generator=train_gen, validation_data=test_gen, steps_per_epoch = train_steps, validation_steps = test_steps, 
    #    epochs = 1000, verbose=2, callbacks=[cnn_checkpoint])
    model.fit_generator(generator=train_gen.flow_from_directory("plates/" + str(tz) + "/train", class_mode="binary", batch_size=1, target_size=INPUT_SIZE), validation_data=test_gen.flow_from_directory("plates/" + str(tz) + "/test", class_mode="binary", batch_size=1, target_size=INPUT_SIZE), steps_per_epoch = 1500, validation_steps = 458, epochs = 1, verbose=2, callbacks=[cnn_checkpoint, es, lc]) 
    time.sleep(90)"""
    
    """time.sleep(90)
    print ("increasing LR")
    sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.7, nesterov=True)
    model.compile(loss='binary_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    model.fit_generator(generator=train_gen.flow_from_directory("plates/" + str(tz) + "/train", class_mode="binary", batch_size=1, target_size=INPUT_SIZE), validation_data=test_gen.flow_from_directory("plates/" + str(tz) + "/test", class_mode="binary", batch_size=1, target_size=INPUT_SIZE), steps_per_epoch = 1500, validation_steps = 458, epochs = 40, verbose=2, callbacks=[cnn_checkpoint, es, lc])
    """
    #model.fit_generator(generator=train_gen.flow_from_directory("plates/" + str(tz) + "/train", class_mode="binary", batch_size=batch_size, target_size=INPUT_SIZE), validation_data=test_gen.flow_from_directory("plates/" + str(tz) + "/test", class_mode="binary", batch_size=batch_size, target_size=INPUT_SIZE), steps_per_epoch = 1500, validation_steps = 458, epochs = 40, verbose=2, callbacks=[cnn_checkpoint, es, lc])
    """
    time.sleep(90)
    model.load_weights(checkpoint1_path)
    print ("back to low LR")
    sgd = keras.optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.7, nesterov=True)
    model.compile(loss='binary_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    checkpoint2 = ModelCheckpoint("weights/plate_models/" + str(tz) +"/best_resnet_exp_low_lr.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    
    model.fit_generator(generator=train_gen.flow_from_directory("plates/" + str(tz) + "/preproc/train", class_mode="binary", batch_size=1, target_size=INPUT_SIZE), validation_data=test_gen.flow_from_directory("plates/" + str(tz) + "/preproc/test", class_mode="binary", batch_size=1, target_size=INPUT_SIZE), steps_per_epoch = 1500, validation_steps = 458, epochs = 40, verbose=2, callbacks=[checkpoint2, es, lc])
    time.sleep(90)
    """
<<<<<<< HEAD

def train_plate_nets():
    #15 already complete[3, 9, 13, 11, [8, 6, 4, 5
    #9
    #change to dictionary w/ the name of the best network
    for tz in [9, 4, 3, 11, 5, 8, 6, 13, 15]:
        try: 
            train_plate_net(tz)
            
        except:
            print("Failed!")
            time.sleep(400)
        K.clear_session()
        time.sleep(30)

#train_plate_nets()

def predict_plates2():
    print ("predictin...")
    sub_file = open("labels/resnet_sub_1_2.csv", "w")
    sub_file.write("Id,Probability\n")
    INPUT_SHAPES = {15: (139, 2115), 3: (139, 1707), 9: (139, 1502), 13: (139, 2225), 11: (139, 2009), 8: (139, 1651), 6: (139, 1818),
        4: (139, 1524), 5: (139, 1337)}
    #INPUT_SHAPES = {15: (2115, 60), 3: (1707, 115), 9: (1502, 110), 13: (2225, 85), 11: (2009, 110), 8: (1651, 95), 6: (1818, 100),
    #    4: (1524, 85), 5: (1337, 75)}
    pred_gen = ImageDataGenerator()
    for tz in [15, 3, 9, 13, 11, 8, 6, 4, 5]:
        print("Tz: " + str(tz))
        model = res_plate(INPUT_SHAPES[tz])
        model.load_weights("weights/plate_models/" + str(tz) + "/best_resnet.h5")
        dir_path = "lb_plates/" + str(tz) 
        plate_files = os.listdir(dir_path + "/ims")
        #print(plate_files[:10])
        i = 0
        for plate_file in plate_files:
            if i % 20 == 0:
                print (str(i))
            plate = np.asarray(Image.open(dir_path + "/ims/" + plate_file).convert("RGB"))
            #print("1")
            plate = scipy.ndimage.zoom(plate, (139.0/plate.shape[0], 1, 1))
            #print("2")
            #plate = plate.reshape((plate.shape[1], plate.shape[0], plate.shape[2]))
            #print("3")
            plate_name = plate_file.split(".")[0]
            #print("4")
            plate_array = []
            #print("5")
            plate_array.append(plate)
            #print("6")
            x = np.array(plate_array)
            #print("7")
            predictions = model.predict(x)
            #print("8")
            predictions = np.clip(predictions, 0.05, 0.95)
            #print("9")
            sub_file.write(plate_name + ", " + str(predictions[0][0]) + "\n")
            #print("10")
            i = i + 1


            #print(dir_path)
            #steps_count = len(plate_files) -1

#predict_plates2()

def predict_plates():
    print ("predictin...")
    sub_file = open("labels/resnet_sub_1_1.csv", "w")
    sub_file.write("Id,Probability\n")
    INPUT_SHAPES = {15: (139, 2115), 3: (139, 1707), 9: (139, 1502), 13: (139, 2225), 11: (139, 2009), 8: (139, 1651), 6: (139, 1818),
        4: (139, 1524), 5: (139, 1337)}
    #INPUT_SHAPES = {15: (2115, 60), 3: (1707, 115), 9: (1502, 110), 13: (2225, 85), 11: (2009, 110), 8: (1651, 95), 6: (1818, 100),
    #    4: (1524, 85), 5: (1337, 75)}
    pred_gen = ImageDataGenerator()
    for tz in [15]:#, 3, 9, 13, 11, 8, 6, 4, 5]:
        model = res_plate(INPUT_SHAPES[tz])
        model.load_weights("weights/plate_models/" + str(tz) + "/best_resnet.h5")
        dir_path = "lb_plates/" + str(tz) 
        plate_files = os.listdir(dir_path + "/ims")
        print(plate_files[:5])
        print(plate_files[-5:])
        i = 0
        if True: #for plate_file in plate_files:
            """plate = np.asarray(Image.open(dir_path + "/" + plate_file).convert("RGB"))
            plate = plate.reshape((plate.shape[1], plate.shape[0], plate.shape[2]))
            plate_name = plate_file.split(".")[0]
            plate_array = []
            plate_array.append(plate)
            x = np.array(plate_array)"""
            #print(dir_path)
            #steps_count = len(plate_files) -1

            gen = pred_gen.flow_from_directory(dir_path, class_mode=None, shuffle=False, target_size=INPUT_SHAPES[tz], batch_size=1, save_to_dir="plates/pred_plates")
            for i in range(0, 5):
                print (next(gen))

            classifications = model.predict_generator(pred_gen.flow_from_directory(dir_path, class_mode=None, shuffle=False, target_size=INPUT_SHAPES[tz], batch_size=1, save_to_dir="plates/pred_plates"), steps=5) #len(plate_files))
            print (len(classifications))
            classifications = np.clip(classifications, 0.05, 0.95)
            for i in range(0, len(classifications)):
                #print(plate_files[i].split(".")[0] + ", " + str(classifications[i][0]) + "\n")
                sub_file.write(plate_files[i].split(".")[0] + ", " + str(classifications[i][0]) + "\n")
                i = i + 1

#predict_plates2()

def evaluate_plates():
    print("evaluate")
    INPUT_SHAPES = {15: (139, 2115), 3: (139, 1707), 9: (139, 1502), 13: (139, 2225), 11: (139, 2009), 8: (139, 1651), 6: (139, 1818),
        4: (139, 1524), 5: (139, 1337)}
    #INPUT_SHAPES = {15: (2115, 60), 3: (1707, 115), 9: (1502, 110), 13: (2225, 85), 11: (2009, 110), 8: (1651, 95), 6: (1818, 100),
    #    4: (1524, 85), 5: (1337, 75)}
    pred_gen = ImageDataGenerator()
    for tz in [15, 3, 9, 13, 11, 8, 6, 4, 5]:
        model = res_plate(INPUT_SHAPES[tz])
        model.load_weights("weights/plate_models/" + str(tz) + "/best_resnet.h5")
        plates_dir = "plates/" + str(tz) + "/test"
        plates = os.listdir(plates_dir + "/threats") + os.listdir(plates_dir+"/non_threats")
        batch_size = 4
        print(len(plates))
        print ("TZ: " + str(tz))
        print(model.metrics_names)
        print(model.evaluate_generator(pred_gen.flow_from_directory(plates_dir, class_mode="binary", batch_size=4, target_size=INPUT_SHAPES[tz]), steps=len(plates)/batch_size))

#evaluate_plates()

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

#test_big_cnn()

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
