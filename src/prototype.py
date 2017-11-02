#Prototype my TSA competition pipeline
# a3d file -> 3D SAE -> TL -> 3D CNN
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')

import json

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, ZeroPadding3D, GlobalAveragePooling3D, Input
from keras.models import Model
from keras import backend as K
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import ModelCheckpoint

import numpy as np
import scipy
import pandas as pd
import cv2
import sys
import os

TEST_FILE_PATH = "stage1_a3d/0a27d19c6ec397661b09f7d5998e0b14.a3d"
SUBJECT_LABELS_PATH = "labels/stage1_labels.csv"
INPUT_SHAPE = (256, 256, 132, 1)
SAE_PATH = "weights/best_sae.h5"
CNN_PATH = "weights/best_cnn.h5"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config = config))

def read_header(infile):
    """Read image header (first 512 bytes)
    """
    h = dict()
    fid = open(infile, 'r+b')
    h['filename'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 20))
    h['parent_filename'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 20))
    h['comments1'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 80))
    h['comments2'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 80))
    h['energy_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['config_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['file_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['trans_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['scan_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['data_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['date_modified'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 16))
    h['frequency'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['mat_velocity'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['num_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
    h['num_polarization_channels'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['spare00'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['adc_min_voltage'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['adc_max_voltage'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['band_width'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['spare01'] = np.fromfile(fid, dtype = np.int16, count = 5)
    h['polarization_type'] = np.fromfile(fid, dtype = np.int16, count = 4)
    h['record_header_size'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['word_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['word_precision'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['min_data_value'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['max_data_value'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['avg_data_value'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['data_scale_factor'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['data_units'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['surf_removal'] = np.fromfile(fid, dtype = np.uint16, count = 1)
    h['edge_weighting'] = np.fromfile(fid, dtype = np.uint16, count = 1)
    h['x_units'] = np.fromfile(fid, dtype = np.uint16, count = 1)
    h['y_units'] = np.fromfile(fid, dtype = np.uint16, count = 1)
    h['z_units'] = np.fromfile(fid, dtype = np.uint16, count = 1)
    h['t_units'] = np.fromfile(fid, dtype = np.uint16, count = 1)
    h['spare02'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['x_return_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_return_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_return_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['scan_orientation'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['scan_direction'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['data_storage_order'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['scanner_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['x_inc'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_inc'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_inc'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['t_inc'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['num_x_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
    h['num_y_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
    h['num_z_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
    h['num_t_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
    h['x_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['x_acc'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_acc'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_acc'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['x_motor_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_motor_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_motor_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['x_encoder_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_encoder_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_encoder_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['date_processed'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 8))
    h['time_processed'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 8))
    h['depth_recon'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['x_max_travel'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_max_travel'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['elevation_offset_angle'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['roll_offset_angle'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_max_travel'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['azimuth_offset_angle'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['adc_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['spare06'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['scanner_radius'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['x_offset'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_offset'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_offset'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['t_delay'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['range_gate_start'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['range_gate_end'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['ahis_software_version'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['spare_end'] = np.fromfile(fid, dtype = np.float32, count = 10)
    return h

def read_data(infile):
    """Read any of the 4 types of image files, returns a numpy array of the image contents
    """
    extension = os.path.splitext(infile)[1]
    h = read_header(infile)
    nx = int(h['num_x_pts'])
    ny = int(h['num_y_pts'])
    nt = int(h['num_t_pts'])
    fid = open(infile, 'rb')
    fid.seek(512) #skip header
    if extension == '.aps' or extension == '.a3daps':
        if(h['word_type']==7): #float32
            data = np.fromfile(fid, dtype = np.float32, count = nx * ny * nt)
        elif(h['word_type']==4): #uint16
            data = np.fromfile(fid, dtype = np.uint16, count = nx * ny * nt)
        data = data * h['data_scale_factor'] #scaling factor
        data = data.reshape(nx, ny, nt, order='F').copy() #make N-d image
    elif extension == '.a3d':
        if(h['word_type']==7): #float32
            data = np.fromfile(fid, dtype = np.float32, count = nx * ny * nt)
        elif(h['word_type']==4): #uint16
            data = np.fromfile(fid, dtype = np.uint16, count = nx * ny * nt)
        data = data * h['data_scale_factor'] #scaling factor
        data = data.reshape(nx, nt, ny, order='F').copy() #make N-d image
    elif extension == '.ahi':
        data = np.fromfile(fid, dtype = np.float32, count = 2* nx * ny * nt)
        data = data.reshape(2, ny, nx, nt, order='F').copy()
        real = data[0,:,:,:].copy()
        imag = data[1,:,:,:].copy()
    fid.close()
    if extension != '.ahi':
        data /= np.max(data)

        return data
    else:
        return real, imag

# Test that a file can be read correctly, print original values vs div by max
#TODO - update bc I moved the /= max bit into read_data
def test_read (should_print):
    data = read_data(TEST_FILE_PATH)
    if (should_print):
        print (data.shape)
        print (data[0:5][0:5][0:5])
    data /= np.max(data)
    if (should_print):
         print (data.shape)
         print (data[0:5][0:5][0:5])
    return data    

#test_read(True)

#define network
# based on - https://arxiv.org/pdf/1706.04970.pdf


def get_sae():
    print (INPUT_SHAPE)
 
    model = Sequential()

    #encoder

    #add stride? 1, see paper
#    model.add(Conv3D(32, kernel_size=(3, 3, 3), padding='same',
    model.add(Conv3D(8, kernel_size=(3, 3, 3), padding='same', input_shape=INPUT_SHAPE))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    
#    model.add(Conv3D(32, kernel_size=(3, 3, 3), padding='same'))
    model.add(Conv3D(16, kernel_size=(3, 3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
   
#    model.add(Flatten())

#    model.add(Dense(8))
#    model.add(Activation('relu'))
    
    model.add(Conv3D(32, kernel_size=(3, 3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    model.add(Conv3D(64, kernel_size=(3, 3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    model.add(Conv3D(64, kernel_size=(3, 3, 3), padding='same'))
    model.add(Activation('relu'))

    model.add(Conv3D(64, kernel_size=(3, 3, 3), padding='same'))
    model.add(Activation('relu'))
 
    model.add(UpSampling3D(size=(2,2,2)))
    model.add(Conv3D(64, kernel_size=(3, 3, 3), padding='same'))
    model.add(Activation('relu'))

    model.add(UpSampling3D(size=(2,2,2)))
    model.add(Conv3D(32, kernel_size=(3, 3, 3), padding='same'))
    model.add(Activation('relu'))



    #decoder

#    model.add(Dense(86507520))
#    model.add(Dense(2703360))#5406720))#10813440))#21626880))#43253760))
#    model.add(Activation('relu'))
#    model.add(Reshape((128,128,165,32)))
#    model.add(Reshape((128,128,165, 1)))
 
    model.add(UpSampling3D(size=(2,2,2)))
#    model.add(Conv3D(32, kernel_size=(3, 3, 3), padding='same'))
    model.add(Conv3D(16, kernel_size=(3, 3, 3), padding='same'))
    model.add(Activation('relu'))

    model.add(ZeroPadding3D((0,0,1)))

    model.add(UpSampling3D(size=(2,2,2)))
    model.add(Conv3D(8, kernel_size=(3, 3, 3), padding='same'))
    model.add(Activation('relu'))

    model.add(Conv3D(1, kernel_size=(3, 3, 3), padding='same'))
    model.add(Activation('linear'))

#    model.add(UpSampling3D(size=(2,2,1)))

    
    print (model.summary())
    #sys.exit(0) #

    model.compile(loss=keras.losses.mean_squared_error,
            optimizer= keras.optimizers.SGD())
             
    return model

def get_cnn():
    sae = load_model(SAE_PATH)
#    sae = get_sae()
    print (sae.summary())
    print ("Wha?")

    x = sae.layers[15].output
    
    x = Conv3D(64, kernel_size=(3, 3, 3), padding='same', name = 'cnn_conv3d_4')(x)
    x = Activation('relu', name='cnn_activation_4')(x)
    x = ZeroPadding3D((1,1,1))(x)
    x = Conv3D(64, kernel_size=(3, 3, 3), padding='same', name = 'cnn_conv3d_5')(x)
    x = Activation('relu', name='cnn_activation_5')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), name='cnn_max_pooling3d_3')(x)
    
    x = ZeroPadding3D((1,1,1))(x)
    x = Conv3D(128, kernel_size=(3, 3, 3), padding='same', name = 'cnn_conv3d_6')(x)
    x = Activation('relu', name='cnn_activation_6')(x)
    x = ZeroPadding3D((1,1,1))(x)
    x = Conv3D(128, kernel_size=(3, 3, 3), padding='same', name = 'cnn_conv3d_7')(x)
    x = Activation('relu', name='cnn_activation_7')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), name='cnn_max_pooling3d_4')(x)
  
    x = Flatten()(x)

    x = Dense(128, activation='relu')(x)
 
    x = Dense(5, activation = 'sigmoid')(x)
 
    model = Model(inputs = sae.input, outputs = x)

    for layer in sae.layers:
        layer.trainable = False

    model.compile(loss=keras.losses.categorical_crossentropy,
            optimizer= keras.optimizers.SGD(lr=0.05))
    
    print (model.summary())
    return model


def get_subject_labels():
    infile = SUBJECT_LABELS_PATH
    df = pd.read_csv(infile)
    # Separate the zone and subject id into a df
    df['Subject'], df['Zone'] = df['Id'].str.split('_',1).str
    #TODO: convert zone to correct int here
    df = df[['Subject', 'Zone', 'Probability']]
    return df

#data generator

def generator(sae):
    dir_path='stage1_a3d'
    image_names = os.listdir(dir_path)
    labels = get_subject_labels()
    if sae is False:
        image_names = list(set(labels["Subject"].values.tolist()))
        """        image_names = [
            'NO - 00360f79fd6e02781457eda48f85da90',
            'NO - 0043db5e8c819bffc15261b1f1ac5e42',
            'NO - 0050492f92e22eed3474ae3a6fc907fa',
            '006ec59fa59dd80a64c85347eef810c7',
            '0097503ee9fa0606559c56458b281a08']
        """
        for i in range(0, len(image_names)):
            image_names[i] = image_names[i] + ".a3d"

#    np.random.shuffle(image_names)
#    image_names = image_names[:10]
    batch_size = 10

    zone_map = {
        1: 2, 2: 2, 3:3, 4:3, 11:4, 13:4, 15:4, 12:5, 14:5, 16:5,
        17:1, 7:1, 6:1, 10:1, 9:1, 8:1, 5:1}

    while True:
        for start in range(0, len(image_names), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(image_names))
            ids_train_batch = image_names[start:end]
            for filename in ids_train_batch:
                if sae is False:
#                    y = np.zeros((17))
                    y = np.zeros((5))
                    threat_list = labels.loc[labels['Subject'] == filename.split(".")[0]]
                    threat_iter = threat_list.iterrows()
                    while True:
                        threat = next(threat_iter, None)
                        if threat is None:
                            break
                        threat = threat[1]
                        if threat['Probability'] is 1:
                            zone = threat['Zone']
                            zone = int(zone[4:])
                            mapped_zone = zone_map[zone]
                            y[mapped_zone-1] = 1
                    y_batch.append(y)                            
                filedata = read_data(dir_path + '/{}'.format(filename))

                data = np.reshape(filedata, (512, 512, 660, 1))
                x_batch.append(scipy.ndimage.zoom(data, (0.5, 0.5, 0.2, 1)))


            x_batch = np.array(x_batch)
            print (x_batch.shape)
            if sae is True:
                yield x_batch, x_batch
            else: 
                y_batch = np.array(y_batch)
                print (y_batch)
                yield x_batch, y_batch


#call train
def load_data():
    dir_path='stage1_a3d'
    image_names = os.listdir(dir_path)
    x = []
    for image_name in image_names[:1]:
        print ("start one")
        filedata = read_data(dir_path + '/' + image_name) 
        data = np.reshape(filedata, (512, 512, 660, 1))
        x.append(scipy.ndimage.zoom(data, (0.5, 0.5, 0.2, 1)))
    x = np.array(x)
    print (x.shape)
    return x

#x = load_data()

sae_checkpoint = ModelCheckpoint(SAE_PATH, monitor='loss', verbose=1, save_best_only=True, mode='min')
cnn_checkpoint = ModelCheckpoint(CNN_PATH, monitor='loss', verbose=1, save_best_only=True, mode='min')

#log = get_sae().fit_generator(generator=generator(True), steps_per_epoch = 200, epochs = 1000, verbose=2, callbacks=[sae_checkpoint])



cnn = load_model(CNN_PATH)
cnn.compile(loss=keras.losses.categorical_crossentropy,
        optimizer= keras.optimizers.SGD(lr=0.01))

#cnn = get_cnn()
gen = generator(False)
x, y = next(gen)
#y = [[1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1]]

log = cnn.fit(x=x, y=y, batch_size=5, epochs=10000, verbose=2, callbacks=[cnn_checkpoint])


predictions = cnn.predict(x)
print(predictions)


#print(cnn.summary())

#log = cnn.fit_generator(generator=generator(False), steps_per_epoch = 1, epochs = 1000, verbose = 2, callbacks=[cnn_checkpoint])

"""log = model.fit_generator(generator=train_generator(),
                    steps_per_epoch=np.ceil(float(len(ids_train_split)) / float(batch_size)),
                    epochs=1000,
                    verbose=2,
                    callbacks=callbacks,
                    validation_data=valid_generator(),
                    validation_steps=np.ceil(float(len(ids_valid_split)) / float(batch_size)))

"""
