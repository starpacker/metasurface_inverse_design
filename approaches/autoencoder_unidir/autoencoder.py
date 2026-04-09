# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 13:26:52 2018

@author: DELL
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] ="2"

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K

input_img = Input(shape=(64, 64, 1))  # adapt this if using `channels_first` image data format

x = Conv2D(128, (3, 3), activation='relu', padding='same', name="Conv_1")(input_img)
x = Conv2D(128, (3, 3), activation='relu', padding='same', name="Conv_2")(x)
x = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(64, (3, 3), activation='relu', padding='same', name="Conv_3")(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same', name="Conv_4")(x)
x = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(32, (3, 3), activation='relu', padding='same', name="Conv_5")(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same', name="Conv_6")(x)
x = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(16, (3, 3), activation='relu', padding='same', name="Conv_5.5")(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same', name="Conv_6.5")(x)
x = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(8, (3, 3), activation='relu', padding='same', name="Conv_7")(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same', name="Conv_8")(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional?

x = Conv2D(128, (3, 3), activation='relu', padding='same', name="Conv_9")(encoded)
x = Conv2D(128, (3, 3), activation='relu', padding='same',name="Conv_10")(x)
x = UpSampling2D((2, 2))(x)

x = Conv2D(64, (3, 3), activation='relu', padding='same',name="Conv_11")(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same',name="Conv_12")(x)
x = UpSampling2D((2, 2))(x)

x = Conv2D(32, (3, 3), activation='relu', padding='same',name="Conv_13")(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same',name="Conv_14")(x)
x = UpSampling2D((2, 2))(x)

x = Conv2D(16, (3, 3), activation='relu', padding='same',name="Conv_13.5")(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same',name="Conv_14.5")(x)
x = UpSampling2D((2, 2))(x)

x = Conv2D(8, (3, 3), activation='relu', padding='same',name="Conv_15")(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same',name="Conv_16")(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same',name="Conv_17")(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

import scipy.io as sio  # 为导入mat格式文件导入组件
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint

mat1 = 'datasize64_1.mat'  
data = sio.loadmat(mat1)
Output = data['data_1']
x_train = Output

mat1 = 'datasize64_2.mat'  
data = sio.loadmat(mat1)
Output = data['data_2']
x_train = np.vstack((x_train,Output))

mat1 = 'datasize64_3.mat'  
data = sio.loadmat(mat1)
Output = data['data_3']
x_train = np.vstack((x_train,Output))

mat1 = 'datasize64_4.mat'  
data = sio.loadmat(mat1)
Output = data['data_4']
x_train = np.vstack((x_train,Output))

mat1 = 'datasize64_val.mat'  
data = sio.loadmat(mat1)
Output = data['data']
x_test = Output

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = np.reshape(x_train, (len(x_train), 64, 64, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 64, 64, 1))  # adapt this if using `channels_first` image data format

#autoencoder.load_weights('D:\Liyu\Project1\Autoencoder\weights.hdf5')
autoencoder_callback = ModelCheckpoint("D:\Liyu\Project1\Autoencoder\weights32dim.hdf5", 
                monitor='val_loss', 
                verbose=1,
                save_best_only=True,
                save_weights_only=True,
                mode='auto',
                period=1
                )

autoencoder.fit(x_train, x_train,
                epochs=120,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks = [autoencoder_callback])

encoder = Model(input_img, encoded)
#
#from keras.utils import multi_gpu_model
#
##Replicates `model` on 8 GPUs.
## This assumes that your machine has 8 available GPUs.
#autoencoder = multi_gpu_model(autoencoder, gpus=2)

decoded_imgs = autoencoder.predict(x_test)
encoded_imgs = encoder.predict(x_test)

autoencoder.save_weights('autoencoder_weights_dim32.h5')

'''
import cv2
decoded_imgs_bi = np.zeros((len(x_test),60,60,1))
for i in range(len(x_test)):
    imin = decoded_imgs[i,:,:,0]
    imout = cv2.threshold(imin, 0.5, 1, cv2.THRESH_BINARY)
    imout = imout[1]
    decoded_imgs_bi[i,:,:,0] = imout 
    # 二值化处理
    '''
n = 10
plt.figure(figsize=(10, 4))
for i in range(n):
    # display original
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(x_test[i].reshape(64, 64))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(64, 64))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    '''
    # 二值化以后，画出二值化图形
    ax = plt.subplot(3, n, i + 1 + 2 * n)
    plt.imshow(decoded_imgs_bi[i].reshape(60, 60))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    '''
plt.show()

decoded_imgs = autoencoder.predict(x_test)

n = 8
plt.figure(figsize=(10, 4))
for i in range(n):
    ax = plt.subplot(1, n, i + 1)
    plt.imshow(encoded_imgs[i].reshape(16, 4 * 4).T)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

