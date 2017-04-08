from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten
from keras.optimizers import SGD
from keras import regularizers
from keras.callbacks import ModelCheckpoint

import numpy as np
import scipy as sp
import scipy.io as sio
import h5py

def DeepSEA():
    nkernels = [320,480,960]
    in_size = (1,1000,4)
    l2_lam = 5e-07 
    l1_lam = 1e-08 

    model = Sequential()
    model.add(Conv2D(nkernels[0], kernel_size=(1,8), strides=(1,1), padding='same', input_shape=in_size, kernel_regularizer=regularizers.l2(l2_lam)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(1,4), strides=(1,4)))
    model.add(Dropout(0.2))
    
    #expecting 
    #(4,250,320) here

    model.add(Conv2D(nkernels[1], kernel_size=(1,8), strides=(1,1), padding='same', kernel_regularizer=regularizers.l2(l2_lam)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(1,4), strides=(1,4)))
    model.add(Dropout(0.2))

    model.add(Conv2D(nkernels[1], kernel_size=(1,8), strides=(1,1), padding='same', kernel_regularizer=regularizers.l2(l2_lam)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(919, kernel_regularizer=regularizers.l1(l1_lam)))
    model.add(Activation('relu'))
    model.add(Dense(919, kernel_regularizer=regularizers.l1(l1_lam)))
    model.add(Activation('sigmoid'))
    
    return model

def loaddata(path="../deepsea_train"):
    valid = sio.loadmat(path+"/valid.mat")
    train = h5py.File(path+"/train.mat")
    test = sio.loadmat(path+"/test.mat")
    
    valid_X = valid["validxdata"]
    valid_X = np.expand_dims(valid_X, 3)
    valid_X = np.transpose(valid_X, axes=(0,3,2,1))
    valid_Y = valid["validdata"]

    train_X = train["trainxdata"][()]
    train_X = np.expand_dims(train_X, 3)
    train_X = np.transpose(train_X, axes=(2,1,0,3))
    train_X = np.transpose(train_X, axes=(0,3,2,1))
    train_Y = np.transpose(train["traindata"][()])
    
    test_X = test["testxdata"]
    test_X = np.expand_dims(test_X, 3)
    test_X = np.transpose(test_X, axes=(0,3,2,1))
    test_Y = test["testdata"]
    
    return valid_X, valid_Y, train_X, train_Y, test_X, test_Y