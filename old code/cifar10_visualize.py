'''
Code for cifar10_v1 network, and visualizing filters.
'''


from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, adadelta, rmsprop
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
import cPickle
import numpy as np


def cifar(weights='MODS_keras_weights_3_he_normal_0.5_rmsprop_24.h5'):

	nb_classes = 2

	#Hyperparameters for tuning
	weight_init = 'he_normal'
	dropout = 0.5

	# input image dimensions
	img_rows, img_cols = 256, 192

	# my images are images are greyscale
	img_channels = 1

	model = Sequential()

	model.add(Convolution2D(128, 3, 3,
		                input_shape=(img_channels, img_rows, img_cols), init=weight_init, name='conv1_1'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Convolution2D(128, 3, 3,init=weight_init, name='conv1_2'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(dropout))

	model.add(Convolution2D(256, 3, 3,init=weight_init, name='conv2_1'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Convolution2D(256, 3, 3,init=weight_init, name='conv2_2'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Convolution2D(256, 3, 3,init=weight_init, name='conv2_3'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))  
	model.add(Dropout(dropout))

	model.add(Convolution2D(512, 3, 3, init=weight_init, name='conv3_1'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Convolution2D(512, 3, 3, init=weight_init, name='conv3_2'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))  
	model.add(Dropout(dropout))

	model.add(Convolution2D(1024, 3,3,border_mode='same',init=weight_init, name='conv4_1'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Convolution2D(1024, 3,3,border_mode='same',init=weight_init, name='conv4_2'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))  
	model.add(Dropout(dropout))

	model.add(Flatten())
	model.add(Dense(120,init=weight_init))
	model.add(Activation('relu'))
	model.add(Dropout(dropout))

	model.add(Dropout(dropout))
	model.add(Dense(nb_classes))
	#model.add(Activation('softmax'))
	model.add(Activation('sigmoid'))

	model.load_weights(weights)
	
	return model




