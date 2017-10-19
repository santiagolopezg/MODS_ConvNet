'''
foo three
'''

from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, adadelta, rmsprop
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
import cPickle
import numpy as np

dropout = 0.5
weight_init='he_normal'

def foo():

    # Determine proper input shape
	if keras.__version__ > '1.0.3':
		K.set_image_dim_ordering('th')
	input_shape = (1, 224, 224)

	#img_input = Input(shape=input_shape)

	model = Sequential()

	model.add(Convolution2D(32, 8, 8,
			        input_shape=input_shape,init=weight_init, name='conv1_1'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Convolution2D(32, 6, 6,init=weight_init, name='conv1_2'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Convolution2D(32, 4, 4,init=weight_init, name='conv1_3'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Convolution2D(32, 2, 2,init=weight_init, name='conv1_4'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2))) # in 208, out 104
	model.add(Dropout(dropout))

	model.add(Convolution2D(64, 8, 8,init=weight_init, name='conv2_1'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Convolution2D(64, 6, 6,init=weight_init, name='conv2_2'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Convolution2D(64, 4, 4,init=weight_init, name='conv2_3'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Convolution2D(64, 2, 2,init=weight_init, name='conv2_4'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2))) # in is 88, out is 44 
	model.add(Dropout(dropout))

	model.add(Flatten())
	model.add(Dense(220, init=weight_init))
	model.add(Activation('relu'))
	model.add(Dropout(dropout))

	model.add(Dense(2))
	model.add(Activation('sigmoid'))

	return model


