# -*- coding: utf-8 -*-
'''
Created on Mon Sep  5 13:50:34 2016
cifar clone with bigger net
"""
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


def get_data(n_dataset):    
    f = file('MODS_dataset_cv_{0}.pkl'.format(n_dataset),'rb')
    data = cPickle.load(f)
    f.close()
    training_data = data[0]
    validation_data = data[1]
    t_data = training_data[0]
    t_label = training_data[1]
    v_data = validation_data[0]
    v_label = validation_data[1]
    
    t_data = np.array(t_data)
    t_label = np.array(t_label)
    v_data = np.array(v_data)
    v_label = np.array(v_label)
    t_data = t_data.reshape(t_data.shape[0], 1, 256, 192)
    v_data = v_data.reshape(v_data.shape[0], 1, 256, 192)
    
    #less precision means less memory needed: 64 -> 32 (half the memory used)
    t_data = t_data.astype('float32')
    v_data = v_data.astype('float32')
    
    return (t_data, t_label), (v_data, v_label)

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        
nb_classes = 2
nb_epoch = 100
data_augmentation = True
n_dataset = 5

#Hyperparameters for tuning
weight_init = 'he_normal' #['glorot_normal']
#regl1 = [1.0, 0.1, 0.01, 0.001, 0.0]
#regl2 = [1.0, 0.1, 0.01, 0.001, 0.0]
dropout = 0.5 #[0.0, 0.25, 0.5, 0.7]
batch_size = 24 #[32, 70, 100, 150]
learning_rate = 0.003 #[0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3]
#optimizer = ['sgd', 'adadelta']

# input image dimensions
img_rows, img_cols = 256, 192
# my images are images are greyscale
img_channels = 1


model = Sequential()

model.add(Convolution2D(128, 3, 3,
                        input_shape=(img_channels, img_rows, img_cols),
			activation='relu',
			#border_mode = 'same',
			init=weight_init, name='conv1_1'))
model.add(BatchNormalization())

model.add(Convolution2D(128, 3, 3,
                        activation='relu', 
                        #border_mode = 'same',
			init=weight_init, name='conv1_2'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(dropout))

model.add(Convolution2D(256, 3, 3,
                        activation='relu', 
                        #border_mode = 'same',
			init=weight_init, name='conv2_1'))
model.add(BatchNormalization())

model.add(Convolution2D(256, 3, 3,
                        activation='relu', 
                        #border_mode = 'same',
			init=weight_init, name='conv2_2'))
model.add(BatchNormalization())

model.add(Convolution2D(256, 3, 3,
                        activation='relu',
                        #border_mode = 'same',
			init=weight_init, name='conv2_3'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))  
model.add(Dropout(dropout))

model.add(Convolution2D(512, 3, 3,
                        activation='relu',
                        #border_mode = 'same',
			init=weight_init, name='conv3_1'))
model.add(BatchNormalization())

model.add(Convolution2D(512, 3, 3,
                        activation='relu',
                        #border_mode = 'same',
			init=weight_init, name='conv3_2'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))  
model.add(Dropout(dropout))

model.add(Convolution2D(1024, 3,3,
			activation='relu',
			#border_mode='same',
			init=weight_init, name='conv4_1'))

model.add(BatchNormalization())

model.add(Convolution2D(1024, 3,3,
			activation='relu',
			#border_mode='same',
			init=weight_init, name='conv4_2'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))  
model.add(Dropout(dropout))

model = Flatten()(model)
model.add(Dense(1000, activation='relu'))
model.add(Dropout(dropout))

model.add(Dropout(dropout))
model.add(Dense(nb_classes))
#model.add(Activation('softmax'))
model.add(Activation('sigmoid'))

for i in xrange(n_dataset):
    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = get_data(i)
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
    
    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    #optimize with adadelta
    model.compile(loss='binary_crossentropy', 
                 optimizer='rmsprop', #adadelta
                 metrics=['accuracy'])

    # let's train the model using SGD + momentum (how original).
    #sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    #model.compile(loss='binary_crossentropy',
    #              optimizer=sgd,
    #              metrics=['accuracy'])
                  
    history = LossHistory()

    X_train /= 255
    X_test /= 255
    
    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(X_train, Y_train,
                  batch_size=batch_size,
                  nb_epoch=nb_epoch,
                  validation_data=(X_test, Y_test),
                  shuffle=True)
    else:
        print('Using real-time data augmentation.')
    
        # this will do preprocessing and realtime data augmentation
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=180,  # randomly rotate images in the range (degrees, 0 to 180)
            #width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            #height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=True,  # randomly flip images
	    fill_mode='nearest')    

        # compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied)
        datagen.fit(X_train)
    
        # fit the model on the batches generated by datagen.flow()
        model.fit_generator(datagen.flow(X_train, Y_train,
                            batch_size=batch_size),
                            samples_per_epoch=X_train.shape[0],
                            nb_epoch=nb_epoch,
                            validation_data=(X_test, Y_test))
                            
                            
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

model.reset_states()

#import matplotlib.pylab as plt
#plt.plot(history.losses,'bo')
#plt.xlabel('Iteration')
#plt.ylabel('loss')
#plt.show()
