# -*- coding: utf-8 -*-
'''
train network
'''

import keras
from keras.optimizers import SGD, adadelta, rmsprop, adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.metrics import matthews_correlation, precision, recall
from keras.callbacks import ModelCheckpoint

import cPickle
import numpy as np

import getpass
username = getpass.getuser()

from little_foo import foo


def get_data(n_dataset):    
    f = file('MODS_all_data_bw_224_224_{0}.pkl'.format(n_dataset),'rb')
    data = cPickle.load(f)
    f.close()
    training_data = data[0]
    validation_data = data[1]
    t_data = training_data[0]
    t_label = training_data[1]
    test_data = validation_data[0]
    test_label = validation_data[1]
    
    t_data = np.array(t_data)
    t_label = np.array(t_label)
    test_data = np.array(test_data)
    test_label = np.array(test_label)
    t_data = t_data.reshape(t_data.shape[0], 1, 224, 224)
    test_data = test_data.reshape(test_data.shape[0], 1, 224, 224)
    
    #less precision means less memory needed: 64 -> 32 (half the memory used)
    t_data = t_data.astype('float32')
    test_data = test_data.astype('float32')
    
    return (t_data, t_label), (test_data, test_label)

class LossAccHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accu = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accu.append(logs.get('acc'))
        
nb_classes = 2
nb_epoch = 100
data_augmentation = True
n_dataset = 5
plot_loss = True

#Hyperparameters for tuning

dropout = 0.5 #[0.0, 0.25, 0.5, 0.7]
batch_size = 72 #[32, 70, 100, 150]
optimizer = 'rmsprop' #['sgd', 'adadelta']
test_metrics = []

for i in xrange(n_dataset):
    #call a model instance
    model = foo()
    #try to load weights corresponding to model trained on this dataset
    try:
	weights='best_weights_lilfoo_{0}_{1}.h5'.format(i,username)
	model.load_weights(weights)
	print ('weights loaded')
    except:
	print ('no weights to load')

    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = get_data(i)

    #take 15% of the training data for validation
    split=int(0.85*len(X_train))
    
    X_val = X_train[split:]
    X_train = X_train[:split]

    y_val = y_train[split:]
    y_train= y_train[:split]
    
    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_val = np_utils.to_categorical(y_val, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    history = LossAccHistory()

    checkpoint = ModelCheckpoint('best_weights_lilfoo_{0}_{1}.h5'.format(i, username), monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    X_train /= 255
    X_val /= 255
    X_test /= 255

    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_val.shape[0], 'validation samples')
    print(X_test.shape[0], 'test samples')
    
    #data augmentation generator for training, with desired settings
    datagen = ImageDataGenerator(
	featurewise_center=False,  # set input mean to 0 over the dataset
	samplewise_center=False,  # set each sample mean to 0
	featurewise_std_normalization=False,  # divide inputs by std of the dataset
	samplewise_std_normalization=False,  # divide each input by its std
	zca_whitening=False,  # apply ZCA whitening
	rotation_range=180,  # randomly rotate images in the range (degrees, 0 to 180)
	width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
	height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
	horizontal_flip=True,  # randomly flip images
	vertical_flip=True,  # randomly flip images
	fill_mode='nearest')  
    datagen.fit(X_train)

    #Shows all layers and names
    for v, layer in enumerate(model.layers):
	print(v, layer.name)

    print('Training of the network, using real-time data augmentation.')
 
    model.compile(loss='binary_crossentropy', 
                 optimizer= rmsprop(lr=0.001), #adadelta
		 metrics=['accuracy', 'matthews_correlation', 'precision', 'recall'])

    # fit the model on the batches generated by datagen.flow()
    model.fit_generator(datagen.flow(X_train, Y_train,
            batch_size=batch_size),
            samples_per_epoch=X_train.shape[0],
            nb_epoch=nb_epoch,
            validation_data=(X_val, Y_val), 
	    nb_val_samples=X_val.shape[0],
	    callbacks=[history, checkpoint])

    print('Finished training network.')
                    
    if plot_loss:
	import matplotlib.pylab as plt
	plt.plot(history.losses,'-k', label='loss')
	plt.xlabel('Iteration')
	plt.ylabel('loss on dataset {0}'.format(i))
	plt.savefig('loss_dset_{0}'.format(i))
	plt.clf()

	plt.plot(history.accu, '-b', label='accuracy')
	plt.ylabel('accuracy on dataset {0}'.format(i))
	plt.xlabel('epoch')
	plt.savefig('acc_dset_{0}'.format(i))
	plt.clf()

    model.reset_states()






