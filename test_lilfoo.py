import keras
from keras.optimizers import SGD, adadelta, rmsprop, adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.metrics import matthews_correlation, precision, recall
import keras.backend as K

import cPickle
import numpy as np

import getpass
username = getpass.getuser()

from little_foo2 import foo


def sens(y_true, y_pred):

	y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    	y_pred_neg = 1 - y_pred_pos

   	y_pos = K.round(K.clip(y_true, 0, 1))
   	y_neg = 1 - y_pos

   	tp = K.sum(y_pos * y_pred_pos)
   	tn = K.sum(y_neg * y_pred_neg)

  	fp = K.sum(y_neg * y_pred_pos)
  	fn = K.sum(y_pos * y_pred_neg)

	se = tp / (tp + fn)
	return se

def spec(y_true, y_pred):

	y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    	y_pred_neg = 1 - y_pred_pos

   	y_pos = K.round(K.clip(y_true, 0, 1))
   	y_neg = 1 - y_pos

   	tp = K.sum(y_pos * y_pred_pos)
   	tn = K.sum(y_neg * y_pred_neg)

  	fp = K.sum(y_neg * y_pred_pos)
  	fn = K.sum(y_pos * y_pred_neg)

	sp = tn / (fp + tn)
	return sp



def get_weights(n_dataset):
    weights='best_weights_lilfoo_{0}_{1}.h5'.format(i,'santiago')
    model = foo()
    model.load_weights(weights)
    print ('weights loaded')
    return model
    

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

def test_net(i):

	model = get_weights(i)
	print 'using weights from net trained on dataset {0}'. format(i)
	history = LossAccHistory()

    	(X_train, y_train), (X_test, y_test) = get_data(i)

    	Y_test = np_utils.to_categorical(y_test, nb_classes)

    	X_test /= 255

    	print(X_test.shape[0], 'test samples')
 
    	model.compile(loss='binary_crossentropy', 
                 optimizer= rmsprop(lr=0.001), #adadelta
		 metrics=['accuracy', 'matthews_correlation', 'precision', 'recall', sens, spec])
          
    	score = model.evaluate(X_test, Y_test, verbose=1)

    	print (model.metrics_names, score)

    	if (len(cvscores[0])==0): #if metric names haven't been saved, do so
		cvscores[0].append(model.metrics_names)
    	else:
		counter = 1
		for k in score: #for each test metric, append it to the cvscores list
			cvscores[counter].append(k)
			counter +=1

    	model.reset_states()


def cv_calc():
#calculate mean and stdev for each metric, and append them to test_metrics file
	test_metrics.append(cvscores[0])

	other_counter = 0
	for metric in cvscores[1:]:
        	v = 'test {0}: {1:.4f} +/- {2:.4f}%'.format(cvscores[0][0][other_counter], np.mean(metric), np.std(metric))
        	print v
		test_metrics.append(v)
		other_counter +=1
		if other_counter == 7:
			other_counter=0
	return cvscores, test_metrics

def save_metrics(cvscores, test_metrics):
#save test metrics to txt file
	file = open('MODS_lilfoo_test_metrics.txt', 'w')
	for j in cvscores:
		file.write('\n%s\n' % j)
	for i in test_metrics:
		file.write('\n%s\n' % i)
	file.close()

	print test_metrics


class LossAccHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accu = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accu.append(logs.get('acc'))
        
nb_classes = 2
nb_epoch = 100
n_dataset = 5

dropout = 0.5
batch_size = 72
optimizer = 'rmsprop' 
test_metrics = []
cvscores = [[],[],[],[],[],[], [], []]
#cvscores = [[metrics],[loss],[acc],[mcc],[precision],[recall], [sens], [spec]]

	
for i in xrange(n_dataset):
	test_net(i)
cvscores, test_metrics = cv_calc()
print cvscores, test_metrics
save_metrics(cvscores, test_metrics)









































