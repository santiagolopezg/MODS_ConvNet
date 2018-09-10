import math
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

from foo_three import foo

def contingency(y_true, y_pred):
	tp = 0
	tn = 0
	fp = 0
	fn = 0
	for i in xrange(len(y_true)):
		if y_true[i] == 0: #if true label is negative:
			if y_pred[i] == y_true[i]: #if pred label = true label = neg:
				tn += 1 #its truly negative
			elif y_pred[i] == 1: # if pred label is pos, but real is neg:
				fp += 1 #its a false positive 
		elif y_true[i] == 1: #if true label is possy:
			if y_pred[i] == y_true[i]: # if pred label = true label = possy:
				tp += 1
			elif y_pred[i] == 0: # if pred label is neg but real is possy:
				fn += 1
	return tp, tn, fp, fn

def get_weights(i, name):
    weights='best_weights_labcrossval_{0}_train_1_{1}.h5'.format(i, username)
    model = foo()
    model.load_weights(weights)
    print ('weights loaded')
    return model
    

def get_data(n_dataset, name):    
    f = file('MODS_224_224_{0}_{1}.pkl'.format(n_dataset, name),'rb')
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
    
    t_data = t_data.astype('float32')
    test_data = test_data.astype('float32')
    
    return (t_data, t_label), (test_data, test_label)

def test_net(i, name):

	model = get_weights(i, name)
	print 'using weights from net trained on dataset {0} for {1}'. format(i, name)
	history = LossAccHistory()

    	(X_train, y_train), (X_test, y_test) = get_data(i, name)

    	Y_test = np_utils.to_categorical(y_test, nb_classes)

    	X_test /= 255

    	print(X_test.shape[0], 'test samples')
 
    	model.compile(loss='binary_crossentropy', 
                 optimizer= rmsprop(lr=0.001), #adadelta
		 metrics=['accuracy', 'matthews_correlation', 'precision', 'recall', sens, spec])
        
	ypred = model.predict_classes(X_test, verbose=1)
	ytrue = Y_test	

	
	tp, tn, fp, fn = contingency(y_test, ypred)

	print '           |     true label\n---------------------------------'
	print 'pred label |  positive | negative'
	print 'positive   |     ', tp, ' |  ', fp
	print 'negative   |     ', fn, '  |  ', tn 

	prec = float(tp)/(tp+fp)
	se = float(tp) / (tp + fn)
	sp = float(tn) / (fp + tn)
	mcc = float(tp*tn - tp*fn)/(math.sqrt((tp + fp)*(tp+fn)*(tn+fp)*(tn+fn)))
	f1 = (2*prec*se)/(prec+se)
	acc = float(tp+tn)/(tp+tn+fp+fn)
	print '     sens     |     spec     |     mcc      |      f1      |      prec      |     acc       '
	print se, sp, mcc, f1, prec, acc

    	model.reset_states()
	return [se, sp, mcc, f1, prec, acc]


def cv_calc(cvscores):
#calculate mean and stdev for each metric, and append them to test_metrics file
	test_metrics.append(cvscores[0])

	other_counter = 0
	for metric in cvscores[1:]:
        	v = 'test {0}: {1:.4f} +/- {2:.4f}%'.format(cvscores[0][other_counter], np.mean(metric), np.std(metric))
        	print v
		test_metrics.append(v)
		other_counter +=1
		if other_counter == 6:
			other_counter=0
	return cvscores, test_metrics

def save_metrics(cvscores, test_metrics):
#save test metrics to txt file
	file = open('MODS_test_metrics_labscrossval.txt', 'w')
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

for name in ['test_1', 'test_2', 'test_3']:
	manualcalc = [['sens', 'spec', 'mcc', 'f1', 'prec', 'acc'], [], [], [], [], [], []]
	#manualcalc = [[metrics], [sens], [spec], [mcc], [f1], [prec], [acc]]
	for i in xrange(n_dataset): # 5 datasets	
		scorez = test_net(i, name)
		for i in xrange(len(manualcalc[1:])):
			#print i
			manualcalc[i+1].append(scorez[i])
	print manualcalc
	cvscores, test_metrics = cv_calc(manualcalc)
	print 'test set ', name
	print cvscores, test_metrics
