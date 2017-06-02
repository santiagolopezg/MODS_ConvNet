"""
code to build dataset for ConvNet training on Tuberculosis MODS images.
Training dataset will have transformations, labels.
Dataset can be used in Theano or Keras.

Updated to use UPCH images
"""

#import libraries
import numpy
import scipy as scipy
import os, glob
import scipy.misc
import cPickle
import random
import math
import datetime
#from skimage import transform as tf
#import gc


class dataset:
	def __init__(self):
	## creates lists for each Dset instance
		self.training_data = []
		self.training_label = []
		self.validation_data = []
		self.validation_label = []
		self.test_data = []
		self.test_label = []
		self.data = []
		self.data_label = []
		self.start_time = datetime.datetime.now()
		self.ndataset = 5


	def DSetGlobal(self, directory = '/home/musk/MODS_data/data/by_lab/'):
          '''
          Function to build a rough dataset of images with labels.
          Returns a pkl file with data and data_labels.
          '''
          ## Find files in folders
	  for v in os.listdir(directory):
		  self.training_data = []
		  self.training_label = []
		  self.validation_data = []
		  self.validation_label = []
		  self.test_data = []
		  self.test_label = []
		  self.data = []
	  	  self.data_label = []
		  print v, directory+v
		  foldername = next(os.walk(directory+v))[1]
		  print foldername
		  #exit()
		  for dirname in foldername: 
		  ##dirname: positive and negative
		     print datetime.datetime.now()
		     f2 = os.path.join(directory+v,dirname)
		     onlyfiles = [ f3 for f3 in os.listdir(f2) if os.path.isfile(os.path.join(f2,f3))]		
		     suffix = dirname
		     if suffix == 'positive':
				label = 1
		     else:
				label = 0
		     for filename in onlyfiles:
		         try: ##reads the image, converts to greyscale, resizes it, appends it to data and adds label too
		             current_image = scipy.misc.imread(os.path.join(f2,filename), mode='L')
		             #current_image = scipy.misc.imread(os.path.join(f2,filename), mode='RGB')
		             #current_image = scipy.misc.imresize(current_image,(256, 192),interp='cubic')
		             current_image = scipy.misc.imresize(current_image,(224,224),interp='bicubic')
		             self.data.append(numpy.hstack(current_image))
		             self.data_label.append(label)
		         except IOError: ##If the image can't be read, or is corrupted
		             print(filename)
		         #scipy.misc.imshow(current_image) ##shows the image being read 
		   
		  ## shuffles de images with their label
		  combined = zip(self.data, self.data_label)
		  random.shuffle(combined)
		  self.data[:], self.data_label[:] = zip(*combined)
		  
		  print len(self.data)

		  dataset = [self.data, self.data_label]
		  f = file('MODS_data_{0}.pkl'.format(v),'wb') ##save images in a pkl
		  cPickle.dump(dataset, f, protocol=cPickle.HIGHEST_PROTOCOL)
		  f.close()
		  print(datetime.datetime.now() - self.start_time)

	def Dset(self, v, ndataset=5, name='MODS_data.pkl'):
         '''
         function to build datasets. ndataset: number of datasets wanted; 
         name: pkl file where the data from DSetGlobal is stored. Code makes sure
         that there is the same ratio of positive/negative images in each dataset.
         This is done, setting a lmda. If you set a really low lmda, you might have
         to stop the program and rerun it a few times.
         Returns a pkl with a segmented dataset. seg_data is a list of n lists, where n
         is the number of datasets desired. These n lists consist of 2 lists: the data
         and its corresponding labels.
         '''
	 ratios = {'train_1': 0.62234,
		'train_2': 0.8499,
		'train_3': 0.53817,
		'test_1': 0.8881987,
		'test_2': 0.51543,
		'test_3': 0.84473
		   }
         f = file(name, 'rb')
         datapapa = cPickle.load(f)
         f.close()    
         w = datapapa[0]
         x = datapapa[1]
         y = range(len(x))
         seg_data = []
         counter = 0
         size = int(len(y)/float(ndataset))
	 print size, 'gamboozle'
         while counter < ndataset:
             z = random.sample(y, size)
             lmda = 0.02
             ratio = float(sum([x[i] for i in z]))/(len([x[i] for i in z if x[i]==0]))
             print(ratio), v
	     #exit()
             dif = math.fabs(ratio-ratios[v]) #ratio of positive to negatives
             if dif < lmda:
                 print('BINGO!', counter, dif)
                 y = [i for i in y if i not in z]
                 current_label = [x[i] for i in z]
                 current_data = [w[i] for i in z]
                 seg_data.append([current_data, current_label])
                 counter+=1
             else:
                 #print('Does not have a acceptable ratio', ratio, dif)
                 #fun+= 1
                 pass 
         f = file('seg_MODS_data_{0}.pkl'.format(v), 'wb')
         cPickle.dump(seg_data, f, protocol=cPickle.HIGHEST_PROTOCOL)
         f.close()
         
	def Djoin(self, v, name='seg_MODS_data.pkl'):
         '''
         Takes as input segmented data from the Dset function. The data is split into
         training and testing. Each list (dataset) in the segmented data is taken,
         once, as the testing set. Then, the rest of the data is shuffled, and put
         into the testing set. Therefore, for each dataset, we have a different testing
         set of images, with also a different set of training images, shuffled twice.     
         Returns n datasets (same amount as in Dset). The datasets are made of two lists:
         training and testing. These lists are made of two lists each: data and labels.
         '''
         f = file(name, 'rb')
         datamama = cPickle.load(f)
         f.close()
         for i in xrange(len(datamama)):
             data_join = []
             data_label_join = []
	     #if 'test' in v:
             validation = datamama[i]
             data_temp = datamama[:i] + datamama[i+1:]
	     #else:
		#validation = []
		#data_temp = datamama[:]
             for j in data_temp:
                 data_join+=j[0]
                 data_label_join+=j[1]
             
             ##Shuffle data
             combined = zip(data_join, data_label_join)
             random.shuffle(combined)
             data_join[:], data_label_join[:] = zip(*combined)                 
            
             training = [data_join,data_label_join]
             dataset_new = [training,validation]
             f = file('MODS_224_224_{0}_{1}.pkl'.format(i, v),'wb')
	     print len(validation), v
             cPickle.dump(dataset_new, f, protocol=cPickle.HIGHEST_PROTOCOL)
             f.close()


directory = '/home/musk/MODS_data/data/by_lab/'
a = dataset()
#a.DSetGlobal()

for v in os.listdir(directory):	
	#a.Dset(v, name='MODS_data_{0}.pkl'.format(v))
	a.Djoin(v, name = 'seg_MODS_data_{0}.pkl'.format(v))

