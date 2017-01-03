"""
code to build dataset for ConvNet training on Tuberculosis MODS images.
Training dataset will have transformations, labels.
Dataset can be used in Theano or Keras.
"""

#import libraries
import numpy
import scipy as scipy
import os
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


	def DSetGlobal(self, directory = '/home/musk/MODS_data/data/shuffled'):
          '''
          Function to build a rough dataset of images with labels.
          Returns a pkl file with data and data_labels.
          '''
          ## Find files in folders
          foldername = next(os.walk(directory))[1]
          for dirname in foldername: 
          ##dirname: positive and negative
	     print datetime.datetime.now()
             f2 = os.path.join(directory,dirname)
             onlyfiles = [ f3 for f3 in os.listdir(f2) if os.path.isfile(os.path.join(f2,f3))]		
             suffix = dirname
             if suffix == 'positive':
			label = 1
             else:
			label = 0
             for filename in onlyfiles:
                 try: ##reads the image, converts to greyscale, resizes it, appends it to data and adds label too
                     #current_image = scipy.misc.imread(os.path.join(f2,filename), mode='L')
                     current_image = scipy.misc.imread(os.path.join(f2,filename), mode='RGB')
                     #current_image = scipy.misc.imresize(current_image,(256, 192),interp='cubic')
                     current_image = scipy.misc.imresize(current_image,0.25,interp='bicubic')
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
          f = file('MODS_data.pkl','wb') ##save images in a pkl
          cPickle.dump(dataset, f, protocol=cPickle.HIGHEST_PROTOCOL)
          f.close()
          print(datetime.datetime.now() - self.start_time)

	def Dset(self, ndataset=5, name='MODS_data.pkl'):
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
         f = file(name, 'rb')
         datapapa = cPickle.load(f)
         f.close()    
         w = datapapa[0]
         x = datapapa[1]
         y = range(len(x))
         seg_data = []
         counter = 0
         size = int(len(y)/float(ndataset))
         while counter < ndataset:
             z = random.sample(y, size)
             lmda = 0.005
             ratio = float(sum([x[i] for i in z]))/(len([x[i] for i in z if x[i]==0]))
             print(ratio)
             dif = math.fabs(ratio-0.6218)
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
         f = file('seg_MODS_data.pkl', 'wb')
         cPickle.dump(seg_data, f, protocol=cPickle.HIGHEST_PROTOCOL)
         f.close()
         
	def Djoin(self, name='seg_MODS_data.pkl'):
         '''
         Takes as input segmented data from the Dset function. The data is split into
         training and validation. Each list (dataset) in the segmented data is taken,
         once, as the validation set. Then, the rest of the data is shuffled, and put
         into the training set. Therefore, for each dataset, we have a different validation
         set of images, with also a different set of training images, shuffled twice.     
         Returns n datasets (same amount as in Dset). The datasets are made of two lists:
         training and validation. These lists are made of two lists each: data and labels.
         '''
         f = file(name, 'rb')
         datamama = cPickle.load(f)
         f.close()
         for i in xrange(len(datamama)):
             data_join = []
             data_label_join = []
             validation = datamama[i]
             data_temp = datamama[:i] + datamama[i+1:]
             for j in data_temp:
                 data_join+=j[0]
                 data_label_join+=j[1]
             
             ##Shuffle data
             combined = zip(data_join, data_label_join)
             random.shuffle(combined)
             data_join[:], data_label_join[:] = zip(*combined)                 
            
             training = [data_join,data_label_join]
             dataset_new = [training,validation]
             f = file('MODS_dataset_rgb_0.25%_{0}.pkl'.format(i),'wb')
             cPickle.dump(dataset_new, f, protocol=cPickle.HIGHEST_PROTOCOL)
             f.close()


a = dataset()
a.DSetGlobal()
a.Dset()
a.Djoin()

'''       
	def aug(self, current_image, label, deg):
         gc.enable
         data = []
         data_label = []
         flip_image = numpy.flipud(current_image)
         data.append(numpy.hstack(flip_image))         
         a = numpy.fliplr(flip_image)
         b = numpy.fliplr(current_image)
         data.append(numpy.hstack(a))
         data.append(numpy.hstack(b))
         data_label += 3*[label]
         
         for i in xrange(int(360/deg -1)):
             data.append(numpy.hstack(tf.rotate(current_image, deg)))
             data.append(numpy.hstack(tf.rotate(flip_image, deg)))
             data.append(numpy.hstack(tf.rotate(a, deg)))
             data.append(numpy.hstack(tf.rotate(b, deg)))
             data_label += 4*[label]
         gc.collect()
         return data, data_label

  
	def data_augment(self, deg=90.0):
         for i in xrange(self.ndataset):
             gc.enable
             f = file('MODS_dataset_cv_{0}.pkl'.format(i),'rb')
             data = cPickle.load(f)
             training = data[0]
             f.close()
             for j in xrange(len(training[0])):
                 current_image = numpy.reshape(training[0][j], (256, 192))
                 label = training[1][j]
                 images, labels = self.aug(current_image, label, deg)
                 training[0] += images
                 training[1] += labels
                 gc.collect()
                 print(j)
             f = file('MODS_dataset_cv_aug_{0}.pkl'.format(i),'wb')
             dataset_new = [training, data[1]]
             cPickle.dump(dataset_new, f, protocol=cPickle.HIGHEST_PROTOCOL)
             f.close()
             gc.collect()
             print('Finished dataset {0}'.format(i))
                                 
        
    		#scipy.misc.imshow(current_image) ##shows the image being read
		#

'''         
