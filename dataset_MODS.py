
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

	def flipupdown(self, current_image, label):
	## transformation that flips the image on the x axis
		flip_image = numpy.flipud(current_image)
		images = numpy.hstack(flip_image)
		self.training_data.append(images)
		self.training_label.append(label)

	def flipleftright(self, current_image,label):
	## transformation that flips the image on the y axis
		current_image = numpy.fliplr(current_image)
		images = numpy.hstack(current_image)
		self.training_data.append(images)
		self.training_label.append(label)


	def DSetGlobal(self, directory = '/home/musk/tb-CNN/data/shuffled'):
	## Find files in folders
         
         foldername = next(os.walk(directory))[1]
         for dirname in foldername: 
         ##dirname: positive and negative
             f2 = os.path.join(directory,dirname)
             onlyfiles = [ f3 for f3 in os.listdir(f2) if os.path.isfile(os.path.join(f2,f3))]		
             suffix = dirname
             if suffix == 'positive':
			label = 1
             else:
			label = 0
			#Con esta linea consigo cada imagen independientemente
             for filename in onlyfiles:
                 try:
                     current_image = scipy.misc.imread(os.path.join(f2,filename), mode='L')
                     current_image = scipy.misc.imresize(current_image,(256, 192),interp='cubic')
                     self.data.append(numpy.hstack(current_image))
                     self.data_label.append(label)
                 except IOError:
                     print(filename)
                 #scipy.misc.imshow(current_image) ##shows the image being read 

           
         combined = zip(self.data, self.data_label)
         random.shuffle(combined)
         self.data[:], self.data_label[:] = zip(*combined)
         print len(self.data)
         dataset = [self.data, self.data_label]
         f = file('MODS_data.pkl','wb')
         cPickle.dump(dataset, f, protocol=cPickle.HIGHEST_PROTOCOL)
         f.close()
         print(datetime.datetime.now() - self.start_time)

	def Dset(self, ndataset=5):
         x = self.data_label
         w = self.data
         y = range(len(x))
         seg_data = []
         counter = 0
         while counter < ndataset:
             z = random.sample(y, int(len(y)/5.0))
             lmda = 0.05
             ratio = float(sum(z))/(float(len(z)*10000))
             dif = math.fabs(ratio-0.621883)
             if dif < lmda:
                 del y[z]
                 current_label = x[z]
                 current_data = w[z]
                 seg_data.append([current_data, current_label])
                 counter+=1
             else:
                 print('Does not have a acceptable ratio', ratio, dif)
         f = file('seg_MODS_data.pkl', 'wb')
         cPickle.dump(seg_data, f, protocol=cPickle.HIGHEST_PROTOCOL)
         f.close()

'''

directory = '/home/musk/tb-CNN/data/shuffled'
for dirname in (next(os.walk('.'))[1]): ##dirname: positive and negative
	f2 = os.path.join(directory,dirname)
	onlyfiles = [ f3 for f3 in os.listdir(f2) if os.path.isfile(os.path.join(f2,f3)) ]		
	suffix = dirname
	if suffix == 'positive':
		label = 1
	else:
		label = 0
	#Con esta linea consigo cada imagen independientemente
	for filename in onlyfiles:
		current_image = scipy.misc.imread(os.path.join(f2,filename))
		current_image = current_image
		training_data.append(numpy.hstack(current_image))
		training_label.append(label)

		flip_image = numpy.flipud(current_image)
		images = numpy.hstack(flip_image)
		training_data.append(images)
		training_label.append(label)

		current_hori_image = numpy.fliplr(current_image)
		images = numpy.hstack(current_hori_image)
		training_data.append(images)
		training_label.append(label)
		
		flip_hori_image = numpy.fliplr(flip_image)
		images = numpy.hstack(flip_hori_image)
		training_data.append(images)
		training_label.append(label)
		trans_new = scipy.misc.imresize(current_image,(462,462),interp='cubic')

for subdirname in validation_folder:
	f2 = os.path.join(directory,dirname,subdirname)
	onlyfiles = [ f3 for f3 in os.listdir(f2) if os.path.isfile(os.path.join(f2,f3)) ]		
	suffix = f2[len(f2)-2:len(f2)]
	if suffix == 'Ne':
		label = 1
	else:
		label = 0
	for filename in onlyfiles:
		current_image = scipy.misc.imread(os.path.join(f2,f3))
		current_image = current_image/255.0
		validation_data.append(numpy.hstack(current_image))
		validation_label.append(label)
for subdirname in test_folder:
	f2 = os.path.join(directory,dirname,subdirname)
	onlyfiles = [ f3 for f3 in os.listdir(f2) if os.path.isfile(os.path.join(f2,f3)) ]		
	suffix = f2[len(f2)-2:len(f2)]
	if suffix == 'Ne':
		label = 1
	else:
		label = 0
	for filename in onlyfiles:
		current_image = scipy.misc.imread(os.path.join(f2,f3))
		current_image = current_image/255.0
		test_data.append(numpy.hstack(current_image))
		test_label.append(label)

combined = zip(training_data, training_label)
random.shuffle(combined)
training_data[:], training_label[:] = zip(*combined)

print len(training_data)


neumonia_dataset = [training_data, training_label],[validation_data, validation_label],[test_data,test_label]

f = file('neumonia_dataset_new_11.pkl','wb')
cPickle.dump(neumonia_dataset, f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()
'''
