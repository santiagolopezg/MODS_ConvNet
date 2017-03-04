'''Adaptation of Keras' 'conv_filter_visualization': 
Visualization of the filters of foo_two, via gradient ascent in input space.
'''


from __future__ import print_function
from scipy.misc import imsave
import numpy as np
import time
from foo_three import foo
from keras import backend as K

import random

#flags to determine what to do
viz=False

# dimensions of the generated pictures for each filter.
img_width = 224
img_height = 224

# the name of the layers we want to visualize - see model definition
#layer_list = ['conv1_2', 'conv2_3', 'conv3_2', 'conv4_2']
layer_name = 'conv4_2'

# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_dim_ordering() == 'th':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x



# build the network with best weights
model = foo()
weights='best_weights_3_santiago.h5'
model.load_weights(weights)

print('Model and weights loaded.')

model.summary()

# this is the placeholder for the input images
input_img = model.input

# get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)



def filter_viz():
	#make a list of 50 random filter indexes 
	randsample = random.sample(xrange(512), 10)

	kept_filters = []
	for filter_index in randsample:
	    # scanning 50 filters
	    print('Processing filter %d' % filter_index)
	    start_time = time.time()

	    # we build a loss function that maximizes the activation
	    # of the nth filter of the layer considered
	    layer_output = layer_dict[layer_name].output
	    if K.image_dim_ordering() == 'th':
		loss = K.mean(layer_output[:, filter_index, :, :])
	    else:
		loss = K.mean(layer_output[:, :, :, filter_index])

	    # we compute the gradient of the input picture wrt this loss
	    grads = K.gradients(loss, input_img)[0]

	    # normalization trick: we normalize the gradient
	    grads = normalize(grads)

	    # this function returns the loss and grads given the input picture
	    iterate = K.function([input_img,
			K.learning_phase()],
			[loss, grads])

	    # step size for gradient ascent
	    step = 1.

	    # we start from a gray image with some random noise
	    if K.image_dim_ordering() == 'th':
		input_img_data = np.random.random((1, 1, img_width, img_height))
	    else:
		input_img_data = np.random.random((1, img_width, img_height, 3))
	    input_img_data = (input_img_data - 0.5) * 20 + 128

	    # we run gradient ascent for 2000 steps
	    for i in range(100):
		loss_value, grads_value = iterate([input_img_data, 1])
		input_img_data += grads_value * step

		print('Current loss value:', loss_value)
		if loss_value <= 0.:
		    # some filters get stuck to 0, we can skip them
		    break

	    # decode the resulting input image
	    if loss_value > 0:
		img = deprocess_image(input_img_data[0])
		kept_filters.append((img, loss_value))
	    end_time = time.time()
	    print('Filter %d processed in %ds' % (filter_index, end_time - start_time))

	# we will stich the best 25 filters on a 5 x 5 grid.
	n = 2

	# the filters that have the highest loss are assumed to be better-looking.
	# we will only keep the top 64 filters.
	kept_filters.sort(key=lambda x: x[1], reverse=True)
	kept_filters = kept_filters[:n * n]

	# build a black picture with enough space for
	# our 8 x 8 filters of size 128 x 128, with a 5px margin in between
	margin = 5
	width = n * img_width + (n - 1) * margin
	height = n * img_height + (n - 1) * margin
	stitched_filters = np.zeros((width, height, 3))

	# fill the picture with our saved filters
	for i in range(n):
	    for j in range(n):
		img, loss = kept_filters[i * n + j]
		stitched_filters[(img_width + margin) * i: (img_width + margin) * i + img_width,
		                 (img_height + margin) * j: (img_height + margin) * j + img_height, :] = img
	return stitched_filters, n


def max_act(output_index):

	#find input that maximizes activation for specific class
	#generate 1000 random positive or negative images, and pick the best 49

	kept_imgs=[]

	for k in xrange(1000):
	    
	    # we build a loss function that maximizes the activation
	    # of the nth filter of the layer considered


		layer_output = model.layers[-1].output
		print (layer_output)
		loss = K.mean(layer_output[:, output_index])


		# we compute the gradient of the input picture wrt this loss
		grads = K.gradients(loss, input_img)[0]

		# normalization trick: we normalize the gradient
		grads = normalize(grads)

		# this function returns the loss and grads given the input picture
		iterate = K.function([input_img,
				K.learning_phase()],
				[loss, grads])

		# step size for gradient ascent
		step = 1.

		# we start from a gray image with some random noise
		if K.image_dim_ordering() == 'th':
		    input_img_data = np.random.random((1, 1, img_width, img_height))
		input_img_data = (input_img_data - 0.5) * 20 + 128

		# we run gradient ascent for 100 steps
		for i in range(100):
		    loss_value, grads_value = iterate([input_img_data, 1])
		    input_img_data += grads_value * step

		    print('Current loss value:', loss_value)
		    #if loss_value <= 0.:
			# some filters get stuck to 0, we can skip them
			#break

		# decode the resulting input image
		if loss_value > 0:
		    img = deprocess_image(input_img_data[0])
		    kept_imgs.append((img, loss_value))
		    end_time = time.time()


	# we will stich the best 49 images on a 7 x 7 grid.
	n = 7

	# the filters that have the highest loss are assumed to be better-looking.
	# we will only keep the top 64 filters.
	kept_imgs.sort(key=lambda x: x[1], reverse=True)
	kept_imgs = kept_imgs[:n * n]

	# build a black picture with enough space for
	# our 8 x 8 filters of size 128 x 128, with a 5px margin in between
	margin = 5
	width = n * img_width + (n - 1) * margin
	height = n * img_height + (n - 1) * margin
	stitched_imgs = np.zeros((width, height, 3))

	# fill the picture with our saved filters
	for i in range(n):
	    for j in range(n):
		img, loss = kept_imgs[i * n + j]
		stitched_imgs[(img_width + margin) * i: (img_width + margin) * i + img_width,
		                 (img_height + margin) * j: (img_height + margin) * j + img_height, :] = img

	return stitched_imgs, output_index

if viz:
	stitched_filters, n =filter_viz()
	# save the result to disk
	imsave('stitched_filters_{0}_{1}_%dx%d.png'.format(layer_name, weights) % (n, n), stitched_filters)

img,output_index = max_act(1)
imsave('max_activation_{0}_{1}.png'.format(output_index, weights),img)




