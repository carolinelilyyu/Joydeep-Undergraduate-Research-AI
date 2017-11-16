from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
from random import randint
import numpy as np

import argparse
import sys
import tempfile

FLAGS = None



folders = ["O", "X"]
datasetFolder = "train"
testFolder = "letters"

def getNumber(letter):
	if(letter == "O"):
		#returns the first row of a 2-d array with ones in the diagonal and zeros elsewhere
		return np.eye(2, dtype=np.float32)[0]
	if(letter == "X"):
		return np.eye(2, dtype=np.float32)[1]

def getListOfImages():
	global folders
	global datasetFolder
	allImagesArray = np.array([], dtype=np.str)
	allImagesLabelsArray = np.array([], dtype=np.str)

	for folder in folders:
		print("Loading Image Name of ", folder)
		currentLetterFolder = datasetFolder+"/"+folder+"/"
		#listdir returns the list containing the names of the entries in the directory given by path
		imagesName = os.listdir(currentLetterFolder)
		allImagesArray = np.append(allImagesArray, imagesName)
		for i in range(0, len(imagesName)):
			print(i)
			#100 images in each folder
			if(i % 100 == 0):
				print("progress -> ", i)
			allImagesLabelsArray = np.append(allImagesLabelsArray, currentLetterFolder)
		#print(allImagesArray)
	return allImagesArray, allImagesLabelsArray

def getListofTestImages():
	global testFolder
	global folders
	allImagesArray = np.array([], dtype=np.str)
	allImagesLabelsArray = np.array([], dtype=np.str)

	for folder in folders:
		print("Loading Test Images of ", folder)
		currentLetterFolder = testFolder + "/" + folder + "/"
		imagesName = os.listdir(currentLetterFolder)
		allImagesArray = np.append(allImagesArray, imagesName)
		for i in range(0, len(imagesName)):
			print(i)
			if(i%20 == 0):
				print("progress -> ", i)
			allImagesLabelsArray = np.append(allImagesLabelsArray, currentLetterFolder)
	return allImagesArray, allImagesLabelsArray

def shuffleImagesPath(imagesPathArray, imagesLabelsArray):
	print("Size of imagesPathArray is ", len(imagesPathArray))
	for i in range(0, 100000):
		if(i % 1000 == 0):
			print("Shuffling in progress -> ", i)
		randomIndex1 = randint(0, len(imagesPathArray)-1)
		randomIndex2 = randint(0, len(imagesPathArray)-1)
		imagesPathArray[randomIndex1], imagesPathArray[randomIndex2] = imagesPathArray[randomIndex2], imagesPathArray[randomIndex1]
		imagesLabelsArray[randomIndex1], imagesLabelsArray[randomIndex2] = imagesLabelsArray[randomIndex2], imagesLabelsArray[randomIndex1]
	return imagesPathArray, imagesLabelsArray

def getBatchOfLetterImages(batchSize, imagesArray, labelsArray):
	global startIndexOfBatch
	dataset = np.ndarray(shape=(0,784), dtype=np.float32)
	labels = np.ndarray(shape=(0,2), dtype=np.float32)
	print("initialized dataset -> ", dataset)
	print("initialized labels -> ", labels)
	print("this is the imagesArray", imagesArray)
	with tf.Session() as sess:
		for i in range(startIndexOfBatch, len(imagesArray)):
			pathToImage = labelsArray[i] + imagesArray[i]
			print("this is the path to image -> " ,pathToImage)
			#rfind returns the last index where substring str is found
			lastIndexOfSlash = pathToImage.rfind("/")
			folder = pathToImage[lastIndexOfSlash - 1]
			print("it is in the folder -> " ,folder)
			print("last index is -> " ,lastIndexOfSlash)
			if(not pathToImage.endswith(".DS_Store")):
				try:
					imageContents = tf.read_file(str(pathToImage))
					print("this is the image contents -> ", imageContents)
					image = tf.image.decode_png(imageContents, dtype=tf.uint8)
					image = tf.image.rgb_to_grayscale(image)
					print("did converting the image to grayscale work? -> ", image)
					resized_image = tf.image.resize_images(image, [28,28])
					imarray = resized_image.eval()
					print("this is imarray -> ", imarray)
					imarray = imarray.reshape(-1) #need to reshape. currently multiplying 3 (channels), 28 (height) and 28 (width)
					#print("this is the size of imarray -> ",len(imarray))
					appendingImageArray = np.array([imarray], dtype=np.float32)
					appendingNumberLabel = np.array([getNumber(folder)], dtype=np.float32)
					#print("appending this image -> ",appendingImageArray)
					#print("appending this label -> ",appendingNumberLabel)

					labels = np.append(labels, appendingNumberLabel, axis=0)
					dataset = np.append(dataset, appendingImageArray, axis=0)
					if(len(labels) < batchSize):
						print(len(labels))
						print("should not be here. length of labels must be greater or equal to the batch size")
						print("this is the batch size -> ", batchSize)
						print("this is the length of the labels -> ", len(labels))
					if(len(labels) >= batchSize):
						print("the label size is more than batch size")
						startIndexOfBatch = i+1
						print("this is the dataset and labels -> ", dataset, labels)
						return dataset, labels
				except:
					print("unexpected image, it's okay, skipping")

def deepnn(x):
	"""builds the graph for a deep net for classifying digits
	
	Args:
		x: an input tensor with the dimension (N_examples, 784) where 784 is the number of 
		pixel in a standard MNIST image

	Returns:
		A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10) with values equal to the
		logits of classifying the digit into one of ten classes (digits 0-9). keep_prob is scalar
		placeholder for the probability of dropout
	"""
	#grayscale only one feature as color channel
	with tf.name_scope('reshape'):
		x_image = tf.reshape(x,[-1, 28,28, 1])

	#first convolutional layer - maps one grayscale image to 32 feature maps
	with tf.name_scope('conv1'):
		W_conv1 = weight_variable([5, 5, 1, 32])
		b_conv1 = bias_variable([32])
		h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

	#pooling layer - downsamples by 2X
	with tf.name_scope('pool1'):
		h_pool1 = max_pool_2x2(h_conv1)

	#second convolutional layer -- maps 32 feature maps to 64
	with tf.name_scope('conv2'):
		W_conv2 = weight_variable([5,5,32,64])
		b_conv2 = bias_variable([64])
		h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2)+b_conv2)

	#second pooling layer
	with tf.name_scope('pool2'):
		h_pool2 = max_pool_2x2(h_conv2)

	#fully connected layer 1 - after 2 round of downsampling, our 28x28 image is down to 7x7x64 features maps
	with tf.name_scope('fc1'):
		W_fc1 = weight_variable([7*7*64, 1024])
		b_fc1 = bias_variable([1024])

		h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
		h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1)+b_fc1)

	#dropout - controls the complexity of the model, prevents co-adaptation of features
	with tf.name_scope('dropout'):
		keep_prob = tf.placeholder(tf.float32)
		h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

	#map the 1024 features to 10 classes, one for each digit
	with tf.name_scope('fc2'):
		W_fc2 = weight_variable([1024, 2])
		b_fc2 = bias_variable([2])

		y_conv = tf.matmul(h_fc1_drop, W_fc2)+ b_fc2
	return y_conv, keep_prob

def conv2d(x,W):
	#conv2d returns a 2d convolution layer with full stride
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
	#max_pool_2x2 downsamples a feature map by 2x
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def weight_variable(shape):
	#weight_variable generates a weight variable of a given shape
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	#bias_variable generates a bias variable of a given shape
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)


startIndexOfBatch = 0
imagesPathArray, imagesLabelsArray = getListOfImages()
imagesPathArray, imagesLabelsArray = shuffleImagesPath(imagesPathArray, imagesLabelsArray)
#print("this is the test image path array -> ", imagesPathArray)
#print("this is the test label path array -> ", imagesLabelsArray)
tf.reset_default_graph()

x = tf.placeholder(tf.float32, shape=[None, 784])
W = tf.Variable(tf.truncated_normal([784, 2]), dtype=tf.float32, name="weights_0")
b = tf.Variable(tf.truncated_normal([2]), dtype=tf.float32, name="bias_0")
y = tf.nn.softmax(tf.matmul(x,W) + b)

trainingRate = 0.001
trainingLoops = 10 #10
batchSize = 64

y_ = tf.placeholder(tf.float32, [None, 2])

#build graph for deep net
y_conv, keep_prob = deepnn(x)


with tf.name_scope('loss'):
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)


crossEntropy = tf.reduce_mean(cross_entropy)

with tf.name_scope('adam_optimizer'):
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)	

with tf.name_scope('accuracy'):
	correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
	correct_prediction = tf.cast(correct_prediction, tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

graph_location = tempfile.mkdtemp()
print('Saving graph to: %s' % graph_location)
train_writer = tf.summary.FileWriter(graph_location)
train_writer.add_graph(tf.get_default_graph())

sess = tf.InteractiveSession()

saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())
for i in range(0, trainingLoops):
	print("Training loop number: ", i)
	batchX, batchY = getBatchOfLetterImages(batchSize, imagesPathArray, imagesLabelsArray)
	print(batchX.shape, batchY.shape)
	if i % 100 == 0:
		train_accuracy = accuracy.eval(feed_dict={x: batchX, y_:batchY, keep_prob:1.0})
		print('step %d, train_accuracy %g' % (i, train_accuracy)) 
	train_step.run(feed_dict={x:batchX, y_:batchY, keep_prob:0.5})

savedPath = saver.save(sess, "models/model.ckpt")
print("Model saved at: ", savedPath)

#test trained model
testImagesPathArray, testImagesLabelsArray = getListofTestImages()
testImagesPathArray, testImagesLabelsArray = shuffleImagesPath(testImagesPathArray, testImagesLabelsArray)
testbatchX, testbatchY = getBatchOfLetterImages(batchSize, testImagesPathArray, testImagesLabelsArray)

print('test accuracy %g' % accuracy.eval(feed_dict={x:testbatchX, y_:testbatchY, keep_prob:1.0}))


