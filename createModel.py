import tensorflow as tf
import os
from random import randint
import numpy as np


folders = ["O", "X"]
datasetFolder = "letters"

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

def getBatchOfLetterImages(batchSize=64):
	global startIndexOfBatch
	global imagesPathArray
	dataset = np.ndarray(shape=(0,784), dtype=np.float32)
	labels = np.ndarray(shape=(0,2), dtype=np.float32)
	with tf.Session() as sess:
		for i in range(startIndexOfBatch, len(imagesPathArray)):
			pathToImage = imagesLabelsArray[i] + imagesPathArray[i]
			#rfind returns the last index where substring str is found
			lastIndexOfSlash = pathToImage.rfind("/")
			folder = pathToImage[lastIndexOfSlash - 1]
			if(not pathToImage.endswith(".DS_Store")):
				try:
					imageContents = tf.read_file(str(pathToImage))
					print("this is the image contents -> ", imageContents)
					image = tf.image.decode_png(imageContents, dtype=tf.uint8)
					image = tf.image.rgb_to_grayscale(image)
					#print("did converting the image to grayscale work? -> ", image)
					resized_image = tf.image.resize_images(image, [28,28])
					imarray = resized_image.eval()
					#print("this is imarray -> ", imarray)
					imarray = imarray.reshape(-1) #need to reshape. currently multiplying 3 (channels), 28 (height) and 28 (width)
					#print("this is the size of imarray -> ",len(imarray))
					appendingImageArray = np.array([imarray], dtype=np.float32)
					appendingNumberLabel = np.array([getNumber(folder)], dtype=np.float32)
					labels = np.append(labels, appendingNumberLabel, axis=0)
					dataset = np.append(dataset, appendingImageArray, axis=0)
					if(len(labels) >= batchSize):
						startIndexOfBatch = i+1
						print(dataset, labels)
						return dataset, labels
				except:
					print("unexpected image, it's okay, skipping")

startIndexOfBatch = 0
imagesPathArray, imagesLabelsArray = getListOfImages()
imagesPathArray, imagesLabelsArray = shuffleImagesPath(imagesPathArray, imagesLabelsArray)

tf.reset_default_graph()

x = tf.placeholder(tf.float32, shape=[None, 784])
W = tf.Variable(tf.truncated_normal([784, 2]), dtype=tf.float32, name="weights_0")
b = tf.Variable(tf.truncated_normal([2]), dtype=tf.float32, name="bias_0")
y = tf.nn.softmax(tf.matmul(x,W)+ b)

trainingRate = 0.001
trainingLoops = 2
batchSize = 100

yTrained = tf.placeholder(tf.float32, [None, 2])

crossEntropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=yTrained, logits=y))

#crossEntropy = -tf.reduce_sum(yTrained * tf.log(y))

trainStep = tf.train.GradientDescentOptimizer(trainingRate).minimize(crossEntropy)

saver = tf.train.Saver()

with tf.Session() as session:
	session.run(tf.global_variables_initializer())
	for i in range(0, trainingLoops):
		print("Training loop number: ", i)
		batchX, batchY = getBatchOfLetterImages(batchSize)
		print(batchX.shape, batchY.shape)
		session.run(trainStep, feed_dict={x: batchX, yTrained: batchY})

	savedPath = saver.save(session, "models/model.ckpt")
	print("Model saved at: ", savedPath)

#add accuracy and correct_predictions