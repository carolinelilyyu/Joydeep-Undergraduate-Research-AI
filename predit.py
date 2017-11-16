import sys
import tensorflow as tf
from PIL import Image, ImageFilter

def predictLetter(imvalue):
	print("testing?")
	# x = tf.placeholder(tf.float32, [None, 784])
	# W = tf.Variable(tf.zeros([784, 2]))
	# b = tf.Variable(tf.zeros([2]))
	# y = tf.nn.softmax(tf.matmul(x,W) + b)
	x = tf.placeholder(tf.float32, shape=[None, 784])
	W = tf.Variable(tf.truncated_normal([784, 2]), dtype=tf.float32, name="weights_0")
	b = tf.Variable(tf.truncated_normal([2]), dtype=tf.float32, name="bias_0")
	y = tf.nn.softmax(tf.matmul(x,W) + b)

	init_op = tf.initialize_all_variables()
	saver = tf.train.Saver()

	with tf.Session() as sess:
		sess.run(init_op)
		saver.restore(sess, "models/model.ckpt")
		prediction = tf.argmax(y, 1)
		print("this is the imvalue -> ", imvalue)
		return prediction.eval(feed_dict={x: [imvalue]}, session=sess)

def imageprepare(argv):
	im = Image.open(argv).convert('L')
	print("this is the image converted -> ", im)
	width = float(im.size[0])
	height = float(im.size[1])
	newImage = Image.new('L', (28, 28), 255)

	if width > height:
		nheight = int(round((20.0/width*height),0))
		if(nheight == 0):
			nheight = 1
		img = im.resize((20,nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
		wtop = int(round(((28-nheight)/2), 0))
		newImage.paste(img, (4, wtop))
	else:
		nwidth = int(round((20.0/height*width), 0))
		if(nwidth == 0):
			nwidth = 1

		img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
		wleft = int(round(((28-nwidth)/2), 0))
		newImage.paste(img, 4)

	tv = list(newImage.getdata())

	tva = [(255-x)*1.0/255.0 for x in tv]
	print("this is the tva -> ", tva)
	return tva

def main(argv):
	imvalue = imageprepare(argv)
	predictedLetter = predictLetter(imvalue)
	if(predictedLetter[0]==0):
		print("the letter is O")
	elif(predictedLetter[0]==1):
		print("the letter is X")
	else:
		print("the letter is neither O nor X")

if __name__ == "__main__":
	main(sys.argv[1])