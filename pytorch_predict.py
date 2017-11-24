import sys
from PIL import Image, ImageFilter
from pytorch_createModel import Net
import torch
from skimage import io, transform
import numpy as np
from torch.autograd import Variable
import argparse
from os import listdir
from os.path import isfile, join



rows = 3
cols = 3
numberOfWinningPositions=rows+cols+2
boardsize = 3
circle=1
cross=2
empty=0

computer=1
human=2


class Player:
	def __init__(self, type, character):
		#type is human or computer
		#character is X or O
		self.type = type
		self.character = character

	def OppositeSign(self, character):
		'''returns the sign of the opponent'''
		if character == "O":
			return "X"
		return "X"

	def SetBoard(self, board):
		'''sets the board on which the player is playing'''
		self.board = board

	def GetMove(self):
		pass


class HumanPlayer(Player):
	def __init__(self, character):
		super().__init__(human, character)

class ComputerPlayer(Player):
	def __init__(self, character):
		self.lastmove = 1

def predictLetter(imvalue, model):
	#volatile: no gradient information computed for output. throws away gradient
	imvalue = Variable(torch.from_numpy(imvalue),volatile=True).unsqueeze(0)#pytorch handles in graph in this line

	output = model(imvalue) #gets the model of the images. this is feed_dict. model serves as a function this is model.forward
	#here's variable, expand the graph. dynamically create graph
	#predefine computation graph and feed dict input variable. outputs idx

	pred = output.data.max(1, keepdim=True)[1]  #for the log probability of each of the classes, this gives maximum
	#returns a tensor of which is the biggest, value
	#print(pred[0])
	return pred[0]

	# x = tf.placeholder(tf.float32, [None, 784])
	# W = tf.Variable(tf.zeros([784, 2]))
	# b = tf.Variable(tf.zeros([2]))
	# # y = tf.nn.softmax(tf.matmul(x,W) + b)
	# x = tf.placeholder(tf.float32, shape=[None, 784])
	# W = tf.Variable(tf.truncated_normal([784, 2]), dtype=tf.float32, name="weights_0")
	# b = tf.Variable(tf.truncated_normal([2]), dtype=tf.float32, name="bias_0")
	# y = tf.nn.softmax(tf.matmul(x,W) + b)

	# init_op = tf.initialize_all_variables()
	# saver = tf.train.Saver()

	# with tf.Session() as sess:
	# 	sess.run(init_op)
	# 	#saver.restore(sess, "models/model.ckpt")
	# 	prediction = tf.argmax(y, 1)
	# 	print("this is the imvalue -> ", imvalue)
	# 	return prediction.eval(feed_dict={x: [imvalue]}, session=sess)

def imageprepare(argv):
	im = Image.open(argv).convert('L')
	#print("this is the image converted -> ", im)
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
	#print("this is the tva -> ", tva)
	return tva


def load_images(folder):
	files = [join(folder, f) for f in listdir(folder) if ".png" in f]  # get_files
	imgs = []
	for f in files:
		original_img = io.imread(f, as_grey=True)
		resize_img = np.expand_dims(transform.resize(original_img, [28, 28]), 0)
		imgs.append(resize_img)
	return np.array(imgs).astype(np.float32)


#prints the board of X's and O's and blanks
def printBoard(boardsize, board):
	boardstr = np.empty([boardsize, boardsize], dtype=str)
	for i in range(0, boardsize): #range is 0 up to, but not including, boardsize
		for j in range(0, boardsize):
			if board[i][j] == 0:
				boardstr[i][j] = '_'
			elif board[i][j] == 1:
				boardstr[i][j] = 'O'
			else:
				boardstr[i][j] = 'X'
	print("This is the board\n", boardstr)
	return boardstr

def readBoard(boardFolder, boardsize):
	row = 0
	column = 0
	#gets the model from the model folder and images from the argv folder
	model = Net()
	model.load_state_dict(torch.load("models/model.nn"))  # variables of network. have all the weights
	folderImages = load_images(boardFolder)
	board = np.empty([boardsize, boardsize], np.int)
	usercharacter = "_"
	chosen = []
	for img in folderImages:
		predictedLetter = predictLetter(img, model)
		if (predictedLetter[0] == 0):
			board[row, column] = 0
		elif (predictedLetter[0] == 1):
			board[row, column] = 1
			usercharacter = 'O'
			chosen.append('O')
		else:
			board[row, column] = 2
			usercharacter = 'X'
			chosen.append('X')
		column += 1
		if column == boardsize:
			row += 1
			column = 0
	if(len(chosen)>1):
		usercharacter = '_'
	return board, usercharacter

def checkColumn(column, boardarray, character):
	for i in range(0, 3):
		if (boardarray[i][column] != character):
			return False
	return True

def checkRow(row, boardarray, character):
	for i in range(0, 3):
		if (boardarray[row][i] != character):
			return False
	return True

def checkDiagonal(boardarray, character):
	diagonalone = True
	diagonaltwo = True
	for i in range(0, 3):
		if (boardarray[i][i] != character):
			diagonalone = False
			break

	for i in range(0, 3):
		if (boardarray[len(boardarray)-i-1][i] != character):
			diagonaltwo = False
			break
	print("The boolean should be true or false", diagonaltwo)
	return diagonalone or diagonaltwo


def checkWin(board, character):
	#loop through every column in the row

	#check cols
	for col in range(0,rows):
		if(checkColumn(col, board, character)==True):
			return True

	#check rows
	for row in range(0,cols):
		if(checkRow(row, board, character)==True):
			return True

	#check diagonals
		if(checkDiagonal(board, character)==True):
			return True
	return False



def main(argv):
	#read the initial board for user's turn 1, the user would draw his character and play, already taking the first turn
	read = readBoard(argv, boardsize)
	board = read[0] #initialize board
	usercharacter = read[1]
	computercharacter = ""
	if(usercharacter == "X"):
		computercharacter = "O"
	elif(usercharacter == "O"):
		computercharacter = "X"
	else:
		print("Please enter a valid board; redo your drawing and choose either X or O.")
		exit()

	computer = ComputerPlayer(self, computer, computercharacter)
	print("The chosen letter for the user is: ", usercharacter)
	print("The letter for the computer is: ", computercharacter)
	for turn in range(2, boardsize * boardsize+1): #8 turns
	#1 - user (initial, read board)
	#2 - computer
	#3 - user
	#4 - computer
	#5 - user
	#6 - computer
	#7 - user
	#8 - computer
		print(printBoard(boardsize, board))
		print("is this global", rows)
		if(turn % 2 == 0): #computer's turn, minimax, update and return board
			#checks if user has won, if not, computer moves
			if(checkWin(board, usercharacter)==True):
				print("The user has won.")
				exit()
			elif (checkWin(board, computercharacter) == True):
				print("The computer has won.")
				exit()
			else:
				print("draw.")
				exit()
			#minimax
			#update the board
			#return a stdout of computer's x onto the spot

		elif(turn % 2 == 1): #user's turn, wait for input, read board
			if (checkWin(board, usercharacter) == True):
				print("The user has won.")
				exit()
			elif (checkWin(board, computercharacter) == True):
				print("The computer has won.")
				exit()
			else:
				print("draw.")
				exit()
			input("Please update the board...Press Enter to continue")
			board = readBoard(argv, boardsize)[0]

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
	parser.add_argument('--f', type=str, help='filename')
	args = parser.parse_args()
	main(args.f)#sys.argv[1])