import sys
from PIL import Image, ImageFilter
from pytorch_createModel import Net
import torch
from skimage import io, transform
import numpy as np
from torch.autograd import Variable
import argparse
from os import listdir, rename
from os.path import isfile, join
from shutil import copy
import zmq
import time

context = zmq.Context()

# get it to work, automate everything, implement tic tac toe algorithm, computer should draw on screen
# running command from c++ file. call a function to execute command
# mimc terminal input string output string what command returned
# string of 9 characters

socket = context.socket(zmq.REQ)
ai_position = ()


class GAME:
	def __init__(self, boardsize):
		self.board = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]  # np.empty([boardsize, boardsize], np.int)
		self.boardsize = boardsize
		self.p1 = ''
		self.p2 = ''
		self.lastmoves = []
		self.previousboard = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
		self.winner = ''

	def predict_letter(self, imvalue, model):
		# volatile: no gradient information computed for output. throws away gradient
		imvalue = Variable(torch.from_numpy(imvalue), volatile=True).unsqueeze(
			0)  # pytorch handles in graph in this line

		output = model(
			imvalue)  # gets the model of the images. this is feed_dict. model serves as a function this is model.forward
		# here's variable, expand the graph. dynamically create graph
		# predefine computation graph and feed dict input variable. outputs idx

		pred = output.data.max(1, keepdim=True)[1]  # for the log probability of each of the classes, this gives maximum
		# returns a tensor of which is the biggest, value
		# print(pred[0])
		return pred[0]

	def image_prepare(self, argv):
		im = Image.open(argv).convert('L')
		# print("this is the image converted -> ", im)
		width = float(im.size[0])
		height = float(im.size[1])
		newImage = Image.new('L', (28, 28), 255)

		if width > height:
			nheight = int(round((20.0 / width * height), 0))
			if (nheight == 0):
				nheight = 1
			img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
			wtop = int(round(((28 - nheight) / 2), 0))
			newImage.paste(img, (4, wtop))
		else:
			nwidth = int(round((20.0 / height * width), 0))
			if (nwidth == 0):
				nwidth = 1

			img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
			wleft = int(round(((28 - nwidth) / 2), 0))
			newImage.paste(img, 4)

		tv = list(newImage.getdata())

		tva = [(255 - x) * 1.0 / 255.0 for x in tv]
		# print("this is the tva -> ", tva)
		return tva

	def load_images(self, folder):
		files = [join(folder, f) for f in listdir(folder) if ".png" in f]  # get_files
		files = sorted(files)
		imgs = []
		for f in files:
			original_img = io.imread(f, as_grey=True)
			resize_img = np.expand_dims(transform.resize(original_img, [28, 28]), 0)
			imgs.append(resize_img)
		return np.array(imgs).astype(np.float32)

	# prints the board of X's and O's and blanks
	def print_board(self):
		boardstr = np.empty([self.boardsize, self.boardsize], dtype=str)
		for i in range(0, self.boardsize):  # range is 0 up to, but not including, boardsize
			for j in range(0, self.boardsize):
				if self.board[i][j] == 0:
					boardstr[i][j] = '_'
				elif self.board[i][j] == 1:
					boardstr[i][j] = 'O'
				else:
					boardstr[i][j] = 'X'
		print(boardstr)

	def convert_move_to_file_number(self, move_position):
		if (move_position == (0, 0)):
			return 1
		elif (move_position == (0, 1)):
			return 2
		elif (move_position == (0, 2)):
			return 3
		elif (move_position == (1, 0)):
			return 4
		elif (move_position == (1, 1)):
			return 5
		elif (move_position == (1, 2)):
			return 6
		elif (move_position == (2, 0)):
			return 7
		elif (move_position == (2, 1)):
			return 8
		else:
			return 9

	def continue_read_board(self, boardFolder):
		print("Hello")
		# iterate through all free positions
		folderImages = self.load_images(boardFolder)
		free_positions = self.get_free_positions()
		# convert every free position into folder positions
		model = Net()
		model.load_state_dict(torch.load("/Users/carolineyu/Documents/school/Undergraduate Research/XO_Classifier/models/model.nn"))  # variables of network. have all the weights

		for changed_position in free_positions:
			print("free position is: ", changed_position)
			row = changed_position[0]
			column = changed_position[1]
			img_number_of_free_position = (self.convert_move_to_file_number(changed_position));
			print("the free position is converted to file number: ", img_number_of_free_position)
			print("this is the file: ", folderImages[img_number_of_free_position-1])
			predictedLetter = self.predict_letter(folderImages[img_number_of_free_position-1], model)
			print("the predicted letter is: ", predictedLetter);
			if (predictedLetter[0] != 0):
				return changed_position
				#return POSITION THAT IS CHANGED, NOT THE POSITION THE AI SHOULD MOVE TO
				# load images
				# look at only the free positions:
				# load model only the positions that are free
				# if one of those free positions is not blank, mark it with human's mark assuming human already has a mark
				# return changed_position

	def read_board(self, boardFolder):
		print("I'm reading the board")
		print("HERE IS THE FIRST PLAYER RIGHT NOW: ", self.p1);
		# if the player has not been assigned yet, use the model
		# otherwise continue assuming players have not changed
		row = 0
		column = 0
		# gets the model from the model folder and images from the argv folder
		model = Net()
		model.load_state_dict(torch.load("/Users/carolineyu/Documents/school/Undergraduate Research/XO_Classifier/models/model.nn"))  # variables of network. have all the weights
		folderImages = self.load_images(boardFolder)
		self.previousboard = self.board
		board = np.empty([self.boardsize, self.boardsize], np.int)
		changedposition = ((),)
		timeschanged = 0
		for img in folderImages:
			predictedLetter = self.predict_letter(img, model)
			if (self.previousboard[row][column] != predictedLetter[0]):
				print(predictedLetter[0])
				changedposition = row, column
				timeschanged += 1
			if (predictedLetter[0] == 0):
				board[row, column] = 0
			elif (predictedLetter[0] == 1):
				board[row, column] = 1
			else:
				board[row, column] = 2
			column += 1
			if column == self.boardsize:
				row += 1
				column = 0
		self.board = board
		if timeschanged == 1:
			print("this is the change position", changedposition)
			return changedposition
		else:
			return "Cannot draw twice in a row or user must draw next move"

	def play(self, player1, player2, argv):
		self.p1 = player1  # should always be human
		self.p2 = player2  # should always be computer

		for i in range(1, 9):  # including 1, not 9
			if i % 2 == 0:  # human's turn
				if self.p1.type == 'H':
					print("human's move")
				else:
					print("ai's move")
				self.p1.move(self, argv)
			else:
				if self.p2.type == 'H':
					print("human's move")
				else:
					print("ai's move")
				self.p2.move(self, argv)

			if self.is_gameover():
				self.print_board()
				if self.winner == 0:
					print("\nGame over with Draw")
				else:
					print("\nWinner : %s" % self.winner)
				return
				# since the board was already read, the ai will play first

	def get_free_positions(self):
		'''Get the list of available positions'''

		moves = []
		for row in range(self.boardsize):
			for column in range(self.boardsize):
				print("the free position is row: ", row)
				print("the free position is column: ", column)
				print("the element on board right now is: ", self.board[row][column])
				if self.board[row][column] == 0:
					moves.append((row, column))
		print("here are the available moves: ", moves)
		return moves

	def mark(self, marker, newposition):
		'''Mark a position with marker 1 or 2'''
		print("newposition first element of tuple", newposition[0])
		print("newposition second element of tuple", newposition[1])
		print("here is the marker to get marked: ", marker)
		self.board[newposition[0]][newposition[1]] = marker
		self.lastmoves.append(newposition)

	def is_gameover(self):
		'''Test whether game has ended'''

		win_positions = [[(0, 0), (0, 1), (0, 2)], [(1, 0), (1, 1), (1, 2)], [(2, 0), (2, 1), (2, 2)],
						 [(0, 0), (1, 0), (2, 0)], [(0, 1), (1, 1), (2, 1)], [(0, 2), (1, 2), (2, 2)],
						 [(0, 0), (1, 1), (2, 2)], [(0, 2), (1, 1), (2, 0)]]
		for i, j, k in win_positions:
			if self.board[i[0]][i[1]] == self.board[j[0]][j[1]] == self.board[k[0]][k[1]] and self.board[i[0]][
				i[1]] != 0:
				self.winner = self.board[i[0]][i[1]]
				print("first place", self.board[i[0]][i[1]])
				print("second place", self.board[j[0]][j[1]])
				print("third place", self.board[k[0]][k[1]])
				return True

		if 0 not in self.board:
			self.winner = 0
			return True

		print(self.print_board())
		return False

	def revert_last_move(self):
		'''Reset the last move'''
		lastposition = self.lastmoves.pop()
		print("we're reverting this move: ", lastposition)
		self.board[lastposition[0]][lastposition[1]] = 0
		self.winner = None


class HUMAN:
	def __init__(self, board):
		self.marker = 0
		self.type = 'H'

	def init_marker(self, board):
		newchosen = []
		for row in range(0, len(board)):
			for column in range(0, len(board)):
				if board[row][column] == 1:
					self.marker = 1
					newchosen.append("O")
				elif board[row][column] == 2:
					self.marker = 2
					newchosen.append("X")
		if (len(newchosen) > 1):
			self.marker = 0
		if (self.marker == 0):
			print("Please enter a valid board; redo your drawing and choose either X or O.")
			exit()

	def move(self, gameinstance, argv):
		# human waits for reply from zmq from python
		# socket.recv(&request)
		# ai sends back the reply (in ai.move)
		input("Please update the board...Press Enter to continue")
		print("this went past pressing Enter")
		if self.marker is 0:
			newposition = gameinstance.read_board(argv)
		else:
			newposition = gameinstance.continue_read_board(argv)
		print(gameinstance.get_free_positions())
		if newposition not in gameinstance.get_free_positions():
			print("Invalid move. Retry")

		gameinstance.mark(self.marker, newposition)
		print("HUMAN HAS MOVED")
		gameinstance.print_board()


class AI:
	def __init__(self):
		self.marker = 0
		self.opponentmarker = 0
		self.type = 'C'

	def init_marker(self, human_marker):
		if (human_marker == 2):
			self.marker = 1
			print("the ai is O")
		elif (human_marker == 1):
			self.marker = 2
			print("the ai is X")
		self.opponentmarker = human_marker

	def convert_move_to_file(self, move_position, previousfilename, argv):
		print("this is the move position: ", move_position)
		arrayfile = previousfilename.split('.')
		print("this is a sample: ", argv + arrayfile[0] + "1.png")
		if (move_position == (0, 0)):
			return argv + arrayfile[0] + "1.png"
		elif (move_position == (0, 1)):
			return argv + arrayfile[0] + "2.png"
		elif (move_position == (0, 2)):
			return argv + arrayfile[0] + "3.png"
		elif (move_position == (1, 0)):
			return argv + arrayfile[0] + "4.png"
		elif (move_position == (1, 1)):
			return argv + arrayfile[0] + "5.png"
		elif (move_position == (1, 2)):
			return argv + arrayfile[0] + "6.png"
		elif (move_position == (2, 0)):
			return argv + arrayfile[0] + "7.png"
		elif (move_position == (2, 1)):
			return argv + arrayfile[0] + "8.png"
		else:
			return argv + arrayfile[0] + "9.png"

	def convert_move_to_file_number(self, move_position):
		if (move_position == (0, 0)):
			return "1"
		elif (move_position == (0, 1)):
			return "2"
		elif (move_position == (0, 2)):
			return "3"
		elif (move_position == (1, 0)):
			return "4"
		elif (move_position == (1, 1)):
			return "5"
		elif (move_position == (1, 2)):
			return "6"
		elif (move_position == (2, 0)):
			return "7"
		elif (move_position == (2, 1)):
			return "8"
		else:
			return "9"

	def move(self, gameinstance, argv):
		print("ai HAS MOVED")
		move_position, score = self.maximized_move(gameinstance)
		gameinstance.mark(self.marker, move_position)  # mark up the moved position, changes the board
		gameinstance.print_board()
		# check what ai.marker is
		if self.marker == 1:
			# opponent is 2 or X
			copy("/Users/carolineyu/Documents/school/Undergraduate Research/XO_Classifier/ai/O.png", argv)
			rename(argv + "O.png", argv + "game.png")
			newname = self.convert_move_to_file(move_position, "game.png", argv)
			rename(argv + "game.png", newname)
		else:
			copy("/Users/carolineyu/Documents/school/Undergraduate Research/XO_Classifier/ai/X.png", argv)
			rename(argv + "X.png", argv + "game.png")
			newname = self.convert_move_to_file(move_position, "game.png", argv)
			rename(argv + "game.png", newname)

		move_position_string = self.convert_move_to_file_number(move_position)
		print(move_position_string)
		socket.send_string(move_position_string)
		message = socket.recv_string()
		print("Received a reply: ", message)

	def maximized_move(self, gameinstance):
		bestscore = None
		bestmove = None

		for position in gameinstance.get_free_positions():  # for a position in all the free positions
			gameinstance.mark(self.marker, position)  # mark that free position with a marker

			if gameinstance.is_gameover():
				score = self.get_score(gameinstance)  # if the game is over, get the score of the gameisntance
			else:
				gameinstance.print_board()
				move_position, score = self.minimized_move(gameinstance)
				# print("testing")
				# print("maxamize exit")
				# exit()

			gameinstance.revert_last_move()

			if bestscore == None or score > bestscore:
				bestscore = score
				bestmove = position
				print("best position is: ", bestmove)
			return bestmove, bestscore

	def minimized_move(self, gameinstance):
		bestscore = None
		bestmove = None

		for position in gameinstance.get_free_positions():  # for a position in all the free positions
			gameinstance.mark(self.marker, position)  # mark that free position with a marker

			if gameinstance.is_gameover():
				score = self.get_score(gameinstance)  # if the game is over, get the score of the gameisntance
			else:
				# gameinstance.print_board()
				move_position, score = self.maximized_move(gameinstance)
				# print("minimize exit")
				# gameinstance.print_board()
				# exit()

			gameinstance.revert_last_move()

			if bestscore == None or score < bestscore:
				bestscore = score
				bestmove = position
				print("best position is: ", bestmove)
		return bestmove, bestscore

	def get_score(self, gameinstance):
		print("the get scored entered")
		if gameinstance.is_gameover():
			if gameinstance.winner == self.marker:
				return 1  # Won

			elif gameinstance.winner == self.opponentmarker:
				return -1  # Opponent won

		return 0  # Draw


def main(argv):
	# read the initial board for user's turn 1, the user would draw his character and play, already taking the first turn


	print("Connecting to hello world server...")
	socket.connect("tcp://localhost:5555")

	# get in the reply

	game = GAME(3)
	game.read_board(argv)  # reads board from folder
	human = HUMAN(game.board)
	human.init_marker(game.board)  # never init_marker afterwards
	ai = AI()
	ai.init_marker(human.marker)  # never init_marker afterwards
	game.print_board()
	game.play(human, ai, argv)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
	parser.add_argument('--f', type=str, help='filename')
	args = parser.parse_args()
	main(args.f)  # sys.argv[1])