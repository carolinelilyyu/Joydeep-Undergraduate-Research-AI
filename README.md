# Joydeep-Undergraduate-Research-AI

Part of my undergraduate research for Joydeep Biswas 2017 (UMass Amherst). This repo handles image detection/machine learning using PyTorch. It also handles the minimax algorithm or AI's turn for Tic Tac Toe. This is written in Python and communicates to C++ or Joydeep-Undergraduate-Research-GUI-Tic-Tac-Toe through ZMQ. Configuring the code to work with your machine will be essential; for example, certain file locations are configured to the creator's desktop: "/User/carolineyu/Documents/....". Change these paths/directories in the code (pytorch_predict.py) to match where your file locations will exist.



## Terminology
- AI: Artificial intelligence: the computer program
- GUI: Graphical User Interface
- Human: User who is using the program
- Pytorch: A framework used for machine-learning. Machine learning is used for character recognition between an X and O. It is an easier version of Tensorflow

## Downloads/Links to help
- **ZeroMQ/ZMQ** - Platform to connect C++ code to Python code
[[ZeroMQ]](http://zeromq.org/)
- **Minimax Algorithm** - A tutorial I followed to implement the minimax algorithm
[[Minimax Algorithm]](http://www.sarathlakshman.com/2011/04/29/writing-a-tic-tac)
- **PyTorch** Framework used for machine-learning for character recognition for X and O. It is a cousin of lua-based Torch framework which is actively used at Facebook. If you want a simple framework to use to get the job done easier than Tensorflow, PyTorch is the framework for you.
[[PyTorch website]](http://pytorch.org/)
- **EMNIST Dataset** Training data, provided by NIST, I used for the character recognition machine learning part. Contains many png pictures of characters
[[EMNIST Dataset]](https://www.nist.gov/itl/iad/image-group/emnist-dataset)
- **ZMQ Basics: Hello World for Python**
[[Hello World Python ZMQ Example]](http://zguide.zeromq.org/py:hwclient)
- **PyCharm** IDE I used for editting things in Python/PyTorch. Can execute terminal through here too. It's a lot prettier than writing in Sublime text editor, I'll give you that.
[[PyCharm]](https://www.jetbrains.com/pycharm/)
- **Tensorflow (OPTIONAL)** Framework used for machine-learning and can also be used for character recognition for X and O. It was developed by Google Brain and actively used at Google. I've also provided some code for Tensorflow, since I attempted it before switching over to PyTorch. The Tensorflow code is commented out inside the PyTorch files, but they can also be found in predict.py and deep-createmodel.py. If you want a deeper understanding of how tensors work and deep learning, TensorFlow is the framework for you.
[[Tensorflow website]](https://www.tensorflow.org/)
- **Understanding Tensorflow: MNIST examples (OPTIONAL)**
[[MNIST for Beginners: Lower Accuracy]](https://www.tensorflow.org/get_started/mnist/beginners)
[[MNIST for Experts: Higher Accuracy]](https://www.tensorflow.org/get_started/mnist/beginners)

## How to Play
1. Press the green play button in the QT IDE.
2. Draw an X or O in one of the squares. DO NOT DRAW IN MORE THAN ONE SQUARE. Also if the user were to choose O, it must look like the O in the **/train/O/** folder. Please provide more training data and create the model to fit the training data.
3. Press "CTRL + S" to save the game board. Do not be afraid if the GUI hangs, ZMQ sockets make the GUI hang because it is waiting for the Python code's response. It will automatically save in the folder **/board/** as 9 separate "game(insert square number).png"s.
4. Start up the python script from Joydeep-Undergraduate-Research-AI inside Pycharm. If the code shows the board state and says "Press Enter to continue...", then the game has moved.
5. Check back on the GUI. The computer should have moved.
6. Draw an X or O again
7. CTRL + S or save the drawing.
8. Go back to Pycharm IDE and Press Enter.
9. Repeat steps 5-9.


## How ZMQ works
In this code, we are using a socket to communicate Python to C++ and vice versa. They are both binded to TCP port: **tcp://127.0.0.1:5555** *(line 13 in mainwindow.cpp)*. You can follow this [Hello World C++ ZMQ Example](http://zguide.zeromq.org/cpp:hwclient) to find out how a simple socket works. Whenever the user saves (or ctrl+s), ZMQ sends a request message from QT/C++ to Python to tell them the human has moved; the GUI hangs while it waits for a reply message from Python to C++/QT. The ZMQ reply message would reply back with the image of the square the AI moved. If it marked "game1.png", that means that it marked square 1, or (0,0). If it marked "game2.png", it marked square 2, or (0,1), etc. The GUI stops hanging and the user can write again. This cycle repeats until the user has won, computer has won, or they end with a draw. 

## Structure of Code and Folders
- **pytorch-predict.py** predicts based upon models within */models/* folder. It takes all the 9 images from */board/* folder to predict the X or O. Execute the python script like: **python pytorch-predict --f board/**
- **pytorch-createmodel.py** creates the model using training and testing data from */train/* and */test/* folder. Makes models and puts into */models/* folder. Execute the python script like: **python pytorch-createmodel.py**
- **pytorch-predict1.py** predicts the same way as pytorch-predict.py but instead of predict the board for the whole folder, it just predicts on one single png file. Execute the python script like: **python pytorch-predict --f board/game1.png**
- **"cppzmq-master" folder** this folder is to provide source code for ZMQ to work
- **"ai" folder** this folder is for the ai to have a preset picture O and X to make it's move and show in the QT GUI.
- **predict.py, deep-model.py**: tensorflow code you can ignore, but use for reference even though it's not written completely correctly, it's not very far off from the right answer. You can debug it if you like, but I switched over to use PyTorch instead of Tensorflow


## Major Bugs
- QT/C++ Bug (GUI): As said before, *system*, or executing shell/terminal commands within the C++ code makes the GUI hang even though it is correctly executing the python script. You can confirm this because the QT IDE prints out the Python code's print statements, but starts hanging halfway through the execution. I've commented out the command execution code part for now until you choose to uncomment it and play around with it so you'll have to start up the python script manually and press Enter after you draw and save. Therefore, this code is not automated.
- Python Bug (AI): Code does not play tic tac toe optimally. Please look at the minimax algorithm within the Python code, or Joydeep-Undergraduate-Research-AI, to debug why it's not doing so. I've also provided a tutorial about how minimax works and the pseudocode I followed under the downloads/links inside the Joydeep-Undergraduate-Research-AI repo. Through the print statements, you can see it trying different game states to see what's the optimal path, but it always just ends up choosing the next available free position, or next square.
