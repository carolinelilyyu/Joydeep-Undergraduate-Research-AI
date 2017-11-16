from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from skimage import io, transform
from torch.autograd import Variable
from os import listdir
from os.path import isfile, join
from torch.utils.data.sampler import RandomSampler
from torch.utils.data import Dataset
import numpy as np
import os
#
# # Training settings
# parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
# parser.add_argument('--batch-size', type=int, default=64, metavar='N',
#                     help='input batch size for training (default: 64)')
# parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
#                     help='input batch size for testing (default: 1000)')
# parser.add_argument('--epochs', type=int, default=30, metavar='N',
#                     help='number of epochs to train (default: 10)')
# parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
#                     help='learning rate (default: 0.01)')
# parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
#                     help='SGD momentum (default: 0.5)')
# parser.add_argument('--no-cuda', action='store_true', default=False,
#                     help='disables CUDA training')
# parser.add_argument('--seed', type=int, default=1, metavar='S',
#                     help='random seed (default: 1)')
# parser.add_argument('--log-interval', type=int, default=10, metavar='N',
#                     help='how many batches to wait before logging training status')
# args = parser.parse_args()
# args.cuda = not args.no_cuda and torch.cuda.is_available()


class OwnDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        Xdir = join(root_dir, 'X')
        Odir = join(root_dir, 'O')
        x_loaded_images = self.load_images(Xdir)
        o_loaded_images = self.load_images(Odir)
        x_labels = np.array([1]*x_loaded_images.shape[0], dtype = np.int)
        o_labels = np.array([0]*o_loaded_images.shape[0], dtype=np.int)
        self.data = np.concatenate([x_loaded_images, o_loaded_images], axis=0)
        self.labels = np.concatenate([x_labels, o_labels], axis=0)

    def __len__(self):
        return self.data.shape[0]

    def load_images(self, folder):
        files = [join(folder,f) for f in listdir(folder) if ".png" in f]#get_files
        imgs = []
        for f in files:
            original_img = io.imread(f,as_grey=True)
            resize_img = np.expand_dims(transform.resize(original_img, [28, 28]), 0)
            imgs.append(resize_img)
        return np.array(imgs).astype(np.float32)

    def __getitem__(self, idx):
        sample = {'image':self.data[idx], 'label':self.labels[idx]}
        return sample




# kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
# train_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=True, download=True,
#                    transform=transforms.Compose([
#                        transforms.ToTensor(),
#                        transforms.Normalize((0.1307,), (0.3081,))
#                    ])),
#     batch_size=args.batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=False, transform=transforms.Compose([
#                        transforms.ToTensor(),
#                        transforms.Normalize((0.1307,), (0.3081,))
#                    ])),
#     batch_size=args.test_batch_size, shuffle=True, **kwargs)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)



folders = ["O", "X"]
datasetFolder = "train"
testFolder = "letters"

def getNumber(letter):
    if(letter == "O"):
        #returns the first row of a 2-d array with ones in the diagonal and zeros elsewhere
        return np.eye(2, dtype=np.float32)[0]
    if(letter == "X"):
        return np.eye(2, dtype=np.float32)[1]


# def getListOfImages():
# 	global folders
# 	global datasetFolder
# 	allImagesArray = np.array([], dtype=np.str)
# 	allImagesLabelsArray = np.array([], dtype=np.str)

# 	for folder in folders:
# 		print("Loading Image Name of ", folder)
# 		currentLetterFolder = datasetFolder+"/"+folder+"/"
# 		#listdir returns the list containing the names of the entries in the directory given by path
# 		imagesName = os.listdir(currentLetterFolder)
# 		allImagesArray = np.append(allImagesArray, imagesName)
# 		for i in range(0, len(imagesName)):
# 			print(i)
# 			#100 images in each folder
# 			if(i % 100 == 0):
# 				print("progress -> ", i)
# 			allImagesLabelsArray = np.append(allImagesLabelsArray, currentLetterFolder)
# 		#print(allImagesArray)
# 	return allImagesArray, allImagesLabelsArray


# def shuffleImagesPath(imagesPathArray, imagesLabelsArray):
# 	print("Size of imagesPathArray is ", len(imagesPathArray))
# 	for i in range(0, 100000):
# 		if(i % 1000 == 0):
# 			print("Shuffling in progress -> ", i)
# 		randomIndex1 = randint(0, len(imagesPathArray)-1)
# 		randomIndex2 = randint(0, len(imagesPathArray)-1)
# 		imagesPathArray[randomIndex1], imagesPathArray[randomIndex2] = imagesPathArray[randomIndex2], imagesPathArray[randomIndex1]
# 		imagesLabelsArray[randomIndex1], imagesLabelsArray[randomIndex2] = imagesLabelsArray[randomIndex2], imagesLabelsArray[randomIndex1]
# 	return imagesPathArray, imagesLabelsArray

# def getBatchOfLetterImages(batchSize, imagesArray, labelsArray):
# 	global startIndexOfBatch
# 	dataset = np.ndarray(shape=(0,784), dtype=np.float32)
# 	labels = np.ndarray(shape=(0,2), dtype=np.float32)
# 	print("initialized dataset -> ", dataset)
# 	print("initialized labels -> ", labels)
# 	print("this is the imagesArray", imagesArray)
# 	with tf.Session() as sess:
# 		for i in range(startIndexOfBatch, len(imagesArray)):
# 			pathToImage = labelsArray[i] + imagesArray[i]
# 			print("this is the path to image -> " ,pathToImage)
# 			#rfind returns the last index where substring str is found
# 			lastIndexOfSlash = pathToImage.rfind("/")
# 			folder = pathToImage[lastIndexOfSlash - 1]
# 			print("it is in the folder -> " ,folder)
# 			print("last index is -> " ,lastIndexOfSlash)
# 			if(not pathToImage.endswith(".DS_Store")):
# 				try:
# 					imageContents = tf.read_file(str(pathToImage))
# 					print("this is the image contents -> ", imageContents)
# 					image = tf.image.decode_png(imageContents, dtype=tf.uint8)
# 					image = tf.image.rgb_to_grayscale(image)
# 					print("did converting the image to grayscale work? -> ", image)
# 					resized_image = tf.image.resize_images(image, [28,28])
# 					imarray = resized_image.eval()
# 					print("this is imarray -> ", imarray)
# 					imarray = imarray.reshape(-1) #need to reshape. currently multiplying 3 (channels), 28 (height) and 28 (width)
# 					#print("this is the size of imarray -> ",len(imarray))
# 					appendingImageArray = np.array([imarray], dtype=np.float32)
# 					appendingNumberLabel = np.array([getNumber(folder)], dtype=np.float32)
# 					#print("appending this image -> ",appendingImageArray)
# 					#print("appending this label -> ",appendingNumberLabel)

# 					labels = np.append(labels, appendingNumberLabel, axis=0)
# 					dataset = np.append(dataset, appendingImageArray, axis=0)
# 					if(len(labels) < batchSize):
# 						print(len(labels))
# 						print("should not be here. length of labels must be greater or equal to the batch size")
# 						print("this is the batch size -> ", batchSize)
# 						print("this is the length of the labels -> ", len(labels))
# 					if(len(labels) >= batchSize):
# 						print("the label size is more than batch size")
# 						startIndexOfBatch = i+1
# 						print("this is the dataset and labels -> ", dataset, labels)
# 						return dataset, labels
# 				except:
# 					print("unexpected image, it's okay, skipping")


def train(epoch, model, train_loader, optimizer):
    model.train()
    correct= 0
    total = 0
    for batch_idx, sample in enumerate(train_loader):
        data= sample["image"]
        target = sample["label"]
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        total += pred.size(0)
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t Accuracy : {}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0], correct/total))

    torch.save(model.state_dict(), "models/model.nn")

def test(model,test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for  sample in test_loader:
        data = sample["image"]
        target = sample["label"]
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.cross_entropy(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    model = Net()
    if args.cuda:
        model.cuda()

    train_dataset = OwnDataset('train/')
    test_dataset = OwnDataset('letters/')
    train_sampler = RandomSampler(train_dataset)
    test_sampler = RandomSampler(test_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, sampler=train_sampler, batch_size=64)
    test_loader = torch.utils.data.DataLoader(test_dataset, sampler=test_sampler, batch_size=64)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(epoch, model, train_loader, optimizer)
        test(model, test_loader)