#3/31/17
#Created by Pratusha Prasad
#Step 1 Feed forward propogation


import sys
import pandas as pd 
import numpy as np
import math
import matplotlib.pyplot as plt

# Read image : Source StackOverflow

def readPGMImage(fileName):
	with open(fileName,'rb') as imageFile:
		image = []
		imageFile.readline()
		imageFile.readline()
		(width, height) = [int(i) for i in imageFile.readline().split()]
		maxVal = int(imageFile.readline().strip())
		for _ in range(width * height):
			image.append(imageFile.read(1)[0]/maxVal)
		return image

# Activation function
def sigmoid(s):
	thetaS = 1.0 / (1.0 + np.exp(-s))
	return thetaS

# Neural network architecture
def optimize(trainSet, trainLabels, nEpochs = 1000):
	trainSet = np.array(trainSet)
	trainLabels = np.array(trainLabels)
	error =[]
	e = []
	w12 = np.random.uniform(low = -0.001, high = 0.001, size = (960, 100))
	w23 = np.random.uniform(low = -0.001, high = 0.001, size = (100, 1))
	for epochs in range(nEpochs):
		for dataPoint in range(trainSet.shape[0]):
			error[:] = []
			# forward propogation
			thetaS1 = sigmoid(np.dot(trainSet[dataPoint], w12))
			thetaS2 = sigmoid(np.dot(thetaS1, w23))
			# Back propogation
			error.append((abs(trainLabels[dataPoint] - thetaS2)))
			deltaL1 =  (trainLabels[dataPoint] - thetaS2) * (sigmoid(thetaS2)) * ( 1 - sigmoid(thetaS2))
			deltaL =   (np.dot(w23, deltaL1) ) * (sigmoid(thetaS1)) * ( 1 - sigmoid(thetaS1))
			w23 += 0.1 * np.outer(thetaS1,deltaL1)
			w12 += 0.1 * np.outer(trainSet[dataPoint],deltaL)
		e.append(sum(error))
	return w12, w23, e


def predict(testSet,w12, w23): 
		y = np.zeros([testSet.shape[0],w23.shape[1]])
		for dataPoint in range(testSet.shape[0]):
				thetaS1 = sigmoid(np.dot(testSet[dataPoint], w12))
				y[dataPoint] = sigmoid(np.dot(thetaS1, w23))
		return y


trainSet = [] 
testSet = []
trainLabels = []


file = 'downgesture_train.list'
gesture = 'down'
readData = pd.DataFrame(pd.read_csv(file,sep = r',', header = None))
for file in readData[0]:
	trainSet.append(readPGMImage(file)) 
	if 'down' in file:
		trainLabels.append(1)
	else:
		trainLabels.append(0)

w12, w23, error = optimize(trainSet, trainLabels)

count = 0
classified = 0
with open('downgesture_test.list') as test:
    count += 1
    for image in test.readlines():
        image = image.strip()
        tester = np.array([readPGMImage(image),])
        decision = predict(tester, w12, w23)
        if decision < 0.5:
        		decision = 0
        else:
        		decision = 1
        if (decision != 0) == ('down' in image):
            classified += 1

print("Classification accuracy = ",( classified / count))
plt.plot(np.arange(0, len(error)), error)
plt.show()



