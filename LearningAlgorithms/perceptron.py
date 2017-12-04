#3/19/17
#Created by Pratusha Prasad and Andrew Hooyman
#Step 1 Percpeptron learning 


import os, sys
import numpy as np
import pandas as pd 
import random
import matplotlib.pyplot as plt
from scipy import linalg

def clip(point):
	if(point < 0):
		return -1
	else: return 1

def classify(predicted, actual):
	if predicted == actual:
	   return True
	else :return False

def findInCorrect(data):
	inCorrect = []
	for i in range(0, len(data)):
		if data[i] == False:
			inCorrect.append(i)
	return inCorrect



def perceptron(data, labels, maxIter = 1, learningRate = 0.001):
	thr = np.ones((data.shape[0],1))
	data = np.concatenate((thr, data), axis = 1)
	weights = np.array([0.1,0.1,0.1,0.1])

	errorCount = []
	iter = 0
	while True:
		c = np.dot(data, weights)
		c = np.array([clip(c[i]) for i in range(0,data.shape[0])])
		classified = np.array([classify(c[j], labels[j]) for j in range(0, data.shape[0])])
		misclassifiedPoints = findInCorrect(classified)
		noOfInCorrect = len(misclassifiedPoints)
		errorCount.append(noOfInCorrect)
		if noOfInCorrect== 0:
			break
		inCorrectPoint = random.choice(misclassifiedPoints)
		weights += (labels[inCorrectPoint] * learningRate * data[inCorrectPoint])
		iter += 1 

	return weights, errorCount


#os.system("cls")
if len(sys.argv) < 2:
		print("Usage : python.exe", sys.argv[0], "FileName.txt")

file = sys.argv[1]
readData = pd.DataFrame(pd.read_csv(file,sep = r',', header = None))
data = readData.loc[:,0:2]
labels = readData.loc[:,3]
weights, errorCount = perceptron(data, labels)
print("weights", weights[1:4])
print("Intercept", weights[0])
plt.plot(np.arange(0, len(errorCount)), errorCount)
plt.show()