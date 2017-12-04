import os, sys
import numpy as np
import pandas as pd 
import random
from numpy import vectorize
import matplotlib.pyplot as plt
from scipy import linalg

def clip(point):
	if(point < 0):
		return -1
	if(point > 0):
	    return 1
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
class pocket:
	weights = []
	noOfErrors = 7000


def perceptron(data, labels, maxIter = 7000, learningRate = 0.01):
	thr = np.ones((data.shape[0],1))
	data = np.concatenate((thr, data), axis = 1)
	weights = np.array([0.1,0.1,0.1,0.1])
	errorCount = []
	iter = 0
	while True:
		if iter  >= maxIter:
			break
		c = np.dot(data, weights.T)
		c = np.array([clip(c[i]) for i in range(0,data.shape[0])])
		classified = np.array([classify(c[j], labels[j]) for j in range(0, data.shape[0])])
		inCorrect = findInCorrect(classified)
		noOfInCorrect = len(inCorrect)
		if noOfInCorrect== 0:
			break
	
		if noOfInCorrect <= pocket.noOfErrors:
			pocket.weights = weights.copy()
			pocket.noOfErrors = noOfInCorrect
		errorCount.append(pocket.noOfErrors)
		inCorrectPoint = random.choice(inCorrect)
		weights += (labels[inCorrectPoint] * learningRate * data[inCorrectPoint])
		iter += 1 
	return pocket.weights, errorCount


os.system("cls")
if len(sys.argv) < 2:
		print("Usage : python.exe", sys.argv[0], "FileName.txt")

file = sys.argv[1]
readData = pd.DataFrame(pd.read_csv(file,sep = r',', header = None))
data = readData.loc[:,0:2]
labels = readData.loc[:,4]
weights, errorCount = perceptron(data, labels)
print(len(errorCount))
plt.plot(np.arange(0,7000), errorCount)
plt.show()