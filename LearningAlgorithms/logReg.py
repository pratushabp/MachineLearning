import sys, os
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from scipy import linalg


def logisticRegression(data, labels, maxIter = 7000, learningRate = 0.01):
	#labels = labels.values.reshape(data.shape[0],1)
	labels = np.array(labels)
	data = np.array(data)
	thr = np.ones((data.shape[0],1))
	error =[]# np.empty((maxIter,1))
	data = np.concatenate((thr, data), axis = 1)
	iter = 0
	weights = np.array([0.1, 0.1, 0.1,0.1])
	while iter < maxIter:
		iter = iter + 1
		s = np.multiply(np.dot(data, weights.T) , labels)
		a = np.multiply(labels.T, data.T) 
		b = a / (1 + np.exp(s)).T
		bi = 1+ np.exp(s)
		dEin = np.sum(b.T, axis = 0) / data.shape[0]
		errorTmp  = np.mean(np.log10(1 + np.exp(np.multiply(-labels , np.dot(data,weights)))))
		error.append(errorTmp)
		v = dEin / np.linalg.norm(dEin)
		weights += learningRate * v


	
	print(weights)
	print(len(error))
	return weights, error




os.system("cls")
if len(sys.argv) < 2:
		print("Usage : python.exe", sys.argv[0], "FileName.txt")

file = sys.argv[1]
readData = pd.DataFrame(pd.read_csv(file,sep = r',', header = None))
data = readData.loc[:,0:2]
labels = readData.loc[:,4]
weights, error = logisticRegression(data, labels)

