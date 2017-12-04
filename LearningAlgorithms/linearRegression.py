import sys, os
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from scipy import linalg


def linearRegression(data, labels, maxIter = 7000, learningRate = 0.01):
	#labels = labels.values.reshape(data.shape[0],1)
	labels = np.array(labels)
	data = np.array(data)
	thr = np.ones((data.shape[0],1))
	error =[]# np.empty((maxIter,1))
	data = np.concatenate((thr, data), axis = 1)
	iter = 0
	weights = np.array([0.1, 0.1, 0.1,0.1])
	weights = np.dot((np.linalg.inv(np.dot(data.T, data))),(np.dot(data.T, labels)))
	return weights


os.system("cls")
if len(sys.argv) < 2:
		print("Usage : python.exe", sys.argv[0], "FileName.txt")

file = sys.argv[1]
readData = pd.DataFrame(pd.read_csv(file,sep = r',', header = None))
data = readData.loc[:,0:1]
labels = readData.loc[:,2]
weights = linearRegression(data, labels)
print(weights)

