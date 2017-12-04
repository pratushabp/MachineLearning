import os, sys 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt


def calcVariance(data, mean):
	var = np.empty((data.shape[1],data.shape[1]))
	for sampleIter in range(data.shape[0]):
		Xi = (data.loc[sampleIter,:] - mean).values.reshape((data.shape[1], 1))
		var = var + (np.dot(Xi, Xi.T))
	return (var/(data.shape[0] -1))

data = pd.read_csv("pca-data.txt", sep = r'\s+', header = None)
#calcMean(data)
#Calc mean 
print(data.shape[0])
mean =  np.array(np.sum(data, axis =0) / data.shape[0])
#Get variance 
print(mean)
var = calcVariance(data, mean)
print(var)
#Do SVD 
eigVal, eigVec = np.linalg.eig(var)
valSort = np.sort_complex(eigVal)
print(valSort)
vecSort = pd.DataFrame(eigVec[:,eigVal.argsort()[::-1]])
print(vecSort)
vecSort.drop(vecSort.columns[len(vecSort.columns) -1], axis = 1, inplace = True)
print(vecSort)
Xnew = np.dot(vecSort.values.T, data.values.T)
Xnew = (Xnew.T)
print(Xnew)
plt.scatter(Xnew[:,0], Xnew[:,1])
#plt.show()


#eigVec = pd.DataFrame(eigVec).set_index([[eigVal[0], eigVal[1], eigVal[2]]])
#print(eigVec)
#Get eigen values nxn 
#Truncate 
#Multiply to map