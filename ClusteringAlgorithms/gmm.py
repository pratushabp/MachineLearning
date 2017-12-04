#2/17/17
#Created by Pratusha Prasad 
#Step 1 Software Familiarization for clustering using Gaussian Mixture Model - EM algorithm 

import os, sys
import pandas as pd 
import numpy as np
import random
from scipy import linalg
import matplotlib.pyplot as plt
import math 

# Function for Guassian normal calculation
# Input  : Data[i] , mean[i] , covariance[i] 
# Output : PDF at the data point
def normalCalc(data, mean, cov):
	cov = np.array(cov).reshape((2,2))
	coeff = 1 / (math.sqrt(((2 * np.pi ) ** (cov.shape[0]) )* np.linalg.det(cov)))
	expOf = math.e ** (- ((np.dot((data - mean).T, np.dot(np.linalg.inv(cov),(data - mean)))/ 2 )))
	pdf = coeff * expOf
	return pdf

# Function : Calculate responsibility matrix for all data points
# Input  : Data, mean, covariance, Amplitude, Num of clusters
# Output : Responsilbility matrix for that iteration

def calcRic(data, mean, cov, pic,k):
	ric = np.empty(shape=(data.shape[0], k))
	for dataIter in range(data.shape[0]):
		for clusterIter in range(k):
			ric[dataIter][clusterIter] = normalCalc(data.loc[dataIter].values.reshape((data.shape[1],1)), mean[clusterIter].reshape((data.shape[1],1)), cov[clusterIter]) #* pic[clusterIter]
			ric[dataIter][clusterIter] = ric[dataIter][clusterIter]  * pic[0][clusterIter]
		ric[dataIter] = ric[dataIter] / (pic[0] * np.sum(ric[dataIter], axis = 0)) # Works!
	return ric 

# Function : Calculate the parameters of Gaussian distributions
# Input  : Data, Responsilbility matrix,  Num of clusters
# Output : Amplitude, Mean, covariance of all clusters

def calcGaussian(data, ric, k):
	amplitude = np.empty(shape = (1, k))
	amplitudeTmp = (np.sum(ric, axis = 0)).reshape(1,k)
	amplitude = (amplitudeTmp/ data.shape[0]).reshape(1,k)
	meanCalc = np.empty(shape = (data.shape[0], data.shape[1]))
	covTmp = []
	cov = []
	covCalc = np.empty(shape = (data.shape[1], data.shape[1]))
	mean = np.empty(shape = (k, data.shape[1]))
	for clusterIter in range(k):
			for dataIter in range(data.shape[0]):
					meanCalc[dataIter] = ((ric[dataIter][clusterIter] *  data.loc[dataIter].values).reshape((1, data.shape[1])))
			mean[clusterIter] = sum(meanCalc).reshape(1,data.shape[1]) / amplitudeTmp[0][clusterIter]
			for dataIter in range(data.shape[0]):
				covCalc = (ric[dataIter][clusterIter] * (np.dot((data.loc[dataIter] - mean[clusterIter]).reshape((data.shape[1],1)), (data.loc[dataIter] - mean[clusterIter]).T.reshape(1,data.shape[1]))))
				covTmp.append(covCalc)
			cov.append(sum(covTmp) / amplitudeTmp[0][clusterIter])
	return amplitude, mean, cov


# Function : Initialize parameters of K- Gaussian distributions
# Input  : Data,  Num of clusters
# Output : Amplitude, Mean, covariance of all clusters

def initGauss(data, k):
	mean = data.sample(n=k).values
	std = []
	for x in range(k):
		std.append(np.eye(data.shape[1]))
	amplitude = []
	amplitudeTmp = [1/k for n in range(k)]
	amplitude.append(amplitudeTmp)
	return mean, std, amplitude

# Function : Initialize responsilbities of each data point to each cluster
# Input  : Data,  Num of clusters
# Output : Initilaized responsibility

def initRic(data, k):
	random.seed(0)
	ric = np.empty(shape = (data.shape[0], k))
	for dataIter in range(data.shape[0]):
			ric[dataIter][0]  = (np.random.uniform(0,1))
			ric[dataIter][1]  = (np.random.uniform(0,1 -ric[dataIter][0]))
			ric[dataIter][2]  = 1 -ric[dataIter][0]  -ric[dataIter][1]	
	return ric

# Function : Main function for E-M algorithm on Gaussian models
# Input  : Data, Num of clusters, maximum iterations
# Output : Convereged Mean, Covariances and amplitudes of K Gaussian

def GMM(data, k, maxIter):
	ric = initRic(data, k)
	#mean , cov, amplitude = initGauss(data, k)
	iter = 0
	while iter < maxIter: 
		#E- Step
		amplitude, mean, cov  = calcGaussian(data, ric, k)
		#M - Step
		ric = calcRic(data, mean, cov, amplitude,k)
		iter += 1 
	return amplitude, mean, cov, ric

# Function : Plot the gaussians
# Input  : Data
# Output : Plotted clsuters
def plotClusters(data, ric):
	plt.scatter(x =data.loc[:,0] , y= data.loc[:,1], color = 'red')
	plt.scatter(x = [x[0] for x in mean], y = [x[1] for x in mean], marker = "+", s = 1000)
	plt.show()
	return None

os.system("cls")
if len(sys.argv) < 2:
		print("Usage : python.exe", sys.argv[0], "FileName.txt")

k = 3
file = sys.argv[1]
data = pd.DataFrame(pd.read_csv(file, header = None))
maxIter = 50
amplitude , mean, cov, ric = GMM(data, k, maxIter)
print("cluster 1 : mean - ", mean[0]) 
print("cluster 2 : mean - ", mean[1]) 
print("cluster 3 : mean - ", mean[2]) 
print("cluster 1 : covariance - ", cov[0])
print("cluster 2 : covariance - ", cov[1])
print("cluster 3 : covariance - ", cov[2])
print("cluster 1 : amplitude - ",  amplitude[0][0])
print("cluster 2 : amplitude - ",  amplitude[0][1])
print("cluster 3 : amplitude - ",  amplitude[0][2])
plotClusters(data, ric)