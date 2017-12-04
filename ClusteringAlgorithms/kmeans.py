#2/17/17
#Created by Pratusha Prasad 
#Step 1 Software Familiarization for clustering using K-means - EM algorithm 

import sys, math, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function : Find the distance between data points and centroids to classify them
# Input  : Data , centroid matrix
# Output : Classified clusters

def getClusters(dataPoints, centroids):
	#minize the variance 
	cluster = {}
	cluster[0] = np.empty(shape = (0,2))
	cluster[1] = np.empty(shape = (0,2))
	cluster[2] = np.empty(shape = (0,2))
	for run in range(0 , dataPoints.shape[0]):
			data = dataPoints.iloc[run,:]
			data = (np.array(data.values.T))
			distance  = (np.sqrt(((centroids - data) ** 2).sum(axis = 1)))
			cluster[np.argmin(distance)] =  np.append(cluster[np.argmin(distance)], [[data[0], data[1]]], axis = 0)
	return cluster

# Function : Re-calculate the centroids of given clusters
# Input  : cluster matrix, centroids, Num of Clusters
# Output : new calculated centroids

def calcCentroids(clusters, centroids, k):
	newCentroids = np.empty(shape = (0,2))
	for x in clusters:
		newX = (sum(clusters[x][:,0])/ (len(clusters[x][:,0])))
		newY = (sum(clusters[x][:,1])/ (len(clusters[x][:,1])))
		newCentroids = np.append(newCentroids, [[newX, newY]], axis = 0)
	return newCentroids

# Function : Initialize the cluster centroids from data points
# Input  : Data, Num of Clusters
# Output : initialized centroids

def initCentroid(datapoints,k):
	centroids = (datapoints.sample(n = k))
	centroids = centroids.values.tolist()
	return centroids

# Function :  Visualize the clusters
# Input  : clusters and centroids


def plotClusters(clusters, centroids):
	plt.scatter(x = clusters[0][:,0], y = clusters[0][:,1] , color = 'red')
	plt.scatter(x = clusters[1][:,0], y = clusters[1][:,1] , color = 'blue')
	plt.scatter(x = clusters[2][:,0], y = clusters[2][:,1] , color = 'green')
	plt.scatter(x = [x[0] for x in centroids], y = [x[1] for x in centroids], marker = "+", s = 1000)
	plt.show()
	return None

# Function : Main function for K-means, EM algorithm
# Input  : Data, Num of clusters, maximum iterations
# Output : Centroids of clusters, clusters

def kmeans(datapoints, k, maxIter):
	iter = 0
	centroids = initCentroid(datapoints,k)
	while iter < maxIter:
		clusters = getClusters(datapoints , centroids)
		newCentroids = calcCentroids(clusters, centroids, k)
		centroids = newCentroids
		iter += 1
	return newCentroids, clusters

os.system('cls')

if len(sys.argv) < 2:
	print("Usage : python.exe", sys.argv[0], "FileName.txt Numberofclusters")
	exit(1)

k = 3
FileName = sys.argv[1]
dataPoints =  pd.DataFrame(pd.read_csv('clusters.txt', header = None))# Get data

maxIter = 100
centroids, clusters = kmeans(dataPoints, k, maxIter)
print(" Cluster 1 centroid :",  centroids[0])
print(" Cluster 2 centroid :", centroids[1])
print(" Cluster 3 centroid :", centroids[2])
plotClusters(clusters, centroids)