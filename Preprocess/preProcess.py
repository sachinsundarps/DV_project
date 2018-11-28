import pandas as pd
import collections
from sklearn import preprocessing
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyclustering.cluster.kmeans import kmeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
def preProcessData(csvFile):
	df = pd.read_csv(csvFile)
	df.drop(df.columns[[2,-4]], axis=1, inplace=True)
	df = df.groupby("institution", as_index=False).sum()
	institutions = df["institution"]
	df.drop(df.columns[[0]], axis=1, inplace=True)
	x = df.values #returns a numpy array
	min_max_scaler = preprocessing.MinMaxScaler()
	x_scaled = min_max_scaler.fit_transform(x)
	scaled = pd.DataFrame(x_scaled)
	return scaled, institutions

def findClusters(df, mappings, numClusters):
	clustersUniversites = collections.defaultdict(list)
	universityFeatures = collections.defaultdict(list)
	kmeans = KMeans(n_clusters=numClusters, random_state=0).fit(df)
	labels = kmeans.labels_
	listDf = df.values.tolist()
	for index in range(len(labels)):
		clustersUniversites[labels[index]].append(mappings[index])
		universityFeatures[mappings[index]].append(listDf[index])

	return clustersUniversites, universityFeatures

def createClusterData(clustersUniversites, universityFeatures):
	for clusterNo in clustersUniversites:
		clusterData = { university: universityFeatures[university][0] for university in clustersUniversites[clusterNo] }
		df = pd.DataFrame(clusterData)	
		df.T.to_csv("../intermediateData/" + str(clusterNo)+".csv", sep=',')

df, institutions = preProcessData("../data/cwurData.csv")
clustersUniversites, universityFeatures = findClusters(df, institutions, 5)
createClusterData(clustersUniversites, universityFeatures)
