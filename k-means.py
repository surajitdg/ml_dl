import numpy as np
import pandas as pd
from random import randint


class KMeans():

    def __init__(self, k, max_iterations, X):
        self.k = k   
        self.max_iterations = max_iterations
        self.X = X
        self.cluster_no = {}

    def init_centroids(self):
        no_of_samples, dimension = np.shape(self.X)
        self.centroids = np.zeros((self.k,dimension))
        for i in range(0,self.k):
            self.centroids[i] = X[randint(0,no_of_samples-1)]
    
    def euclidean_distance(self, x1, x2):
        """
        x = sqr_root(x^2-y^2)
        """
        assert np.shape(x1) == np.shape(x2)
        dist = np.sqrt(np.sum(np.power(np.subtract(x1,x2),2)))
        return dist
    
    def nearest_neighbours(self, sample):
        min_dist = 9999999999
        cluster_no = 0
        for i in range(0,len(self.centroids)):
            dist = self.euclidean_distance(sample, self.centroids[i])
            if dist < min_dist:
                min_dist = dist
                cluster_no = i
     
        return cluster_no, min_dist
    
    def assign_cluster(self):
        self.cluster_no = {}
        for i in range(0,len(self.X)):
            c , _ = self.nearest_neighbours(self.X[i])
            # print ('Cluster no',c)
            if c not in self.cluster_no:
                self.cluster_no[c] = [self.X[i]]
            else:
                self.cluster_no[c].append(self.X[i])    

    def cluster_means(self):
        for k, v in self.cluster_no.items():
            self.centroids[k] = np.mean(self.cluster_no[k])

    def kmeans(self):
        self.init_centroids()
        for i in range(0,self.max_iterations):
            self.assign_cluster()
            self.cluster_means()
            
            

    def predict(self, sample):
        min_dist = 999999
        predict_cluster = 0
        for i in range(0,len(self.centroids)):
            dist = self.euclidean_distance(sample,self.centroids[i])
            if dist < min_dist:
                min_dist = dist
                predict_cluster = i
        return predict_cluster,min_dist



            


if __name__ == "__main__":
    X = [[0.1,0.3,0.5,0.6],
         [0.1,0.2,0.5,0.6],
         [0.1,0.3,0.9,0.6],
         [0.1,0.3,0.8,0.6],
         [0.2,0.2,0.5,0.6],
         [0.1,0.3,0.5,0.8],
         [0.2,0.3,0.5,0.6],
         [0.1,0.3,0.7,0.6],
         [0.1,0.5,0.5,0.6],
         [0.1,0.6,0.5,0.6],
         [0.41,0.41,0.41,0.41],
         [0.23,0.01,0.54,0.91],
         [0.11,0.21,0.58,0.09],
         [0.09,0.45,0.5,0.9],
         [0.01,0.32,0.99,0.72],
         [0.12,0.99,0.34,0.81],
         [0.89,0.25,0.53,0.05],
         [0.65,0.01,0.92,0.54],
         [0.03,0.81,0.5,0.9],
         ]
    sample = [0.9, 0.9, 0.9, 0.9]
    km = KMeans(4, 200, X)
    km.kmeans()
    print (km.cluster_no)
    print (km.centroids)
    # print (km.euclidean_distance(r[0],r[1]))
    predict_cluster,min_dist = km.predict(sample)
    print (f'predicted cluster {predict_cluster} with distance {min_dist}')




