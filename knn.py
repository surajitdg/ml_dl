import numpy as np

class KNN():

    def __init__(self, k, X, y):
        self.k = k
        self.X = X
        self.y = y

    def majority(self, values):
        freq = {}
        for v in values:
            if v not in freq:
                freq[v] = 1
            else:
                freq[v] = freq[v]+1

        freq = dict(sorted(freq.items(), key= lambda item:item[1], reverse=True))
        print ('label distribution -- ',freq)
        return list(freq.keys())[0]
    
    def euclidean_distance(self, x1, x2):
        """
        x = sqrt(Σ (x_i - y_i)²)
        """
        assert np.shape(x1) == np.shape(x2)
        dist = np.sqrt(np.sum(np.power(np.subtract(x1,x2),2)))
        return float(dist)
    
    def get_neighbours(self, x1):
        dist = [] #use list other the dict, since same dist can be possible
        for i in range(0,len(self.X)):
            d = self.euclidean_distance(x1,self.X[i])
            dist.append((d, self.y[i]))
        
        dist.sort(key=lambda x:x[0])
        # print (m)
        return [l for _,l in dist[:self.k]]


        
    
    def predict(self, X_test):
        y_pred = np.zeros(shape=(np.shape(X_test)[0],1))
        for i in range(0,len(X_test)):
            candidates = self.get_neighbours(X_test[i])
            l = self.majority(candidates)
            y_pred[i] = l

        return y_pred


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
    y =  [0,
          0,
          1,
          0,
          0,
          0,
          0,
          0,
          0,
          1,
          1,
          1,
          1,
          0,
          0,
          1,
          1,
          1,
          1]
    
    knn = KNN(k=5, X=X,y=y)
    X_test = [[0.2,0.3,0.5,0.6],
         [0.15,0.3,0.71,0.6],
         [0.1,0.53,0.5,0.6],
         [0.8,0.76,0.5,0.67]]
    res = knn.predict(X_test)
    print (res)
