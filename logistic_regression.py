import numpy as np
import pandas as pd
from activate_functions import Sigmoid

class LogisticRegression():
    """
    X.w = y
    w = X^-1 * y

    X = (n_samples,)
    y = ()
    """
    def __init__(self, learning_rate) -> None:
        self.param = None
        self.learning_rate = learning_rate

    def init_params(self, X):
        n_features = np.shape(X)[1]
        self.param = np.random.randint(0,99, size=(n_features,))
        self.param = self.param/100

    def binary_cross_entropy(self, y_true, y_pred):
        # print (y_true, y_pred)
        bce = -np.mean(np.dot(y_true,np.log(y_pred)) + np.dot(np.subtract(1,y_true), np.log(np.subtract(1,y_pred))))
        #print (bce)
        return bce

    def fit(self, X, y, iterations=1000):
        self.init_params(X)
        sigmoid = Sigmoid()
        for i in range(0,iterations):
            y_pred = sigmoid(np.dot(X,self.param))
            print (f'loss for iteration {i} -->', self.binary_cross_entropy(y,y_pred))
            # gradient for bce  d(bce)/dw = -(^y-y)x
            # gradient descent  for LR
            # W(t+1) = W - @*(-(^y-y)x)
            # with regularization
            # W(t+1) = W - @*(-(^y-y)x + 2 * W)
            self.param = self.param - self.learning_rate*(np.dot(-(y-y_pred),X)+ 2*self.param)
            

    def predict(self,X):
        sigmoid = Sigmoid()
        y_pred = sigmoid(np.dot(X,self.param))
        return y_pred
    

if __name__ == "__main__":
    lr = LogisticRegression(learning_rate=0.1)
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
    # y = [1,
    #      1,
    #      0,
    #      1,
    #      1,
    #      1,
    #      1,
    #      1,
    #      1,
    #      0,
    #      0,
    #      0,
    #      0,
    #      1,
    #      1,
    #      0,
    #      0,
    #      0,
    #      0]
    lr.fit(X,y)
    print(lr.param)
    print (lr.predict([0.02,0.81,0.54,0.9]))
        


        


