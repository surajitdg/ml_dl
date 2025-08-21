import numpy as np

class Sigmoid():

    def __call__(self,x):
        sig = 1/(1+np.exp(-x))
        return sig