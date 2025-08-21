import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.src import initializers

class MSE(keras.Loss):
    def call(self, y_true, y_pred):
        return tf.reduce_mean(((y_true-y_pred)**2))
    
class MAE(keras.Loss):
    def call(self, y_true, y_pred):
        return tf.reduce_mean(abs(y_true-y_pred))
    
