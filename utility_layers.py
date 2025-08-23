import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.src import initializers


class LayerNorm(keras.Layer):
    def __init__(self, epsilon, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        
    def build(self, input_shape):
        self.gamma = self.add_weight(shape=(input_shape[-1],),
                                     initializer='uniform',
                                     trainable=True,
                                     name='gamma_scale')
        self.beta = self.add_weight(shape=(input_shape[-1],),
                                    initializer='zeros',
                                    trainable=True,
                                    name='beta_shift')
        super().build(input_shape)

    
    def call(self, inputs):
        # features = tf.shape(inputs)[2]
        mean = tf.math.reduce_mean(inputs, axis=-1, keepdims=True) #(bath, seq, 1)
        variance = tf.math.reduce_variance(inputs, axis=-1, keepdims=True) # (batch, seq, 1)
        normalized = tf.subtract(inputs, mean)/tf.math.sqrt(variance+self.epsilon) #(batch, seq, features)
        mean_scaled = self.gamma * normalized
        out = tf.add(mean_scaled,self.beta)
        return out
    

if __name__ == "__main__":
    x = tf.random.normal((2,4,16))
    l = LayerNorm(epsilon=0.0001)
    out = l(x)
    print (out.shape)






        