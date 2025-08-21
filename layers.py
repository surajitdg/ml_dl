import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.src import initializers


class DenseLayer(keras.Layer):
    """
    Dense layer
    """

    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        # print (input_shape[-1])
        self.weight = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='uniform',
                                      trainable=True,
                                      name='weight')
        self.bias = self.add_weight(shape=(self.units,),
                                      initializer='zeros',
                                      trainable=True,
                                      name='bias')
        
    def call(self, inputs):
        return tf.matmul(inputs, self.weight)+self.bias
    

class RNNCell(keras.Layer):

    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        # self.states = states
    
    def build(self, input_shape):
        # print (input_shape)
        self.wht = self.add_weight(shape=(input_shape[-1],self.units),
                                   initializer='uniform',
                                   trainable=True,
                                   name='weight_input')
        self.b = self.add_weight(shape=(self.units,),
                                initializer='zeros',
                                trainable=True,
                                name='bias')
        self.uht = self.add_weight(shape=(self.units,self.units),
                                    initializer='uniform',
                                    trainable=True,
                                    name='weight_recurrent')
        super().build(input_shape)
        # self.oht = self.add_weight(shape=(self.units,1),
        #                             initializer='uniform',
        #                             trainable=True,
        #                             name='weight_out')
        
    def call(self, inputs, states):
        #inputs expected as (batch, seq, features) if not please reshape before fitting the same
        prev_state = states[0]  # (batch, units)
        temp = tf.matmul(inputs, self.wht) + self.b
        output = tf.matmul(prev_state, self.uht) + temp
        output = tf.nn.tanh(temp)
        return output, [output]
    

class LSTMCell(keras.Layer):
    
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        #forget gate
        self.wht_f = self.add_weight(shape=(input_shape[-1],self.units),
                                   initializer='uniform',
                                   trainable=True,
                                   name='weight_forget')
        self.b_f = self.add_weight(shape=(self.units,),
                                initializer='zeros',
                                trainable=True,
                                name='bias_forget')
        self.uht_f = self.add_weight(shape=(self.units,self.units),
                                    initializer='uniform',
                                    trainable=True,
                                    name='weight_recurrent_forget')
        

        #input gate

        self.wht_i = self.add_weight(shape=(input_shape[-1],self.units),
                                   initializer='uniform',
                                   trainable=True,
                                   name='weight_input')
        self.b_i = self.add_weight(shape=(self.units,),
                                initializer='zeros',
                                trainable=True,
                                name='bias_input')
        self.uht_i = self.add_weight(shape=(self.units,self.units),
                                    initializer='uniform',
                                    trainable=True,
                                    name='weight_recurrent_input')
        
        #candidate weights
        self.wht_c = self.add_weight(shape=(input_shape[-1],self.units),
                                   initializer='uniform',
                                   trainable=True,
                                   name='weight_candidate')
        self.b_c = self.add_weight(shape=(self.units,),
                                initializer='zeros',
                                trainable=True,
                                name='bias_candidate')
        self.uht_c = self.add_weight(shape=(self.units,self.units),
                                    initializer='uniform',
                                    trainable=True,
                                    name='weight_recurrent_candidate')
        
        #output weights
        self.wht_o = self.add_weight(shape=(input_shape[-1],self.units),
                                   initializer='uniform',
                                   trainable=True,
                                   name='weight_output')
        self.b_o = self.add_weight(shape=(self.units,),
                                initializer='zeros',
                                trainable=True,
                                name='bias_output')
        self.uht_o = self.add_weight(shape=(self.units,self.units),
                                    initializer='uniform',
                                    trainable=True,
                                    name='weight_recurrent_output')
        super().build(input_shape)
        
    def call(self, inputs, states):
        """
        inputs = (batch, seq, features)
        
        """
        h_prev = states[0] #previous hidden
        c_prev = states[1] #previous carry
        ft = tf.nn.sigmoid((tf.matmul(inputs, self.wht_f)+self.b_f) + 
                                    tf.matmul(h_prev, self.uht_f))
        kt = tf.multiply(c_prev, ft)
        gt = tf.nn.tanh((tf.matmul(inputs, self.wht_i)+self.b_i) + 
                                    tf.matmul(h_prev, self.uht_i))
        it = tf.nn.sigmoid((tf.matmul(inputs, self.wht_c)+self.b_c) + 
                                    tf.matmul(h_prev, self.uht_c))
        jt = tf.multiply(gt, it)
        ct = jt+kt

        ot = tf.nn.sigmoid((tf.matmul(inputs, self.wht_o)+self.b_o) + 
                                    tf.matmul(h_prev, self.uht_o))
        ht = tf.multiply(ot, tf.nn.tanh(ct))

        return ht, [ht, ct]
        

        
    

    


if __name__ == "__main__":
    rnn = RNNCell(5,None)
    out = rnn.build(input_shape=(3,4,16))
    print (out)
    # rnn.summary()