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
    
class AttentionCustom(keras.Layer):

    def __init__(self, max_seq_length, units, **kwargs):
        super().__init__(**kwargs)
        self.max_seq_length = max_seq_length
        self.units = units

    def build(self, input_shape):

        # key
        self.weight_key = self.add_weight(shape=(input_shape[-1], self.units),
                                            initializer='uniform',
                                            trainable=True,
                                            name='weight_key')

        # query
        self.weight_query = self.add_weight(shape=(input_shape[-1], self.units),
                                            initializer='uniform',
                                            trainable=True,
                                            name='weight_query')

        # value
        self.weight_value = self.add_weight(shape=(input_shape[-1], self.units),
                                            initializer='uniform',
                                            trainable=True,
                                            name='weight_value')
        

        super().build(input_shape)

    
    def apply_mask(self, logits, mask):
        if mask is not None:
            logits = tf.add(logits, (mask)*-1e9)
        return logits

    def call(self, inputs, mask=None):
        q = tf.matmul(inputs, self.weight_key)
        k = tf.matmul(inputs, self.weight_query)
        v = tf.matmul(inputs, self.weight_value)

        scaled_logits = tf.matmul(q, k, transpose_b=True)/tf.math.sqrt(tf.cast(self.units, tf.float32))
        masked_logits = self.apply_mask(scaled_logits, mask)

        attention_weights = tf.nn.softmax(masked_logits)
        attention = tf.matmul(attention_weights, v)
        return attention
    


class MultiHeadAttention(keras.Layer):

    def __init__(self, max_seq_length, units, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.max_seq_length = max_seq_length
        self.units = units
        self.num_heads = num_heads
        self.depth_head = self.units//self.num_heads

    def build(self, input_shape):

        # key
        self.weight_key = self.add_weight(shape=(input_shape[-1], self.units),
                                            initializer='uniform',
                                            trainable=True,
                                            name='weight_key')

        # query
        self.weight_query = self.add_weight(shape=(input_shape[-1], self.units),
                                            initializer='uniform',
                                            trainable=True,
                                            name='weight_query')

        # value
        self.weight_value = self.add_weight(shape=(input_shape[-1], self.units),
                                            initializer='uniform',
                                            trainable=True,
                                            name='weight_value')
        
        #final projection
        self.weight_o = self.add_weight(shape=(self.units, self.units),
                                            initializer='uniform',
                                            trainable=True,
                                            name='weight_o')
        

        super().build(input_shape)


    def split_heads(self, inp, batch_size):
        """
        inp: (batch, max_seq, units)
        out: (batch, num_heads, max_seq, depth_head)
        
        """
        seq = tf.shape(inp)[1]
        out = tf.reshape(inp, shape=(batch_size, seq , self.num_heads, self.depth_head))
        out = tf.transpose(out, perm=[0, 2, 1, 3])
        return out
    
    def apply_mask(self, logits, mask):
        if mask is not None:
            logits = tf.add(logits, (mask)*-1e9)
        return logits

    def call(self, inputs, mask=None):

        batch_size = tf.shape(inputs)[0]
        seq = tf.shape(inputs)[1]
        q = tf.matmul(inputs, self.weight_key)
        k = tf.matmul(inputs, self.weight_query)
        v = tf.matmul(inputs, self.weight_value)

        #split heads
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        #scaled attention
        logits = tf.matmul(q, k, transpose_b=True)/tf.math.sqrt(tf.cast(self.units, tf.float32))
        logits_masked = self.apply_mask(logits, mask)
        attention_weights = tf.nn.softmax(logits_masked)
        attention = tf.matmul(attention_weights, v) #output now: batch, head, seq, depth

        #combine heads
        attention = tf.transpose(attention, perm=[0,2,1,3]) # back to batch, seq, head, depth
        concat_attention = tf.reshape(attention, shape=(batch_size, seq, self.units))

        #final output
        output = tf.matmul(concat_attention, self.weight_o)

        return output






        

        
    

    


if __name__ == "__main__":
    # rnn = RNNCell(5,None)
    # out = rnn.build(input_shape=(3,4,16))
    # print (out)
    # rnn.summary()

    def create_causal_mask(seq_len):
        mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        return mask

    # layer = AttentionCustom(max_seq_length=10, units=8)
    x = tf.random.normal((2, 10, 16))  # batch=2, seq=10, features=16
    mask = create_causal_mask(10)
    layer = MultiHeadAttention(max_seq_length=10, units=32, num_heads=4)
    out = layer(x, mask=mask)
    print(out.shape)