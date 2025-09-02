import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.src import initializers
from layers import DenseLayer, RNNCell, LSTMCell, TransformerEncoder
from loss import MSE, MAE

class LinReg(keras.Model):
    """
    Single layer , ideal for regressions
    units: no of units for output layer, e.g. 1 for normal linear regression
    """
    def __init__(self, units):
        super().__init__()
        self.dense = DenseLayer(units)

    def call(self, inputs):
        return self.dense(inputs)
    
class FeedForwardNetwork(keras.Model):
    """
    FF network
    layers: type:dict -- {'<layer_name>':<layer_units>}
    out_units: int -- output layer, e.g 1 for regression.
    
    """
    def __init__(self, layers_hidden={'layer_1':32,'layer_2':16}, out_units = 1):
        super().__init__()
        self.dense = {}
        for k,v in layers_hidden.items():
            self.dense[k] = DenseLayer(v)
        self.out = DenseLayer(out_units)

    def call(self, inputs):
        temp = inputs
        for k, v in self.dense.items():
            temp = self.dense[k](temp)
        out = self.out(temp)
        return out
    
    

class CustomRNN(keras.Model):
    def __init__(self, units, return_sequences=False, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.return_sequences = return_sequences
        self.cell = RNNCell(units)

    def call(self, inputs):
        """
        inputs: (batch, timesteps, features)

        """

        batch_size = tf.shape(inputs)[0]
        timesteps = tf.shape(inputs)[1]
        # print (inputs.shape)

        state = tf.zeros((batch_size, self.units))

        outputs = []

        for t in range(timesteps):
            # loop over each timestamp, take all of batch, only t -th time index of timestamp
            # and all of features dimensions
            x_t = inputs[:, t , :]
            state, [state] = self.cell(x_t, [state])
            if self.return_sequences:
                # append states if return sequences true
                outputs.append(state)

        #give recurrence output
        if self.return_sequences:
            return tf.stack(outputs,axis=1)
        else:
            return state


        #return state

class CustomLSTM(keras.Model):

    def __init__(self, units, return_sequences=False, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.return_sequences = return_sequences
        self.cell = LSTMCell(units)

    def call(self, inputs):

        batch_size = tf.shape(inputs)[0]
        timesteps = tf.shape(inputs)[1]

        ht = tf.zeros((batch_size, self.units))
        ct = tf.zeros((batch_size, self.units))

        outputs = []

        for t in range(inputs.shape[1]):
            x_t = inputs[ : , t , : ]
            ht, [ht, ct] = self.cell(x_t, [ht, ct])
            if self.return_sequences:
                outputs.append(ht)

        
        #give recurrence output
        if self.return_sequences:
            return tf.stack(outputs,axis=1) # (batch, timesteps, units)
        else:
            return ht


class TransformerModel(keras.Model):
    def __init__(self, no_layers, max_seq_length, feature_dim, vocab_size, num_heads, dropout_rate, dff, no_classes, **kwargs):
        super().__init__(**kwargs)
        self.max_seq_length = max_seq_length
        self.transformer_encoder_1 = TransformerEncoder(no_layers,max_seq_length,feature_dim,vocab_size,num_heads,dropout_rate,dff)
        self.classifier = DenseLayer(no_classes)

    def create_causal_mask(self, seq_len):
        mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        return mask
    
    def build(self, input_shape):
        super().build(input_shape)


    def call(self, inputs, training=True, mask=None):

        # layer = AttentionCustom(max_seq_length=10, units=8)
        # x = tf.random.normal((10000, 200, 128))  # batch=2, seq=10, features=16
        mask = self.create_causal_mask(self.max_seq_length)
        enc = self.transformer_encoder_1(inputs, training=training, mask=mask)
        final_logits = tf.reduce_mean(enc, axis=1)
        out = self.classifier(final_logits)
        distributions = tf.nn.softmax(out)
        return distributions









    



if __name__ == "__main__":
    # ## linear regression with one output
    # lr = LinReg(1)
    # X = [[0.1,0.3,0.5,0.6],
    #      [0.1,0.2,0.5,0.6],
    #      [0.1,0.3,0.9,0.6],
    #      [0.1,0.3,0.8,0.6],
    #      [0.2,0.2,0.5,0.6],
    #      [0.1,0.3,0.5,0.8],
    #      [0.2,0.3,0.5,0.6],
    #      [0.1,0.3,0.7,0.6],
    #      [0.1,0.5,0.5,0.6],
    #      [0.1,0.6,0.5,0.6],
    #      [0.41,0.41,0.41,0.41],
    #      [0.23,0.01,0.54,0.91],
    #      [0.11,0.21,0.58,0.09],
    #      [0.09,0.45,0.5,0.9],
    #      [0.01,0.32,0.99,0.72],
    #      [0.12,0.99,0.34,0.81],
    #      [0.89,0.25,0.53,0.05],
    #      [0.65,0.01,0.92,0.54],
    #      [0.03,0.81,0.5,0.9],
    #      ]
    # y = [0.34,
    #      0.42,
    #      0.85,
    #      0.90,
    #      0.83,
    #      0.97,
    #      0.88,
    #      0.54,
    #      0,65,
    #      0.99,
    #      0.91,
    #      0.87,
    #      0.95,
    #      0.11,
    #      0.23,
    #      0.45,
    #      0.70,
    #      0.48]
    # # lr.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),loss='mse', metrics=['mse'])
    # # lr.fit(x=np.array(X),y=np.array(y),epochs=100)
    # # lr.summary()

    # #ff-network
    # layers = {'layer_1':32,'layer_2':16,'layer_3':4}
    # ff = FeedForwardNetwork(layers_hidden=layers)
    # ff.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),loss=MSE(), metrics=['mse', MAE()])
    # ff.fit(x=np.array(X),y=np.array(y),epochs=100)
    # ff.summary()


    # Parameters
    # batch_size = 3
    # seq_length = 5
    # dimension = 4
    # num_classes = 2

    # # Generate random data
    # X = np.random.rand(batch_size, seq_length, dimension)

    # # Generate random labels
    # y = np.random.randint(0, num_classes, size=(batch_size,))
    # m = CustomRNN(2)
    # m.compile()
    # m.fit(X,y, epochs=2)
    # m.summary()

    # config
    batch_size = 2048
    timesteps = 10
    features = 32
    num_classes = 5

    # random input (batch, timesteps, features)
    X = tf.random.normal((batch_size, timesteps, features))

    # random integer labels (batch,)
    y = tf.random.uniform((batch_size,num_classes))
    # print (y)

    # x = CustomRNN(32, return_sequences=True)(inputs)
    # inputs = tf.keras.Input(shape=(timesteps,features))
    # out = CustomRNN(512, return_sequences=True)(inputs)
    # out = CustomRNN(1024, return_sequences=False)(out)

    # outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(out)

    # model = tf.keras.Model(inputs, outputs)
    # model.summary()
    # model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0),loss='categorical_crossentropy',metrics=['accuracy'])
    # model.fit(X, y, epochs=10, batch_size=16)


    # inputs = tf.keras.Input(shape=(timesteps,features))
    # out = CustomLSTM(512, return_sequences=True)(inputs)
    # out = CustomLSTM(1024, return_sequences=False)(out)

    # outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(out)

    # model = tf.keras.Model(inputs, outputs)
    # model.summary()
    # model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0),loss='categorical_crossentropy',metrics=['accuracy'])
    # model.fit(X, y, epochs=10, batch_size=16)


    #transformer

    model = TransformerModel(
        no_layers=2, feature_dim=128, num_heads=4, dff=512,
        vocab_size=100000, max_seq_length=1024, no_classes=5,
        dropout_rate=0.001
    )

    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    # fake data for demo
    X = np.random.randint(0, 100000, (32, 1024))
    y = np.random.randint(0, 5, (32,))
    
    model.fit(X, y, epochs=2)
    model.summary()
