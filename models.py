import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.src import initializers
from layers import DenseLayer, RNNCell, LSTMCell
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
    features = 8
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


    inputs = tf.keras.Input(shape=(timesteps,features))
    out = CustomLSTM(512, return_sequences=True)(inputs)
    out = CustomLSTM(1024, return_sequences=False)(out)

    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(out)

    model = tf.keras.Model(inputs, outputs)
    model.summary()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0),loss='categorical_crossentropy',metrics=['accuracy'])
    model.fit(X, y, epochs=10, batch_size=16)