# ml_dl
ML and DL models with base modules like numpy, tensorflow. The intent of the repo is to understand how we can build some popular models from grounds up. 


# ML -- Creating all basic ML models using base packages only like numpy, pandas, list etc.

k-means = K means algo using numpy
pca = pca using numpy
logistic regression = using numpy simple gradient descent of log loss
linear regression = using grad desc as well pseudoinverse
decision tree = binary feature based simple decision tree
knn = k-nearest neighbours

# DL -- Creating popular deep learning models like DNN, RNN, LSTM , Multi-Head attention using base tensorflow packages.

layers = Base layers for model subclassed from keras.Layer
       - DenseLayer -  Basic FF network layer 
       - RNNCell -  single RNN cell
       - LSTMCell -  single LSTM cell
       - AttentionCustom - single head attention
       - MultiHeadAttention - multi head attention
       - SingleTransformerBlock -  single transformer block
       - TransformerEncoder -  transformer encoder block using single transformers blocks
loss = loss functions
utility_layers =  Other utility layers subclassed from keras.Layer
models = models by subclassing keras.Models
       - LinReg - simple linear regression model
       - FeedForwardNetwork - FF network with multiple layers
       - CustomRNN - RNN model using RNN cell
       - CustomLSTM - LSTM model using LSTM cells
       - TransformerModel - Transformer model using TransformerEncoder



