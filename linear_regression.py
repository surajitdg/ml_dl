import tensorflow as tf
from tensorflow import keras
import numpy as np
from models import LinReg
from d2l import tensorflow as d2l

 

class LR(d2l.Module):

    def __init__(self, features, lr=0.01):
        super().__init__()
        self.lr = lr
        w = tf.zeros(shape=(features,1))
        b = tf.zeros(shape=(1))
        self.w = tf.Variable(w, trainable=True)
        self.b = tf.Variable(b, trainable=True)

    def forward(self, X):
        return tf.matmul(X,self.w)+self.b

    # def call(self, X):
    #     y_pred = tf.matmul(X,self.w)+self.b
    #     mse = self.mean_sq_error(self.y_true, y_pred)
    #     self.add_loss(mse)
    #     return y_pred
    #     # return tf.matmul(X,self.w)+self.b
    
    def loss(self, y_hat, y):
        return tf.reduce_mean(((y_hat-y)**2)/2)
    
    def configure_optimizers(self):
        return SGD(self.lr)
    

class SGD(d2l.HyperParameters):  #@save
    """Minibatch stochastic gradient descent."""
    def __init__(self, lr):
        self.save_hyperparameters()

    def apply_gradients(self, grads_and_vars):
        for grad, param in grads_and_vars:
            param.assign_sub(self.lr * grad)

@d2l.add_to_class(d2l.Trainer)  #@save
def prepare_batch(self, batch):
    return batch

@d2l.add_to_class(d2l.Trainer)  #@save
def fit_epoch(self):
    self.model.training = True
    for batch in self.train_dataloader:
        with tf.GradientTape() as tape:
            loss = self.model.training_step(batch)
        grads = tape.gradient(loss, self.model.trainable_variables)
        if self.gradient_clip_val > 0:
            grads = self.clip_gradients(self.gradient_clip_val, grads)
        self.optim.apply_gradients(zip(grads, self.model.trainable_variables))
        self.train_batch_idx += 1
    if self.val_dataloader is None:
        return
    self.model.training = False
    for batch in self.val_dataloader:
        self.model.validation_step(self.prepare_batch(batch))
        self.val_batch_idx += 1

    

if __name__ == "__main__":
    # l_r = LR(features=4, learning_rate=0.01)
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
    # l_r.compile(optimizer='adam')
    # l_r.set_y(np.array(y))
    # l_r.fit(x=np.array(X),y=np.array(y),epochs=100,batch_size=4)
    model = LR(2, lr=0.03)
    data = d2l.SyntheticRegressionData(w=tf.constant([2, -3.4]), b=4.2)
    print (data)
    trainer = d2l.Trainer(max_epochs=3)
    trainer.fit(model, data)
    
