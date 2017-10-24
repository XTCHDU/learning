import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

n_inputs = 28
n_steps = 28
n_hidden_units = 128

def RNN(X,weights,biases):
    X = tf.reshape(X,[-1,n_inputs])
    X_in = tf.matmul(X,weights['in']+biases['in'])
    X_in = tf.reshape(X_in,[-1,n_steps,n_hidden_units])


    lstm