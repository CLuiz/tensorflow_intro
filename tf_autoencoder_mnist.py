"""MNIST dataset autoencoder walkthrough.
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorails.mnist import input_data

# -- Set parameters ---
learning_rate = 0.01
epochs = 20
batch_size = 256
display_step = 1
examples_to_show = 10

# First layer number of features
n_hidden_1 = 256
# Second layer number of features
n_hidden_2 = 128
# MNIST data input (img shape: 28 * 28)
n_input = 784

# Set tf placeholder for images
X = tf.placeholder('float', [None, n_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input. n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input]))
    }

biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([n_input]))
    }


# Build encoder
def encoder(x):
    # Encode first layer with sigmoid activation function
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Encode second layer with sigmoid activation function
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2


# Build decoder
def decoder(x):
    # First layer with sigmoid activation
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Second layer with sigmoid activation
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h1']),
                                   biases['decoder_b1']))
    return layer_2

# Construct madel
