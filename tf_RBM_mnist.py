"""Restricted Boltzman Machine applied to MNIST dataset
"""
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from utils import tile_raster_images

# Load and split dataset
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
train_X, train_Y, test_X, test_Y = (mnist.train.images,
                                    mnist.train.labels,
                                    mnist.test.images,
                                    mnist.test.labels)
# Define visual unit and hidden unit biases
vb = tf.placeholder('float', [784])
hb = tf.placeholder('float', [500])

# Define weights
W = tf.placeholder('float', [784, 500])

# --- Define forward pass ---

X = tf.placeholder('float', [None, 784])

# Probabilities of the hidden units
_h0 = tf.nn.sigmoid(tf.matmul(X, W) + hb)

# Sample X given h
h0 = tf.nn.relu(tf.sign(_h0 - tf.random_uniform(tf.shape(_h0))))

# Print out example of sampling
with tf.Session() as sess:
    a = tf.constant([0.7, 0.1, 0.8, 0.2])
    print(sess.run(a))
    b = sess.run(tf.random_uniform(tf.shape(a)))
    print(b)
    print(sess.run(a-b))
    print(sess.run(tf.sign(a - b)))
    print(sess.run(tf.nn.relu(tf.sign(a - b))))

# --- Define backward pass (reconstruction) ---
_v1 = tf.nn.sigmoid(tf.matmul(h0, tf.transpose(W)) + vb)

# Sample v given h
v1 = tf.nn.relu(tf.sign(_v1 - tf.random_uniform(tf.shape(_v1))))
h1 = tf.nn.sigmoid(tf.matmul(v1, W) + hb)

# Learning rate & contrastive divergence
alpha = 1.0
w_pos_grad = tf.matmul(tf.transpose(X), h0)
w_neg_grad = tf.matmul(tf.transpose(v1), h1)
CD = (w_pos_grad - w_neg_grad) / tf.to_float(tf.shape(X)[0])
update_w = W + alpha * CD
update_vb = vb + alpha * tf.reduce_mean(X - v1, 0)
update_hb = hb + alpha * tf.reduce_mean(h0 - h1, 0)

# Set objective function
err = tf.reduce_mena(tf.square(X - v1))

# Start session and intitialize variables
cur_w = np.zeros([784, 500], np.float32)
cur_vb = np.zeros([784], np.float32)
cur_hb = np.zeros([500], np.float32)
