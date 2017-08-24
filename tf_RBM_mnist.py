"""Restricted Boltzman Machine applied to MNIST dataset"""

import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from utils import tile_raster_images

# Quash annoying tf instruction errors:
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
print('Simple sampling example:')
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
err = tf.reduce_mean(tf.square(X - v1))

# Start session and intitialize variables
cur_w = np.zeros([784, 500], np.float32)
cur_vb = np.zeros([784], np.float32)
cur_hb = np.zeros([500], np.float32)
prv_w = np.zeros([784, 500], np.float32)
prv_vb = np.zeros([784], np.float32)
prv_hb = np.zeros([500], np.float32)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

print(f'First run error: {sess.run(err, feed_dict={X: train_X, W: prv_w, vb: prv_vb, hb: prv_hb})}')

# Define training loop
epochs = 5
batchsize = 100
weights = []
errors = []

for epoch in range(epochs):
    for start, end in zip(range(0, len(train_X), batchsize),
                          range(batchsize, len(train_X), batchsize)):
        batch = train_X[start:end]
        fdict = {X: batch, W: prv_w, vb: prv_vb, hb: prv_hb}

        cur_w = sess.run(update_w, feed_dict=fdict)
        cur_vb = sess.run(update_vb, feed_dict=fdict)
        cur_hb = sess.run(update_hb, feed_dict=fdict)
        prv_w = cur_w
        prv_vb = cur_vb
        prv_hb = cur_hb
        if start % 10000 == 0:
            errors.append(sess.run(err, feed_dict={X: train_X,
                                                   W: cur_w,
                                                   vb: cur_vb,
                                                   hb: cur_hb}))
            weights.append(cur_w)
    print(f'Epoch: {epoch}, reconstruction err: {errors[-1]}')
plt.plot(errors)
plt.xlabel('Batch Number')
plt.ylabel('Error')
plt.show()

sess.close()
