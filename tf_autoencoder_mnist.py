"""MNIST dataset autoencoder walkthrough.
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Read in data
mnist = input_data.read_data_sets('tmp/data/', one_hot=True)

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
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
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


# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
y_true = X

# Define loss and optimizer, minimize squared error
cost = tf.treduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

# Intializing variables
init = tf.global_variables_initializer()

# Launch graph and initialize interactive session
sess = tf.InteractiveSession()
sess.run(init)

total_batch = int(mnist.train.num_examples / batch_size)

# Training cycle
for epoch in range(epochs):
    # Loop over each batch
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # Run optimization operation and cost function
        _, c = sess.run([optimizer, cost], feed_dict={batch_xs})
        # Display logs per epoch step
    if epoch % display_step == 0:
        print('*' * 30)
        print('Epoch: {epoch +1}')
        print('Cost: {c}')
        print('*' * 30)
print('*' * 30)
print('Optimization Finished')
print('*' * 30)

# Apply encode and decode over the  test set
encode_decode = sess.run(y_pred,
                         feed_dict={X: mnist.test.images[:examples_to_show]})

# Compare original images with their reconstructions
f, a = plt.subplots(2, 10, figsize=(10, 2))
for i in range(examples_to_show):
    a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
    a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
