import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from utils import tile_raster_images

# Deep Learning applied to MNIST

# Intialize interactive session
sess = tf.InteractiveSession()

# Read in mnist data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Create general parameters for the model

# Width of the image in pixels
width = 28
# Height of the image in pixels
height = 28
# Number of total pixels
flat = width * height
# Number of possible classifications
class_output = 10

# Create placeholders for inputs and outputs
x = tf.placeholder(tf.float32, shape=[None, flat])
y_ = tf.placeholder(tf.float32, shape=[None, class_output])

# Convert images of the data set to tensors
x_image = tf.reshape(x, [-1, 28, 28, 1])

# --- Convolutional layer 1 ---

# Define kernel weights & biases
W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))

# Define function to create convolutional layers
convolve1 = (tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1],
             padding='SAME') + b_conv1)

# Apply the ReLU activation function
h_conv1 = tf.nn.relu(convolve1)

# Define a max pooling function
h_pool1 = tf.nn.max_pool(h_conv1,
                         ksize=[1, 2, 2, 1],
                         strides=[1, 2, 2, 1],
                         padding='SAME')

# First layer completed
layer1 = h_pool1

# --- Convolutional layer 2 ---

# Define kernel weights & biases
W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))

# Define function to create convolutional layers
convolve2 = (tf.nn.conv2d(layer1,
                          W_conv2,
                          strides=[1, 1, 1, 1],
                          padding='SAME')) + b_conv2

# Apply ReLU activation function
h_conv2 = tf.nn.relu(convolve2)

# Apply max pooling
h_pool2 = tf.nn.max_pool(h_conv2,
                         ksize=[1, 2, 2, 1],
                         strides=[1, 2, 2, 1],
                         padding='SAME')

# Second layer complete
layer2 = h_pool2

# --- Layer 3 ---

# Flatten second layer
layer2_matrix = tf.reshape(layer2, [-1, 7 * 7 * 64])

# Weights and biases between level 2 and 3
W_fcl = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1))
# Number of biases must match number of outputs
b_fcl = tf.constant(0.1, shape=[1024])

# Apply weightws and biases
fcl3 = tf.matmul(layer2_matrix, W_fcl) + b_fcl

# Apply ReLU activation function
h_fcl = tf.nn.relu(fcl3)

# Third layer complete
layer3 = h_fcl

# --- Define dropout to reduce overfitting ---

keep_prob = tf.placeholder(tf.float32)
layer3_drop = tf.nn.dropout(layer3, keep_prob)

# --- Layer 4, softmax readout layer ---

# Weights & Biases
W_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))

# Matrix multiplication to apply wights & biases
fcl4 = tf.matmul(layer3_drop, W_fc2) + b_fc2

# Apply softmax activation function
y_conv = tf.nn.softmax(fcl4)
layer4 = y_conv

# --- Define functions and train the model ---

# Define loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ *
                                              tf.log(layer4),
                                              reduction_indices=[1]))


# Define optimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# Define prediction
correct_prediction = tf.equal(tf.argmax(layer4, 1), tf.argmax(y_, 1))

# Define accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initialize variables
sess.run(tf.global_variables_initializer())

# Run the model
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
                                                  x: batch[0],
                                                  y_: batch[1],
                                                  keep_prob: 1.0})
        print('step %d, training accuracy %g' % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: .5})

print('test accuracy: %g' % accuracy.eval(feed_dict={x: mnist.test.images,
                                                     y_: mnist.test.labels,
                                                     keep_prob: 1.0}))
# --- Viz ---

kernels = sess.run(tf.reshape(tf.transpose(W_conv1,
                                           perm=[2, 3, 0, 1]),
                              [32, -1]))
image = Image.fromarray(tile_raster_images(kernels,
                                           img_shape=(5, 5),
                                           tile_shape=(4, 8),
                                           tile_spacing=(1, 1)))
# Plot image
plt.rcParams['figure.figsize'] = (18.0, 18.0)
imgplot = plt.imshow(image)
imgplot.set_cmap('gray')

# Output of image passing through the first convolution layer
plt.rcParams['figure.figsize'] = (5.0, 5.0)
sampleimage = mnist.test.images[1]
plt.imshow(np.reshape(sampleimage, [28, 28]), cmap='gray')

ActivatedUnits = sess.run(convolve1,
                          feed_dict={x: np.reshape(sampleimage, [1, 784],
                                                   order='F'),
                                     keep_prob: 1.0})
filters = ActivatedUnits.shape[3]
plt.figure(1, figsize=(20, 20))
n_columns = 6
n_rows = np.math.ceil(filters / n_columns) + 1
for i in range(filters):
    plt.subplot(n_rows, n_columns, i + 1)
    plt.title('Filter ' + str(i))
    plt.imshow(ActivatedUnits[0, :, :, i],
               interpolation='nearest',
               cmap='gray')

# Output of image passing through the second convolution layer

ActivatedUnits = sess.run(convolve2,
                          feed_dict={x: np.reshape(sampleimage,
                                                   [1, 784], order='F'),
                                     keep_prob: 1.0})
filters = ActivatedUnits.shape[3]
plt.figure(1, figsize=(20, 20))
n_columns = 8
n_rows = np.math.ceil(filters / n_columns) + 1
for i in range(filters):
    plt.subplot(n_rows, n_columns, i+1)
    plt.title('Filter ' + str(i))
    plt.imshow(ActivatedUnits[0, :, :, i],
               interpolation="nearest",
               cmap="gray")
sess.close()
