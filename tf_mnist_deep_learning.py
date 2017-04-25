import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

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
x_image = tf.reshape(x, [-1,28,28,1])

#### Convolutional layer 1 ####

# Define kernel weights & biases
W_conv1 = tf.Variable(tf.truncated_normal([5,5,1,32], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))

# Define function to create convolutional layers
convolve1 = (tf.nn.conv2d(x_image, W_conv1, strides=[1,1,1,1], padding='SAME')
            + b_conv1)

# Apply the ReLU activation function
h_conv1 = tf.nn.relu(convolve1)

# Define a max pooling function
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1,2,2,1], strides=[1,2,2,1])

# First layer completed
layer1 = h_pool1

#### Convolutional layer 2 ####

# Define kernel weights & biases
W_conv2 = tf.Variable(tf.truncated_normal([5,5,32,64], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))

# Define function to create convolutional layers
convolve2 = (tf.nn.conv2d(layer1, W_conv2, strides=[1,1,1,1], padding='SAME'))
            + b_conv2)

# Apply ReLU activation function
h_conv2 = tf.nn.relu(convolve2)

# Apply max pooling
h_pool2 = tf.nn.max_pool(h_conv2, ksize==[1,2,2,1], strides=[1,2,2,1])

# Second layer complete
layer2 = h_pool2
