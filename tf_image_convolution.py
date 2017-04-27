import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import misc
from scipy import signal
import tensorflow as tf

# img download location :
# https://ibm.box.com/shared/static/o4x2cvlqfre9lax05ihbn47cxpnzwlvb.png

# read the image as Float data type
im = misc.imread('lena.png')

# Convert image to gray scale
grayim = np.dot(im[..., :3], [0.299, 0.587, 0.144])

# Plot images
plt.subplot(1, 2, 1)
plt.imshow(im)
plt.xlabel(' Float Image ')

plt.subplot(1, 2, 2)
plt.imshow(grayim, cmap=plt.get_cmap('gray'))
plt.xlabel(' Gray Scale Image ')

# print shape of grayscale image
print(f"The shape of the grayscale image is: {grayim.shape}")

# --- Extend the Dimensions of the Gray Scale Image ---

""" For convolution Tensorflow acepts images in dimensions:
[num of images, width, height, channels]
In this case we are looking for dimensions of [1, 512, 512, 1] from the sahpe
of the image, (512, 512)
"""

image = np.expand_dims(np.expand_dims(grayim, 0), -1)

print(f"The shape of the expanded grayscale image is: {image.shape}")

# Create placeholder for the input image and prnt the shape
img_placeholder = tf.placeholder(tf.float32, shape=[None, 512, 512, 1])
print(f"The shape of the image placeholder is: {img_placeholder.get_shape()}")

# Create a variable for the weight matrix and print out the shape
weights = tf.Variable(tf.truncated_normal([5, 5, 1, 1],
                      stddev=0.05),
                      dtype=tf.float32)

print(f"The shape of the weight matrix is: {weights.get_shape().as_list()}")

# Create two convolution graphs in tensorflow

convolve1 = tf.nn.conv2d(input=img_placeholder,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

convolve2 = tf.nn.conv2d(input=img_placeholder,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='VALID')

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Run sessions and get results for the two convolutional operations

result1 = sess.run(convolve1, feed_dict={img_placeholder: image})
result2 = sess.run(convolve2, feed_dict={img_placeholder: image})

# Reduce dimensions & reshape images to original dimensions
vec1 = np.reshape(result1, (1, -1))
image1 = np.reshape(vec1, (512, 512))

vec2 = np.reshape(result2, (1, -1))
image2 = np.reshape(vec2, (508, 508))
print('\nThe shape of the resulting images are:')
print(f'Image 1: {image1.shape}')
print(f'Image 2: {image2.shape}')

# Plot the images post convolution
plt.subplot(1, 2, 1)
plt.imshow(image1, cmap=plt.get_cmap('gray'))
plt.xlabel(' SAME Padding')

plt.subplot(1, 2, 2)
plt.imshow(image2, cmap=plt.get_cmap('gray'))
plt.xlabel(" VALID Padding ")
# plt.show()

# --- Create first convolutional neural network layer ---


def conv2d(X, W):
    conv = tf.nn.conv2d(input=X,
                        filter=W,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    return conv


def max_pool(X):
    mpool = tf.nn.max_pool(X,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME')
    return mpool


W_conv1 = tf.Variable(tf.random_normal([5, 5, 1, 32]))
b_conv1 = tf.Variable(tf.random_normal(shape=[32]))

# Define a tensorflow graph for relu, convolution, and max pooling

conv1 = tf.nn.relu(conv2d(image, weights['W_conv1']) + biases['b_conv1'])
maxpool = max_pool(conv1)

# Intialize all variables and run the session
init = tf.global_variables_initializer()
sess.run(init)

sess.close()
