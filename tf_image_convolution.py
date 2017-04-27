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

""" For convolution Tensorflow acepts images if dimensions:
[num of images, width, height, channels]
In this case we are looking for dimensions of [1, 512, 512, 1] from the sahpe
of the image, (512, 512)
"""

img = np.expand_dims(np.expand_dims(grayim, 0), -1)

print(f"The shape of the expanded grayscale image is: {img.shape}")

# Create placeholder for the input image and prnt the shape
img_placeholder = tf.placeholder(tf.float32, shape=[None, 512, 512, 1])
print(f"The shape of the image placeholder is: {img_placeholder.get_shape().as_list()}")

# Create a variable for the weight matrix and print out the shape
weights = tf.Variable(tf.truncated_normal([5, 5, 1, 1], stddev=0.05))
print(f"The shape of the weight matrix is: {weights.get_shape().as_list()}")

# Create two convolution graphs in tensorflow

convolve1 = tf.nn.conv2d(input=img,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

convolve2 = tf.nn.convolv2d(input=img,
                            filter=weights,
                            strides=[1, 1, 1, 1],
                            padding='VALID')

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
