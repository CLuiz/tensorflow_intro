import matplotlib.pyplot as plt
import numpy as np
# from PIL import Image look for alternative python3 compatible
from scipy import signal as sg
from scipy import misc
import tensorflow as tf

h = [2, 1, 0]
x = [3, 4, 5]

y1 = np.convolve(x, h)
print(f'Standard numpy implementation of convolve: {y1}')
print(f'Compare with the following values in Python: \
        {y1[0]}; {y1[1]}; {y1[2]}; {y1[3]}; {y1[4]}')

# Effects of using zero padding
h = [1, 2, 5, 4]
x = [6, 2]

y2 = np.convolve(x, h, 'full')
print(f'"Full" numpy implementation of convolve: {y2}')

# Effect of 'same' in numpy convolve
y3 = np.convolve(x, h, 'same')
print(f'"Same" numpy implementation of convolve: {y3}')

# Effect of 'valid' in numpy convolve
y4 = np.convolve(x, h, 'valid')
print(f'"Valid" numpy implementation of convolve: {y4} \n')

print('*' * 50)
print('2d Scipy.signal implementation of convolution')
print('(1x2 kernel)')
print('*' * 50)

# 2d convolutional operations with scipy.signal
I = [[255,   7,   3],
     [212, 240,   4],
     [218, 216, 230],
     ]
g = [[-1, 1]]

print('Without zero padding, "valid" & 1x2 kernel: \n')
print(f'{sg.convolve(I, g, "valid")} \n')

# The 'valid' argument states that the output consists only of
# those elements that do not rely on the zero-padding.
print('With zero padding, "full", & 1x2 kernel: \n')
print(f'{sg.convolve(I, g)} \n')

# 2x2 kernel implementation w/ scipy.signal
print('*' * 50)
print('2d Scipy.signal implementation of convolution')
print('(2x2 kernel)')
print('*' * 50)

I = [[255,   7,   3],
     [212, 240,   4],
     [218, 216, 230],
     ]
g = [[-1, 1],
     [2, 3]]

# The output of the below is the full discreet linear convolution of the inputs
# It will use zero to complete the linear matrix.
print('With zero padding, "full", & 2x2 kernel: \n')
print(f"{sg.convolve(I, g, 'full')} \n")

# The output of the below is the full discreet linear convolution of the inputs
# It will use zero to complete the linear matrix.
print('With zero padding, "same", & 2x2 kernel: \n')
print(f"{sg.convolve(I, g, 'same')} \n")

# The 'valid argument states that the output consists of only those elements
# that do not really on zero padding
print('Without zero padding, "valid", & 2x2 kernel: \n')
print(f"{sg.convolve(I, g, 'valid')} \n")

# Tensorflow implementatation of convolution
print('*' * 50)
print('Tensorflow implementatation of convolution')
print('*' * 50)

# Build graph

input = tf.Variable(tf.random_normal([1, 10, 10, 1]))
filter = tf.Variable(tf.random_normal([3, 3, 1, 1]))
op = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')
op2 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')

# Initialize variables and session
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    print('Input \n')
    print(f'{input.eval()} \n')
    print('Filter/Kernel \n')
    print(f'{filter.eval()} \n')
    print('Result/Feature Map with valid positions \n')
    print(f'{sess.run(op)}')
    print('\n')
    print(f'{sess.run(op2)}')

# Convolution applied on images

# obtain image and save to local directory
# wget --quiet https://ibm.box.com/shared/static/cn7yt7z10j8rx6um1v9seagpgmzzxnlz.jpg --output-document bird.jpg
"""Need to replace this code with python3.6 compatible code"""
# im = Image.open('bird.jpg')
#
# # Use the ITU-R 601-2 Luma transform to convert image to grey scale
#
# image_gr = im.comvert('L')
# print('\n Original type: %r \n\n' % image_gr)
#
# # Convert image to a matrix with values from 0 to 255 (uint8)
# arr = np.asarray(image_gr)
# print('After conversion to numerical representation: \n\n %r' % arr)
#
# # Plot image
# imgplot = plt.imshow(arr)
# imgplot.set_cmap('gray')
# print('\n Input image converted to gray scale: \n')
# plt.show(imgplot)


# Create edge detector kernel

kernel = np.array([
                    [0, 1, 0],
                    [1, -4, 1],
                    [0, 1, 0]
                                ])
grad = sg.convolve2d(arr, kernel, mode='same', boundary='symm')

grad_biases = np.absolute(grad) + 100
grad_biases[grad_biases > 255] = 255

print('GRADIENT MAGNITUDE - Feature map')

fig, aux = plt.subplots(figsize=(10, 10))
aux.imshow(np.absolute(grad_biases), cmap='gray')
