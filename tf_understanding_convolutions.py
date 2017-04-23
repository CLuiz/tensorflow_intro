import numpy as np
from scipy import signal as sg
import tensorflow as tf

h = [2,1,0]
x = [3,4,5]

y1 = np.convolve(x,h)
print(f'Standard numpy implementation of convolve: {y1}')
print(f'Compare with the following values in Python: {y1[0]} ; {y1[1]}; {y1[2]}; {y1[3]}; {y1[4]}')

# Effects of using zero padding
h = [1,2,5,4]
x = [6,2]

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
     [218, 216, 230],]
g = [[-1,1]]

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
     [218, 216, 230],]
g = [[-1,1],
     [2 ,3]]

# The output of the below is the full discreet linear convolution of the inputs.
# It will use zero to complete the linear matrix.
print('With zero padding, "full", & 2x2 kernel: \n')
print(f"{sg.convolve(I, g, 'full')} \n")

# The output of the below is the full discreet linear convolution of the inputs.
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
op2 = tf.nn.conv2d(input, filter, strides=[1 ,1, 1, 1], padding='SAME')

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
