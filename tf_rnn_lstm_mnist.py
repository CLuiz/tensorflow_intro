import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import warnings


# Set warnings to ignore
warnings.filterwarnings('ignore')

# Read data in with one-hot encoding
mnist = input_data.read_data_sets(".", one_hot=True)

# Assign data to train and test sets
train_imgs = mnist.train.images
train_labels = mnist.train.labels
test_imgs = mnist.test.images
test_labels = mnist.test.labels

# Inspect & print out dimensions of the dataset
ntrain = train_imgs.shape[0]
ntest = test_imgs.shape[0]
dim = train_imgs.shape[1]
print(f'Train Images Shape: {train_imgs.shape}')
print(f'Train Labels Shape: {train_labels.shape}')
print(f'Test Images Shape: {test_imgs.shape}')
print(f'Test Labels Shape: {test_labels.shape}')

# Look at one sample
samples_idx = [100, 101, 102]
fig = plt.figure()

ax1 = fig.add_subplot(121)
ax1.imshow(test_imgs[samples_idx[0]].reshape([28, 28]),
           cmap='gray')

xx, yy = np.meshgrid(np.linspace(0, 28, 28),
                     np.linspace(0, 28, 28))
X = xx
Y = yy
Z = 100 * np.ones(X.shape)

img = test_imgs[77].reshape([28, 28])
ax = fig.add_subplot(122, projection='3d')
ax.set_zlim((0, 200))

offset = 200
for i in samples_idx:
    img = test_imgs[i].reshape([28, 28]).transpose()
    ax.contourf(X, Y, img, 200, zdir='z', offset=offset, cmap='gray')
    offset -= 100
plt.show()

for i in samples_idx:
    print(f"""Sample: {i} -
          Class: {np.nonzero(test_labels[i])[0]} -
          Label Vector: {test_labels[i]}""")

"""Build simple RNN with three layers.

   Layer1: Input layer which converts 28 dimensional imput to a 128
   dimensional hidden layer.

   Layer2: One intermediate recurrent neural network (LSTM)

   Layer3: One output layer which converts a 128 dimensional output
   of the LSTM to a 10 dimensional output indicating a class label
"""
# Set parameters
n_input = 28
n_steps = 28
n_hidden = 128
n_classes = 10

learning_rate = 0.001
training_iters = 100000
batch_size = 100
display_step = 10

# Construct RNN
x = tf.placeholder(dtype='float', shape=[None, n_steps, n_input], name='x')
y = tf.placeholder(dtype='float', shape=[None, n_classes], name='y')

weights = {'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))}
biases = {'out': tf.Variable(tf.random_normal([n_classes]))}
