import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

def plot_act(i=1.0, actfunc=lambda x: x):
    ws = np.arange(-0.5, 0.5, 0.5)
    bs = np.arange(-0.5, 0.5, 0.5)

    X, Y = np.meshgrid(ws, bs)

    os = np.array([actfunc(tf.constant(w * i + b)).eval(session=sess)
                   for w, b in zip(np.ravel(X), np.ravel(Y))])

    Z = os.reshape(X.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1)

# Start a session
sess = tf.Session()
# Create a simple input of 3 real values
i = tf.constant([1.0, 2.0, 3.0], shape=[1, 3])
# Create a matrix of weights
w = tf.random_normal(shape=[3, 3])
# Create a vector of biases
b = tf.random_normal(shape=[1, 3])
# Dummy activation function
def func(x): return x
# tf.matmul will multiply the input(i) tensor and the weight(w) tensor then
# sum the result with the bias(b) tensor.
act = func(tf.matmul(i, w) + b)
# Evaluate the tensor to a numpy array
act.eval(session=sess)

# Plot step function
plot_act(1.0, func)

# Plot sigmoid activation function
plot_act(1, tf.sigmoid)

# Use of sigmoid in neural net layer
act = tf.sigmoid(tf.matmul(i, w) + b)
act.eval(session=sess)

# Plot hyperbolic tangent
plot_act(1, tf.tanh)

# Use tanh in a neural net layer
act = tf.tanh(tf.matmul(i, w) + b)
act.eval(session=sess)

# Parametric rectified linear units (ReLU)
plot_act(1, tf.nn.relu)

# Using ReLU in a neural net layer
act = tf.nn.relu(tf.matmul(i, w) + b)
act.eval(session=sess)
