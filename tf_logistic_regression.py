import tensorflow as tf
import pandas as pd
import numpy as np
import time
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

iris = load_iris()
iris_X, iris_y = iris.data[:-1,:], iris.target[:-1]
iris_y = pd.get_dummies(iris_y).values
trainX, testX, trainY, testY = train_test_split(iris_X, iris_y,
                                                test_size=.33,
                                                random state=42)

# numFeatures is the number of features in our input dataset
# In the Iris Dataset, there are 4
numFeatures = trainX.shape[1]

# numLabels is the number of classes our data points can be in
# The Iris dataset has three possible classes
numLabels = trainY.shape[1]

# Placeholders
# 'None' means Tensorflow shouldn't expect a fixed number in that dimension
X = tf.placeholder(tf.float32, [None, numFeatures])
yGold = tf.placeholder(tf.float32, [None, numLabels])

W = tf.Variable(tf.zeros([4,3])) # 4 dimensional input and 3 classes
b = tf.Variable(tf.zeros([3]))# 3-dimensional output [0,0,1],[1,1,0],[1,0,0]

# Randomly sample form a normal distribution with sd of .01
weights = tf.Variable(tf.random_normal([numFeatures, numLabels],
                                        mean=0,
                                        stddev=0.01,
                                        name='weights'))
bias = tf.Variable(tf.random_normal([1, numLabels],
                                     mean=0,
                                     stddev=0.01
                                     name='bias'))

# Three component breakdown of the Logistic Regression equation.
# Note that these feed into each other
apply_weights_OP = tf.matmul(X, weights, name='apply_weights')
add_bias_OP = tf.add(apply_weights_OP, bias, name='add_bias')
activation_OP = tf.nn.sigmoid(add_bias_OP, name='activation')

# Number of Epochs
numEpochs = 700

# Define learning rate
learningRate = tf.train.exponential_decay(learning_rate=0.0008,
                                          global_step=1,
                                          decay_steps=trainX.shape[0],
                                          decay_rate=0.95,
                                          staircase=True)
# Define cost function
cost_OP = tf.nn.12_loss(activation_OP-yGold, name='square_error_cost')

# Define Gradient Decent
training_OP = tf.train.GradientDescentOptimizer(learningRate).minimize(cost_OP)

# Create session object
sess = tf.Session()

# Initialize weights and biases
init_OP = tf.global_variables_initializer()

# Initialize all tf variables
sess.run(init_OP)
