import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read data into a pandas DataFrame (add check for local csv)
df = pd.read_csv(
    'https://ibm.box.com/shared/static/fjbsu8qbwm1n5zsw90q6xzfo4ptlsw96.csv')

# Set chirps and temp as our x and y values
x_data, y_data = (df['Chirps'].values, df['Temp'].values)

# Initialize plot to examine relationship
plt.plot(x_data, y_data, 'ro')

# Set axes labels
plt.xlabel('# Chirps per 15 sec')
plt.ylabel('Temp in Farenheit')

# plt.show()

# Create placeholders
X = tf.placeholder(tf.float32, shape=(x_data.size))
Y = tf.placeholder(tf.float32, shape=(y_data.size))

# Create updateable tf variables and prepare to initialize with arbtrary values

m = tf.Variable(3.0)
c = tf.Variable(2.0)

# Construct a model
Y_pred = tf.add(tf.multiply(X, m), c)

# Create session and initialize variables
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    pred = sess.run(Y_pred, feed_dict={X: x_data})

# Plot initial prediction against datapoints
plt.plot(x_data, pred)
plt.plot(x_data, y_data, 'ro')

# Set axes labels
plt.xlabel('# Chirps per 15 sec')
plt.ylabel('Temp in Farenheit')

# plt.show()
