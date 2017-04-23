import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read data into a pandas DataFrame (add check for local csv)
df = pd.read_csv(
    'https://ibm.box.com/shared/static/fjbsu8qbwm1n5zsw90q6xzfo4ptlsw96.csv')

# Set chirps and temp as our x and y values
x_data, y_data = (df['Chirps'].values, df['Temp'].values)

# # Initialize plot to examine relationship
# plt.plot(x_data, y_data, 'ro')
#
# # Set axes labels
# plt.xlabel('# Chirps per 15 sec')
# plt.ylabel('Temp in Farenheit')

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
sess = tf.Session()
sess.run(tf.global_variables_initializer())
pred = sess.run(Y_pred, feed_dict={X: x_data})

# # Plot initial prediction against datapoints
# plt.plot(x_data, pred)
# plt.plot(x_data, y_data, 'ro')
#
# # Set axes labels
# plt.xlabel('# Chirps per 15 sec')
# plt.ylabel('Temp in Farenheit')

# plt.show()

# Define a graph for the loss function

# Normalization factor
nf = 1e-1

# Set up loss function
loss = tf.reduce_mean(tf.squared_difference(Y_pred*nf, Y*nf))

# Define an optimization graph w/ gradient descent & adagrad
optimizer1 = tf.train.GradientDescentOptimizer(learning_rate=.01)
optimizer2 = tf.train.AdagradOptimizer(0.01)

# Pass the loss function to the optimizer
train = optimizer1.minimize(loss)

# Initialize variables again
sess.run(tf.global_variables_initializer())

convergenceTolerance = 0.0001
previous_m = np.inf
previous_c = np.inf

steps = {}
steps['m'] = []
steps['c'] = []

losses=[]

for k in range(100000):
    _, _m, _c, _l = sess.run([train, m, c, loss],
                            feed_dict={X: x_data, Y: y_data})
    steps['m'].append(_m)
    steps['c'].append(_c)
    losses.append(_l)
    if (np.abs(previous_m - _m) or np.abs(previous_c - _c) ) <= convergenceTolerance :

        print("Finished by Convergence Criterion")
        print(k)
        print(_l)
        break
    previous_m = _m,
    previous_c = _c,

plt.plot(losses[:])
plt.show()
sess.close()
