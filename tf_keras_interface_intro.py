"""
Walthrough of code from 'The Keras Blog'.

Full walthrough of code for using Keras as a simplified backend for Tensorflow.
"""
from keras import backend as K
from keras.layers import Dense
from keras.layers import Dropout
from keras.metrics import categorical_accuracy as accuracy
from keras.objectives import categorical_crossentropy
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

sess = tf.Session()
K.set_session(sess)

# Set placeholder
img = tf.placeholder(tf.float32, shape=(None, 784))

# --- Keras layers can be called on tf tensors ---

# Fully connected layer with 128 untis and ReLU activation
x = Dense(128, activation='relu')(img)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)

# Output layer with 10 units and softmax activation
preds = Dense(10, activation='softmax')(x)

# Define label placeholder and cost function
labels = tf.placeholder(tf.float32, shape=(None, 10))
loss = tf.reduce_mean(categorical_crossentropy(labels, preds))

# Train model with tf optimizer
mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# Initialize variables
init_op = tf.global_variables_initializer()
sess.run(init_op)

# Run training loop
with sess.as_default():
    for i in range(100):
        batch = mnist_data.train.next_batch(50)
        train_step.run(feed_dict={img: batch[0],
                                  labels: batch[1],
                                  K.learning_phase(): 1})

# Evaluate model
acc_value = accuracy(labels, preds)
with sess.as_default():
    print(acc_value.eval(feed_dict={img: mnist_data.test.images,
                                    labels: mnist_data.test.labels,
                                    K.learning_phase(): 0}))
