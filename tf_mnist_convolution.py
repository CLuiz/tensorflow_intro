import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Initialize session object.
sess = tf.InteractiveSession()

# Set placeholders

"""Placeholder 'x': represents the 'space' allocatoted input or the images.

 Each input has 784 pixels distributed in a 28 x 28 matrix
 The 'shape' arg defines the the tensor size by its dimensions
 1st dim = None. Indicates thet the abtch size can be of any size
 2nd dim = 784. Indicates the number of pixels in a single flattened mnist image
"""
x = tf.placeholder(tf.float32, shape=[None, 784])

"""Placeholder 'y_' represents the final output or the labels.
10 possible classes (0,1,2,3,4,5,6,7,8,9)
The 'shape' arg defines the tensor size by its dimensions
1st dimension = None. Same as above
 2nd dimension = 10. Indicates the number of targets/outcomes.
"""
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# Assign weights and biases to null tensors

# Weight tensor
W = tf.Variable(tf.zeros([784,10], tf.float32))

# Bias tensor
b = tf.Variable(tf.zeros([10], tf.float32))

# Initialize variables
# sess.run(tf.initialize_all_variables())
sess.run(tf.global_variables_initializer())
# Set softmax activation function
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Set cost function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),
                                              reduction_indices=[1]))
# Set gradient descent as the type of optimization
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Train using minibatch gradient descent

# Load 50 training examples for each training iteration
for i in range(1000):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

# Test
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
acc = accuracy.eval(feed_dict={x: mnist.test.images,
                               y_: mnist.test.labels}) * 100
print(f'The final accuracy for the simple ANN model is: {acc}.')

sess.close()
