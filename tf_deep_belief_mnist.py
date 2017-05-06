import math
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from utils import tile_raster_images


class RBM(object):

    def __init__(self, input_size, output_size):
        # Define hyperparameters
        self._input_size = input_size
        self._output_size = output_size
        self.epochs = 5
        self.learning_rate = 1.0
        self.batchsize = 100

        # Initialize weights and biases as all zero matrices
        self.w = np.zeros([input_size, output_size], np.float32)
        self.hb = np.zeros([output_size], np.float32)
        self.vb = np.zeros([output_size], np.float32)

    def prob_h_given_v(self, visible, w, hb):
        # Sigmoid
        return tf.nn.sigmoid(tf.matmul(visible, w) + hb)

    def prob_v_given_h(self, hidden, w, vb):
        return tf.nn.sigmoid(tf.matmul(hidden, tf.transpose(w)) + vb)

    def sample_prob(self, probs):
        return tf.nn.relu(tf.sign(probs - tf.random_uniform(tf.shape(probs))))

    def train(self, X):
        # Create placeholders for our parameters
        _w = tf.placeholder('float', [self._input_size, self._output_size])
        _hb = tf.placeholder('float', [self._output_size])
        _vb = tf.placeholder('float', [self._input_size])

        prv_w = np.zeros([self._input_size, self._output_size], np.float32)
        prv_hb = np.zeros([self._output_size], np.float32)
        prv_vb = np.zeros([self._input_size], np.float32)

        cur_w = np.zeros([self._input_size, self._output_size], np.float32)
        cur_hb = np.zeros([self._output_size], np.float32)
        cur_vb = np.zeros([self._input_size], np.float32)
        v0 = tf.placeholder('float', [None, self._input_size])

        # Initialize with sample probabilities
        h0 = self.sample_prob(self.prob_h_given_v(v0, _w, _hb))
        v1 = self.sample_prob(self.prob_v_given_h(h0, _w, _vb))
        h1 = self.prob_h_given_v(v1, _w, _hb)

        # Create gradients
        positive_grad = tf.matmul(tf.transpose(v0), h0)
        negative_grad = tf.matmul(tf.transpose(v1), h1)

        # Update learning rates for layers
        update_w = (_w + self.learning_rate * (positive_grad - negative_grad)
                    / tf.to_float(tf.shape(v0[0])))
        update_vb = _vb + self.learning_rate * tf.reduce_mean(v0 - v1, 0)
        update_hb = _hb + self.learning_rate * tf.reduce_mean(h0 - h1, 0)

        # Find the error rates
        err = tf.reduce_mean(tf.square(v0 - v1))

        # Training Loop
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            # For each epoch
            for epoch in range(self.epochs):
                # For each step/batch
                for start, end in zip(range(0, len(X), self.batchsize), range(self.batchsize, len(X), self.batchsize)):
                    batch = X[start:end]
                    # update rates
                    fdict = {v0: batch, _w: prv_w, _hb: prv_hb, _vb: prv_vb}

                    cur_w = sess.run(update_w, feed_dict=fdict)
                    cur_hb = sess.run(update_hb, feed_dict=fdict)
                    cur_vb = sess.run(update_vb, feed_dict=fdict)
                    prv_w = cur_w
                    prv_hb = cur_hb
                    prv_vb = cur_vb
                error = sess.run(err, feed_dict={v0: X,
                                                 _w: cur_w,
                                                 _vb: cur_vb,
                                                 _hb: cur_hb})
                print(f'Epoch: {epoch}, reconstruction error: {error}')
            self.w = prv_w
            self.hb = prv_hb
            self.vb = prv_vb

    def rbm_output(self, X):
        input_X = tf.constant(X)
        _w = tf.constant(self.w)
        _hb = tf.constant(self.hb)
        out = tf.nn.sigmoid(tf.matmul(input_X, _w) + _hb)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            return sess.run(out)


class NN(object):

    def __init__(self, sizes, X, Y):
        # Initialize hyperparameters
        self._sizes = sizes
        self._X = X
        self._Y = Y
        self.w_list = []
        self.b_list = []
        self._learning_rate = 1.0
        self._momentum = 0.0
        self._epochs = 10
        self._batchsize = 100
        input_size = X.shape[1]

        # Initialization loop
        for size in self.sizes + [Y.shape[1]]:
            # Define upper limit for the uniform distribution range
            max_range = 4 * math.sqrt(6. / (input_size + size))

            # Initialize weights thrugh a random uniform distribution
            self.w_list.append(np.random.uniform(-max_range,
                               max_range,
                               [input_size, size]).astype(np.float32))

            # Intitalize bias as zeros
            self.b_list.append(np.zeros([size], np.float32))
            input_size = size

    def load_from_rbms(self, dbn_sizes, rbm_list):
        # Check if expected sizes are correct
        assert len(dbn_sizes) == len(self._sizes)

        for i in range(len(self._sizes)):
            # Check if for each RBN the expected sizes are correct
            assert dbn_sizes[i] == self._sizes[i]

        # If everything is correct, bring over the weights and biases
        for i in range(len(self._sizes)):
            self.w_list[i] = rbm_list[i].w
            self.b_list[i] = rbm_list[i].hb

    def train(self):
        # Create placeholders for input, weights, biases, output
        _a = [None] * (len(self._sizes) + 2)
        _w = [None] * (len(self._sizes) + 1)
        _b = [None] * (len(self._sizes) + 1)
        _a[0] = tf.placeholder('float', [None, self._X.shape[1]])
        y = tf.placeholder('float', [None, self._Y.shape[1]])

        # Define variables and actiavation function
        for i in range(len(self._sizes) + 1):
            _w[i] = tf.Variable(self.w_list[i])
            _b[i] = tf.Variable(self.b_list[i])
        for i in range(1, len(self._sizes) + 2):
            _a[i] = tf.nn.sigmoid(tf.matmul(_a[i - 1], _w[i - 1]) + _b[i - 1])

        # Define the cost function
        cost = tf.reduce_mean(tf.square(_a[-1] - y))

        # Define the training operation
        train_op = tf.train.MomentumOptimizer(self._learning_rate, self._momentum).minimize(cost)

        # Prediction operation
        predict_op = tf.argmax(_a[-1], 1)

        # Training Loop
        with tf.Session() as sess:
            # Initailize Variables
            sess.run(tf.global_variables_initializer())
            # For each epoch
            for i in range(self._epochs):
                # For each step
                for start, end in zip(range(0,
                                            len(self.X),
                                            self._batchsize),
                                      range(self._batchsize,
                                            len(self._X),
                                            self._batchsize)):
                    # Run the training operation on the input data
                    sess.run(train_op, feed_dict={_a[0]: self._X[start:end],
                                                  y: self._Y[start:end]})
                for j in range(len(self._sizes) + 1):
                    # Retrieve weights and biases
                    self.w_list[j] = sess.run(_w[j])
                    self.b_list[j] = sess.run(_b[j])

                print("Accuracy rating for epoch " + str(i) + ": " + str(np.mean(np.argmax(self._Y,
                                        axis=1) ==
                                        sess.run(predict_op,
                                                 feed_dict={_a[0]: self._X,
                                                            y: self._Y}))))

                # print(f'Accuracy rating for epoch {str(i)}: ')


if __name__ == '__main__':
    mnist = input_data.read_data_sets('MNIST_Data/', one_hot=True)
    trX, trY, teX, teY = mnist.train.images, mnist.train.labels,
    mnist.test.images, mnist.test.labels

    # Create 2 layers of RBM with size of 400 and 100
    RBM_hidden_sizes = [500, 200, 50]

    # Set input as training data
    inpX = trX

    # Create list to hold our RBM's
    rbm_list = []

    # Size of inputs is the number of inputs in the traiing set
    input_size = inpX.shape[1]

    # For each RBM we want to generate
    for i, size in enumerate(RBM_hidden_sizes):
        print(f'RBM: {i}, {input_size} -> {size}')
        rbm_list.append(RBM(input_size, size))
        input_size = size

    # Training loop
    for rbm in rbm_list:
        print('New RBM:')
        # Train a new rbm
        rbm.train(inpX)
        # Return the ouput layer
        inpX = rbm.rbm_output(inpX)

    # Execute the code
    nNet = NN(RBM_hidden_sizes, trX, trY)
    nNet.load_from_rbms(RBM_hidden_sizes, rbm_list)
    nNet.train()
