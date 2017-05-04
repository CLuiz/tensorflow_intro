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


if __name__ == '__main__':
    mnist = input_data.read_data_sets('MNIST_Data/', one_hot=True)
    trX, trY, teX, teY = mnist.train.images, mnist.train.labels,
    mnist.test.images, mnist.test.labels

    # Create 2 layers of RBM with size of 400 and 100
    RBM_hidden_sizes = [500, 200, 50]
    
