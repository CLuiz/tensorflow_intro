import numpy as np
import tensorflow as tf

# Build micro network with one LSTM cell
sample_input = tf.constant([[1, 2, 3, 4, 3, 2],
                            [3, 2, 2, 2, 2, 2]],
                           dtype=tf.float32)
LSTM_CELL_SIZE = 3

lstm_cell = tf.contrib.rnn.BasicLSTMCell(LSTM_CELL_SIZE,
                                         state_is_tuple=True)
state = (tf.zeros([2, LSTM_CELL_SIZE]),) * 2

with tf.variable_scope("LSTM_sample4"):
    output, state_new = lstm_cell(sample_input, state)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print(f'Sample input: {sess.run(sample_input)}')
print(f'State: {sess.run(state)}')
print(f'Output: {sess.run(output)}')
print(f'State new: {sess.run(state_new)}')

# --- Stacked LSTM basecs ---

sample_LSTM_CELL_SIZE = 3
sample_batch_size = 2
num_layers = 2
sample_input = tf.constant([[[1, 2, 3, 4, 3, 2],
                            [1, 2, 1, 1, 1, 2],
                            [1, 2, 2, 2, 2, 2]],
                            [[1, 2, 3, 4, 3, 2],
                             [3, 2, 2, 1, 1, 2],
                             [0, 0, 0, 0, 3, 2]]],
                           dtype=tf.float32)
lstm_cell = tf.contrib.rnn.BasicLSTMCell(sample_LSTM_CELL_SIZE,
                                         state_is_tuple=True)
stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell] * num_layers)
_initial_state = stacked_lstm.zero_state(sample_batch_size, tf.float32)

with tf.variable_scope("Stacked_LSTM_sample8"):
    outputs, new_state = tf.nn.dynamic_rnn(stacked_lstm,
                                           sample_input,
                                           dtype=tf.float32,
                                           initial_state=_initial_state)
sess.run(tf.global_variables_initializer())

print('*' * 30)
print('Stacked LSTM info')
print('*' * 30)
print(f'Sample input: {sess.run(sample_input)}')
sess.run(_initial_state)
print(f'New state: {sess.run(new_state)}')
print(f'Output: {sess.run(output)}')

sess.close()
