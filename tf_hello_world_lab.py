import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

a = tf.constant([5])
b = tf.constant([2])

# Simple addition
c = tf.add(a, b)

with tf.Session() as sess:
    result = sess.run(c)
    print('The addition of these two constants {0} and {1} is: {2}'.format(a,b,result))

# Multiplication
c = a * b

with tf.Session() as sess:
    result = sess.run(c)
    print('The multiplication of these two constants is: {0}'.format(result))

# Elementwise & Matrix Multiplication
matrixA = tf.constant([[2,3], [3,4]])
matrixB = tf.constant([[2,3],[3,4]])

elementwise_mult_result = tf.multiply(matrixA, matrixB)
matrix_mult_result = tf.matmul(matrixA, matrixB)

with tf.Session() as sess:
    result1 = tf.multiply(matrixA, matrixB)
    print(f'The result of elementwise multiplication of these two matrices is: {result1}')

    result2 = tf.matmul(matrixA, matrixB)
    print(f'The result of multiplying these two matrices is: {result2}')

# updating variables
a = tf.constant(1000)
b = tf.Variable(0)
init_opt = tf.global_variables_initializer()
update = tf.assign(b,a)

with tf.Session() as sess:
    sess.run(init_opt)
    sess.run(update)
    print(f'The new value of b is: {sess.run(b)}')

# Fibonnacci sequence
f = [tf.constant(1), tf.constant(1)]

for i in range(2,10):
    temp = f[i-1] + f[i-2]
    f.append(temp)

with tf.Session() as sess:
    result = sess.run(f)
    print(f'The result is: {result}')

# placeholders
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

c = 2 * a - b

dictionary = {a:[2,2], b:[3,4]}
with tf.Session() as sess:
    print(f'The result is: {sess.run(c, feed_dict=dictionary)}')

# aasdsd
a = tf.cosntant(5.)
b = tf.constant(2.)
