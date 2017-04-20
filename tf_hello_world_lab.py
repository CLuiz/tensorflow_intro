import tensorflow as tf

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

a = tf.constant(1000)
b = tf.Variable(0)
init_opt = tf.global_variables_initializer()
update = tf.assign(b,a)

with tf.Session() as sess:
    sess.run(init_opt)
    sess.run(update)
    print(f'The new value of b is: {sess.run(b)}')
