import tensorflow as tf

a = tf.constant([2])
b = tf.constant([3])

c = tf.add(a,b)

with tf.Session() as session:
    result = session.run(c)
    print(result)

# defining arrays
Scalar = tf.constant([2])
Vector = tf.constant([5,6,2])
Matrix = tf.constant([[1,2,3],[2,3,4],[3,4,5]])
Tensor = tf.constant([ [[1,2,3,], [2,3,4],[3,4,5]], [[4,5,6],[5,6,7],[6,7,8]],[[7,8,9], [8,9,10], [9,10,11]] ])

with tf.Session() as session:
    result = session.run(Scalar)
    print("Scalar (1 entry):\n %s \n" % result)
    result = session.run(Vector)
    print("Vector (3 entries) :\n %s \n" % result)
    result = session.run(Matrix)
    print("Matrix (3x3 entries):\n %s \n" % result)
    result = session.run(Tensor)
    print("Tensor (3x3x3 entries) :\n %s \n" % result)
