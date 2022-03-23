import tensorflow as tf

shape0 = tf.constant(1)
shape1 = tf.constant([1, 2])
shape2 = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
tf.print(tf.shape(shape0))
tf.print(tf.shape(shape1))
tf.print(tf.shape(shape2))