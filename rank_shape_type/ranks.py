import tensorflow as tf

rank0 = tf.constant(1)
rank1 = tf.constant([1, 2])
rank2 = tf.constant([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]])
tf.print(tf.rank(rank0))
tf.print(tf.rank(rank1))
tf.print(tf.rank(rank2))