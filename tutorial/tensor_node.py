import tensorflow as tf

node1 = tf.constant(3.0)
node2 = tf.constant(4.0)
print(node1)
print(node2)

tf.print(node1)
tf.print(node2)

node3 = tf.add(node1, node2)
print(node3)

tf.print(node3)