import tensorflow as tf

node1 = tf.Variable(3.0)
node2 = tf.Variable(4.0)
print(node1)
print(node2)
tf.print(node1)
tf.print(node2)

node1.assign(1.0)
node2.assign(2.0)
print(node1)
print(node2)
tf.print(node1)
tf.print(node2)