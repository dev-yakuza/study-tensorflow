import os
import shutil
import tensorflow as tf

path = "./logs/"
if os.path.isdir(path):
  shutil.rmtree(path)

graph = tf.Graph()
with graph.as_default():
  node1 = tf.constant(3.0, name="node1")
  node2 = tf.constant(4.0, name="node2")

  node3 = tf.add(node1, node2, name="node3")
  tf.print(node3)

writer = tf.summary.create_file_writer(path)
with writer.as_default():
  tf.summary.graph(graph)
