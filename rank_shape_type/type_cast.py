import tensorflow as tf

float32Type = tf.constant(1.0)
tf.print(tf.cast(float32Type, tf.int32).dtype)