import tensorflow as tf

int32Type = tf.constant(1)
int64Type = tf.constant(1, dtype=tf.int64)
float32Type = tf.constant(1.0)
float64Type = tf.constant(1.0, dtype=tf.float64)
stringType = tf.constant("Hello, world")
boolType = tf.constant(False)
tf.print(int32Type.dtype)
tf.print(int64Type.dtype)
tf.print(float32Type.dtype)
tf.print(float64Type.dtype)
tf.print(stringType.dtype)
tf.print(boolType.dtype)
