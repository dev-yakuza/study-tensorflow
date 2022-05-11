import tensorflow as tf

i = tf.constant(3)
f = tf.constant(2.2)

print(i.dtype)
print(f.dtype)

# print(i + f) # Error
print(tf.cast(i, tf.float32) + f)