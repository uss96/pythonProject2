import tensorflow as tf

print(tf.__version__)


hello = tf.constant("Hello, TensorFlow!")

tf.print(hello)