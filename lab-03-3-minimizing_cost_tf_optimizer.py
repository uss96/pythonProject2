import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
sess = tf.Session()

X = [1, 2, 3]
Y = [1, 2, 3]

W = tf.Variable(5.0)
hypothesis = X * W

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

sess.run(tf.global_variables_initializer())

for step in range(100):
    print(step, sess.run(W))
    sess.run(train)