import tensorflow as tf
import numpy as np

%matplotlib inline

import pylab

x_data = np.random.rand(100).astype(np.float32)
noise = np.random.normal(scale=0.2, size=len(x_data))

# equation is 4.5*x^2 + 3 * x + 1 + noise
y_data = 4.5 * x_data **2 + 3 * x_data + noise

# plot the data
pylab.plot(x_data, y_data, '.')

# build the inference graph
W = tf.Variable(np.random.rand(2).astype(np.float32), name="weights")
b = tf.Variable(np.asarray([0], dtype=np.float32), name="bias")
y = W[0] * x_data ** 2 + W[1] * x_data + b

# build the training graph
# loss function will be mean square error
loss = tf.reduce_mean(tf.square(y - y_data))
# use a gradient optimizer
optimizer = tf.train.GradientDescentOptimizer(0.4)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

# create a Session (tensorflow runtime) to run the graphs
sess = tf.Session()
sess.run(init)

# perform the training iteratively
for i in xrange(3000):
    sess.run(train)
    
    # print the process in each 30 iterations
    if not i % 30:
        print "Step",i,sess.run([W, b])

# now the training is complete

# let's evaluate the trained model
print "Mean Square Loss:", sess.run(loss)

# show the evaluation using a graph
pylab.plot(x_data, y_data, ".", label = "target")
pylab.plot(x_data, sess.run(y), ".", label = "trained")
pylab.legend()

# ylim is added otherwise the legend hides some data points
pylab.ylim(-2, 12)
