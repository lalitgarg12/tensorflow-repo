import numpy as np 
import tensorflow as tf 

coefficients = np.array([[1.],[-10.0],[25.]])

#Solve for y = w(sq) - 10w + 25 = (w-5)^2
w = tf.Variable(0, dtype = tf.float32)
x = tf.placeholder(tf.float32, [3,1])
#cost = tf.add(tf.add(w**2, tf.multiply(-10.,w)),25)
cost = w**2 - 10*w + 25
cost = x[0][0]*w**2 + x[1][0]*w + x[2][0]
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)
session.run(w)

session.run(train, feed_dict = {x:coefficients})
print(session.run(w))

for i in range(1000):
    session.run(train, feed_dict = {x:coefficients})
print(session.run(w))
