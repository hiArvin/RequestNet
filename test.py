from inits import *
sess=tf.Session()
w=glorot([10,2])
sess.run(tf.global_variables_initializer())
print(sess.run(w))