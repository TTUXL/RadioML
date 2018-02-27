import tensorflow as tf

m1=tf.constant([[3,3]])
sess= tf.InteractiveSession()
print(sess.run(m1))

m2=tf.constant([[2],[3]])
product=tf.matmul(m1,m2)
with tf.Session() as sees:
     result=sees.run(product)
print(result)
