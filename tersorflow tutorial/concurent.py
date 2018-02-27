import tensorflow as tf

input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)

#model 1
add = tf.add(input2,input3)
#model 2
mul = tf.multiply(input1,add)

with tf.Session() as sess:
    #concurent
    result = sess.run([mul,add])
    print(result)
