import tensorflow as tf

input1 = tf.placeholder(tf.float32) 
input2 = tf.placeholder(tf.float32)

#model 1
output = tf.multiply(input1,input2)

with tf.Session() as sess:
    #feed_dict data
    print(sess.run(output,feed_dict={input1:[7,2],input2:[2,7]}))
