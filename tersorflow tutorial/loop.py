import tensorflow as tf

#model 1
state=tf.Variable(0,name="counter")

#model 2
new_state=tf.add(state,1)
update=tf.assign(state,new_state)

init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(state))
    for i in range(5):
        sess.run(update)
        print(sess.run(state))
