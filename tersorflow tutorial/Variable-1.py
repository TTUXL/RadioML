import tensorflow as tf

x=tf.Variable([2,3])
a=tf.Variable([3,3])

sub=tf.subtract(x,a)
add=tf.add(x,a)

init=tf.global_variables_initializer()

with tf.Session() as sees:
    sees.run(init)
    print(sees.run(sub))
    print(sees.run(add))
