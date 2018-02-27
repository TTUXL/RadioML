import tensorflow as tf
import numpy as np

x_data=np.random.randn(100)
y_data=0.1*x_data+0.2

#构建一个线性模型
k=tf.Variable(0.)
b=tf.Variable(0.)
y=k*x_data+b

#定义一个二次代价函数
loss=tf.reduce_mean(tf.square(y_data-y))

#定义一个梯度下降法训练的优化器 运用已经封装好的GradientDescentOptimizer，学习率是0.2
optimizer=tf.train.GradientDescentOptimizer(0.2)

#训练的目的就是最小化代价函数，用minimize.如果loss越小，那么就应该k越接近0.1，b越接近0.2.
train=optimizer.minimize(loss)

#本程序用到了Variable，所以需要初始化变量
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(200): #迭代200次
        sess.run(train) #每次会最小化loss
#每20次打印k和b的值：
        if step%20 == 0:
            print(step,sess.run([k,b]))
