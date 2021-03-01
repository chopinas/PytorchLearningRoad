#针对迷宫问题 tensorflow版本
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()  #因为我是2.1的版本，这里调用了t1版本的api，所以需要用这段代码
import numpy as np
from collections import deque
import random


input_num=6
output_num=6

x_data=np.linspace(-1,1,300).reshape(-1,input_num)  #转为列向量
noise=np.random.normal(0,0.05,x_data.shape)
y_data=np.square(x_data)+0.5+noise

xs = tf.placeholder(tf.float32, [None, input_num])  # 样本数未知，特征数为 6，占位符最后要以字典形式在运行中填入
ys = tf.placeholder(tf.float32, [None, output_num])

neuro_layer_1 = 3
w1 = tf.Variable(tf.random_normal([input_num, neuro_layer_1]))
b1 = tf.Variable(tf.zeros([1, neuro_layer_1]) + 0.1)
l1 = tf.nn.relu(tf.matmul(xs, w1) + b1)

neuro_layer_2 = output_num
w2 = tf.Variable(tf.random_normal([neuro_layer_1, neuro_layer_2]))
b2 = tf.Variable(tf.zeros([1, neuro_layer_2]) + 0.1)
l2 = tf.matmul(l1, w2) + b2

# reduction_indices=[0] 表示将列数据累加到一起。
# reduction_indices=[1] 表示将行数据累加到一起。
loss = tf.reduce_mean(tf.reduce_sum(tf.square((ys - l2)), reduction_indices=[1]))

# 选择梯度下降法
train = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
# train = tf.train.AdamOptimizer(1e-1).minimize(loss)


init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(100000):
    sess.run(train, feed_dict={xs: x_data, ys: y_data})
    if i % 1000 == 0:
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))

class DeepQNetword:
    def __init__(self):
        self.q_eval_input = tf.placeholder(shape=[None, self.state_num], dtype=tf.float32)
        self.action_input = tf.placeholder(shape=[None, self.action_num], dtype=tf.float32)
        self.q_target = tf.placeholder(shape=[None], dtype=tf.float32)

        neuro_layer_1 = 3
        w1 = tf.Variable(tf.random_normal([self.state_num, neuro_layer_1]))
        b1 = tf.Variable(tf.zeros([1, neuro_layer_1]) + 0.1)
        l1 = tf.nn.relu(tf.matmul(self.q_eval_input, w1) + b1)

        w2 = tf.Variable(tf.random_normal([neuro_layer_1, self.action_num]))
        b2 = tf.Variable(tf.zeros([1, self.action_num]) + 0.1)
        self.q_eval = tf.matmul(l1, w2) + b2

        # 取出当前动作的得分。
        self.reward_action = tf.reduce_sum(tf.multiply(self.q_eval, self.action_input), reduction_indices=1)
        self.loss = tf.reduce_mean(tf.square((self.q_target - self.reward_action)))
        self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

        self.predict = tf.argmax(self.q_eval, 1)
