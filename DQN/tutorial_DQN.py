"""
Deep Q-Network Q(a, s)
# The FrozenLake v0 environment
SFFF       (S: starting point, safe)
FHFH       (F: frozen surface, safe)
FFFH       (H: hole, fall to your doom)
HFFG       (G: goal, where the frisbee is located)
The episode ends when you reach the goal or fall in a hole. You receive a reward
of 1 if you reach the goal, and zero otherwise.
Prerequisites
--------------
tensorflow>=2.0.0a0
tensorlayer>=2.0.0
python tutorial_DQN.py --train/test
"""
import argparse
import os
import time

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import tensorlayer as tl

# add arguments in command  --train/test
parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='test', action='store_true', default=True)
args = parser.parse_args()

tl.logging.set_verbosity(tl.logging.DEBUG)

#####################  hyper parameters  ####################
env_id = 'FrozenLake-v0'
alg_name = 'DQN'
lambd = .99  # decay factor
e = 0.1  # e-Greedy Exploration, the larger the more random
num_episodes = 10000
render = False  # display the game environment

##################### DQN ##########################

"""
把原来的编码改为热独编码，现在我们有16个不同的state,按照一般的分类，会把16个state分成1，2，3...16类
在机器学习中，0，1，2，3这些数字是包含了一个大小关系但是在分类问题中，每一项都是平等的，没有大小关系
因此变成热独编码形式
我们构造一个矩阵，只有该分类的标志位是1，其他都是0，矩阵大小就是类别个数，譬如1：[1 0 0 0 ...] 16:[0，0...1]

"""
def to_one_hot(i, n_classes=None):
    a = np.zeros(n_classes, 'uint8')
    a[i] = 1
    return a


## Define Q-network q(a,s) that ouput the rewards of 4 actions by given state, i.e. Action-Value Function.
# encoding for state: 4x4 grid can be represented by one-hot vector with 16 integers.
def get_model(inputs_shape):
    ni = tl.layers.Input(inputs_shape, name='observation')#输入矩阵形状
    #dense作为输出，没有加一个Dense层，因为16个状态空间，模型拟合能力已经足够
    nn = tl.layers.Dense(4, act=None, W_init=tf.random_uniform_initializer(0, 0.01), b_init=None, name='q_a_s')(ni)
    return tl.models.Model(inputs=ni, outputs=nn, name="Q-Network")


def save_ckpt(model):  # save trained weights
    path = os.path.join('model', '_'.join([alg_name, env_id]))
    if not os.path.exists(path):
        os.makedirs(path)
    tl.files.save_weights_to_hdf5(os.path.join(path, 'dqn_model.hdf5'), model)


def load_ckpt(model):  # load trained weights
    path = os.path.join('model', '_'.join([alg_name, env_id]))
    tl.files.save_weights_to_hdf5(os.path.join(path, 'dqn_model.hdf5'), model)


if __name__ == '__main__':

    qnetwork = get_model([None, 16]) #前面是状态的数量（输入一个batch大小的数）
    qnetwork.train()
    train_weights = qnetwork.trainable_weights

    optimizer = tf.optimizers.SGD(learning_rate=0.1)
    env = gym.make(env_id)#创建环境

    t0 = time.time()
    if args.train:
        all_episode_reward = []
        for i in range(num_episodes):
            ## Reset environment and get first new observation
            s = env.reset()  # 步骤3.初始化环境，返回一个初始状态s（0-15）
            rAll = 0
            if render: env.render()#如果render为true,则渲染
            for j in range(99):  # step index, maximum step is 99
                ## Choose an action by greedily (with e chance of random action) from the Q-network
                #步骤4.输入状态s，得到所有Q值
                allQ = qnetwork(np.asarray([to_one_hot(s, 16)], dtype=np.float32)).numpy()
                #步骤5.1 获得最大Q值，返回最大数的索引值
                a = np.argmax(allQ, 1)
                print("j"+str(j)+'\n')
                print(allQ)
                print(a[0])

                ##步骤5.2  e-Greedy Exploration策略 如果随机数小于门槛则进行样本随机动作
                if np.random.rand(1) < e:
                    a[0] = env.action_space.sample()
                #步骤6从环境中获得reward和新状态,标志位d表示是否进入最终状态
                s1, r, d, _ = env.step(a[0])
                if render: env.render()
                ## 步骤7.1 通过S1新状态来确定Q的值
                Q1 = qnetwork(np.asarray([to_one_hot(s1, 16)], dtype=np.float32)).numpy()

                ## 步骤7.2获取maxQ，这里的maxQ用来代表s1状态的v值，更新Q值.
                maxQ1 = np.max(Q1)  # in Q-Learning, policy is greedy, so we use "max" to select the next action.
                targetQ = allQ
                targetQ[0, a[0]] = r + lambd * maxQ1
                ## 步骤8训练网络Train network using target and predicted Q values
                # it is not real target Q value, it is just an estimation,
                # but check the Q-Learning update formula:
                #    Q'(s,a) <- Q(s,a) + alpha(r + lambd * maxQ(s',a') - Q(s, a))
                # minimizing |r + lambd * maxQ(s',a') - Q(s, a)|^2 equals to force Q'(s,a) ≈ Q(s,a)
                with tf.GradientTape() as tape:
                    _qvalues = qnetwork(np.asarray([to_one_hot(s, 16)], dtype=np.float32))
                    _loss = tl.cost.mean_squared_error(targetQ, _qvalues, is_mean=False)
                grad = tape.gradient(_loss, train_weights)
                optimizer.apply_gradients(zip(grad, train_weights))

                rAll += r
                s = s1
                ## Reduce chance of random action if an episode is done.
                if d ==True:
                    e = 1. / ((i / 50) + 10)  # reduce e, GLIE: Greey in the limit with infinite Exploration
                    break

            ## Note that, the rewards here with random action
            print('Training  | Episode: {}/{}  | Episode Reward: {:.4f} | Running Time: {:.4f}' \
                  .format(i, num_episodes, rAll, time.time() - t0))

            if i == 0:
                all_episode_reward.append(rAll)
            else:
                all_episode_reward.append(all_episode_reward[-1] * 0.9 + rAll * 0.1)

        save_ckpt(qnetwork)  # save model
        plt.plot(all_episode_reward)
        if not os.path.exists('image'):
            os.makedirs('image')
        plt.savefig(os.path.join('image', '_'.join([alg_name, env_id])))

    if args.test:
        load_ckpt(qnetwork)  # load model
        for i in range(num_episodes):
            ## Reset environment and get first new observation
            s = env.reset()  # observation is state, integer 0 ~ 15
            rAll = 0
            if render: env.render()
            for j in range(99):  # step index, maximum step is 99
                ## Choose an action by greedily (with e chance of random action) from the Q-network
                allQ = qnetwork(np.asarray([to_one_hot(s, 16)], dtype=np.float32)).numpy()
                a = np.argmax(allQ, 1)  # no epsilon, only greedy for testing

                ## Get new state and reward from environment
                s1, r, d, _ = env.step(a[0])
                rAll += r
                s = s1
                if render: env.render()
                ## Reduce chance of random action if an episode is done.
                if d: break

            print('Testing  | Episode: {}/{}  | Episode Reward: {:.4f} | Running Time: {:.4f}' \
                  .format(i, num_episodes, rAll, time.time() - t0))