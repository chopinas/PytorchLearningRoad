#针对小车平衡的源码
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

# .unwrapped让开发人员能调整模型的底层数据，例如解开reward的最大限制等。在这里加入或删除unwrapped无影响
env = gym.make('CartPole-v0').unwrapped

# 建立 matplotlib，如果使用pycharm等IDE则可注释
is_ipython = 'inline' in matplotlib.get_backend()
print(is_ipython)
if is_ipython:
    from IPython import display

plt.ion()

# 如果可以使用gpu，则将全局device设为gpu。device决定了pytorch会将tensor放在哪里运算。
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# namedtuple是一种特殊的数据结构，类似于C中的struct，也可以理解为只有属性的类
# Transition可以理解为类名，state，action，next_state，reward都为这个类的属性
# 声明一个namedtuple：Transition(state,action,next_state,reward)
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


# 经验回放池。把每条Transition（包括state，action，next_state，reward）放入池中，从中随机抽样来训练网络
# 使用经验回放目的是打破每一条数据之间的相关性。神经网络应该只关心在一条训练数据中，输入state与action所得到的reward，借此来调整神经网络的参数
# 例如在一个游戏中相邻两帧之间的数据就是有“有相关性的”数据。经验回放就是从一场游戏中抽取若干不相邻的帧
class ReplayMemory(object):

    def __init__(self, capacity):
        # 经验回放池的最大容量
        self.capacity = capacity
        # 使用列表作为存储结构
        self.memory = []
        # 列表下标
        self.position = 0

    # 若池中尚有空间，直接追加新数据；若无空间，则覆盖旧数据
    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    # 随机抽样一个batch_size大小的数据集
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# DQN本体，一个卷积神经网络，继承于nn.Module，必须实现forward方法，该方法决定了数据如何前向通过网络
# 神经网络的输入为40*90游戏图像（图像包含着状态信息），输出为两个动作的Q值
class DQN(nn.Module):

    def __init__(self, h, w):
        super(DQN, self).__init__()
        # nn.Conv2d的参数依次为：输入维度，输出维度，卷积核大小，步长
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # 线性输入连接的数量取决于conv2d层的输出，因此需要计算输入图像的大小。
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, 2)  # 448 或者 512

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.head(x.view(x.size(0), -1))
        return x


# 定义了一个流水线函数，把以下三个步骤整合到一起：1. 把tensor转为图像；2.调整图像大小，把较短一条边的长度设为40；3.将图像转为tensor
resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])


# 获取小车的中心位置，用于裁剪图像
def get_cart_location(screen_width):
    # 小车左右横跳的宽度是4.8
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # 车子的中心


# 获取环境的图像，可以不关注细节
def get_screen():
    # 返回 gym 需要的400x600x3 图片, 但有时会更大，如800x1200x3. 将其转换为torch (CHW).
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    # 车子在下半部分，因此请剥去屏幕的顶部和底部。
    _, screen_height, screen_width = screen.shape
    # print(screen.shape)
    screen = screen[:, int(screen_height * 0.4):int(screen_height * 0.8)]
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(screen_width)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    # 去掉边缘，这样我们就可以得到一个以车为中心的正方形图像。
    screen = screen[:, :, slice_range]
    # 转化为 float, 重新裁剪, 转化为 torch 张量(这并不需要拷贝)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # 重新裁剪,加入批维度 (BCHW)
    return resize(screen).unsqueeze(0).to(device)


env.reset()
plt.figure()
# gpu中的tensor转成numpy数组必须移入cpu
plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),
           interpolation='none')
plt.title('Example extracted screen')
plt.show()

# BATCH_SIZE大小
BATCH_SIZE = 128

# gamma
GAMMA = 0.999
# 选择随机动作的概率，就是论文中的epsilon，上界是0.9
EPS_START = 0.9
# 选择随机动作的概率，下界是0.05
EPS_END = 0.05
# 从上界到下界的衰减速率
EPS_DECAY = 200
# 每个多少回合，更新target网络的Q值
TARGET_UPDATE = 10

# 获取屏幕大小，以便我们可以根据从ai-gym返回的形状正确初始化层。这一点上的典型尺寸接近3x40x90，这是在get_screen（）中抑制和缩小的渲染缓冲区的结果。
init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape

# 创建两个网络
# 实时训练的网络
policy_net = DQN(screen_height, screen_width).to(device)
# 目标网络，每隔若干回合更新一次
target_net = DQN(screen_height, screen_width).to(device)
# 把policy_net的权重参数复制给目标网络
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# RMSprop优化器
optimizer = optim.RMSprop(policy_net.parameters())
# 经验回放池大小
memory = ReplayMemory(10000)
# 全局变量，记录了网络训练的步数
steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    # 与论文不同的是，这里的epsilon是变化的，随着训练的进行，选取随机动作的概率会越来越小
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max（1）将为每行的列返回最大值。max result的第二列是找到max元素的索引，因此我们选择预期回报较大的操作。
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(2)]], device=device, dtype=torch.long)


episode_durations = []


# 画图函数，横坐标是训练批次（一次游戏结束为一个批次），纵坐标为游戏的reward，也就是持续时间
def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # 平均 100 次迭代画一次
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # 暂定一会等待屏幕更新
    # if is_ipython:
    #     display.clear_output(wait=True)
    #     display.display(plt.gcf())


# 该函数手动实现了一个minibatch梯度下降
# 如果结合注释还是看不懂，建议手动输出一下看看
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return

    # 抽样
    transitions = memory.sample(BATCH_SIZE)
    # 将BATCH_SIZE个的四元组转换成一个大的四元组，每一个元素都是大小为BATCH_SIZE的列表
    batch = Transition(*zip(*transitions))
    # non_final_mask由0和1组成，0表示该索引位置对应的state是结束状态，1表示不为结束状态
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.uint8)
    # 直接存储状态图像，把不为空的state首尾拼接（cat()：拼接函数）
    # 大小小于等于BATCH_SIZE
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # 计算Q(s_t, a)，在后面用于计算损失
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # next_state_values[non_final_mask]这一句的作用是，若对应位置的mask为1，则将该值设为下一个状态的最大Q动作的Q值
    # 否则保持初始值0
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # 根据贝尔曼公式计算期望 Q 值
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # 计算两个网络的两个Q值的Huber 损失
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # 优化模型
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        # 梯度大小限制在-1和1之间，防止梯度爆炸
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


num_episodes = 300
for i_episode in range(num_episodes):

    # 初始化环境和状态
    env.reset()
    last_screen = get_screen()
    current_screen = get_screen()
    # 这里把两帧图像之间的差异设为state，我个人不太理解
    state = current_screen - last_screen
    for t in count():
        # 选择并执行动作
        action = select_action(state)
        _, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        # 观察新状态
        last_screen = current_screen
        current_screen = get_screen()
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None

        # 在内存中储存当前参数
        memory.push(state, action, next_state, reward)

        # 进入下一状态
        state = next_state

        # 进行一步优化
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            print("Episode {} finished after {} timesteps".format(i_episode, t + 1))
            plot_durations()
            break
    # 更新目标网络, 复制在 DQN 中的所有权重参数
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()