import os
import argparse
from mozi_utils.pyfile import read_start_epoch
from mozi_utils.pyfile import write_start_epoch_file
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math
import numpy as np
from mozi_ai_sdk.test.env import Antiship as Environment
from mozi_ai_sdk.test.env import etc
from collections import OrderedDict
from mozi_utils import pylog
import matplotlib.pyplot as plt

#  设置墨子安装目录下bin目录为MOZIPATH，程序会自动启动墨子
os.environ['MOZIPATH'] = 'D:\\MoZiSystem\\Mozi\\MoziServer\\bin'
parser = argparse.ArgumentParser()
parser.add_argument('--avail_ip_port', type=str, default='127.0.0.1:6060')
parser.add_argument('--platform_mode', type=str, default='development')
parser.add_argument('--side_name', type=str, default='蓝方')
parser.add_argument('--agent_key_event_file', type=str, default=None)

""" 后面这些全都可以加进argument里面 """
BATCH_SIZE = 64
LR = 0.0001  # 学习率
GAMMA = 0.95  # 折扣因子，对未来奖励 的折扣
EPSILON_START = 0.90  # e-greedy策略中初始epsilon
EPSILON_END = 0.01  # e-greedy策略中的终止epsilon
EPSILON_DECAY = 500  # e-greedy策略中epsilon的衰减率
TARGET_UPDATE = 4  # 目标网络的更新频率
MEMORY_CAPACITY = 100000  # 经验回放的容量
HIDDEN_DIM = 256  # 网络隐藏层
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检测GPU

# 导入环境
args = parser.parse_args()
if args.platform_mode == 'versus':
    print('比赛模式')
    ip_port = args.avail_ip_port.split(':')
    ip = ip_port[0]
    port = ip_port[1]
    env = Environment(ip,
                      port,
                      duration_interval=etc.DURATION_INTERVAL,
                      app_mode=2,
                      agent_key_event_file=args.agent_key_event_file,
                      platform_mode=args.platform_mode)

else:
    print('开发模式')
    env = Environment(etc.SERVER_IP,
                      etc.SERVER_PORT,
                      None,
                      etc.DURATION_INTERVAL,
                      etc.app_mode,
                      etc.SYNCHRONOUS,
                      etc.SIMULATE_COMPRESSION,
                      etc.SCENARIO_NAME,
                      platform_mode=args.platform_mode)
env.start(env.server_ip, env.aiPort)
env.step_count = 0
epoch_file_path = "%s/epoch.txt" % etc.OUTPUT_PATH
#start_epoch = read_start_epoch(epoch_file_path)

STATE_DIM = 9  # 状态;216  env.get_obs_agent()
ACTION_DIM = 94  # 动作维度;94   env.get_avail_agent_actions()


class Net(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        """ 初始化q网络，为全连接网络
            state_dim: 输入的特征数即环境的状态维度
            action_dim: 输出的动作维度
        """
        super(Net, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)  # 输入层
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # 隐藏层
        self.fc3 = nn.Linear(hidden_dim, action_dim)  # 输出层

    def forward(self, x):
        # 各层对应的激活函数
        x = F.relu(self.fc1(x))  # {Tensor:(64,4)}tensor(([[ 0.1834,  0.0130,  0.1694,  0.6715],...])
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQN(object):
    def __init__(self):
        self.device = DEVICE
        # 神经网络参数设置
        self.policy_net = Net(STATE_DIM, ACTION_DIM, hidden_dim=HIDDEN_DIM).to(self.device)
        self.target_net = Net(STATE_DIM, ACTION_DIM, hidden_dim=HIDDEN_DIM).to(self.device)

        for target_param, param in zip(self.target_net.parameters(),
                                       self.policy_net.parameters()):  # 复制参数到目标网路targe_net
            target_param.data.copy_(param.data)

        # buffer参数设置
        self.learn_step_counter = 0  # 统计步数
        self.capacity = MEMORY_CAPACITY  # 经验回放的容量
        self.buffer = []  # 缓冲区
        self.position = 0  # 缓冲区中的位置

        # e-greedy策略相关参数
        self.frame_idx = 0  # 用于epsilon的衰减计数
        self.epsilon = lambda frame_idx: EPSILON_END + \
                                         (EPSILON_START - EPSILON_END) * \
                                         math.exp(-1. * frame_idx / EPSILON_DECAY)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)  # 优化器

    def update_target_network(self, episode_idx):
        if (episode_idx + 1) % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(OrderedDict(self.policy_net.state_dict()))
            #print("目标网络已更新！")

    def choose_action(self, state, avail_actions):
        self.frame_idx += 1

        state = torch.tensor(state, device=self.device, dtype=torch.float32)
        q_values = self.policy_net(state).detach().cpu().numpy()

        q_values[avail_actions == 0] = float('-inf')

        if random.random() > self.epsilon(self.frame_idx):
            action = np.argmax(q_values)
        else:
            action = random.choice(np.flatnonzero(avail_actions))  # 从可用动作中随机选择一个

        return action

    def store_transition(self, state, action, reward, next_state, done):   # 将当前的经验存储到经验回放缓冲区中 删除了done
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)  # 插入一个空值
        self.buffer[self.position] = (state, action, reward, next_state, done)   # 将当前的经验存储到经验回放缓冲区的位置 self.position 处       删除了done
        self.position = (self.position + 1) % self.capacity                      # 更新经验回放缓冲区的位置，采用循环队列的方式，当达到缓冲区的容量时，会重新从头开始覆盖旧的经验

    def learn(self):
        if len(self.buffer) < BATCH_SIZE:  # 当memory中不满足一个批量时，不更新策略
            return
        # 采样
        batch = random.sample(self.buffer, BATCH_SIZE)  # 随机采出小批量转移
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)  # 解压成状态，动作等

        # 转为张量
        state_batch = torch.tensor(state_batch, device=self.device, dtype=torch.float)
        action_batch = torch.tensor(action_batch, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(reward_batch, device=self.device, dtype=torch.float)
        next_state_batch = torch.tensor(next_state_batch, device=self.device, dtype=torch.float)
        done_batch = torch.tensor(np.float32(done_batch), device=self.device)

        # q_values = self.policy_net(state)    # learn和choose action处的区别
        q_values = self.policy_net(state_batch).gather(dim=1, index=action_batch)  # 计算当前状态(s_t,a)对应的Q(s_t, a)
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach()  # 计算下一时刻的状态(s_t_,a)对应的Q值

        # 计算期望的Q值，对于终止状态，此时done_batch[0]=1, 对应的expected_q_value等于reward
        expected_q_values = reward_batch + GAMMA * next_q_values * (1 - done_batch)
        loss = nn.MSELoss()(q_values, expected_q_values.unsqueeze(1))  # 计算均方根损失
        # 优化更新模型
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():  # clip防止梯度爆炸
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()


dqn = DQN()
episode_rewards = []
for i_episode in range(5000):
    env.reset()
    s = env.get_obs_agent()  # 得到环境的反馈，现在的状态
    # print("Observation shape:", s.shape)
    ep_r = 0
    # while True:
    for i in range(20):
        env.get_ship_missile_list()
        avail_actions = env.get_avail_agent_actions()
        a = dqn.choose_action(s, avail_actions)  # 根据dqn来接受现在的状态，得到一个行为
        step_use_mount = env.do_action(a)  #
        s_ = env.get_obs_agent()
        r = env.health_reward()      #env.reward_battle(step_use_mount)  reward_battle 输入是mount_use_ep， step_use_mount还是有问题
        done = False
        dqn.store_transition(s, a, r, s_, done)  # dqn存储现在的状态，行为，反馈，和环境导引的下一个状态
        s = s_
        dqn.learn()
        ep_r += r  # 累加奖励
        dqn.update_target_network(i_episode)
        if done:
            break
    print(f"回合：{i_episode + 1}，奖励：{ep_r:.1f}")
    episode_rewards.append(ep_r)
plt.plot(range(1, len(episode_rewards) + 1), episode_rewards, label='回合奖励')
plt.xlabel('episode')
plt.ylabel('reward')
plt.title('1')
plt.legend()
plt.savefig("./output" + '/plt_{}.png'.format(5000), format='png')
print('完成测试！')
