import math
import random
import time
import numpy as np
from collections import namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
import os

from model.fc3 import DeepQNetwork
from model.CNN_E import CNN_E

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


#######################################

use_cuda = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

###########################

random.seed(time.time())

Transition = namedtuple('Transition',
                        ('state', 'next_state', 'action', 'reward', 'E'))
# 'state'--numpy, 'next_state'--numpy, 'action'--int, 'reward'--float

Transition_chain = namedtuple('Transition_chain',
                              ('net_input', 'next_net_input', 'action', 'reward', 'E'))


# 'net_input'--numpy, 'next_net_input'--numpy, 'action'--int, 'reward'--float


# class Variable(autograd.Variable):
#     def __init__(self, data, *args, **kwargs):
#         # if USE_CUDA:
#         #     data = data.cuda()
#         super(Variable, self).__init__(data, *args, **kwargs)
#


class ReplayMemory(object):
    def __init__(self, capacity, window, input_length):

        self.__window = window
        self.__input_length = input_length

        self.__capacity = capacity
        self.__memory = []
        self.__memory_chain = []

    def __len__(self):
        return len(self.__memory_chain)

    def reset(self):
        self.__memory = []
        self.__memory_chain = []

    def get_net_input(self, state):
        memory_length = len(self.__memory)
        if (memory_length <= self.__window):
            return None

        else:
            net_input = []
            for i in range(memory_length - self.__window, memory_length):
                net_input += self.__memory[i].state.tolist()  # state
                net_input.append(self.__memory[i].action)  # action
            net_input += state.tolist()
            net_input = np.array(net_input).reshape(-1)  # 转为numpy一维链
            return net_input

    def push(self, state, next_state, action, R, E):
        # 输出net_input和next_net_input

        net_input = self.get_net_input(state)
        # 存储单状态经验池。注意前后的get_net_input顺序
        self.__memory.append(Transition(state, next_state, action, R, E))
        if (len(self.__memory) > self.__capacity):
            self.__memory.pop(0)
        next_net_input = self.get_net_input(next_state)
        # 存储chain
        if ((None is not net_input) and (None is not next_net_input)):
            self.__memory_chain.append(Transition_chain(net_input, next_net_input, action, R, E))
            if (len(self.__memory_chain) > self.__capacity):
                self.__memory_chain.pop(0)
        return net_input, next_net_input

    def sample(self, batch_size):
        return random.sample(self.__memory_chain, batch_size)


class CNN_Q(nn.Module):
    def __init__(self, input_length, num_action):
        super(CNN_Q, self).__init__()

        # self.size_input = size_input    # (a,b)
        # self.input_length = input_length
        self.num_action = num_action
        self.cov1 = nn.Conv1d(1, 20, kernel_size=3)
        self.bn1 = nn.BatchNorm1d(20)
        self.cov2 = nn.Conv1d(20, 40, kernel_size=2)
        self.bn2 = nn.BatchNorm1d(40)
        self.fc1 = nn.Linear(40 * (input_length + 1 - 3 + 1 - 2), 180)
        self.fc2 = nn.Linear(180, self.num_action)

    def forward(self, x):
        # x = x.resize((6, 6))
        # print(x)
        # x = x.view(x.size(0), -1)
        # print(x)
        x = x.view(x.size(0), -1, x.size(-1))  # attention! a big keng here
        x = F.leaky_relu(self.bn1(self.cov1(x)))
        x = F.leaky_relu(self.bn2(self.cov2(x)))
        x = x.view(x.size(0), -1)

        # x=self.fc1(x)
        # x = F.leaky_relu(x)

        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        return x

    def reset(self):
        self.cov1.reset_parameters()
        self.bn1.reset_parameters()
        self.cov2.reset_parameters()
        self.bn2.reset_parameters()
        self.fc1.reset_parameters()
        # self.fc2 = nn.Linear(180, self.num_action)
        self.fc2.reset_parameters()


class DQN_E:
    def __init__(self, input_length, Num_action, memory_capacity, window, lr, GAMMA=0.5, learning_begin=50, beta=0.7,
                 safe_mode=True):

        self.Num_action = Num_action
        self.memory = ReplayMemory(memory_capacity, window, input_length)
        self.GAMMA = GAMMA
        self.learning_begin = learning_begin
        self.beta = beta
        self.safe_mode = safe_mode
        self.lr = lr
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.safe_mode:
            print('running on safe mode!')

        # if use_cuda:
        #     self.CNN_Q_0 = DeepQNetwork(self.lr, output_dim=Num_action, input_dim=input_length)
        #     # # 多卡
        #     # self.CNN_Q_model = torch.nn.DataParallel(self.CNN_Q_0.cuda())
        #
        #     self.CNN_E_0 = CNN_E((input_length + 1) * window + input_length, Num_action)
        #     self.CNN_E_model = torch.nn.DataParallel(self.CNN_E_0.cuda())
        # else:
        # self.CNN_Q_model = CNN_Q((input_length + 1) * window + input_length, Num_action)
        self.CNN_Q_0 = DeepQNetwork(self.lr, output_dim=Num_action, input_dim=input_length)
        self.CNN_E_model = CNN_E((input_length + 1) * window + input_length, Num_action)

        # self.optimizer_Q = optim.SGD(self.CNN_Q_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-7)
        # self.optimizer_E = optim.SGD(self.CNN_E_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-7)
        self.optimizer_Q = optim.Adam(self.CNN_Q_model.parameters(), lr=0.001)
        self.optimizer_E = optim.Adam(self.CNN_E_model.parameters(), lr=0.001)

    def reset(self):
        self.memory.reset()
        # self.model.reset()
        # self.steps_done = 0

        # if use_cuda:
        #     self.CNN_Q_0.reset()
        #     self.CNN_Q_model = torch.nn.DataParallel(self.CNN_Q_0.cuda())  ## 每个episode结束后在GPU上重置网络模型
        #     self.CNN_E_0.reset()
        #     self.CNN_E_model = torch.nn.DataParallel(self.CNN_E_0.cuda())  ## 每个episode结束后在GPU上重置网络模型
        # else:
        self.CNN_Q_model.reset()
        self.CNN_E_model.reset()

    def choose_action(self, state):

        state = state.reshape(-1)
        net_input = self.memory.get_net_input(state)

        # if (step <= self.learning_begin):
        #     return np.random.choice(range(self.Num_action), 1, replace=False)[0]

        if (None is not net_input):  # 链非空，则经过网络计算 否则随机选动作

            Q_value = self.CNN_Q_model(torch.from_numpy(net_input.reshape(1, -1)).float())
            E_value = self.CNN_E_model(torch.from_numpy(net_input.reshape(1, -1)).float())

            temp = (math.e ** (Q_value * E_value)).reshape(-1) if self.safe_mode else (math.e ** Q_value).reshape(-1)
            # temp = (math.e ** (Q_value / (E_value + 1))).reshape(-1) if self.safe_mode else (math.e ** Q_value).reshape(-1)
            prob = temp / sum(temp)
            prob = prob.cpu().detach().numpy()
            # print(prob)
            try:
                idx = np.random.choice(range(self.Num_action), 1, replace=False, p=prob)[0]

            except:
                # print(prob)
                idx = int(random.randrange(self.Num_action))
            return idx

        else:
            return int(random.randrange(self.Num_action))

    def update_memory(self, state, next_state, action, R, E):

        # 检查输入类型
        state = state.reshape(-1)
        next_state = next_state.reshape(-1)
        action = int(action)
        # R=int(R)

        self.memory.push(state, next_state, action, R,
                         E)  # Transition = namedtuple('Transition',('state', 'next_state', 'action', 'reward'))

    def gamma_anneal(self, step, gamma_start, gamma_end, anneal_step, learning_begin):
        gamma_temp = gamma_start - (step - learning_begin) * (gamma_start - gamma_end) / anneal_step
        self.GAMMA = max(gamma_end, min(gamma_start, gamma_temp))
    def optimize_net_para(self, BATCH_SIZE=32):
        if len(self.memory) < BATCH_SIZE:  # 这里应该是chain的长度
            return

        experience = self.memory.sample(BATCH_SIZE)
        batch = Transition_chain(*zip(*experience))

        def _cat_to_tensor(data, dev, dtype):
            return torch.cat([torch.tensor(np.array(data), dtype=dtype).to(dev)])

        next_states = _cat_to_tensor(batch.next_net_input, self.device, torch.float32)
        state_batch = _cat_to_tensor(batch.net_input, self.device, torch.float32)
        action_batch = _cat_to_tensor(batch.action, self.device, torch.long).view(-1, 1)
        reward_batch = _cat_to_tensor(batch.reward, self.device, torch.float32)
        E_batch = _cat_to_tensor(batch.E, self.device, torch.float32)

        ########################## update #############################
        # self.gamma_anneal

        state_action_Q_values = self.CNN_Q_model(state_batch).gather(1, action_batch)

        next_state_action_Q_values = \
        (self.CNN_Q_model(next_states) * self.beta ** (1 / self.CNN_E_model(next_states))).max(1)[0].detach()
        expected_state_action_Q_values = (next_state_action_Q_values * self.GAMMA) + reward_batch
        lossQ = F.smooth_l1_loss(state_action_Q_values, expected_state_action_Q_values.unsqueeze(1))  # 原版loss

        # print("loss",loss)
        # # Optimize the model
        self.optimizer_Q.zero_grad()
        lossQ.backward()
        for param in self.CNN_Q_model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer_Q.step()

        state_action_E_values = self.CNN_E_model(state_batch).gather(1, action_batch)
        next_state_action_E_values = self.CNN_E_model(next_states).min(1)[0].detach()
        expected_state_action_E_values = -(next_state_action_E_values * self.GAMMA) + E_batch
        lossE = F.smooth_l1_loss(state_action_E_values, expected_state_action_E_values.unsqueeze(1))  # 原版loss
        self.optimizer_E.zero_grad()
        lossE.backward()
        for param in self.CNN_E_model.parameters():
            param.grad.data.clamp_(-1, 1)
            self.optimizer_E.step()

    def optimize_model(self, state, next_state, action, R, E, step, gamma_start=0.8, gamma_end=0.3, anneal_step=2000,
                       learning_begin=50, BATCH_SIZE=32):
        self.update_memory(state, next_state, action, R, E)
        # self.gamma_anneal(gamma_start=gamma_start, gamma_end=gamma_end, anneal_step=anneal_step,
        #                        learning_begin=learning_begin)
        self.optimize_net_para(BATCH_SIZE=BATCH_SIZE)

    def hotbooting(self, times, HotbootingMemory, BATCH_SIZE=32):
        print('hotbooting...')
        self.memory = copy.deepcopy(HotbootingMemory)
        for _ in range(times):
            self.optimize_net_para(BATCH_SIZE)
