import numpy as np
from torch import nn
import torch.optim as optim
import torch
from model.fc3 import DeepQNetwork as Brain


class Guide(object):
    def __init__(self, gamma, lr, batchSize, input_num, output_num, total_num, dev, epoch=5):
        self.gamma = gamma
        self.lr = lr
        self.batchSize = batchSize
        self.brain = Brain(lr, input_num + 1, output_num)  # 1 for warning signal
        self.ins_risk = 0
        self.long_risk = 0
        self.total_num = total_num
        self.warning_signal = 0
        self.epoch = epoch
        self.beta = 1
        self.memory = {}


        self.loss_func = nn.MSELoss()
        self.dev = dev
        self.optim = optim.Adam(self.brain.parameters(), lr=self.lr) # previous sgd

    def provide_guidance(self, state, Q_values, accu, late):
        # Step 1 : choose action
        state_exp = state.copy()
        # print("state_for_guide = ", state_exp)
        self.eval_ins_risk(state_exp, accu, late)
        state_exp.append(self.warning_signal)
        guidance = self.eval_long_risk(state_exp)
        # print("Q_values = ", Q_values.detach().cpu().numpy()[0])
        # print("guidance = ", guidance.detach().cpu().numpy())
        final_action_values = Q_values + guidance * 1
        safest_action = np.argmax(final_action_values.detach().cpu().numpy())

        # Step 2 : learn
        self.brain.train()
        for e in range(self.epoch):
            guidance = self.eval_long_risk(state_exp)
            target_val = 1 + 2 * self.beta * guidance.sum() / len(guidance) if self.warning_signal == 0 else -self.warning_signal
            target = guidance.clone()
            target[safest_action] = target_val
            # print(f"target = {target}")
            loss = self.loss_func(guidance, target)
            # print(f"guide loss = {loss}")
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()

        return safest_action

    def eval_long_risk(self, state):
        state = torch.tensor(state, dtype=torch.float).to(self.dev)
        G_values = self.brain.forward(state)
        return G_values

    def eval_ins_risk(self, s, accu_pre, late):
        self.warning_signal = 0
        state = np.array(s[:-1]).reshape(self.total_num, -1)
        duty = state[:, state.shape[1] - 2]
        accu = state[:, 0]
        c_latency = state[:, 1]
        t_latency = state[:, 2]
        for v in range(self.total_num):
            if duty[v] == 1:
                if accu[v] < accu_pre:
                    self.warning_signal += 1
                if c_latency[v] + t_latency[v] > late:
                    self.warning_signal += 1

    def learn(self):
        self.brain.learn()
