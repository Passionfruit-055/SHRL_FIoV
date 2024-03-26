import itertools
import random

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from itertools import permutations, combinations

from DQN_agent import DQN
from model.ClientNN import Mnist_2NN
from Guide import Guide


def normalize_array(source_array):
    source_array = np.array(source_array.copy(), dtype=np.float32)
    source_mean = source_array.mean()
    mul = 1
    while source_mean * mul < 1:
        if source_mean == 0:
            break
        mul *= 10
    mul /= 10.0
    source_array *= mul
    return source_array.tolist()


class Server(object):
    def __init__(self, net, mode, lr, total_num, duty_num, setsizes, accu_threshold, dataset, mul=1, batchSize=32,
                 single_state_num=4):
        self.net = net
        self.mode = mode
        self.lr = lr
        self.global_paras = net.state_dict()
        self.total_num = total_num
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batchSize = batchSize
        self.brain1 = None
        self.brain2 = None
        self.guide = None
        self.setsizes = setsizes
        self.duty_comb = list(combinations(range(self.total_num), duty_num))
        self.size_comb = list(itertools.combinations_with_replacement(setsizes, duty_num))
        self.work_mode(lr, single_state_num)
        self.memory = {}
        self.accu = 0
        self.accu_threshold = accu_threshold
        self.latency_threshold = (0.0432 + 1.76e-4) * mul  # * 10  # + 3.8e-6  # *10 for cifar10
        self.dataset = dataset

    def work_mode(self, lr, one_state_num=4):
        state_num = self.total_num * one_state_num + 1  # 1 for server model accuracy
        if self.mode == 'DQN':
            action_num = len(self.duty_comb)
            self.brain1 = DQN(gamma=0.8, lr=lr, INITIAL_EPSILON=0.2, FINAL_EPSILON=0.001,
                              action_num=action_num,
                              state_num=state_num,
                              batch_size=self.batchSize, buffer_size=1000)
        elif self.mode == 'SafeHRL':
            action1_num = len(self.duty_comb)
            self.brain1 = DQN(gamma=0.8, lr=lr, INITIAL_EPSILON=0.2, FINAL_EPSILON=0.001,
                              action_num=action1_num,
                              state_num=state_num,
                              batch_size=self.batchSize, buffer_size=1000)
            action2_num = len(self.size_comb)
            self.brain2 = DQN(gamma=0.8, lr=lr, INITIAL_EPSILON=0.2, FINAL_EPSILON=0.001,
                              action_num=action2_num,
                              state_num=state_num + self.total_num,
                              batch_size=self.batchSize, buffer_size=1000)
            self.guide = Guide(lr=self.lr, batchSize=self.batchSize, output_num=len(self.size_comb),
                               input_num=state_num + self.total_num,
                               total_num=self.total_num, gamma=0.8, dev=self.dev)

        elif self.mode == 'Reward reshaping':
            action1_num = len(self.duty_comb)
            self.brain1 = DQN(gamma=0.8, lr=lr, INITIAL_EPSILON=0.2, FINAL_EPSILON=0.001,
                              action_num=action1_num,
                              state_num=state_num,
                              batch_size=self.batchSize, buffer_size=1000)
            action2_num = len(self.size_comb)
            self.brain2 = DQN(gamma=0.8, lr=lr, INITIAL_EPSILON=0.2, FINAL_EPSILON=0.001,
                              action_num=action2_num,
                              state_num=state_num + self.total_num,
                              batch_size=self.batchSize, buffer_size=1000)
            self.guide = Guide(lr=self.lr, batchSize=self.batchSize, output_num=len(self.size_comb),
                               input_num=state_num + self.total_num,
                               total_num=self.total_num, gamma=0.8, dev=self.dev)

    def cumulate(self, local_paras):
        sum_parameters = None
        for local_para in local_paras:
            if sum_parameters is None:
                sum_parameters = {key: var.clone() for key, var in local_para}
            else:
                for key, var in local_para:
                    sum_parameters[key] = sum_parameters[key] + var
        if sum_parameters is None:
            # print("0 CONNECTED.")
            pass
        else:
            # print(f"{len(local_paras)} CONNECTED.")
            for key, var in sum_parameters.items():
                sum_parameters[key] = var / len(local_paras)
            self.global_paras = sum_parameters
        self.net.load_state_dict(self.global_paras)

    def server_eval(self, local_paras, test_set, comms, eval):
        self.cumulate(local_paras)

        if comms.count(1) == 0 and not eval:
            return self.accu

        if self.global_paras is None:
            pass
        else:
            self.net.eval()
            indices = np.random.choice(len(test_set), 1000, replace=False)
            subtest_set = Subset(test_set, indices)
            correct = 0
            test_loader = DataLoader(subtest_set, batch_size=self.batchSize, shuffle=True)
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(self.dev), target.to(self.dev)
                    output = self.net(data)
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
            self.accu = correct / len(subtest_set)
            print('Server Accuracy: {}/{} ({:.0f}%)'.format(correct, len(subtest_set),
                                                            100. * correct / len(subtest_set)))
        return self.accu

    def assign_task(self, duty_num, vehicles):
        duty = [0 for i in range(self.total_num)]
        setsize = [0 for i in range(self.total_num)]

        if self.mode == 'FedAvg': 
            indices = np.random.choice(range(self.total_num), duty_num, replace=False)
            duty = [1 if i in indices else 0 for i in range(self.total_num)]
            setsize = [random.choice(self.setsizes) if i in indices else 0 for i in range(self.total_num)]

        elif self.mode == 'Fixed FedAvg':
            indices = np.random.choice(range(1, min(1 + duty_num, self.total_num) + 1), duty_num, replace=False)
            duty = [1 if i in indices else 0 for i in range(self.total_num)]
            setsize = [random.choice(self.setsizes) if i in indices else 0 for i in range(self.total_num)]

        elif self.mode == 'DQN':  # DQN
            state = [self.accu]
            for vehicle in vehicles:
                state.extend(vehicle.report_raw_state())
            self.memory['state'] = state
            action_index, _ = self.brain1.choose_action(state)
            self.memory['action'] = action_index
            duty_index = action_index
            indices = self.duty_comb[duty_index]
            j = 0
            for i, vehicle in enumerate(vehicles):
                if i in indices:
                    duty[i] = 1
                    setsize[i] = random.choice(self.setsizes)
                    j += 1
                else:
                    duty[i] = 0
                    setsize[i] = 0
                vehicle.duty = duty[i]
                vehicle.datasize = setsize[i]

        elif self.mode == 'Reward reshaping':  # HRL
            # level 1
            state = []
            for vehicle in vehicles:
                state.extend(vehicle.report_raw_state())
            state.append(self.accu)
            self.memory['state1'] = state
            duty_index, _ = self.brain1.choose_action(state)
            indices = self.duty_comb[duty_index]
            for i, vehicle in enumerate(vehicles):
                if i in indices:
                    duty[i] = 1
                else:
                    duty[i] = 0
                vehicle.duty = duty[i]
            self.memory['action1'] = duty_index
            # level 2
            state = []
            for vehicle in vehicles:
                state.extend(vehicle.report_duty_state())
            state.append(self.accu)
            size_index, _ = self.brain2.choose_action(state)
            self.memory['state2'] = state
            j = 0
            for i, vehicle in enumerate(vehicles):
                if i in indices:
                    setsize[i] = self.size_comb[size_index][j]
                    j += 1
                else:
                    setsize[i] = 0
                vehicle.datasize = setsize[i]
            self.memory['action2'] = size_index

        elif self.mode == 'SafeHRL':  # Gnetwork
            # level 1
            state = []
            for vehicle in vehicles:
                state.extend(vehicle.report_raw_state())
            state.append(self.accu)
            self.memory['state1'] = state
            duty_index, _ = self.brain1.choose_action(state)
            indices = self.duty_comb[duty_index]
            for i, vehicle in enumerate(vehicles):
                if i in indices:
                    duty[i] = 1
                else:
                    duty[i] = 0
                vehicle.duty = duty[i]
            self.memory['action1'] = duty_index
            # level 2
            state = []
            for vehicle in vehicles:
                state.extend(vehicle.report_duty_state())
            state.append(self.accu)
            size_index, Q_values = self.brain2.choose_action(state)
            self.memory['state2'] = state.copy()
            state.pop()
            state.append(duty_index)
            if self.accu >= self.accu_threshold:
                final_size_index = size_index
            else:
                final_size_index = self.guide.provide_guidance(state, Q_values, self.accu, self.latency_threshold)
            j = 0
            for i, vehicle in enumerate(vehicles):
                if i in indices:
                    setsize[i] = self.size_comb[final_size_index][j]
                    j += 1
                else:
                    setsize[i] = 0
                vehicle.datasize = setsize[i]
            self.memory['action2'] = final_size_index

        else:
            raise ValueError("Wrong mode!")

        return duty, setsize

    def compute_reward(self, vehicles, detect_rate):
        c_lates = []
        t_lates = []
        par = []
        risk = 0
        for vehicle in vehicles:
            if vehicle.duty == 1:
                c_lates.append(vehicle.compute_latency)
                t_lates.append(vehicle.trans_latency)
                risk_late = 0 if vehicle.compute_latency + vehicle.trans_latency < self.latency_threshold else 1
                risk_accu = 0 if vehicle.accu >= self.accu else 1
                risk += (risk_late + risk_accu)
                par.append(vehicle.par)
        if len(c_lates) == 0:
            reward = self.memory['reward']
        else:
            c_lates = normalize_array(c_lates)

            if self.mode in ['SafeHRL', 'Reward reshaping']:
                reward = self.accu + 2 * detect_rate * 5 - 20 * (max(c_lates) + max(t_lates)) + max(par) * 0.1  # 10->20
            else:
                reward = self.accu - 10 * (max(c_lates) + max(t_lates)) + max(par) * 0.1

        self.memory['reward'] = reward
        return reward 

    def estimate_latency(self, vehicles, dutys, comms, attacks):

        latency = []
        for (vehicle, duty, comm, atk) in zip(vehicles, dutys, comms, attacks):
            if not duty:
                latency.append(0)
            else:
                la = vehicle.compute_latency + vehicle.trans_latency
                latency.append(la)

        return max(latency)

    def learn(self, vehicles):
        if self.mode == 'DQN':
            next_state = [self.accu]
            for vehicle in vehicles:
                next_state.extend(vehicle.report_raw_state())
            self.brain1.store_transition(self.memory['state'], self.memory['action'], self.memory['reward'],
                                         next_state)
            self.brain1.learn()

        elif self.mode in ['SafeHRL', 'Reward reshaping']:
            next_state = []
            for vehicle in vehicles:
                next_state.extend(vehicle.report_raw_state())
            next_state.append(self.accu)
            warning_signal = self.guide.eval_ins_risk(self.memory['state1'], self.accu, self.latency_threshold)
            # print(len(self.memory['state1']), self.memory['action1'], self.memory['reward'], len(next_state))
            self.brain1.store_transition(self.memory['state1'], self.memory['action1'], self.memory['reward'],
                                         next_state, warning_signal)
            self.brain1.learn()
            next_state = []
            for vehicle in vehicles:
                next_state.extend(vehicle.report_duty_state())
            next_state.append(self.accu)
            # print(len(self.memory['state2']), self.memory['action2'], self.memory['reward'], len(next_state))
            self.brain2.store_transition(self.memory['state2'], self.memory['action2'], self.memory['reward'],
                                         next_state)
            self.brain2.learn()

    def logging(self, vehicles, path, info, args, log=True):
        file = open(path, 'w')
        if not log:
            return
        for arg in args.keys():
            print(f"{arg}:{args[arg]}", file=file)
        print(info, file=file)
        for v, vehicle in enumerate(vehicles):
            print(f"Vehicle {v + 1} \naccu: {vehicle.accu}, compute latency: {vehicle.compute_latency}, "
                  f"trans latency: {vehicle.trans_latency}, par rate: {vehicle.par_rate}", file=file)
        print("Server\naccu: {}".format(self.accu), file=file)


if __name__ == '__main__':
    server = Server(net=Mnist_2NN(), mode=2, lr=0.01, total_num=6, duty_num=2)
    server.work_mode(mode=2, lr=0.01)
    duty, sizeset = server.assign_task(mode=2, duty_num=2, vehicles=None)
    print(duty, sizeset)
