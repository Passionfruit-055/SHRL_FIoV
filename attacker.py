import random
from collections import deque

import numpy as np


class Attacker:
    def __init__(self, args, vehicle_num=0, duty_num=0):
        self.recent_attack_success_times = None
        self.recent_assign_times = None
        self.recent_record = args['memolen']
        self.mode = 'greedy'  # ['random', 'greedy', 'learning']
        self.epsilon = 0.2
        self.epsilon_decay = 0.001
        self.epsilon_min = 0.01

        self.vehicle_num = vehicle_num
        self.duty_num = duty_num
        self.target_num = 0
        self.success_num = 0

        self.cost = [1 for _ in range(vehicle_num)]
        self.bounty = 20 / self.vehicle_num # 50

        self.timestep = 0
        self.all_assign_times = [38 for _ in range(self.vehicle_num)]
        self.assign_times = deque(maxlen=self.recent_record)
        self.assign_times.append([1 for _ in range(self.vehicle_num)])
        self.assign_rate = [0.0 for _ in range(self.vehicle_num)]  
        self.attack_success_times = deque(maxlen=self.recent_record)
        self.attack_success_times.append([1 if i > 2 else 0 for i in range(self.vehicle_num)])
        self.attack_success_rate = [0.0 for _ in range(self.vehicle_num)]

        self.attack_rate = 1

        self.attack_policy = self.all_attack_policy

        self.game = args['game']

    def random_attack_success_rate(self):
        self.attack_success_rate = [np.random.choice(np.arange(0, 1, 0.1)) for _ in range(self.vehicle_num)]

    def annealing(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

    @property
    def all_attack_policy(self):
        import itertools

        results = []
        for r in range(0, self.vehicle_num - 1 + 1):
            combinations = list(itertools.combinations(range(1, self.vehicle_num + 1), r))
            results.extend(combinations)

        result_array = []
        for combination in results:
            result_array.append(list(combination))

        binary_sequences = []
        for combination in results:
            binary_sequence = [0] * self.vehicle_num
            for num in combination:
                binary_sequence[num - 1] = 1
            binary_sequences.append(binary_sequence)

        return binary_sequences

    @property
    def rewards(self):
        rewards = np.zeros_like(np.array(self.attack_policy)).tolist()
        for policy, reward in zip(self.attack_policy, rewards):
            cost_num_weight = (policy.count(1) / len(policy)) / 10 + 1
            for i, (p, succ_rate, assign_rate) in enumerate(zip(policy, self.attack_success_rate, self.assign_rate)):
                cost = self.cost[i] 
                reward[i] = p * (succ_rate * self.bounty - cost * cost_num_weight)
                self.cost[i] += (p * 0.001)

        policy_rewards = np.sum(np.array(rewards), axis=1)
        # print(f"cost = {self.cost}")
        return policy_rewards

    def game_reward(self, duty, server_reward):
        rewards = np.zeros_like(np.array(self.attack_policy).tolist(), dtype=np.float32)
        for policy, reward in zip(self.attack_policy, rewards):
            cost_num_weight = (policy.count(1) / len(policy)) + 1
            for i, (p, d, cost) in enumerate(zip(policy, duty, self.cost)):
                reward[i] = p * (- server_reward + self.bounty * self.attack_rate - cost * cost_num_weight)

        policy_rewards = np.sum(np.array(rewards), axis=1)
        # print(f"cost = {self.cost}")
        return policy_rewards

    def game_reward_v2(self, duty, server_reward):
        rewards = np.zeros_like(np.array(self.attack_policy).tolist(), dtype=np.float32)
        for policy, reward in zip(self.attack_policy, rewards):
            cost_num_weight = 1
            for p, d in zip(policy, duty):
                if p and not d:
                    cost_num_weight += policy.count(1) / (len(policy) * 2)
            for i, (p, d, cost) in enumerate(zip(policy, duty, self.cost)):
                reward[i] = p * (- server_reward + self.bounty * self.attack_rate - cost * cost_num_weight)

        policy_rewards = np.sum(np.array(rewards), axis=1)
        # print(f"cost = {self.cost}")
        return policy_rewards

    def update_attack_success_rate(self, success_time):
        self.attack_success_times.append(success_time)
        self.recent_attack_success_times = np.sum(np.array(self.attack_success_times), axis=0)

        self.attack_success_rate = [success / max(assign, 1) for success, assign in
                                    zip(self.recent_attack_success_times, self.recent_assign_times)]

        # print(f'success_times:{self.recent_attack_success_times}')
        # print(f'success_rate:{self.attack_success_rate}')

    def update_assign_rate(self, assign):
        self.assign_times.append(assign)
        self.recent_assign_times = np.sum(np.array(self.assign_times), axis=0)
        self.assign_rate = [assign / min((self.timestep + 1), self.recent_record) for assign in
                            self.recent_assign_times]

        # print(f'assign_times:{self.recent_assign_times}')
        # print(f'assign_rate:{self.assign_rate}')

    def update_attack_rate(self, attack_policy):
        self.all_assign_times = np.add(self.all_assign_times, attack_policy)
        self.attack_rate = sum(self.all_assign_times) / (self.vehicle_num * (self.timestep + 50))

        # print(f'attack_times:{self.all_assign_times}')
        # print(f'attack_rate:{self.attack_rate}')

    def launch_attack(self, duty, server_reward):
        self.timestep += 1
        if self.mode == 'greedy':
            self.annealing()
            if random.random() < self.epsilon:
                index = random.choice(range(len(self.attack_policy)))
            else:
                # reward = self.rewards if not self.game else self.game_reward(duty, server_reward)
                reward = self.game_reward_v2(duty, server_reward)
                # print(f'attacker rewards = {reward}')
                max_reward_indices = np.where(reward == np.max(reward))[0]
                # print(max_reward_indices)
                index = random.choice(max_reward_indices)
            policy = self.attack_policy[index]
        else:
            raise NotImplementedError('Attack mode not implemented yet')

        # self.update_cost(policy)

        return policy

    def update_cost(self, policy):
        for i, p in enumerate(policy):
            self.cost[i] += (p * 0.01)

    def update_level_cost(self, atk_policy, server_policy):
        for i, (a_p, s_p) in enumerate(zip(atk_policy, server_policy)):
            if a_p:
                if not s_p:
                    self.cost[i] += 0.01
                else:
                    self.cost[i] += 0.005

    @property
    def mean_success_rate(self):
        return np.mean(self.attack_success_rate)


if __name__ == '__main__':
    pass
