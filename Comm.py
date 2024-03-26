import os
import random
from collections import deque
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
import torch

import model.ClientNN as ClientNN
import utils.DataLoader as DataLoader
import utils.parser as parser
from Server import Server
from Vehicle import Vehicle
from attacker import Attacker

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

pars = parser.init_parser()
args = pars.parse_args()
args = args.__dict__

# set random seed
seed = random.randint(0, 10000)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

plt.style.use('bmh')
plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["font.size"] = 20
plt.rcParams['axes.unicode_minus'] = True
plt.rcParams['text.color'] = 'black'
plt.set_cmap('jet')


def comm():
    info = args['info']

    now = datetime.now()
    dt_string = now.strftime("%m-%d-%H-%M")
    root_path = 'results/' + now.strftime('%m.%d') + '/' + str(seed) + '_' + str(args['dataset']) + '_' + info + '/'
    file_kind = ['xlsx', 'mat', 'png', 'pdf', 'report']
    for fk in file_kind:
        if not os.path.exists(root_path + fk):
            os.makedirs(root_path + fk)
    category = ['reward', 'accuracy', 'latency', 'attack_rate', 'detection_rate', 'atk_success_rate']

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    if torch.cuda.is_available():
        print('run on GPU, num of GPU is', torch.cuda.device_count())
    else:
        print('run on CPU')

    IID = args['IID']
    batchSize = args['batch_size']
    epoch = args['epoch']
    timestep = args['timestep']

    total_num = args['total_num']
    duty_num = args['duty_num']

    # load data
    train_set, test_set = DataLoader.load_data(dataset=args['dataset'])

    setsizes = [16, 32, 64, 96, 128]
    mul = 1
    accu_threshold = 0.9
    epochmul = 1

    if args['dataset'] == 'cifar10':
        epochmul = 5 
        setsizes = [mul * s for s in setsizes]
        batchSize = 32
        epoch *= epochmul
        timestep = 2000 # 2500

    elif args['dataset'] == 'stanford':
        epochmul = 5
        setsizes = [mul * s for s in setsizes]
        batchSize = 32
        epoch *= epochmul
        timestep = 600

    mode_set = ['FedAvg', 'Fixed FedAvg', 'DQN', 'Reward reshaping', 'SafeHRL']

    reward_set = []
    accu_set = []
    latency_set = []
    root_set = []

    attack_set = []
    detect_set = []
    success_attack_set = []

    log_add = " "
    for mode in mode_set:
        # init server
        net = ClientNN.init_net(args['dataset'], dev)
        server = Server(net, mode, args['lrServer'], total_num, duty_num, setsizes, accu_threshold, args['dataset'],
                        mul,
                        batchSize=batchSize)
        print(f"Server working in {mode} mode.")

        attacker = Attacker(args, 6, duty_num)

        # init clients
        vehicles = []
        cpu_freq = [5, 4, 3, 3, 2, 1]  # GHz
        compute_speed = [s / cycle for s in cpu_freq]

        comm_file_path = './data/comm/'

        for i, comm_file in enumerate(comm_files):
            df = pd.read_excel(comm_file_path + comm_file)
            data_dict = {}
            for column in df.columns:
                data_dict[column] = df[column].tolist()
            vehicles.append(
                Vehicle(i, par[i], data_dict['Tx-Power'][0], data_dict['Bit Rate'][0], 0, compute_speed[i], 0,
                        args['lrClient'], dev, 0, data_dict, args['dataset'], args['batch_size'], net=server.net))

        print(f'Init {total_num} vehicles done.')

        rewards = []
        accus = []
        lates = []
        attacks = []
        detects = []
        success_attacks = []
        all_detected = deque(maxlen=args['memolen'])
        all_attacked = deque(maxlen=args['memolen'])
        all_not_detected = deque(maxlen=args['memolen'])

        # history padding
        for _ in range(args['memolen']):
            # all_attacked.append(random.choice(range(2, 5)))
            all_attacked.append(random.choice(range(3, 6)))
            all_detected.append(random.choice(range(1, 3)))
            all_not_detected.append(random.choice(range(1, 4)))

        duty = [1] * duty_num + [0] * (total_num - duty_num)
        random.shuffle(duty)

        for t in range(timestep):
            print(f'\nRound {t}')

            # launch attacks
            r_pre = 0.0 if t == 0 else rewards[-1]
            attack_policy = attacker.launch_attack(duty, r_pre)
            attack_num = attack_policy.count(1)
            attacker.update_attack_rate(attack_policy)
            attacks.append(attacker.attack_rate)
            print(f'Duty:{duty}')
            print(f'Attacks:{attack_policy}')

            attacker.update_assign_rate(attack_policy)

            # attack detection

            atk_success = []
            detected_num = 0

            # feedback for attacker
            attacker.update_attack_success_rate(atk_success)

            all_attacked.append(attack_num)
            all_detected.append(attack_num - detected_num)
            all_not_detected.append(atk_success.count(1))

            detect_success_rate = 1 - sum(all_not_detected) / sum(all_attacked)

            detects.append(detect_success_rate)
            success_attacks.append(1 - detect_success_rate)

            # assign task
            duty, setsize = server.assign_task(args['duty_num'], vehicles)
            train_loaders = DataLoader.allocate_data(train_set, setsize, IID, batchSize)

            attacker.update_level_cost(attack_policy, duty)
            local_paras = []
            comm_num = 0
            comms = []
            print(f"Assigned datasize: {setsize}")
            for i, (vehicle, d, s, loader, atk) in enumerate(
                    zip(vehicles, duty, setsize, train_loaders, attack_policy)):
                communicate = vehicle.respond(d, s, t, server.latency_threshold, epoch / epochmul, atk)
                comms.append(communicate)
                if duty:
                    vehicle.param = server.global_paras
                    if communicate:
                        comm_num += 1
                        vehicle.local_training(loader, test_set, epochs=epoch)
                    local_paras.append(vehicle.param.items())
            print(f"{comm_num} CONNECTED")

            accus.append(server.server_eval(local_paras, test_set, comms, True if t % 100 == 0 else False) * 100)

            lates.append(server.estimate_latency(vehicles, duty, comms, attack_policy) * 1e3)

            rewards.append(server.compute_reward(vehicles, detect_success_rate))

            server.learn(vehicles)

            source = [rewards, accus, lates, attacks, detects, success_attacks]
            save_data = {}
            for i, (cate, src) in enumerate(zip(category, source)):
                save_data[cate] = src
            sio.savemat(root_path + 'mat/' + str(seed) + '_' + dt_string + '_' + mode + '.mat', save_data,
                        appendmat=True)
            # save_data = pd.DataFrame(save_data)
            # save_data.to_excel(root_path + 'xlsx/' + dt_string + '_' + mode + '.xlsx', index=False)

        # plot (episode)
        ylabels = ['Reward', 'Accuracy (%)', 'latency (ms)', 'Attack Rate', 'Detection Success Rate',
                   'Attack Success Rate']
        for c, ylable in zip(category, ylabels):
            plt.figure()
            plt.plot(save_data[c])
            plt.title(mode)
            if c == 'accuracy':
                plt.yticks(np.arange(0, 110, 10))
            plt.xlabel('timestep'.title())
            plt.ylabel(ylable)
            plt.tight_layout()
            plt.savefig(root_path + 'png/' + mode + '_' + c + '.png')
            # plt.savefig(root_path + 'pdf/' + dt_string + '_' + modeset[m] + '_' + c + '.pdf')
            plt.close()

        reward_set.append(rewards)
        accu_set.append(accus)
        latency_set.append(lates)
        attack_set.append(attacks)
        detect_set.append(detects)
        success_attack_set.append(success_attacks)
        # ending report
        server.logging(vehicles, root_path + 'report/' + dt_string + '_' + mode + '.txt', log_add, args)

    # plot exp
    colors = ['red', 'royalblue', 'black', 'forestgreen', 'orange', 'darkviolet', ]
    data_set = [reward_set, accu_set, latency_set, attack_set, detect_set, success_attack_set]
    ylabels = ['Reward', 'Accuracy (%)', 'Latency (ms)', 'Attack rate', 'Detection success rate', 'Attack success rate']
    for c, d, yl in zip(category, data_set, ylabels):
        plt.figure()
        for color, data, m in zip(colors, d, mode_set):
            plt.plot(data, color=color, label=m)
        plt.legend(loc='best')
        if c == 'accuracy':
            plt.yticks(np.arange(0, 110, 10))
        plt.xlabel('timestep'.title())
        plt.ylabel(yl)
        plt.title(seed)
        plt.tight_layout()
        plt.savefig(root_path + 'png/' + 'All_' + c + '.png')
        # plt.savefig(root_path + 'pdf/' + dt_string + '_sum_' + c + '.pdf')
        if yl in ['Accuracy (%)']:
            plt.show()
        plt.close()


if __name__ == '__main__':
    comm()
