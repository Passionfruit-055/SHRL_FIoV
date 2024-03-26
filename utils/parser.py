import argparse
import random


def init_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="SafeHRL")
    parser.add_argument('-nvt', '--total_num', type=int, default=6, help='num of vehicles')
    parser.add_argument('-nvd', '--duty_num', type=int, default=2, help='num of vehicles on duty')
    parser.add_argument('-lrs', '--lrServer', type=float, default=0.008,
                        help='the learning rate of Server')
    parser.add_argument('-lrc', '--lrClient', type=float, default=0.05,
                        help='the learning rate of Client')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='local train batch size')
    parser.add_argument('-ds', "--dataset", type=str, default="cifar10", help="name of dataset") # mnist, cifar10, stanford

    parser.add_argument('-ts', '--timestep', type=int, default=1000, help='timesteps for one epoch')
    parser.add_argument('-e', '--epoch', type=int, default=1, help='Total epoch for one communication round')
    parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints', help='the saving path of checkpoints')
    parser.add_argument('-IID', '--IID', type=bool, default=True, help='the data distribution is IID or not')
    parser.add_argument('-mn', '--model_name', type=str, default='mnist_2nn', help='local model name')
    parser.add_argument('-m', '--mode', type=int, default=3, help='''
                                                                  Working mode of server
                                                                  1. Fedavg
                                                                  2. MPL
                                                                  2. DQN
                                                                  3. reward reshaping
                                                                  4. SHRL_FIoV
                                                                  ''')
    parser.add_argument('-s', '--seed', type=int, default=5, help='random seed')
    # attacker related
    parser.add_argument('--game', type=bool, default=True, help='whether attacker_reward related to server')

    parser.add_argument('--info', type=str, default='2 vehicle', help='experiment info')
    parser.add_argument('--memolen', type=int, default=50, help='save recent n records') # 200
    return parser


if __name__ == '__main__':
    pass
