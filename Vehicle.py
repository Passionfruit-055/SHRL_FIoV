import random
from collections import deque

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import model.ClientNN as MnistNN
from utils import DataLoader
from torch.autograd import Variable
from utils.stanfordcars import *
from tqdm import tqdm


class Vehicle(object):
    def __init__(self, index, par, power, trans_speed, trans_latency, compute_speed, compute_latency, lr, dev, datasize,
                 commDataDict, dataset, batch_size, net=None):
        self.index = int(index)
        self.par = par
        self.power = power
        self.trans_speed = trans_speed
        self.compute_speed = compute_speed
        self.trans_latency = trans_latency
        self.compute_latency = compute_latency
        self.model_para_size = 2200 * 8
        self.duty = 0
        self.net = net
        self.dev = dev
        # self.net.to(self.dev)
        self.lr = lr
        self.dev = dev
        self.datasize = datasize
        self.accu = 0
        self.commDataDict = commDataDict
        self.duty_times = 1
        self.par_times = 0
        self.par_rate = 0.
        self.reputation = 0.
        self.param = self.net.state_dict()
        self.dataset = dataset
        self.batch_size = batch_size

    @property
    def participate(self):
        return random.random() < self.par_rate

    def report_raw_state(self):
        return [self.trans_latency, self.compute_latency, self.par_rate, self.accu]

    def report_duty_state(self):
        return [self.trans_latency, self.compute_latency, self.par_rate, self.accu, self.duty]

    def local_training(self, train_loader, test_set, epochs):
        self.net.load_state_dict(self.param)
        self.net.train()
        loss_func = nn.CrossEntropyLoss()
        opti = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9) if self.dataset in ['mnist',
                                                                                              'cifar10'] else optim.Adam(
            self.net.parameters(), lr=1e-4)
        for epoch in range(epochs):
            for data, label in train_loader:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = self.net(data)
                loss = loss_func(preds, label)
                loss.backward()
                opti.step()
                opti.zero_grad()
        self.accu = self.local_val(test_set)
        self.param = self.net.state_dict()

    def local_val(self, test_set):
        self.net.eval()
        indices = np.random.choice(len(test_set), 1000, replace=False)
        subtest_set = torch.utils.data.Subset(test_set, indices)
        test_loader = torch.utils.data.DataLoader(subtest_set, batch_size=self.batch_size, shuffle=True,
                                                  num_workers=0) 
        correct = 0
        total = 0
        with torch.no_grad():
            for data, label in test_loader:
                data, label = data.to(self.dev), label.to(self.dev)
                outputs = self.net(data)
                _, predicted = torch.max(outputs.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
            accu = self.accu = correct / total
        return accu

    def ROOT(self, timestep, threshold, epoch):
        self.trans_latency = self.model_para_size / (self.commDataDict['Bit Rate'][timestep] * 1e6)
        self.compute_latency = 18000 * self.datasize * epoch / (self.compute_speed * 1e9)
        return self.trans_latency + self.compute_latency > threshold

    def respond(self, duty, datasize, timestep, threshold, epoch, attacked):
        self.duty = duty
        self.datasize = datasize
        respond = False
        if duty:
            root = self.ROOT(timestep, threshold, epoch)
            self.update_reputation(attacked, duty, root)
            print(f"Vehicle {self.index} ", end='')
            if attacked: 
                print('Being attacked!')
            elif root: 
                print('ROOT!')
            else:
                print("On call!")
                respond = True
        self.duty_times += duty
        self.par_times += respond
        self.par_rate = self.reputation  # for APIs
        return respond

    def update_reputation(self, attacked, duty, root):
        par_rate = self.par_times / self.duty_times
        if attacked and duty:
            weight = -0.2
        elif root and duty:
            weight = -0.1
        elif duty:
            weight = 0.1
        else:
            weight = 0
        self.reputation += par_rate * weight


def stanford_train(net, nb_epoch, train_loader, device):
    netp = torch.nn.DataParallel(net, device_ids=[0])

    CELoss = nn.CrossEntropyLoss()
    optimizer = optim.SGD([
        {'params': net.classifier_concat.parameters(), 'lr': 0.002},
        {'params': net.conv_block1.parameters(), 'lr': 0.002},
        {'params': net.classifier1.parameters(), 'lr': 0.002},
        {'params': net.conv_block2.parameters(), 'lr': 0.002},
        {'params': net.classifier2.parameters(), 'lr': 0.002},
        {'params': net.conv_block3.parameters(), 'lr': 0.002},
        {'params': net.classifier3.parameters(), 'lr': 0.002},
        {'params': net.Features.parameters(), 'lr': 0.0002}

    ],
        momentum=0.9, weight_decay=5e-4)

    max_val_acc = 0
    lr = [0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.0002]
    for epoch in tqdm(range(nb_epoch)):
        # print('\nEpoch: %d' % epoch)
        net.train()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            idx = batch_idx
            # if inputs.shape[0] < batch_size:
            #     continue

            inputs, targets = inputs.to(device), targets.to(device)
            inputs, targets = Variable(inputs), Variable(targets)

            for nlr in range(len(optimizer.param_groups)):
                optimizer.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, nb_epoch, lr[nlr])

            # Train the experts from deep to shallow with data augmentation by multiple steps
            # e3
            optimizer.zero_grad()
            inputs3 = inputs
            output_1, output_2, output_3, _, map1, map2, map3 = netp(inputs3)
            loss3 = CELoss(output_3, targets) * 1
            loss3.backward()
            optimizer.step()

            p1 = net.state_dict()['classifier3.1.weight']
            p2 = net.state_dict()['classifier3.4.weight']
            att_map_3 = map_generate(map3, output_3, p1, p2)
            inputs3_att = attention_im(inputs, att_map_3)

            p1 = net.state_dict()['classifier2.1.weight']
            p2 = net.state_dict()['classifier2.4.weight']
            att_map_2 = map_generate(map2, output_2, p1, p2)
            inputs2_att = attention_im(inputs, att_map_2)

            p1 = net.state_dict()['classifier1.1.weight']
            p2 = net.state_dict()['classifier1.4.weight']
            att_map_1 = map_generate(map1, output_1, p1, p2)
            inputs1_att = attention_im(inputs, att_map_1)
            inputs_ATT = highlight_im(inputs, att_map_1, att_map_2, att_map_3)

            # e2
            optimizer.zero_grad()
            flag = torch.rand(1)
            if flag < (1 / 3):
                inputs2 = inputs3_att
            elif (1 / 3) <= flag < (2 / 3):
                inputs2 = inputs1_att
            elif flag >= (2 / 3):
                inputs2 = inputs

            _, output_2, _, _, _, map2, _ = netp(inputs2)
            loss2 = CELoss(output_2, targets) * 1
            loss2.backward()
            optimizer.step()

            # e1
            optimizer.zero_grad()
            flag = torch.rand(1)
            if flag < (1 / 3):
                inputs1 = inputs3_att
            elif (1 / 3) <= flag < (2 / 3):
                inputs1 = inputs2_att
            elif flag >= (2 / 3):
                inputs1 = inputs

            output_1, _, _, _, map1, _, _ = netp(inputs1)
            loss1 = CELoss(output_1, targets) * 1
            loss1.backward()
            optimizer.step()

            # Train the experts and their concatenation with the overall attention region in one go
            optimizer.zero_grad()
            output_1_ATT, output_2_ATT, output_3_ATT, output_concat_ATT, _, _, _ = netp(inputs_ATT)
            concat_loss_ATT = CELoss(output_1_ATT, targets) + \
                              CELoss(output_2_ATT, targets) + \
                              CELoss(output_3_ATT, targets) + \
                              CELoss(output_concat_ATT, targets) * 2
            concat_loss_ATT.backward()
            optimizer.step()

            # Train the concatenation of the experts with the raw input
            optimizer.zero_grad()
            _, _, _, output_concat, _, _, _ = netp(inputs)
            concat_loss = CELoss(output_concat, targets) * 2
            concat_loss.backward()
            optimizer.step()


def standford_test(net, test_set, device):
    testloader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True, num_workers=0)

    # criterion = nn.CrossEntropyLoss()

    use_cuda = torch.cuda.is_available()
    test_loss = 0
    correct = 0
    correct_com = 0
    correct_com2 = 0
    total = 0
    idx = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            idx = batch_idx
            if use_cuda:
                inputs, targets = inputs.to(device), targets.to(device)
            inputs, targets = Variable(inputs), Variable(targets)
            output_1, output_2, output_3, output_concat, map1, map2, map3 = net(inputs)

            p1 = net.state_dict()['classifier3.1.weight']
            p2 = net.state_dict()['classifier3.4.weight']
            att_map_3 = map_generate(map3, output_3, p1, p2)

            p1 = net.state_dict()['classifier2.1.weight']
            p2 = net.state_dict()['classifier2.4.weight']
            att_map_2 = map_generate(map2, output_2, p1, p2)

            p1 = net.state_dict()['classifier1.1.weight']
            p2 = net.state_dict()['classifier1.4.weight']
            att_map_1 = map_generate(map1, output_1, p1, p2)

            inputs_ATT = highlight_im(inputs, att_map_1, att_map_2, att_map_3)
            output_1_ATT, output_2_ATT, output_3_ATT, output_concat_ATT, _, _, _ = net(inputs_ATT)

            outputs_com2 = output_1 + output_2 + output_3 + output_concat
            outputs_com = outputs_com2 + output_1_ATT + output_2_ATT + output_3_ATT + output_concat_ATT

            # loss = criterion(output_concat, targets)
            #
            # test_loss += loss.item()
            _, predicted = torch.max(output_concat.data, 1)
            _, predicted_com = torch.max(outputs_com.data, 1)
            _, predicted_com2 = torch.max(outputs_com2.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            # correct_com += predicted_com.eq(targets.data).cpu().sum()
            # correct_com2 += predicted_com2.eq(targets.data).cpu().sum()

            # print('Step: %d | Loss: %.3f |Combined Acc: %.3f%% (%d/%d)' % (
            #     batch_idx, test_loss / (batch_idx + 1),
            #     100. * float(correct_com) / total, correct_com, total))

        # test_acc_en = 100. * float(correct_com) / total
        test_acc_en = float(correct) / total
        print(f'local accu: {test_acc_en * 100. :.1f}% ({correct}/{total})')
    return test_acc_en


if __name__ == '__main__':
    data_dict = {'latency': 1, 'par_rate': 0.5}
    for d in data_dict.keys():
        print(d, data_dict[d])
