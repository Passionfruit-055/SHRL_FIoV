import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class CNN_E(nn.Module):
    def __init__(self, input_length, num_action):
        super(CNN_E, self).__init__()

        self.num_action = num_action
        self.cov1 = nn.Conv1d(1, 20, kernel_size=3)
        self.bn1 = nn.BatchNorm1d(20)
        self.cov2 = nn.Conv1d(20, 40, kernel_size=2)
        self.bn2 = nn.BatchNorm1d(40)
        self.fc1 = nn.Linear(40 * (input_length + 1 - 3 + 1 - 2), 180)
        self.fc2 = nn.Linear(180, self.num_action)

    def forward(self, x):
        x = x.view(x.size(0), -1, x.size(-1))  # attention! a big keng here
        x = F.leaky_relu(self.bn1(self.cov1(x)))
        x = F.leaky_relu(self.bn2(self.cov2(x)))
        x = x.view(x.size(0), -1)

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