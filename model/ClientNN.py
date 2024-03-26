import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models


class Mnist_2NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, inputs):
        tensor = inputs.view(inputs.size(0), -1)
        tensor = F.relu(self.fc1(tensor))
        tensor = F.relu(self.fc2(tensor))
        tensor = F.softmax(self.fc3(tensor), dim=1)
        return tensor


class Mnist_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(7 * 7 * 64, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, inputs):
        tensor = inputs.view(-1, 1, 28, 28)
        tensor = F.relu(self.conv1(tensor))
        tensor = self.pool1(tensor)
        tensor = F.relu(self.conv2(tensor))
        tensor = self.pool2(tensor)
        tensor = tensor.view(-1, 7 * 7 * 64)
        tensor = F.relu(self.fc1(tensor))
        tensor = F.relu(self.fc2(tensor))
        tensor = F.softmax(self.fc3(tensor), dim=1)
        return tensor


class Better_MNIST(nn.Module):
    def __init__(self):
        super(Better_MNIST, self).__init__()

        # Convolution layer 1 (（w - f + 2 * p）/ s ) + 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.batch1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.batch2 = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1_drop = nn.Dropout(0.25)

        # Convolution layer 2
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.relu3 = nn.ReLU()
        self.batch3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.relu4 = nn.ReLU()
        self.batch4 = nn.BatchNorm2d(64)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2_drop = nn.Dropout(0.25)

        # Fully-Connected layer 1

        self.fc1 = nn.Linear(576, 256)
        self.fc1_relu = nn.ReLU()
        self.dp1 = nn.Dropout(0.5)

        # Fully-Connected layer 2
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        # conv layer 1 的前向计算，3行代码
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.batch1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.batch2(out)

        out = self.maxpool1(out)
        out = self.conv1_drop(out)

        # conv layer 2 的前向计算，4行代码
        out = self.conv3(out)
        out = self.relu3(out)
        out = self.batch3(out)

        out = self.conv4(out)
        out = self.relu4(out)
        out = self.batch4(out)

        out = self.maxpool2(out)
        out = self.conv2_drop(out)

        # Flatten拉平操作
        out = out.view(out.size(0), -1)

        # FC layer的前向计算（2行代码）
        out = self.fc1(out)
        out = self.fc1_relu(out)
        out = self.dp1(out)

        out = self.fc2(out)

        return F.log_softmax(out, dim=1)


class Cifar10(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Cifar10, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.softmax(out)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = nn.ReLU()(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = nn.AvgPool2d(4)(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


Half_width = 256
layer_width = 512


def conv3x3(in_channels, out_channels, stride=1):
    """3x3 kernel size with padding convolutional layer in ResNet BasicBlock."""
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


class VGG(nn.Module):

    def __init__(self, features, num_class=10):
        super().__init__()
        self.features = features

        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)

        return output


class SpinalVGG(nn.Module):

    def __init__(self, features, num_class=10):
        super().__init__()
        self.features = features

        self.fc_spinal_layer1 = nn.Sequential(
            nn.Dropout(), nn.Linear(Half_width, layer_width),
            nn.ReLU(inplace=True),
        )
        self.fc_spinal_layer2 = nn.Sequential(
            nn.Dropout(), nn.Linear(Half_width + layer_width, layer_width),
            nn.ReLU(inplace=True),
        )
        self.fc_spinal_layer3 = nn.Sequential(
            nn.Dropout(), nn.Linear(Half_width + layer_width, layer_width),
            nn.ReLU(inplace=True),
        )
        self.fc_spinal_layer4 = nn.Sequential(
            nn.Dropout(), nn.Linear(Half_width + layer_width, layer_width),
            nn.ReLU(inplace=True),
        )
        self.fc_out = nn.Sequential(
            nn.Dropout(), nn.Linear(layer_width * 4, num_class)
        )

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        x = output
        x1 = self.fc_spinal_layer1(x[:, 0:Half_width])
        x2 = self.fc_spinal_layer2(torch.cat([x[:, Half_width:2 * Half_width], x1], dim=1))
        x3 = self.fc_spinal_layer3(torch.cat([x[:, 0:Half_width], x2], dim=1))
        x4 = self.fc_spinal_layer4(torch.cat([x[:, Half_width:2 * Half_width], x3], dim=1))

        x = torch.cat([x1, x2], dim=1)
        x = torch.cat([x, x3], dim=1)
        x = torch.cat([x, x4], dim=1)

        x = self.fc_out(x)

        return x


def make_layers(cfg, batch_norm=False):
    layers = []

    input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]

        layers += [nn.ReLU(inplace=True)]
        input_channel = l

    return nn.Sequential(*layers)


def vgg11_bn():
    return VGG(make_layers(cfg['A'], batch_norm=True))


def vgg13_bn():
    return VGG(make_layers(cfg['B'], batch_norm=True))


def vgg16_bn():
    return VGG(make_layers(cfg['D'], batch_norm=True))


def vgg19_bn():
    return VGG(make_layers(cfg['E'], batch_norm=True))


def Spinalvgg11_bn():
    return SpinalVGG(make_layers(cfg['A'], batch_norm=True))


def Spinalvgg13_bn():
    return SpinalVGG(make_layers(cfg['B'], batch_norm=True))


def Spinalvgg16_bn():
    return SpinalVGG(make_layers(cfg['D'], batch_norm=True))


def Spinalvgg19_bn():
    return SpinalVGG(make_layers(cfg['E'], batch_norm=True))


# For updating learning rate
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class Network_Wrapper(nn.Module):
    def __init__(self, net_layers, num_classes):
        super().__init__()
        self.Features = Features(net_layers)

        self.max_pool1 = nn.MaxPool2d(kernel_size=46, stride=1)
        self.max_pool2 = nn.MaxPool2d(kernel_size=23, stride=1)
        self.max_pool3 = nn.MaxPool2d(kernel_size=12, stride=1)

        self.conv_block1 = nn.Sequential(
            BasicConv(512, 512, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(512, 1024, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier1 = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Linear(512, num_classes)
        )

        self.conv_block2 = nn.Sequential(
            BasicConv(1024, 512, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(512, 1024, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier2 = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Linear(512, num_classes),
        )

        self.conv_block3 = nn.Sequential(
            BasicConv(2048, 512, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(512, 1024, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier3 = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Linear(512, num_classes),
        )

        self.classifier_concat = nn.Sequential(
            nn.BatchNorm1d(1024 * 3),
            nn.Linear(1024 * 3, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x1, x2, x3 = self.Features(x)

        x1_ = self.conv_block1(x1)
        map1 = x1_.clone().detach()
        x1_ = self.max_pool1(x1_)
        x1_f = x1_.view(x1_.size(0), -1)

        x1_c = self.classifier1(x1_f)

        x2_ = self.conv_block2(x2)
        map2 = x2_.clone().detach()
        x2_ = self.max_pool2(x2_)
        x2_f = x2_.view(x2_.size(0), -1)
        x2_c = self.classifier2(x2_f)

        x3_ = self.conv_block3(x3)
        map3 = x3_.clone().detach()
        x3_ = self.max_pool3(x3_)
        x3_f = x3_.view(x3_.size(0), -1)
        x3_c = self.classifier3(x3_f)

        x_c_all = torch.cat((x1_f, x2_f, x3_f), -1)
        x_c_all = self.classifier_concat(x_c_all)

        return x1_c, x2_c, x3_c, x_c_all, map1, map2, map3


from model.STC import *


def TResNet():
    model_params = {'num_classes': 196}

    model = TResnetL_V2(model_params)
    weights_url = \
        'https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/tresnet/stanford_cars_tresnet-l-v2_96_27.pth'
    weights_path = "tresnet-l-v2.pth"

    if not os.path.exists(weights_path):
        print('downloading weights...')
        r = requests.get(weights_url)
        with open(weights_path, "wb") as code:
            code.write(r.content)
    pretrained_weights = torch.load(weights_path)
    model.load_state_dict(pretrained_weights['model'])

    net_layers = list(model.children())
    net_layers = net_layers[0]
    net_layers = list(net_layers.children())
    net = Network_Wrapper(net_layers, 196)

    return net


def init_net(dataset, dev):
    net = None
    if dataset == 'mnist':
        net = Better_MNIST()
    elif dataset == 'cifar10':
        net = Spinalvgg19_bn()
    elif dataset == 'stanford':
        # net = TResNet()
        net = models.efficientnet_v2_s(pretrained=True)
        for params in net.parameters():
            params.requires_grad = True
        net.classifier[1] = nn.Linear(in_features=1280, out_features=196)
    net = net.to(dev)
    return net
