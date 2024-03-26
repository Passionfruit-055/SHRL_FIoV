import numpy as np
import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Sampler
import os


def load_data(dataset):
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    transform_mnist = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    transform_train_cifar10 = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform_test_cifar10 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor()])
    # transform_train_cifar10 = transforms.Compose([
    #     transforms.Resize((272, 272)),
    #     transforms.RandomRotation(15, ),
    #     transforms.RandomCrop(256),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    # ])
    # transform_test_cifar10 =  transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    # ])
    transform_train_stanford = transforms.Compose([
        transforms.Resize((421, 421)),
        transforms.RandomCrop(368, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    transform_test_stanford = transforms.Compose([
        transforms.Resize((421, 421)),
        transforms.CenterCrop(368),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # following this tutorial https://debuggercafe.com/stanford-cars-classification-using-efficientnet-pytorch/
    IMAGE_SIZE = 224
    stanford_train_efficient = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(35),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        transforms.RandomGrayscale(p=0.5),
        transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
        transforms.RandomPosterize(bits=2, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    stanford_test_efficient = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_set = torch.empty(0)
    test_set = torch.empty(0)
    if dataset == 'mnist':
        train_set = datasets.MNIST(root=parent_dir + '/data', train=True, download=False, transform=transform_mnist)
        test_set = datasets.MNIST(root=parent_dir + '/data', train=False, download=False, transform=transform_mnist)
    elif dataset == 'cifar10':
        train_set = datasets.CIFAR10(root='./data', train=True,
                                     download=False, transform=transform)

        test_set = datasets.CIFAR10(root='./data', train=False,
                                    download=False, transform=transforms.ToTensor())
    elif dataset == 'stanford':
        train_set = datasets.StanfordCars(root='./data/', split='train', transform=stanford_train_efficient,
                                          download=False)
        test_set = datasets.StanfordCars(root='./data/', split='test', transform=stanford_test_efficient,
                                         download=False)
    print('\n' + dataset.upper() + ' loaded!')
    print('train_set size: ', len(train_set))
    print('test_set size: ', len(test_set))
    return train_set, test_set


def allocate_data(train_set, subset_sizes, IID=True, batch_size=64):
    train_loaders = []
    for subset_size in subset_sizes:
        if subset_size == 0:
            train_loaders.append(None)
            continue
        indices = np.random.choice(len(train_set), subset_size, replace=False)
        subset = Subset(train_set, indices)
        train_loaders.append(
            torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=IID,
                                        num_workers=0, pin_memory=True))
    return train_loaders


if __name__ == '__main__':
    Datasets = ['mnist', 'cifar10']
    for ds in Datasets:
        train_loaders, test_loader = load_data(ds)
        for train_loader in train_loaders:
            for batch_idx, (data, target) in enumerate(train_loader):
                print(data.shape)
                print(target.shape)
                print(target)
                break
        for batch_idx, (data, target) in enumerate(test_loader):
            print(data.shape)
            print(target.shape)
            print(target)
            break
        print(ds + ' successfully loaded!')
