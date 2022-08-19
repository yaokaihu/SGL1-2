'''获取数据集'''

import torchvision.datasets as datasets
import torchvision.transforms as T
from torch.utils.data import Dataset
import os
from PIL import Image


def get_dataset(dataset):
    if dataset == 'mnist':
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST('./dataset', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./dataset', train=False, download=True, transform=transform)
    elif dataset == 'cifar10':
        transform_train = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = datasets.CIFAR10(root='./dataset/cifar10', train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(root='./dataset/cifar10', train=False, download=True, transform=transform_test)
    elif dataset == 'cifar100':
        transform_train = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = datasets.CIFAR100(root='./dataset/cifar100', train=True, download=True,
                                          transform=transform_train)
        test_dataset = datasets.CIFAR100(root='./dataset/cifar100', train=False, download=True,
                                         transform=transform_test)
    else:
        raise ValueError('Unknown dataset')

    return train_dataset, test_dataset
