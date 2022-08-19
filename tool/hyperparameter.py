from functools import partial
from network import *
import torch.nn as nn
import torch
import random
import numpy as np
import torch.optim as optim


def get_hyperparameters(network_type, dataset):
    if network_type == 'lenet':
        network = LeNet5().cuda()
        # Xaiver initialization
        for m in network.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
        optimizer = partial(optim.Adam, lr=0.001, betas=(0.9, 0.999), weight_decay=0.0005)
        train_iteration = 10
        train_batch_size = 64
        test_batch_size = 100
        gamma = 0.9
        stepvalue = [5000, 7000, 8000, 9000, 9500]

    elif network_type == 'resnet20':
        if dataset == 'cifar10':
            network = ResNet20(num_classes=10).cuda()
        elif dataset == 'cifar100':
            network = ResNet20(num_classes=100).cuda()
        # He initialization
        for m in network.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
        optimizer = partial(optim.Adam, lr=0.001, betas=(0.9, 0.999), weight_decay=0.0001)
        train_iteration = 64000
        train_batch_size = 128
        test_batch_size = 100
        gamma = 0.1
        stepvalue = [32000, 48000]

    elif network_type == 'resnet50':
        if dataset == 'cifar10':
            network = resnet50(num_class=10).cuda()
        if dataset == 'cifar100':
            network = resnet50(num_class=100).cuda()
        # He initialization
        for m in network.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
        optimizer = partial(optim.Adam, lr=0.001, betas=(0.9, 0.999), weight_decay=0.0001)
        train_iteration = 100000
        train_batch_size = 128
        test_batch_size = 100
        gamma = 0.1
        stepvalue = [50000, 75000]

    elif network_type == 'alexnet':
        network = AlexNet().cuda()
        optimizer = partial(optim.Adam, lr=0.001, betas=(0.9, 0.999), weight_decay=0.0001)
        train_iteration = 64000
        prn_iter = 32000
        train_batch_size = 100
        test_batch_size = 100
        gamma = 0.1
        stepvalue = [32000, 48000]

    elif network_type == 'vgg16':
        if dataset == 'cifar10':
            network = vgg16(num_classes=10).cuda()
        if dataset == 'cifar100':
            network = vgg16(num_classes=100).cuda()
        optimizer = partial(optim.Adam, lr=0.001, betas=(0.9, 0.999), weight_decay=0.0001)
        train_iteration = 64
        train_batch_size = 128
        test_batch_size = 100
        gamma = 0.1
        stepvalue = [32000, 48000]
    else:
        raise ValueError('Unknown network')

    return network, optimizer, train_iteration, train_batch_size, test_batch_size, gamma, stepvalue
