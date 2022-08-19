import argparse
import os
import random
import numpy as np

import torch
import tool
import time
from tool.dataset import get_dataset
from tool.train import train, test
from tool.hyperparameter import get_hyperparameters
from tool.sparsity import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='dataset (mnist|cifar10|cifar100)')
    parser.add_argument('--network', type=str, required=True, help='network (lenet|resnet20|resnet50|vgg16')
    parser.add_argument('--penalty', type=int, required=True, help='regularization type(1(GL1/2) | 2(GL) | 3(SGL1,2) | 4(GL1,2))')
    parser.add_argument('--reg_param', type=float, default=0.002, help='regularization parameter')
    parser.add_argument('--thre', type=float, default=0.009, help='threshold ')
    args = parser.parse_args()

    # dataset
    train_dataset, test_dataset = get_dataset(args.dataset)

    prunedPath = f'./checkpoint/{args.dataset}_{args.network}/{args.penalty}/{args.reg_param}'
    os.makedirs(f'./checkpoint/{args.dataset}_{args.network}', exist_ok=True)
    os.makedirs(f'./checkpoint/{args.dataset}_{args.network}/{args.penalty}', exist_ok=True)
    os.makedirs(prunedPath, exist_ok=True)

    '''1. Pre Train'''
    print(f'{args.network}-{args.dataset}-{args.penalty}\t ʎ:{args.reg_param}\t thre:{args.thre}\t {time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())}')

    # get hyper-parameter
    network, optimizer, train_iteration, train_batch_size, test_batch_size, gamma, stepvalue = get_hyperparameters(args.network, args.dataset)
    # train
    train(train_dataset, network, optimizer, train_iteration, train_batch_size,
          args.penalty, args.reg_param, gamma, stepvalue, args.network, path=prunedPath)

    '''2. Prune'''
    # Sets the weights below the threshold to 0
    filter_num = 0
    zero_filter_num = 0
    for name, param in network.named_parameters():
        if 'conv' in name and 'weight' in name:
            filter_num += param.shape[0]
            for i in range(param.shape[0]):
                param.data[i, :, :, :] = zero_out(param.data[i, :, :, :], args.thre)  # prune
                ith_filter_L1_score = torch.sum(abs(param.data[i, :, :, :]))  # 统计移除的filter的数目
                if ith_filter_L1_score == 0: zero_filter_num += 1

    # save weight
    with open(os.path.join(prunedPath, 'Weight.txt'), "w")as f:
        for name, param in network.named_parameters():
            f.write(f"{name}{param}")
    # save model
    torch.save(network.state_dict(), os.path.join(prunedPath, 'model.pth'))

    '''3.Pruned Test'''
    # test
    top1_acc, top5_acc = test(network, test_dataset)
    # compute sparsity
    weight_sparsity = get_sparsity(network, args.network)
    filter_sparsity = 100 * zero_filter_num / filter_num

    print(f'top1-acc:{top1_acc:.2f}\t top5-acc:{top5_acc:.2f}\t weight_sparsity:{weight_sparsity:.2f}\t filter_sparsity:{filter_sparsity:.2f}')


if __name__ == '__main__':
    main()
