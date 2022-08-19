'''train、test'''

import torch
import time
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler
from tool.regularization import Regularization

import random
from tool.sparsity import *
device = 'cuda:0'


# The dataset is divided into several batch_size according to the num_iterations
class BatchSampler(Sampler):
    def __init__(self, dataset, num_iterations, batch_size):
        self.dataset = dataset
        self.num_iterations = num_iterations
        self.batch_size = batch_size

    def __iter__(self):
        for _ in range(self.num_iterations):
            indices = random.sample(range(len(self.dataset)), self.batch_size)
            yield indices

    def __len__(self):
        return self.num_iterations


def train(train_dataset, network, optimizer, num_iterations, train_batch_size, penalty,
          reg_param, gamma,
          stepvalue, net_name, path=""):
    network.train()
    batch_sampler = BatchSampler(train_dataset, num_iterations, train_batch_size)  # train by iteration, not epoch
    train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=0, pin_memory=True)

    optimizer = optimizer(network.parameters())
    if gamma != 0:
        sched = torch.optim.lr_scheduler.MultiStepLR(optimizer, stepvalue, gamma=gamma, last_epoch=-1)
    else:
        sched = None
    start = time.time()
    for i, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()

        out = network(x).to(device)
        loss = F.cross_entropy(out, y)
        reg = Regularization(network).to(device)
        reg_loss = reg(network, penalty, net_name) * reg_param
        loss1 = loss + reg_loss
        loss1.backward()
        optimizer.step()
        if sched is not None:
            sched.step()
        if (i + 1) % 100 == 0:
            print(f'{i+1}th iteration\t loss:{loss:.2f}\t reg_loss:{reg_loss:.2f}')


def test(network, dataset, batch_size=100):
    network.eval()
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, pin_memory=True)

    with torch.no_grad():
        total = 0
        top1 = 0
        top5 = 0
        for i, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)
            preds = network(x)

            _, max5 = torch.topk(preds, 5, dim=-1)
            total += y.size(0)
            y = y.view(-1, 1)

            top1 += (y == max5[:, 0:1]).sum().item()
            top5 += (y == max5).sum().item()

    top1_acc = top1 / len(dataset) * 100.0
    top5_acc = top5 / len(dataset) * 100.0

    return top1_acc, top5_acc
