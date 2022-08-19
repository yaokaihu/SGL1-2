import numpy as np
import torch

dict = {"lenet": 25500, "resnet20": 267696,"resnet18": 10987200, "vgg16": 14710464, "alexnet": 3320352, "resnet50": 20676608}


def zero_out(X, thre):
    X = X.cpu().detach().numpy()
    zero_out_idx = np.nonzero(abs(X) < thre)
    X[zero_out_idx] = 0
    return torch.tensor(X)

def get_sparsity(model, net_name):
    zero = 0
    total = dict[net_name]

    for name, param in model.state_dict().items():
        if 'conv' in name and 'weight' in name:
            p = param.cpu().detach().numpy()
            zero += np.count_nonzero(p == 0)

    return 100 * (zero / total)
