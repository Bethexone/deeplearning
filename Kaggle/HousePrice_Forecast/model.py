# -*- coding: utf-8 -*-
# @Author  : Zhangwei
# @Time    : 2024/11/12 下午8:07
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Baseline_Net(nn.Module):
    def __init__(self, num_features, num_class=1):
        super().__init__()
        self.Linear = nn.Linear(in_features=num_features, out_features=num_class)

    def forward(self, x):
        return self.Linear(x)


class MLP_Net(nn.Module):
    def __init__(self, num_features, num_hidden, num_class=1):
        super().__init__()
        self.hidden = nn.Linear(in_features=num_features, out_features=num_hidden)
        self.Linear = nn.Linear(in_features=num_hidden, out_features=num_class)

    def forward(self, x):
        return self.Linear(F.relu(self.hidden(x)))


def loss_fn(output, target):
    output = output.reshape(target.shape)
    output = torch.clamp(output, 1, float('inf'))
    rmse = torch.sqrt(torch.mean(torch.pow((torch.log(output) - torch.log(target)), 2)))
    return rmse
    # output = np.clip(output, 1, np.inf)
    # return ((np.log(output) - np.log(target)) ** 2).mean()


def Optimizer(parameters, lr, weight_decay):
    return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
