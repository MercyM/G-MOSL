import math
from functools import partial
from copy import deepcopy

import torch as th
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from harl.utils.models_tools import get_active_func


class PlainMLP(nn.Module):
    """Plain MLP"""

    def __init__(self, sizes, activation_func, final_activation_func="identity"):
        super().__init__()
        layers = []
        for j in range(len(sizes) - 1):
            act = activation_func if j < len(sizes) - 2 else final_activation_func
            layers += [nn.Linear(sizes[j], sizes[j + 1]), get_active_func(act)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class compare_STE(th.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return F.relu(th.sign(input))

    @staticmethod
    def backward(ctx, grad_output):
        return th.tanh_(grad_output)


