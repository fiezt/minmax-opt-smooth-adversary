from __future__ import print_function
from __future__ import division
from builtins import range
from builtins import int
from builtins import dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torch.backends.cudnn as cudnn

import torch.nn.functional as F

import torchvision.datasets as dset
import torchvision.transforms as T

import numpy as np


def cw_train_unrolled(model, X, y, dtype):

    N = X.shape[0]
    X = X.repeat(1, 10, 1, 1).reshape(N * 10, 1, 28, 28)
    X_copy = X.clone()
    X.requires_grad = True
    

    eps = 0.4
    
    y = y.view(-1, 1).repeat(1, 10).view(-1, 1).long().cuda()

    index = torch.tensor([jj for jj in range(10)] * N).view(-1, 1).cuda().long()

    MaxIter_max = 11
    step_size_max = 0.1

    for i in range(MaxIter_max):

        output = model(X)
        
        maxLoss = (output.gather(1, index) - output.gather(1, y)).mean()

        X_grad = torch.autograd.grad(maxLoss, X, retain_graph=True)[0]
        X = X + X_grad.sign() * step_size_max
        
        X.data = X_copy.data + (X.data - X_copy.data).clamp(-eps, eps)
        X.data = X.data.clamp(0, 1)

    preds = model(X)

    # loss = (-F.log_softmax(preds)).gather(1, y).view(-1, 10).max(dim=1)[0].mean()

    loss = (-F.log_softmax(preds)).gather(1, y.view(-1, 1)).view(-1, 10).max(dim=1)[0].mean()

    return loss