from dataprocess import loadData
from experiments import load_config
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from model import LinearNet
import higher
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import sys

def minmax_higher_grad_eval(model, args, loader_train, dtype, num_inner_steps):

    model.train()
        
    loss_function = nn.CrossEntropyLoss()    
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate_outer)
    
    norms = []
    for i, (X_, y_) in enumerate(loader_train):

        X = Variable(X_.type(dtype), requires_grad=True)
        X_original = Variable(X_.type(dtype), requires_grad=False)
        y = Variable(y_.type(dtype), requires_grad=False).long()

        max_player = LinearNet(X)
        max_optimizer = optim.SGD(max_player.parameters(), lr=args.learning_rate_inner)
        max_optimizer.zero_grad()

        inner_loop_loss_history = []
        with higher.innerloop_ctx(max_player, max_optimizer) as (max_player_functional, higher_max_optimizer):
            for j in range(num_inner_steps):
                max_player_loss = -loss_function(model(max_player_functional.params), y) 
                inner_loop_loss_history.append(-max_player_loss.item())
                returned = higher_max_optimizer.step(max_player_loss)

            optimizer.zero_grad()
            loss = loss_function(model(max_player_functional.params), y) 
            loss.backward()
            grad_norm = 0
            for p in model.parameters():
                param_norm = p.grad.data.norm(2)
                grad_norm += param_norm.item() ** 2
            grad_norm = grad_norm**(1./2)
            norms.append(grad_norm)
            
        if i == 100:
            break
    return norms

sim = 'minmax_10_1_higher_sgd_100'

if len(sys.argv) == 1:
    pass
else:
    learning_rate_inner = sys.argv[1]

for seed in range(1, 6):
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', type=str, default='minmax_10_1_higher_sgd_100')
    argparser.add_argument('--seed', type=int, default=0)
    argparser.add_argument('--gpu', type=int, default=0)
    if seed <= 3: 
        gpu = 0
    else:
        gpu = 1
    args = argparser.parse_args(['--config', sim, '--seed', str(seed), '--gpu', str(gpu)])
    seed = args.seed
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    args = load_config(args.config)  
    if len(sys.argv) == 1:
        learning_rate_inner = args.learning_rate_inner
    else:
        learning_rate_inner = float(sys.argv[1])
    args.learning_rate_inner = learning_rate_inner
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    loader_train, loader_test = loadData(args)
    dtype = torch.cuda.FloatTensor
    num_inner_steps = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    folder = os.path.join(os.getcwd(), 'results', sim+'_seed_'+str(seed))

    for epoch in range(25, 125, 25):
        fi = os.path.join(folder, 'model', 'model_'+str(epoch))
        model = torch.load(fi)
        results_list = []
        for steps in num_inner_steps:
            norms = minmax_higher_grad_eval(model, args, loader_train, dtype, steps)
            results_list.append(norms)        
        if not os.path.exists(os.path.join(folder, 'grad_norm_results')):
            os.makedirs(os.path.join(folder, 'grad_norm_results'))
        save_path = os.path.join(folder, 'grad_norm_results', 'grad_norm_results_'+str(learning_rate_inner)+'_'+str(epoch))
        print('saving to', save_path)
        pickle.dump(results_list, open(save_path, "wb"))