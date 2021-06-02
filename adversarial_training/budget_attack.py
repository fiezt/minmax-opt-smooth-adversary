from experiments import test
from experiments import load_config
from dataprocess import loadData
from adversary import budgetgdAttackTest
from experiments import load_config
import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

sims = ['minmax_10_1_normal_sgd_100', 'minmax_10_1_no_higher_sgd_100', 'minmax_10_1_higher_sgd_100']

for sim in sims:
    for seed in range(1, 6):
        argparser = argparse.ArgumentParser()
        argparser.add_argument('--config', type=str, default='minmax_10_1_normal_sgd_100')
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
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        loader_train, loader_test = loadData(args)
        dtype = torch.cuda.FloatTensor
        step_size = [1,2,8, 20]
        num_steps = [40, 20, 5, 2]
        folder = os.path.join(os.getcwd(), 'results', sim+'_seed_'+str(seed))

        print(folder)
        for epoch in range(25, 105, 5):
            fi = os.path.join(folder, 'model', 'model_'+str(epoch))
            model = torch.load(fi)
            accuracies = budgetgdAttackTest(model, loader_test, dtype, step_size, num_steps)
            save_results = {'gd': (step_size, num_steps, accuracies)}
            if not os.path.exists(os.path.join(folder, 'attack_results')):
                os.makedirs(os.path.join(folder, 'attack_results'))
            save_path = os.path.join(folder, 'attack_results', 'budget_attack_results_'+str(epoch))
            print('saving to', save_path)
            pickle.dump(save_results, open(save_path, "wb"))