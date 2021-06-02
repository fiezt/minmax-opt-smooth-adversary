import seaborn
import os
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from data import gaussian_data_generator, noise_sampler
import utils
from model import Generator, Discriminator
from time import gmtime, strftime
import higher
import argparse
from tqdm import tqdm
import time


def d_loop(G, D, d_optimizer, criterion):
    
    # Zero discriminator gradient.
    d_optimizer.zero_grad()

    #  Get loss on real data component.
    d_real_data = torch.from_numpy(dset.sample(config.minibatch_size)).cuda()
    d_real_decision = D(d_real_data)
    target = torch.ones_like(d_real_decision).cuda()
    d_real_error = criterion(d_real_decision, target)  

    #  Get loss on fake data component.
    d_gen_input = torch.from_numpy(noise_sampler(config.minibatch_size, config.g_inp)).cuda()
    with torch.no_grad():
        d_fake_data = G(d_gen_input)
    d_fake_decision = D(d_fake_data)
    target = torch.zeros_like(d_fake_decision).cuda()
    d_fake_error = criterion(d_fake_decision, target)  

    # Compute loss and backpropogate.
    d_loss = d_real_error + d_fake_error
    d_loss.backward()
    d_optimizer.step()  

    return d_loss.cpu().item()


def d_unrolled_loop_higher(G, D, d_optimizer, criterion):

    #  Get loss on real data component.
    d_real_data = torch.from_numpy(dset.sample(config.minibatch_size)).cuda()
    d_real_decision = D(d_real_data)
    target = torch.ones_like(d_real_decision).cuda()
    d_real_error = criterion(d_real_decision, target)  # ones = true

    #  Get loss on fake data component.
    d_gen_input = torch.from_numpy(noise_sampler(config.minibatch_size, config.g_inp)).cuda()
    d_fake_data = G(d_gen_input)
    d_fake_decision = D(d_fake_data)
    target = torch.zeros_like(d_fake_decision).cuda()
    d_fake_error = criterion(d_fake_decision, target)  # zeros = fake

    # Compute loss and backpropogate.
    d_loss = d_real_error + d_fake_error
    d_optimizer.step(d_loss) 

    return d_loss.cpu().item()


def sgan(G, D, gen_input, disc_input, criterion):
    
    dg_real_decision = D(disc_input)
    target = torch.ones_like(dg_real_decision).cuda()
    g_real_error = -criterion(dg_real_decision, target)        

    g_fake_data = G(gen_input)
    dg_fake_decision = D(g_fake_data)
    target = torch.zeros_like(dg_fake_decision).cuda()
    g_fake_error = -criterion(dg_fake_decision, target)  
    g_error = g_real_error + g_fake_error

    return g_error

def nsgan(G, D, gen_input, disc_input, criterion):
    
    g_fake_data = G(gen_input)
    dg_fake_decision = D(g_fake_data)
    target = torch.ones_like(dg_fake_decision).cuda()
    g_error = criterion(dg_fake_decision, target)  
    
    return g_error


def g_loop(G, D, g_optimizer, d_optimizer, criterion):
    
    g_optimizer.zero_grad()
    d_optimizer.zero_grad()

    gen_input = torch.from_numpy(noise_sampler(config.minibatch_size, config.g_inp)).cuda()
    disc_input = torch.from_numpy(dset.sample(config.minibatch_size)).cuda()
    
    # This is the unrolling.
    if config.unrolled_steps > 0:
        
        # This is differentiating through the unrolling.
        if config.use_higher:
            backup = copy.deepcopy(D)

            with higher.innerloop_ctx(D, d_optimizer) as (functional_D, diff_D_optimizer):
                start = time.time()
                for i in range(config.unrolled_steps):
                    d_unrolled_loop_higher(G, functional_D, diff_D_optimizer, criterion)
                                
                if config.objective == 'sgan':
                    g_error = sgan(G, functional_D, gen_input, disc_input, criterion)
                    
                elif config.objective == 'nsgan':    
                    g_error = nsgan(G, functional_D, gen_input, disc_input, criterion)
                
                g_error.backward()
                g_optimizer.step()  

            D.load(backup)
            del backup
            
        # This is not differentiating through the unrolling.
        else:
            backup = copy.deepcopy(D)
            for i in range(config.unrolled_steps):
                d_loop(G, D, d_optimizer, criterion)

            if config.objective == 'sgan':
                g_error = sgan(G, D, gen_input, disc_input, criterion)

            elif config.objective == 'nsgan':    
                g_error = nsgan(G, D, gen_input, disc_input, criterion)
            
            g_error.backward()
            g_optimizer.step()  # Only optimizes G's parameters
            
            # This is different compared to no unrolling.
            D.load(backup)
            del backup

    else:
        if config.objective == 'sgan':
            g_error = sgan(G, D, gen_input, disc_input, criterion)

        elif config.objective == 'nsgan':    
            g_error = nsgan(G, D, gen_input, disc_input, criterion)
        
        g_error.backward()
        g_optimizer.step()  

    return g_error.cpu().item()


def g_sample(gen_input=None):
    with torch.no_grad():
        if gen_input is None:
            gen_input = torch.from_numpy(noise_sampler(2048, config.g_inp)).cuda()
        g_fake_data = G(gen_input)
        return g_fake_data.cpu().numpy()


def load_config(name):
    import importlib
    config = importlib.import_module('configs.' + name )
    return config
    
if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', type=str, default='sim1_sgan')
    argparser.add_argument('--seed', type=int, default=0)
    args = argparser.parse_args()
    
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False

    config = load_config(args.config)

    exp_dir = os.path.join('./experiments', "{}_{}".format(config.prefix, seed))
    os.makedirs(exp_dir, exist_ok=True)

    #dset = gaussian_data_generator(config.seed)
    dset = gaussian_data_generator()
    dset.uniform_distribution()
    fixed_fake_data = torch.from_numpy(noise_sampler(2048, config.g_inp)).cuda()

    sample_points = dset.sample(2048)
    utils.plot_samples([sample_points], config.log_interval, config.unrolled_steps, path='{}/samples_{}.png'.format(exp_dir, 'real'))

    def binary_cross_entropy(x, y):
        loss = -(x.log() * y + (1 - x).log() * (1 - y))
        return loss.mean()
    criterion = binary_cross_entropy

    G = Generator(input_size=config.g_inp, hidden_size=config.g_hid, output_size=config.g_out).cuda()
    D = Discriminator(input_size=config.d_inp, hidden_size=config.d_hid, output_size=config.d_out).cuda()

    if config.optimizer == "adam":
        g_optimizer = optim.Adam(G.parameters(), lr=config.g_learning_rate, betas=config.optim_betas)
        d_optimizer = optim.Adam(D.parameters(), lr=config.d_learning_rate, betas=config.optim_betas)
    elif config.optimizer == "sgd":
        g_optimizer = optim.SGD(G.parameters(), lr=config.g_learning_rate)
        d_optimizer = optim.SGD(D.parameters(), lr=config.d_learning_rate)
    elif config.optimizer == 'mix_g':
        g_optimizer = optim.Adam(G.parameters(), lr=config.g_learning_rate, betas=config.optim_betas)
        d_optimizer = optim.SGD(D.parameters(), lr=config.d_learning_rate)
    elif config.optimizer == 'mix_d':
        g_optimizer = optim.SGD(G.parameters(), lr=config.g_learning_rate)
        d_optimizer = optim.Adam(D.parameters(), lr=config.d_learning_rate, betas=config.optim_betas)

    samples = []
    g_infos = []
    start = time.time()
    for it in range(1, config.num_iterations+1):

        if it % config.log_interval == 0:
            start = time.time()

        if config.restart == True:
            D = Discriminator(input_size=config.d_inp, hidden_size=config.d_hid, output_size=config.d_out).cuda()

            if config.optimizer == "adam":
                d_optimizer = optim.Adam(D.parameters(), lr=config.d_learning_rate, betas=config.optim_betas)
            elif config.optimizer == "sgd":
                d_optimizer = optim.SGD(D.parameters(), lr=config.d_learning_rate)
            elif config.optimizer == 'mix_g':
                d_optimizer = optim.SGD(D.parameters(), lr=config.d_learning_rate)
            elif config.optimizer == 'mix_d':
                d_optimizer = optim.Adam(D.parameters(), lr=config.d_learning_rate, betas=config.optim_betas)
        else:
            # Optimize D parameters.
            d_infos = []
            for d_index in range(config.d_steps):
                d_info = d_loop(G, D, d_optimizer, criterion)
                d_infos.append(d_info)

        # Optimize G parameters.
        for g_index in range(config.g_steps):
            g_info = g_loop(G, D, g_optimizer, d_optimizer, criterion)
            g_infos.append(g_info)

        if it % config.log_interval == 0:
            if not os.path.isdir(os.path.join(exp_dir, 'model')):
                os.mkdir(os.path.join(exp_dir, 'model'))
            if not os.path.isdir(os.path.join(exp_dir, 'loss')):
                os.mkdir(os.path.join(exp_dir, 'loss'))
            torch.save(G, os.path.join(exp_dir, 'model', 'model_' + str(it)))
            np.save(os.path.join(exp_dir, 'loss', 'loss_' + str(it)), g_infos)
            g_fake_data = g_sample(fixed_fake_data)
            samples.append(g_fake_data)
            utils.plot_samples(samples, config.log_interval, config.unrolled_steps, path='{}/samples_full_{}.png'.format(exp_dir, it))
            utils.plot_samples([samples[-1]], config.log_interval, config.unrolled_steps, path='{}/samples_{}.png'.format(exp_dir, it))

    utils.plot_samples(samples, config.log_interval, config.unrolled_steps, path='{}/samples_full_{}.png'.format(exp_dir, 'final'))
    utils.plot_samples([samples[-1]], config.log_interval, config.unrolled_steps, path='{}/samples_{}.png'.format(exp_dir, 'final'))

