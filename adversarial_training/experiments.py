import importlib
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import time
import higher
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as T
from model import ConvNet
from model import LinearNet
import numpy as np
from lossfns import cw_train_unrolled
from adversary import pgdAttackTest, fgsmAttackTest, gdAttackTest
from dataprocess import loadData
import os
import pickle


def main(args):
    
    loader_train, loader_test = loadData(args)
    dtype = torch.cuda.FloatTensor
    
    if args.alg == 'normal':
        model = normal_train(args, loader_train, loader_test, dtype)
    if args.alg == 'concave':
        model = unrolled(args, loader_train, loader_test, dtype)
    if args.alg == 'pgd':
        model = pgd_train(args, loader_train, loader_test, dtype)
    if args.alg == 'minmax':
        model = minmax_train(args, loader_train, loader_test, dtype)
    if args.alg == 'minmax_higher':
        model = minmax_higher_train(args, loader_train, loader_test, dtype)
        
    print("Training done!")
    
    
def normal_train(args, loader_train, loader_test, dtype):

    model = ConvNet()
    model = model.type(dtype)
    model.train()
        
    loss_function = nn.CrossEntropyLoss()    
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate_outer)
    
    outer_loss_history = []

    for epoch in range(args.num_epochs):

        for i, (X_, y_) in enumerate(loader_train):

            X = Variable(X_.type(dtype), requires_grad=False)
            y = Variable(y_.type(dtype), requires_grad=False).long()

            loss = loss_function(model(X), y)
            outer_loss_history.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if (epoch + 1) % args.loss_every == 0:
            print('\nTraining epoch %d / %d ... Loss = %.3f\n' % (epoch + 1, args.num_epochs, loss.item()))
            if not os.path.exists(os.path.join(args.folder, 'loss_results')):
                os.makedirs(os.path.join(args.folder, 'loss_results'))
            np.save(os.path.join(args.folder, 'loss_results', 'outer_loss_results_'+str(epoch+1)), outer_loss_history)

        if (epoch + 1) % args.attack_every == 0:
            gd_eps_list, gd_attack_acc = gdAttackTest(args, model, loader_test, dtype)
            pgd_eps_list, pgd_attack_acc = pgdAttackTest(model, loader_test, dtype)
            fgsm_eps_list, fgsm_attack_acc = fgsmAttackTest(model, loader_test, dtype) 
            save_results = {'gd': (gd_eps_list, gd_attack_acc), 'pgd': (pgd_eps_list, pgd_attack_acc), 
                            'fgsm': (fgsm_eps_list, fgsm_attack_acc)}
            if not os.path.exists(os.path.join(args.folder, 'attack_results')):
                os.makedirs(os.path.join(args.folder, 'attack_results'))
            pickle.dump(save_results, open(os.path.join(args.folder, 'attack_results', 'attack_results_'+str(epoch+1)), "wb"))

        if (epoch + 1) % args.save_every == 0:
            if not os.path.exists(os.path.join(args.folder, 'model')):
                os.makedirs(os.path.join(args.folder, 'model'))

            torch.save(model, os.path.join(args.folder, 'model', 'model_' + str(epoch+1)))

    return model


def unrolled(args, loader_train, loader_test, dtype):

    model = ConvNet()
    model = model.type(dtype)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate_outer)
    
    outer_loss_history = []
    
    for epoch in range(args.num_epochs):
                
        for i, (X_, y_) in enumerate(loader_train):

            X = Variable(X_.type(dtype), requires_grad=False)
            y = Variable(y_.type(dtype), requires_grad=False)
            
            loss = cw_train_unrolled(model, X, y, dtype)
            outer_loss_history.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % args.loss_every == 0:
            print('\nTraining epoch %d / %d ... Loss = %d\n' % (epoch + 1, args.num_epochs, loss.item()))
            if not os.path.exists(os.path.join(args.folder, 'loss_results')):
                os.makedirs(os.path.join(args.folder, 'loss_results'))
            np.save(os.path.join(args.folder, 'loss_results', 'outer_loss_results_'+str(epoch+1)), outer_loss_history)

        if (epoch + 1) % args.attack_every == 0:
            gd_eps_list, gd_attack_acc = gdAttackTest(model, loader_test, dtype)
            pgd_eps_list, pgd_attack_acc = pgdAttackTest(model, loader_test, dtype)
            fgsm_eps_list, fgsm_attack_acc = fgsmAttackTest(model, loader_test, dtype) 
            save_results = {'gd': (gd_eps_list, gd_attack_acc), 'pgd': (pgd_eps_list, pgd_attack_acc), 
                            'fgsm': (fgsm_eps_list, fgsm_attack_acc)}
            if not os.path.exists(os.path.join(args.folder, 'attack_results')):
                os.makedirs(os.path.join(args.folder, 'attack_results'))
            pickle.dump(save_results, open(os.path.join(args.folder, 'attack_results', 'attack_results_'+str(epoch+1)), "wb"))

        if (epoch + 1) % args.save_every == 0:
            if not os.path.exists(os.path.join(args.folder, 'model')):
                os.makedirs(os.path.join(args.folder, 'model'))

            torch.save(model, os.path.join(args.folder, 'model', 'model_' + str(epoch+1)))

    return model


def pgd_train(args, loader_train, loader_test, dtype):

    model = ConvNet()
    model = model.type(dtype)
    model.train()
        
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate_outer)
    
    outer_loss_history = []
    inner_loss_history = []
    perturbation_history = []

    for epoch in range(args.num_epochs):

        for i, (X_, y_) in enumerate(loader_train):

            X = Variable(X_.type(dtype), requires_grad=True)
            X_original = Variable(X_.type(dtype), requires_grad=False)
            y = Variable(y_.type(dtype), requires_grad=False).long()

            inner_loop_loss_history = []
            for j in range(args.num_inner_steps):
                loss = loss_function(model(X), y)
                inner_loop_loss_history.append(loss.item())
                loss.backward()
                
                with torch.no_grad():
                    X.data = X.data + args.learning_rate_inner * X.grad.sign()
                    X.data = X_original + (X.data - X_original).clamp(min=-args.eps, max=args.eps)
                    X.data = X.data.clamp(min=0, max=1)
                    X.grad.zero_()
                    
            X.requires_grad = False
            inner_loss_history.append(inner_loop_loss_history)
            perturbation_history.append(torch.norm(X-X_original, float('inf')).item())

            loss = loss_function(model(X), y)
            outer_loss_history.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % args.loss_every == 0:
            print('\nTraining epoch %d / %d ... Loss = %.3f\n' % (epoch + 1, args.num_epochs, loss.item()))
            if not os.path.exists(os.path.join(args.folder, 'loss_results')):
                os.makedirs(os.path.join(args.folder, 'loss_results'))
            np.save(os.path.join(args.folder, 'loss_results', 'outer_loss_results_'+str(epoch+1)), outer_loss_history)
            np.save(os.path.join(args.folder, 'loss_results', 'inner_loss_results_'+str(epoch+1)), inner_loss_history)
            np.save(os.path.join(args.folder, 'loss_results', 'norm_results_'+str(epoch+1)), perturbation_history)

        if (epoch + 1) % args.attack_every == 0:
            gd_eps_list, gd_attack_acc = gdAttackTest(model, loader_test, dtype)
            pgd_eps_list, pgd_attack_acc = pgdAttackTest(model, loader_test, dtype)
            fgsm_eps_list, fgsm_attack_acc = fgsmAttackTest(model, loader_test, dtype) 
            save_results = {'gd': (gd_eps_list, gd_attack_acc), 'pgd': (pgd_eps_list, pgd_attack_acc), 
                            'fgsm': (fgsm_eps_list, fgsm_attack_acc)}
            if not os.path.exists(os.path.join(args.folder, 'attack_results')):
                os.makedirs(os.path.join(args.folder, 'attack_results'))
            pickle.dump(save_results, open(os.path.join(args.folder, 'attack_results', 'attack_results_'+str(epoch+1)), "wb"))

        if (epoch + 1) % args.save_every == 0:
            if not os.path.exists(os.path.join(args.folder, 'model')):
                os.makedirs(os.path.join(args.folder, 'model'))
            torch.save(model, os.path.join(args.folder, 'model', 'model_' + str(epoch+1)))

    return model


def minmax_train(args, loader_train, loader_test, dtype):

    model = ConvNet()
    model = model.type(dtype)
    model.train()
        
    loss_function = nn.CrossEntropyLoss()    
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate_outer)
    
    outer_loss_history = []
    inner_loss_history = []
    perturbation_history = []

    for epoch in range(args.num_epochs):

        for i, (X_, y_) in enumerate(loader_train):

            X = Variable(X_.type(dtype), requires_grad=True)
            X_original = Variable(X_.type(dtype), requires_grad=False)
            y = Variable(y_.type(dtype), requires_grad=False).long()

            inner_loop_loss_history = []
            for j in range(args.num_inner_steps):
                loss = loss_function(model(X), y) 
                loss.backward()
                inner_loop_loss_history.append(loss.item())
                
                with torch.no_grad():
                    X.data = X.data + args.learning_rate_inner*X.grad
                    X.grad.zero_()

            X.requires_grad = False
            perturbation_history.append(torch.norm(X-X_original, float('inf')).item())
            
            inner_loss_history.append(inner_loop_loss_history)

            loss = loss_function(model(X), y)
            outer_loss_history.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if (epoch + 1) % args.loss_every == 0:
            print('\nTraining epoch %d / %d ... Loss = %.3f\n' % (epoch + 1, args.num_epochs, loss.item()))
            if not os.path.exists(os.path.join(args.folder, 'loss_results')):
                os.makedirs(os.path.join(args.folder, 'loss_results'))
            np.save(os.path.join(args.folder, 'loss_results', 'outer_loss_results_'+str(epoch+1)), outer_loss_history)
            np.save(os.path.join(args.folder, 'loss_results', 'inner_loss_results_'+str(epoch+1)), inner_loss_history)
            np.save(os.path.join(args.folder, 'loss_results', 'norm_results_'+str(epoch+1)), perturbation_history)

        if (epoch + 1) % args.attack_every == 0:
            gd_eps_list, gd_attack_acc = gdAttackTest(args, model, loader_test, dtype)
            pgd_eps_list, pgd_attack_acc = pgdAttackTest(model, loader_test, dtype)
            fgsm_eps_list, fgsm_attack_acc = fgsmAttackTest(model, loader_test, dtype) 
            save_results = {'gd': (gd_eps_list, gd_attack_acc), 'pgd': (pgd_eps_list, pgd_attack_acc), 
                            'fgsm': (fgsm_eps_list, fgsm_attack_acc)}
            if not os.path.exists(os.path.join(args.folder, 'attack_results')):
                os.makedirs(os.path.join(args.folder, 'attack_results'))
            pickle.dump(save_results, open(os.path.join(args.folder, 'attack_results', 'attack_results_'+str(epoch+1)), "wb"))

        if (epoch + 1) % args.save_every == 0:
            if not os.path.exists(os.path.join(args.folder, 'model')):
                os.makedirs(os.path.join(args.folder, 'model'))

            torch.save(model, os.path.join(args.folder, 'model', 'model_' + str(epoch+1)))

    return model


def minmax_higher_train(args, loader_train, loader_test, dtype):

    model = ConvNet()
    model = model.type(dtype)
    model.train()
        
    loss_function = nn.CrossEntropyLoss()    
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate_outer)
    
    outer_loss_history = []
    inner_loss_history = []
    perturbation_history = []

    for epoch in range(args.num_epochs):

        for i, (X_, y_) in enumerate(loader_train):

            X = Variable(X_.type(dtype), requires_grad=True)
            X_original = Variable(X_.type(dtype), requires_grad=False)
            y = Variable(y_.type(dtype), requires_grad=False).long()
            
            max_player = LinearNet(X)
            max_optimizer = optim.SGD(max_player.parameters(), lr=args.learning_rate_inner)
            max_optimizer.zero_grad()

            inner_loop_loss_history = []
            with higher.innerloop_ctx(max_player, max_optimizer) as (max_player_functional, higher_max_optimizer):
                for j in range(args.num_inner_steps):
                    max_player_loss = -loss_function(model(max_player_functional.params), y) 
                    inner_loop_loss_history.append(-max_player_loss.item())
                    returned = higher_max_optimizer.step(max_player_loss)
                    
                inner_loss_history.append(inner_loop_loss_history)
                perturbation_history.append(torch.norm(max_player_functional.params-X_original, float('inf')).item())

                optimizer.zero_grad()
                loss = loss_function(model(max_player_functional.params), y) 
                outer_loss_history.append(loss.item())
                loss.backward()
                optimizer.step()
            
        if (epoch + 1) % args.loss_every == 0:
            print('\nTraining epoch %d / %d ... Loss = %.3f\n' % (epoch + 1, args.num_epochs, loss.item()))
            if not os.path.exists(os.path.join(args.folder, 'loss_results')):
                os.makedirs(os.path.join(args.folder, 'loss_results'))
            np.save(os.path.join(args.folder, 'loss_results', 'outer_loss_results_'+str(epoch+1)), outer_loss_history)
            np.save(os.path.join(args.folder, 'loss_results', 'inner_loss_results_'+str(epoch+1)), inner_loss_history)
            np.save(os.path.join(args.folder, 'loss_results', 'norm_results_'+str(epoch+1)), perturbation_history)

        if (epoch + 1) % args.attack_every == 0:
            gd_eps_list, gd_attack_acc = gdAttackTest(args, model, loader_test, dtype)
            pgd_eps_list, pgd_attack_acc = pgdAttackTest(model, loader_test, dtype)
            fgsm_eps_list, fgsm_attack_acc = fgsmAttackTest(model, loader_test, dtype) 
            save_results = {'gd': (gd_eps_list, gd_attack_acc), 'pgd': (pgd_eps_list, pgd_attack_acc), 
                            'fgsm': (fgsm_eps_list, fgsm_attack_acc)}
            if not os.path.exists(os.path.join(args.folder, 'attack_results')):
                os.makedirs(os.path.join(args.folder, 'attack_results'))
            pickle.dump(save_results, open(os.path.join(args.folder, 'attack_results', 'attack_results_'+str(epoch+1)), "wb"))

        if (epoch + 1) % args.save_every == 0:
            if not os.path.exists(os.path.join(args.folder, 'model')):
                os.makedirs(os.path.join(args.folder, 'model'))

            torch.save(model, os.path.join(args.folder, 'model', 'model_' + str(epoch+1)))

    return model


def test(model, loader_test, dtype):
    num_correct = 0
    num_samples = 0
    model.eval()
    for X_, y_ in loader_test:

        X = Variable(X_.type(dtype), requires_grad=False)
        y = Variable(y_.type(dtype), requires_grad=False).long()

        logits = model(X)
        _, preds = logits.max(1)

        num_correct += (preds == y).sum()
        num_samples += preds.size(0)

    accuracy = float(num_correct)/num_samples * 100
    print('\nAccuracy = %.2f%%' % accuracy)
    model.train()

    
def load_config(name):
    config = importlib.import_module('configs.' + name)
    return config

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', type=str, default='pgd')
    argparser.add_argument('--seed', type=int, default=0)
    argparser.add_argument('--gpu', type=int, default=0)
    args = argparser.parse_args()
    seed = args.seed
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    args = load_config(args.config)
    args.folder = args.folder+ '_seed_'+str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    main(args)


