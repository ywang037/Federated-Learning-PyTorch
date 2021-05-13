#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar':
        data_dir = './data/cifar/'
        
        # # default normalization
        # mean = (0.5, 0.5, 0.5)
        # std = (0.5, 0.5, 0.5)
        
        # # default train transform
        # train_transform = transforms.Compose(
        #     [transforms.ToTensor(),
        #      transforms.Normalize(mean, std)])

        # # default test transform
        # test_transform = transforms.Compose(
        #     [transforms.ToTensor(),
        #      transforms.Normalize(mean, std)])

        # alternative normilzation
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)

        # alternative train transform
        train_transform = transforms.Compose([
            transforms.RandomCrop(32,padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ])

        # alternative test transform
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
        ])        

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=train_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=test_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == 'mnist' or 'fmnist':
        if args.dataset == 'mnist':
            data_dir = './data/mnist/'
        else:
            data_dir = './data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def exp_details(args):
    print('\nFedAvg experiment details:')
    print(f'    Model               : {args.model}')
    print(f'    Loss function       : {args.loss}')
    print(f'    Optimizer           : {args.optimizer}')    
    print(f'    Learning rate       : {args.lr}')
    print(f'    Momentum            : {args.momentum}')
    print(f'    Nesterov accleration: {args.nag}')
    print(f'    Global Rounds       : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users   : {args.frac}')
    print(f'    Local Batch size    : {args.local_bs}')
    print(f'    Local Epochs        : {args.local_ep}\n')
    return

# the function added by WY:
def get_count_params(model):
    """
    Use this function to get number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def make_learning_curve(x):
    """
    Use this function to get monotonically incerasing learning curve
    from an input list x
    """
    running_max=0
    y=[0]*len(x)
    for idx, value in enumerate(x):
        running_max = max(x[:idx+1])
        y[idx] = running_max if x[idx] < running_max else x[idx]
    return y