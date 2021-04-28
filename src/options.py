#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # general training arguments
    parser.add_argument('--bs',type=int, default=64, help='batch size')
    parser.add_argument('--epochs', type=int, default=10,
                        help="number of rounds of training")    
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--lr_decay', type=float, default=1.0, help='decaying rate of learning rate')
    parser.add_argument('--momentum', type=float, default=0.0, help='set momentum (default 0.0)')
    parser.add_argument('--nag', type=bool, default=False, help='if to use nesterov acceralted gradient')    
    parser.add_argument('--gpu', type=bool, default=False, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.") 
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer: sgd, or adam") 
    parser.add_argument('--loss',type=str,default='ce',help='Select between nll(negative log likelihood) and ce(crossentropy)')
    
    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--num_users', type=int, default=100,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=10,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10,
                        help="local batch size: B")


    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9,
                        help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to \
                        use for convolution')
    parser.add_argument('--num_channels', type=int, default=1, help="number \
                        of channels of imgs")
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for \
                        mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than \
                        strided convolutions")

    # dataset arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name \
                        of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')

    # other arguments
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    
    # results argument
    parser.add_argument('--plot', type=bool, default=False, help='If to plot learning curves')
    parser.add_argument('--save_fig', type=bool, default=False, help='If to save the figure to local repository')
    parser.add_argument('--save_model', type=bool, default=False, help='If to save the trained model weights')
    parser.add_argument('--save_record', type=bool, default=False, help='If to save the training records to csv file')
    args = parser.parse_args()
    return args
