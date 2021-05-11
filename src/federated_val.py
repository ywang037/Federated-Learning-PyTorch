#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, LocalUpdateVal, test_inference
from models import MLP, TwoNN, CNNMnist, CNNMnistWy, CNNFashion_Mnist, CNNCifarTorch,  CNNCifarTfDp
from utils import get_dataset, average_weights, exp_details

import time, csv
from itertools import zip_longest

if __name__ == '__main__':
     # define paths
    path_project = os.path.abspath('..')
    # logger = SummaryWriter('./logs')

    args = args_parser()
    exp_details(args)

    if args.gpu:
        torch.cuda.device(torch.cuda.current_device()) 
    device = 'cuda' if args.gpu else 'cpu'

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # BUILD MODEL
    if args.model == 'cnn':  # Convolutional neural netork
        if args.dataset == 'mnist':
            # global_model = CNNMnist(args=args)
            global_model = CNNMnist() # use WY's modification no args are needed          
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifarTorch() # use WY's edition, no args are needed
            # global_model = CNNCifar(args=args)
    elif args.model == 'wycnn':
        if args.dataset == 'mnist':
            global_model = CNNMnistWy() # use WY's cnn for learning mnist
        elif args.dataset == 'cifar':
            global_model = CNNCifarTfDp() # cnn from current TF tutorial
    elif args.model == 'mlp':  # Multi-layer preceptron
        if args.dataset == 'mnist':            
            img_size = train_dataset[0][0].shape
            len_in = 1
            for x in img_size:
                len_in *= x
                global_model = MLP(dim_in=len_in, dim_hidden=64,dim_out=args.num_classes)
        else:
            exit('Error: MLP/2NN can only be trained with MNIST dataset')
    elif args.model == 'wymlp':
        if args.dataset == 'mnist':
            global_model = TwoNN() # WY's mlp for learning mnist        
        else:
            exit('Error: MLP/2NN can only be trained with MNIST dataset')
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model,'\n')

    # print a message to confirm currently runs validation mode
    print('\nWorking on validation mode...\n')

    # pause and print message for user to confirm the hyparameter are good to go
    answer = input("Press n to abort, press any other key to continue, then press ENTER: ")
    if answer == 'n':
        exit('\nTraining is aborted by user')
    print('\nTraining starts...\n')

    # start the tensorboard writer
    logger_path_iid = f'runs_val/fedavg-{args.dataset}-IID/E{args.local_ep}-B{args.local_bs}-C{args.frac}-Lr{args.lr}-R{args.epochs}-{args.model}'
    logger_path_noniid = f'runs_val/fedavg-{args.dataset}-non-IID/E{args.local_ep}-B{args.local_bs}-C{args.frac}-Lr{args.lr}-R{args.epochs}-{args.model}'
    logger = SummaryWriter(logger_path_iid) if args.iid else SummaryWriter(logger_path_noniid)

    # copy weights
    global_weights = global_model.state_dict()

    # Validation training
    train_loss, test_loss, test_acc = [], [], []

    start_time = time.time()
    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        print('\n')
        print('\n| Global Round : {:>4}/{} | Learning rate : {:.6f}'.format(epoch+1, args.epochs, args.lr))

        # randomly pick m clients from num_users
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        # work on validation mode
        # perform per-user update, in a round-robin fashion
        global_model.train()
        for idx in idxs_users:
            # to use the entire training set for grid search of lr, toggle the following line
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
            w, loss = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch)

            # # to use the validation set, i.e., 20% training data, toggle the following line
            # local_model = LocalUpdateVal(args=args, dataset=train_dataset, idxs=user_groups[idx])
            # # local_model = LocalUpdateVal(args=args, dataset=train_dataset, idxs=user_groups[idx], logger=logger)
            # w, loss = local_model.update_weights_validate(model=copy.deepcopy(global_model), global_round=epoch)
            
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
        
        # update global weights
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(np.around(loss_avg,4))

        # decay the learning rate
        args.lr *= args.lr_decay

        # test per round
        round_test_acc, round_test_loss = test_inference(args, global_model, test_dataset)
        test_acc.append(np.around(round_test_acc,4)) 
        test_loss.append(np.around(round_test_loss,4))

        # show training performance after each round
        print('\n| Global Round : {:>4}/{} | Validation loss: {:.2f} | Test loss: {:.2f}| Test acc = {:.2f}%'.format(
            epoch+1, args.epochs, loss_avg, round_test_loss, 100*round_test_acc))

        # write training loss, test loss, and test acc to tensorboard writer
        logger.add_scalar('Train loss', loss_avg, epoch+1)
        logger.add_scalar('Test loss', round_test_loss, epoch+1)
        logger.add_scalar('Test acc', round_test_acc, epoch+1)
  
    # print the wall-clock-time used
    end_time=time.time() 
    time_elapsed = end_time-start_time
    print('\nValidation training completed, highest test acc: {:.2f}%, time elapsed: {:.2f}s ({:.2f}hrs)'.format(100*max(test_acc),time_elapsed,time_elapsed/3600))

    logger.flush()
    logger.close()

    # Saving the objects train_loss and train_accuracy:
    if args.save_record:
        # write results to csv file
        if args.save_record:
            results = [torch.arange(1,args.epochs+1).tolist(), train_loss, test_loss, test_acc]
            export_data = zip_longest(*results, fillvalue = '')
            record_path_save = f'./save-val/{args.dataset}-{args.model}/validation-fedavg-{args.dataset}-{args.model}-r{args.epochs}-le{args.local_ep}-lb{args.local_bs}-fr{args.frac}-lr{args.lr}.csv'
            with open(record_path_save, 'w', newline='') as file:
                writer = csv.writer(file,delimiter=',')
                writer.writerow(['Dataset', 'Model', 'Num of rounds', 'E', 'B', 'C', 'Lr', 'Time elapsed'])
                writer.writerow([args.dataset, args.model, args.epochs, args.local_ep, args.local_bs, args.frac, args.lr, end_time-start_time])
                writer.writerow(['Epoch', 'Training loss', 'Test loss', 'Test acc'])
                writer.writerows(export_data)
        


