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
from update import LocalUpdateVal, test_inference
from models import MLP, TwoNN, CNNMnist, CNNMnistWy, CNNFashion_Mnist, CNNCifar, CNNCifarTorch
from utils import get_dataset, average_weights, exp_details

import time, csv
from itertools import zip_longest

if __name__ == '__main__':
     # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('./logs')

    args = args_parser()
    exp_details(args)

    if args.gpu:
        torch.cuda.device(torch.cuda.current_device()) 
    device = 'cuda' if args.gpu else 'cpu'

    # if args.gpu_id:
    #     torch.cuda.set_device(args.gpu_id)
    # device = 'cuda' if args.gpu else 'cpu'

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # BUILD MODEL
    if args.model == 'cnn':  # Convolutional neural netork
        if args.dataset == 'mnist':
            # global_model = CNNMnist(args=args)
            # global_model = CNNMnist() # use WY's edition, no args are needed
            global_model = CNNMnistWy() # use WY's cnn for learning mnist            
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifarTorch() # use WY's edition, no args are needed
            # global_model = CNNCifar(args=args)
    elif args.model == 'mlp':  # Multi-layer preceptron
        if args.dataset == 'mnist':
            global_model = TwoNN() # WY's mlp for learning mnist        
            # img_size = train_dataset[0][0].shape
            # len_in = 1
            # for x in img_size:
            #     len_in *= x
            #     global_model = MLP(dim_in=len_in, dim_hidden=64,dim_out=args.num_classes)
        else:
            exit('Error: MLP/2NN can only be trained with MNIST dataset')
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model,'\n')

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

        # perform per-user update, in a round-robin fashion
        global_model.train()
        for idx in idxs_users:
            local_model = LocalUpdateVal(args=args, dataset=train_dataset, idxs=user_groups[idx], logger=logger)

            # work on validation mode
            w, loss = local_model.update_weights_validate(model=copy.deepcopy(global_model), global_round=epoch)
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


    logger.flush()
    logger.close()
    
    # print the wall-clock-time used
    end_time=time.time() 
    print('\nValidation training completed, time elapsed: {:.2f}s'.format(end_time-start_time))

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
        
        # # write to pkl file
        # file_name = f'./save/objects/{args.dataset}_{args.model}_{args.epochs}_C[{args.frac}]_iid[{args.iid}]_E[{args.local_ep}]_B[{args.local_bs}].pkl'
        # with open(file_name, 'wb') as f:
        #     pickle.dump([train_loss, test_loss, test_acc], f)
        #     # pickle.dump([train_loss, train_accuracy], f)

    # # visualize the training results
    # if args.plot:
    #     matplotlib.use('Agg')
    #     # Plot Loss curve
    #     plt.figure()
    #     plt.title('Training Loss vs Communication rounds')
    #     plt.plot(range(len(train_loss)), train_loss, color='r')
    #     plt.ylabel('Training loss')
    #     plt.xlabel('Communication Rounds')
    #     plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
    #                 format(args.dataset, args.model, args.epochs, args.frac,
    #                     args.iid, args.local_ep, args.local_bs))
        
    #     # Plot Average Accuracy vs Communication rounds
    #     plt.figure()
    #     plt.title('Average Accuracy vs Communication rounds')
    #     plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    #     plt.ylabel('Average Accuracy')
    #     plt.xlabel('Communication Rounds')
    #     plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
    #                 format(args.dataset, args.model, args.epochs, args.frac,
    #                     args.iid, args.local_ep, args.local_bs))

    # PLOTTING (optional)
    # import matplotlib
    # import matplotlib.pyplot as plt

