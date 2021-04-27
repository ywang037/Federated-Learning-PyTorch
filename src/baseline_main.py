#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from utils import get_dataset
from options import args_parser
from update import test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar

import time


if __name__ == '__main__':
    start = time.time()
    args = args_parser()

    # print some welcome messsages to confirm the settings
    print('\nBaseline implementation')
    print(f'Dataset:\t{args.dataset}')
    if args.optimizer == 'sgd':
        print('Optimizer:\tvanilla sgd')
    elif args.optimizer == 'acc-sgd':
        print('Optimizer:\tsgd with nesterov momentum=0.9')
    elif args.optimizer == 'adam':
        print('Optimizer:\tadam')
    print(f'Learning rate:\t{args.lr}')
    print(f'Batch size:\t{args.bs}')
    print(f'Number of epochs:\t{args.epochs}')
    print('Model to train:')  
        
    if args.gpu:
        torch.cuda.device(torch.cuda.current_device())  # this line is changed by wy on 21-April-2021 
        #torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'

    # load datasets
    train_dataset, test_dataset, _ = get_dataset(args)

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)
    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model, '\n')

    # Training
    # Set optimizer
    if args.optimizer == 'sgd':
        if args.momentum:
            if args.nag:
                optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr, momentum=0.9, nesterov=True) # nesterov momentum sgd
            else:
                optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr, momentum=args.momentum) # momentum accelerated sgd
        else:
            optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr) # vanilla sgd
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(global_model.parameters(), lr=args.lr, weight_decay=1e-4) # adam

    # set loss function 
    if args.loss == 'nll':
        criterion = torch.nn.NLLLoss().to(device)
    elif args.loss == 'ce':
        criterion = torch.nn.CrossEntropyLoss().to(device)
    
    trainloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
    epoch_loss = []
    epoch_loss_test = []
    epoch_acc_test = []

    for epoch in tqdm(range(args.epochs)):
        batch_loss = []
        print('\n')

        for batch_idx, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = global_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}\t/{}\t({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1, batch_idx * len(images), len(trainloader.dataset),
                    100. * batch_idx / len(trainloader), loss.item()))
            batch_loss.append(loss.item())

        loss_avg = sum(batch_loss)/len(batch_loss)
        print(f'\nTrain loss after Epoch {epoch+1}:\t{loss_avg}')
        epoch_loss.append(loss_avg)

        test_acc, test_loss = test_inference(args, global_model, test_dataset)
        print('Test loss after Epoch{}:\t{:.2f}'.format(epoch+1,test_loss))
        print("Test accuracy after Epoch{}:\t{:.2f}%".format(epoch+1,100*test_acc))
        print(f'Test on {len(test_dataset)} samples\n')
        epoch_acc_test.append(test_acc)
        epoch_loss_test.append(test_loss)

    # Plot training loss
    fig = plt.figure()    
    plt.subplot(1,3,1)
    plt.plot(range(len(epoch_loss)), epoch_loss, label='training loss')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('Train loss')
    # plt.show()

    # Plot test loss
    # plt.figure()
    plt.subplot(1,3,2)
    plt.plot(range(len(epoch_loss_test)), epoch_loss_test, label='test loss')
    plt.plot(range(len(epoch_loss)), epoch_loss, label='training loss') # includes training loss for comparison
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('Test loss')
    # plt.show()

    # plot test accuracy
    # plt.figure()
    plt.subplot(1,3,3)
    plt.plot(range(len(epoch_acc_test)), epoch_acc_test, label='test accruacy')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('Test accuracy')
    fig.tight_layout()
    plt.show()

    # save resulted figures
    if args.savefig:
        plt.savefig(f'./save/train_loss_{args.dataset}_{args.model}_{args.optimizer}_{args.lr}_{args.epochs}_{args.bs}.png')
        plt.savefig(f'./save/test_loss_{args.dataset}_{args.model}_{args.optimizer}_{args.lr}_{args.epochs}_{args.bs}.png')
        plt.savefig(f'./save/test_acc_{args.dataset}_{args.model}_{args.optimizer}_{args.lr}_{args.epochs}_{args.bs}.png')

    '''
    # one-time testing
    test_acc, test_loss = test_inference(args, global_model, test_dataset)
    print('\nTest on', len(test_dataset), 'samples')
    print("Test Accuracy: {:.2f}%".format(100*test_acc))
    print('Test loss: {:.2f}'.format(test_loss))
    '''

    # print the wall-clock-time used
    end=time.time() 
    print('\nWall clock time elapsed: {:.2f}s'.format(end-start))