#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import get_dataset, get_count_params
from options import args_parser
from update import test_inference
from models import MLP, TwoNN, CNNMnist, CNNMnistWy, CNNFashion_Mnist, CNNCifarTorch, CNNCifarTfDp

import time, csv
from itertools import zip_longest

if __name__ == '__main__':
    args = args_parser() 
    torch.manual_seed(args.seed)    
        
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
            global_model = CNNMnist() # use WY's modification, no args are needed
            # global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            # global_model = CNNCifarTf() # cnn borrowed from current TF tutorial
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
    print(global_model, '\n')

    # Training
    # Set optimizer
    if args.optimizer == 'sgd':
        if args.nag:
            optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True) # nesterov momentum sgd
        else:
            optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr, momentum=args.momentum) # vanilla sgd or momentum accelerated sgd
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(global_model.parameters(), lr=args.lr, weight_decay=1e-4) # adam

    # set loss function 
    if args.loss == 'nll':
        criterion = torch.nn.NLLLoss().to(device)
    elif args.loss == 'ce':
        criterion = torch.nn.CrossEntropyLoss().to(device)
    
   # print some welcome messsages to confirm the setup
    print('\nBaseline experiment settings:')
    print('{:<18}: {}'.format('Dataset',args.dataset))
    print('{:<18}: {}'.format('Loss',args.loss))
    if args.optimizer == 'sgd':
        if args.momentum:
            if args.nag:
                print('{:<18}: sgd with nesterov momentum={}'.format('Optimizer',args.momentum)) # nesterov accelerated sgd
            else:
                print('{:<18}: sgd with momentum={}'.format('Optimizer',args.momentum)) # sgd with momentum                
        else:
            print('{:<18}: vanilla sgd'.format('Optimizer')) # vanilla sgd
    elif args.optimizer == 'adam':
        print('{:<18}: adam'.format('Optimizer')) # adam
    print('{:<18}: {}'.format('Learning rate',args.lr))
    print('{:<18}: {}'.format('Batch size',args.bs))
    print('{:<18}: {}'.format('Number of epochs',args.epochs))
    print('{:<18}: {}-{}'.format('Model to train',args.dataset,args.model))  
    print('{:<18}: {}'.format('Parameter amount',get_count_params(global_model)))

    # pause and print message for user to confirm the hyparameter are good to go
    answer = input("Press n to abort, press any other key to continue, then press ENTER: ")
    if answer == 'n':
        exit('\nTraining is aborted by user')
    print('\nTraining starts...\n')

    # start the tensorboard writer
    logger_path = f'runs/fedavg-{args.dataset}-SGD-tf/B{args.bs}-Lr{args.lr}-R{args.epochs}-{args.model}'
    logger = SummaryWriter(logger_path)

    # start training
    trainloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
    epoch_loss = []
    epoch_loss_test = []
    epoch_acc_test = []

    start = time.time()
    for epoch in tqdm(range(args.epochs)):
        # set a learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=args.lr_decay, verbose=False)
        
        batch_loss = []
        print('\n')

        for batch_idx, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = global_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if args.verbose and (batch_idx % 100 == 0):
                print('Train Epoch: {} [{:>5}/{} ({:>2.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1, batch_idx * len(images), len(trainloader.dataset),
                    100. * batch_idx / len(trainloader), loss.item()))
            batch_loss.append(loss.item())

        loss_avg = sum(batch_loss)/len(batch_loss)
        epoch_loss.append(loss_avg)
        
        test_acc, test_loss = test_inference(args, global_model, test_dataset)
        epoch_acc_test.append(test_acc)
        epoch_loss_test.append(test_loss)

        # update the learning rate
        scheduler.step()

        print('\nEpoch: {:>4}/{} | Learning rate: {:.6f}| Training Loss: {:.2f} | Test Loss: {:.2f} | Test accuracy = {:.2f}%'.format(
            epoch+1, args.epochs, args.lr, loss_avg, test_loss, 100*test_acc))
        # print(f'\nTrain loss after Epoch {epoch+1}:\t{loss_avg}')       
        # print('Test loss after Epoch{}:\t{:.2f}'.format(epoch+1,test_loss))
        # print("Test accuracy after Epoch{}:\t{:.2f}%".format(epoch+1,100*test_acc))
        print(f'Tested on {len(test_dataset)} samples\n')
    
        # write training loss, test loss, and test acc to tensorboard writer
        logger.add_scalar('Train loss', loss_avg, epoch+1)
        logger.add_scalar('Test loss', test_loss, epoch+1)
        logger.add_scalar('Test acc', test_acc, epoch+1)

    # print the wall-clock-time used
    end=time.time() 
    time_elapsed = end-start
    # print('\nTraining completed, time elapsed: {:.2f}s'.format(end-start))
    print('\nTraining completed, highest test acc: {:.2f}%, time elapsed: {:.2f}s ({:.2f}hrs)'.format(100*max(epoch_acc_test),time_elapsed,time_elapsed/3600))

    # flush the event and close the tensoarboard writer
    logger.flush()
    logger.close()

    # write results to csv file
    if args.save_record:
        results = [torch.arange(1,args.epochs+1).tolist(), epoch_loss, epoch_loss_test, epoch_acc_test]
        export_data = zip_longest(*results, fillvalue = '')
        with open('./save/results-baseline.csv', 'w', newline='') as file:
            writer = csv.writer(file,delimiter=',')
            writer.writerow(['Epoch', 'training loss', 'test loss', 'test acc'])
            writer.writerows(export_data)
        
    # visualize the training results
    if args.plot:
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
    if args.save_fig:
        plt.savefig(f'./save/train_loss_{args.dataset}_{args.model}_ep{args.epochs}_bs{args.bs}_lr{args.lr}.png')
        plt.savefig(f'./save/test_loss_{args.dataset}_{args.model}_ep{args.epochs}_bs{args.bs}_lr{args.lr}.png')
        plt.savefig(f'./save/test_acc_{args.dataset}_{args.model}_ep{args.epochs}_bs{args.bs}_lr{args.lr}.png')

    # save trained weights
    if args.save_model:
        save_path = f'./save/weights-baseline-{args.dataset}-{args.model}-ep{args.epochs}-bs{args.bs}-lr{args.lr}.pth'
        torch.save(global_model.state_dict(), save_path)

    '''
    # one-time testing
    test_acc, test_loss = test_inference(args, global_model, test_dataset)
    print('\nTest on', len(test_dataset), 'samples')
    print("Test Accuracy: {:.2f}%".format(100*test_acc))
    print('Test loss: {:.2f}'.format(test_loss))
    '''


