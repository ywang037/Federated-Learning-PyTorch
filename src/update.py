#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        self.device = 'cuda' if args.gpu else 'cpu'        
        self.criterion = nn.CrossEntropyLoss().to(self.device) if args.loss == 'ce' else nn.NLLLoss().to(self.device)
        self.trainloader = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        # self.criterion = nn.CrossEntropyLoss().to(self.device)
        # # Default criterion set to NLL loss function (WY: WHY???)
        # self.criterion = nn.NLLLoss().to(self.device)
        # self.trainloader, self.validloader, self.testloader = self.train_val_test(dataset, list(idxs))    

    '''
    # WY's comment: 
    # Below is the original function which splits the entire dataset into training, validation, and test.        
    # However, there's no point to use the following method for MNIST and CIFIAR, 
    # because dedicated test dataset is available,
    # so that one does not need to manually split train, val, test dataset
    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=int(len(idxs_val)/10), shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=int(len(idxs_test)/10), shuffle=False)
        return trainloader, validloader, testloader
    '''
    def update_weights(self, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Below is WY'es edition for optimizer setup
        if self.args.optimizer == 'sgd':
            if self.args.nag:
                # nesterov momentum sgd
                optimizer = torch.optim.SGD(model.parameters(), 
                                            lr=self.args.lr, 
                                            momentum=self.args.momentum, 
                                            nesterov=True) 
            else:
                # vanilla or momentum accelerated sgd
                optimizer = torch.optim.SGD(model.parameters(), 
                                            lr=self.args.lr, 
                                            momentum=self.args.momentum) 
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=1e-4) # adam
        
        '''
        # Below is the original code section
        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)
        '''

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:>2.0f}%)]\tLoss: {:.6f}'.format(
                        global_round+1, 
                        iter+1, 
                        batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    '''
    # WY's comment: 
    # There's no point to return the loss using the following method for MNIST and CIFIAR, 
    # because the loss below is computed from a manually splitted test dataset, 
    # which is used for the case where no dedicated test dataest is available
    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """
        model.eval()
        total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()*len(labels) # corrected by WY

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss/total
    '''
class LocalUpdateVal(object):
    '''
    This class is duplicated from the class LocalUpdate defined above, 
    the only difference is that this class is used for validation training only
    '''
    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        self.device = 'cuda' if args.gpu else 'cpu'        
        self.criterion = nn.CrossEntropyLoss().to(self.device) if args.loss == 'ce' else nn.NLLLoss().to(self.device)
        _, self.validloader = self.train_val(dataset, list(idxs))
        # self.trainloader, self.validloader = self.train_val(dataset, list(idxs))    
    
    def train_val(self, dataset, idxs):
        """
        This function is only used to build a validation dataset for doing grid searches of learning rate.
        Returns train, validation and test dataloaders for a given dataset and user indexes.
        """
        # split indexes for train, validation (80, 20)
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=int(len(idxs_val)/5), shuffle=False)
        return trainloader, validloader
    
    def update_weights_validate(self, model, global_round):
        '''
        This function is used only for doing grid searches of learning rate.
        This function performs the same training procedure as the function update_weights, 
        but on the validation dataset using validloader instead of trainloader
        '''
        # print a message to confirm currently runs validation mode
        print(f'Working on validation dataset of size {len(self.validloader.dataset)}')

        # Set mode to train model
        model.train()
        epoch_loss = []

        # Below is WY'es edition for optimizer setup
        if self.args.optimizer == 'sgd':
            if self.args.nag:
                # nesterov momentum sgd
                optimizer = torch.optim.SGD(model.parameters(), 
                                            lr=self.args.lr, 
                                            momentum=self.args.momentum, 
                                            nesterov=True) 
            else:
                # vanilla or momentum accelerated sgd
                optimizer = torch.optim.SGD(model.parameters(), 
                                            lr=self.args.lr, 
                                            momentum=self.args.momentum) 
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=1e-4) # adam

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.validloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:>2.0f}%)]\tLoss: {:.6f}'.format(
                        global_round+1, 
                        iter+1, 
                        batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                # self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)    

def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    device = 'cuda' if args.gpu else 'cpu'

    if args.loss == 'nll':
        criterion = nn.NLLLoss(reduction='sum').to(device)
    elif args.loss == 'ce':
        criterion = nn.CrossEntropyLoss(reduction='sum').to(device)
    
    testloader = DataLoader(test_dataset, batch_size=200, shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)

        # Accumulate loss over batches
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item() 

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, loss/total
