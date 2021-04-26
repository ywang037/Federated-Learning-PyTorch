import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

import torchvision
from torchvision import transforms
from torchvision import datasets

# from models import CNNCifar
# from data_preparation import data_setup

class HyperParam():
    def __init__(self,path,learning_rate=0.1, batch_size=64, epoch=10, momentum=0.9, nesterov=False):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.datapath = path
        self.lr=learning_rate
        self.bs=batch_size
        self.epoch=epoch
        self.momentum=momentum
        self.nesterov=nesterov        

class CNNCifar(nn.Module):
    def __init__(self):
        super(CNNCifar,self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(16*5*5,120),
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84,10)
        )

    def forward(self,x):
        x=self.conv_layer(x)
        x=x.view(-1, 16 * 5 * 5)
        logits=self.fc_layer(x)
        return F.log_softmax(logits,dim=1)
    
def data_setup(path, batch_size=64):
    """
    returns training data loader and test data loader
    """
    # no brainer normalization used in the pytorch tutorial
    mean_0 = (0.5, 0.5, 0.5)
    std_0 = (0.5, 0.5, 0.5)

    # alternative normilzation
    mean_1 = (0.4914, 0.4822, 0.4465)
    std_1 = (0.2023, 0.1994, 0.2010)

    # configure tranform for training data
    # standard transform used in the pytorch tutorial 
    transform_train_0 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean_0,std_0),
    ])

    # enhanced transform, random crop and flip is optional
    transform_train_1 = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean_1,std_1),
    ])

    # alternative, only random crop is used
    transform_train_2 = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean_1,std_1),
    ])

    # configure transform for test data
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean_1,std_1),
    ])

    # setup the training dataset
    data_train = datasets.CIFAR10(root=path, train=True, download=False, transform=transform_train_0)
    loader_train = data.DataLoader(data_train, batch_size=batch_size, shuffle=True)

    # setup the test dataset
    data_test = datasets.CIFAR10(root=path, train=False, download=False, transform=transform_test)
    loader_test = data.DataLoader(data_test, batch_size=100, shuffle=False)

    return loader_train, loader_test

        
if __name__ == '__main__':
    settings = HyperParam(path='..\data\cifar')
    model = CNNCifar().to(settings.device)
    loader_train, loader_test = data_setup(path=settings.datapath,batch_size=settings.bs)
    loss_fn = nn.CrossEntropyLoss().to(settings.device)
    if settings.nesterov:
        optimizer = torch.optim.SGD(model.parameters(), lr=settings.lr, momentum=settings.momentum, nesterov=settings.nesterov)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=settings.lr)
    
    # print some welcome messages
    print('\nModel training initiated...\n')
    print('Dataset:\tCIFAR10')
    print(f'Loss function:\t{loss_fn}')
#     optimizer_selection = 'SGD with Nesterov momentum=0.9' if settings.nesterov else 'vanilla SGD'
#     print('Optimizer:\t',optimizer_selection)
    print('Optimizer:\tSGD with Nesterov momentum=0.9') if settings.nesterov else print('Optimizer:\tvanilla SGD')
    print(f'learning rate:\t{settings.lr}')
    print(f'Batch size:\t{settings.bs}')
    print(f'Num of epochs:\t{settings.epoch}')
    print('Model to train:\n', model)

    # start training
    for epoch in range(1, settings.epoch+1):
        train_loss = 0.0
        test_loss = 0.0
        test_acc = 0.0
        
        # training of each epoch
        model.train()
        for batch, (images, labels) in enumerate(loader_train):
            images, labels = images.to(settings.device), labels.to(settings.device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= len(loader_train.dataset)

        # test after each epoch
        model.eval()
        num_correct = 0 
        with torch.no_grad():
            for batch, (images, labels) in enumerate(loader_test):
                images, labels = images.to(settings.device), labels.to(settings.device)
                outputs = model(images)
                loss = loss_fn(outputs,labels)
                test_loss += loss.item() * images.size(0)
                pred = outputs.argmax(dim=1)
                num_correct += pred.eq(labels.view_as(pred)).sum().item()
        test_loss /= len(loader_test.dataset)
        test_acc = 100*num_correct/len(loader_test.dataset)
        print('Epoch: {} | Training Loss: {:.2f} | Test Loss: {:.2f} | Test accuracy = {:.2f}%'.format(epoch, train_loss, test_loss, test_acc))
