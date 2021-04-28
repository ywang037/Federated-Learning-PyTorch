import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

import torchvision
from torchvision import transforms
from torchvision import datasets

# from models import CNNCifar
# from data_preparation import data_setup

import time
import csv
import numpy as np

class TaskMnist():
    def __init__(self, nn='cnn_wy'):
        self.path = './data/mnist'
        self.name = 'mnist'
        self.nn = nn
        
class TaskCifar():
    def __init__(self,nn='cnn_torch'):
        self.path = './data/cifar'
        self.name = 'cifar'
        self.nn = nn

class HyperParam():
    def __init__(self,path,learning_rate=0.1, batch_size=100, epoch=10, momentum=0.0, nesterov=False):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.datapath = path
        self.lr=learning_rate
        self.bs=batch_size
        self.epoch=epoch
        self.momentum=momentum
        self.nesterov=nesterov        

# the 2NN model described in the vanilla FL paper for experiments with MNIST
class TwoNN(nn.Module):
    def __init__(self):
        super(TwoNN,self).__init__()
        self.nn_layer=nn.Sequential(
            nn.Linear(in_features=28*28,out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100,out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100,out_features=10)
        )
    def forward(self,x):
        x = x.view(-1,28*28)
        logits = self.nn_layer(x)
        return F.log_softmax(logits,dim=1)
                 
# the 2NN model in AshwinRJ's repository
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(28*28, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return F.log_softmax(x,dim=1)

# the CNN model describted in the vanilla FL paper for experiments with MNIST
class CNNMnistWy(nn.Module):
    def __init__(self):
        super(CNNMnistWy,self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(in_features=1024,out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512,out_features=10),
        )
    
    def forward(self,x):
        x=self.conv_layer(x)
        x=x.view(-1,1024)
        logits = self.fc_layer(x)
        return F.log_softmax(logits,dim=1)

# the CNN model in AshwinRJ's repository
class CNNMnist(nn.Module):
    def __init__(self):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# the example model used in the official CNN training tutorial of PyTorch using CIFAR10
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
class CNNCifarTorch(nn.Module):
    def __init__(self):
        super(CNNCifarTorch,self).__init__()
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

# the exmaple model used in the official CNN tutorial of TensorFlow using CIFAR10
# https://www.tensorflow.org/tutorials/images/cnn
class CNNCifarTf(nn.Module):
    def __init__(self):
        super(CNNCifarTf,self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32, kernel_size=3), # output size 30*30, i.e., (32, 30 ,30)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2), # output size 15*15, i.e., (32, 15 ,15)
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3), # output size 13*13, i.e., (64, 13 ,13)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2), # output size 6*6, i.e., (64, 6, 6)
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3), # output size 4*4, i.e., (64, 4, 4)
            nn.ReLU()
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(in_features=1024,out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64,out_features=10),
        )

    def forward(self,x):
        x=self.conv_layer(x)
        x=x.view(-1,1024)
        logits=self.fc_layer(x)
        return F.log_softmax(logits,dim=1)
    
def data_cifar(path, batch_size=64):
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
    # configure transform for test data
    transform_test_0 = transforms.Compose([
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
    transform_test_1 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean_1,std_1),
    ])

    # setup the CIFAR10 training dataset
    data_train = datasets.CIFAR10(root=path, train=True, download=False, transform=transform_train_0)
    loader_train = data.DataLoader(data_train, batch_size=batch_size, shuffle=True)

    # setup the CIFAR10 test dataset
    data_test = datasets.CIFAR10(root=path, train=False, download=False, transform=transform_test_0)
    loader_test = data.DataLoader(data_test, batch_size=100, shuffle=False)

    return loader_train, loader_test

def data_mnist(path,batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # setup the MNIST training dataset
    data_train = datasets.MNIST(root=path, train=True, download=False, transform=transform)
    loader_train = data.DataLoader(data_train, batch_size=batch_size, shuffle=True) 
    
    # setup the MNIST training dataset
    data_test = datasets.MNIST(root=path, train=False, download=False, transform=transform)
    loader_test = data.DataLoader(data_test, batch_size=100, shuffle=False)
    return loader_train,loader_test
    
# the function used to count the number of trainable parameters
def get_count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# training function
def train_model(loader_train, loader_test, epochs, loss_fn, optimizer, device):
    with open('./save/results-mybaseline.csv', 'w', newline='') as file:
        writer = csv.writer(file,delimiter=',')
        writer.writerow(['epoch','training loss', 'test acc'])
        for epoch in range(1, epochs+1):
            train_loss = 0.0
            test_loss = 0.0
            test_acc = 0.0

            # training of each epoch
            model.train()
            for batch, (images, labels) in enumerate(loader_train):
                images, labels = images.to(device), labels.to(device)
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
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = loss_fn(outputs,labels)
                    test_loss += loss.item() * images.size(0)
                    pred = outputs.argmax(dim=1)
                    num_correct += pred.eq(labels.view_as(pred)).sum().item()
            test_loss /= len(loader_test.dataset)
            test_acc = 100*num_correct/len(loader_test.dataset)
            print('Epoch: {:>4}/{} | Training Loss: {:.2f} | Test Loss: {:.2f} | Test accuracy = {:.2f}%'.format(epoch, epochs, train_loss, test_loss, test_acc))
            writer.writerow([
                epoch, 
                np.round(train_loss,2), 
                np.round(test_acc,2)
                ])

if __name__ == '__main__':
    torch.manual_seed(1)
    # configure the task and training settings
    task = TaskMnist(nn='cnn')    
    # task = TaskCifar(nn='cnn_tf')
    settings = HyperParam(path=task.path, learning_rate=0.01, epoch=200, momentum=0.0, nesterov=False)  
    
    if task.name == 'mnist':
        if task.nn == 'cnn_wy':
            model = CNNMnistWy().to(settings.device) # use CNN createdy by WY for MNIST learning
        elif task.nn == 'cnn':
            model = CNNMnist().to(settings.device) # use CNN in the original repository for MNIST learning
        elif task.nn == '2nn_wy':
            model = TwoNN().to(settings.device) # use 2NN MLP created by WY for MNIST learning
        else:
            model = MLP().to(settings.device) # use 2NN MLP in the original repository for MNIST learning
        loader_train, loader_test = data_mnist(path=settings.datapath,batch_size=settings.bs)
    elif task.name == 'cifar':
        if task.nn == 'cnn_torch':
            model = CNNCifarTorch().to(settings.device)
        else:
            model = CNNCifarTf().to(settings.device)
        loader_train, loader_test = data_cifar(path=settings.datapath,batch_size=settings.bs)
    
    # set the loss function and optimizer
    loss_fn = nn.CrossEntropyLoss().to(settings.device)
    if settings.nesterov:
        optimizer = torch.optim.SGD(model.parameters(), lr=settings.lr, momentum=settings.momentum, nesterov=settings.nesterov)
    elif settings.momentum:
        optimizer = torch.optim.SGD(model.parameters(), lr=settings.lr, momentum=settings.momentum)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=settings.lr)
    
    # print some welcome messages
    print('\nModel training initiated...\n')
    print(f'Dataset:\t{task.name}')
    print(f'Loss function:\t{loss_fn}')
    print('Optimizer:\tSGD with Nesterov momentum=0.9') if settings.nesterov else print('Optimizer:\tvanilla SGD')
    print(f'learning rate:\t{settings.lr}')
    print(f'Batch size:\t{settings.bs}')
    print(f'Num of epochs:\t{settings.epoch}')
    print('Model to train:\n', model)
    print(f'Trainable model parameters:\t{get_count_params(model)}')

    # start training
    start = time.time()
    train_model(loader_train=loader_train,
                loader_test=loader_test,
                loss_fn=loss_fn,
                optimizer=optimizer,
                epochs=settings.epoch,
                device=settings.device)

    # print the wall-clock-time used
    end=time.time() 
    print('\nWall clock time elapsed: {:.2f}s'.format(end-start))
    # if settings.nesterov:
    #     save_path = f'./save/weights-{task.nn}-{task.name}-ep{settings.epoch}-bs{settings.bs}-lr{settings.lr}-nag.pth'
    # else:
    #     save_path = f'./save/weights-{task.nn}-{task.name}-ep{settings.epoch}-bs{settings.bs}-lr{settings.lr}.pth'
    save_path = f'./save/weights-mybaseline-{task.nn}-{task.name}-ep{settings.epoch}-bs{settings.bs}-lr{settings.lr}.pth'
    torch.save(model.state_dict(), save_path)