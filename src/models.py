#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return self.softmax(x)

'''
MNIST data has fixed number channels=1, and fixed number of classes=10,
Therefore, WY slightly changed the orignal definition of class CNNMnist,
to explictly specify the values for these two arguments.

# the following is the original
class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
'''
# the following is WY's edition of CNNMnist
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

class CNNFashion_Mnist(nn.Module):
    def __init__(self, args):
        super(CNNFashion_Mnist, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    
'''
CIFAR10 has a fixed number of classes=10, 
so WY sightly changed original definition for the class CNNCifar,
to explicity specify this value of argument

# the following is the original 
class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.fc3 = nn.Linear(84, args.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
'''

# the following is WY's edition of CNNCifar
class CNNCifar(nn.Module):
    def __init__(self):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
    
    
class modelC(nn.Module):
    def __init__(self, input_size, n_classes=10, **kwargs):
        super(AllConvNet, self).__init__()
        self.conv1 = nn.Conv2d(input_size, 96, 3, padding=1)
        self.conv2 = nn.Conv2d(96, 96, 3, padding=1)
        self.conv3 = nn.Conv2d(96, 96, 3, padding=1, stride=2)
        self.conv4 = nn.Conv2d(96, 192, 3, padding=1)
        self.conv5 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv6 = nn.Conv2d(192, 192, 3, padding=1, stride=2)
        self.conv7 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv8 = nn.Conv2d(192, 192, 1)

        self.class_conv = nn.Conv2d(192, n_classes, 1)


    def forward(self, x):
        x_drop = F.dropout(x, .2)
        conv1_out = F.relu(self.conv1(x_drop))
        conv2_out = F.relu(self.conv2(conv1_out))
        conv3_out = F.relu(self.conv3(conv2_out))
        conv3_out_drop = F.dropout(conv3_out, .5)
        conv4_out = F.relu(self.conv4(conv3_out_drop))
        conv5_out = F.relu(self.conv5(conv4_out))
        conv6_out = F.relu(self.conv6(conv5_out))
        conv6_out_drop = F.dropout(conv6_out, .5)
        conv7_out = F.relu(self.conv7(conv6_out_drop))
        conv8_out = F.relu(self.conv8(conv7_out))

        class_out = F.relu(self.class_conv(conv8_out))
        pool_out = F.adaptive_avg_pool2d(class_out, 1)
        pool_out.squeeze_(-1)
        pool_out.squeeze_(-1)
        return pool_out

'''
Below are the models created by Wang Yuan for experiment with CIFAR10 dataset
'''
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
    
'''
Below are the models created by Wang Yuan for experiment with MNIST dataset
'''
# the 2-NN model described in the vanilla FL paper for experiments with MNIST
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

# the CNN model describted in the vanilla FL paper for experiments with MNIST
class CNNMnistWyBnDp(nn.Module):
    def __init__(self):
        super(CNNMnistWyBnDp,self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
            nn.Dropout2d(),
            nn.BatchNorm2d(64),
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