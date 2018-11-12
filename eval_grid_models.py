from __future__ import print_function
from sklearn.decomposition import PCA

import setGPU
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import time

import numpy as np

import resnet
import vgg
import copy
import ipdb
import itertools

chunk_id = 3 #choose one of [0,1,2,3] - 4 chunks of the grid\

deltax = 1
deltay = 0.5
x= np.arange(-20, 20, deltax)
y = np.arange(-6, 8, deltay)

X, Y = np.meshgrid(x, y) #X and Y are of shape (28, 40)

job_len = int(len(x)*len(y)/4)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
criterion = nn.CrossEntropyLoss()

unique_run_str = str(time.time()).replace('.','') #timestamp to store log files

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


train_acc_matrix = np.zeros(X.shape)
train_loss_matrix = np.zeros(X.shape)
test_acc_matrix = np.zeros(X.shape)
test_loss_matrix = np.zeros(X.shape)


for i in range(chunk_id*int(X.shape[0]/4) , (chunk_id+1)*int(X.shape[0]/4)):
    print('Now on i '+str(i))
    for j in range(X.shape[1]):

        net_on_grid = resnet.ResNet18()
        net_on_grid.load_state_dict(
            torch.load('results/curve_find/15418861362468774/net_i_' +
                       str(i)+'_j_'+str(j) + '.t7')) 
        net_on_grid = net_on_grid.to(device)
        net_on_grid.train()

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net_on_grid(inputs)

        net_on_grid.eval()

        for loaderid,loader in enumerate([testloader,trainloader]):
            if loaderid == 0:
                print('evaluating on test set')
            else:
                print('evaluating on train set')

            cummloss = 0
            correct = 0
            total = 0
            for batch_idx, (inputs, targets) in enumerate(loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net_on_grid(inputs)

                loss = criterion(outputs, targets)

                cummloss += loss.item()
                _, predicted = torch.max(outputs.data, 1)

                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum().item()

                #progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                #    % (loss/(batch_idx+1), 100.*correct/total, correct, total))


            if loaderid == 0:
                test_acc_matrix[i,j] = 100. * correct/total
                test_loss_matrix[i,j] = cummloss/(batch_idx + 1)

            else:
                train_acc_matrix[i,j] = 100. * correct/total
                train_loss_matrix[i,j] = cummloss/(batch_idx + 1)
            
info_grid = {
            'train_acc_mat': train_acc_matrix,
             'train_loss_mat': train_loss_matrix,
             'test_acc_mat': test_acc_matrix,
             'test_loss_matrix':test_loss_matrix,
            }
            
np.save('results/curve_find/15418861362468774/info_grid_part_'+str(chunk_id)+'.npy',info_grid)