''' Script to find \theta (curve's single bend) given w1 and w2 '''

from __future__ import print_function

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
from utils import progress_bar

import resnet
import vgg
import copy
import ipdb

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

#Using fixed bsz of 100 here! For other bsz values, modify many 500's to 50k/bsz

parser.add_argument('--lr', default=0.1, type=float, help='lr for curve finding/training')
#divided by 10 at 45-th and 80-th epoch

parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--dataparallel', '-dpar', action='store_true', help='resume from checkpoint')

parser.add_argument('--w1', type=str, default='results/15417101052776022/epoch_70.t7', help='name/path of file from where we load w1')
parser.add_argument('--w2', type=str, default='results/15417101052776022/epoch_150.t7', help='name/path of file from where we load w2')

parser.add_argument('--numthetasteps',type=int, default=500*101, metavar = 'numthetasteps',
                    help='number of steps to take on the theta parameter that defines the curve')
parser.add_argument('--numcurvsample',type=int, default=30, metavar = 'numcurvsample',
                    help='number of samples to take on the curve for evaluating it on train and test sets')
parser.add_argument('--log_interval', type=int, default=20*500, metavar='Nloginterval', # evaluate the curve every 5 epochs
                    help='number of iterations to wait before logging - sampling pts on curve and evaluating') 


parser.add_argument('--donotsave', action='store_true', help='dont save models and logs | by default we save everything') 
args = parser.parse_args()

if '15417101052776022' in args.w1: load_mode = 'state_dict'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

unique_run_str = str(time.time()).replace('.','') #timestamp to store log files

if not os.path.isdir('./results/curve_find'):
    os.mkdir('./results/curve_find/')
    
if not os.path.isdir('./results/curve_find' + unique_run_str):
    os.mkdir('./results/curve_find/' + unique_run_str)

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

net1 = resnet.ResNet18()
net2 = resnet.ResNet18()

print('==> Loading W1 (net1) from checkpoint..')
if load_mode == 'state_dict':
    net1.load_state_dict(torch.load(args.w1))
else:
    net1 = torch.load(args.w1)
    
print('==> Loading W2 (net2) from checkpoint..')    
if load_mode == 'state_dict':
    net2.load_state_dict(torch.load(args.w2))
else:
    net2 = torch.load(args.w2)
net1.to(device)
net2.to(device)
net1.eval()
net2.eval()

'''\theta model - initialized to (w1 + w2) / 2'''
net3 = copy.deepcopy(net1)
net3 = net3.to(device)
for param2,param3 in zip(net2.parameters(),net3.parameters()):
    if param3.requires_grad:        
        param3.data.mul_(1.0 - 0.5)
        param3.data.add_(param2.data * 0.5)
#net3 still needs accurate BN parameters for the new weights
#therefore we require one pass over training data before any evaluation

'''\phi(t) model used for curve training'''
net4 = copy.deepcopy(net1)
net4 = net4.to(device)
net4.train()

'''\phi(t) model used for curve evaluation '''
net5 = copy.deepcopy(net1)
net5 = net5.to(device)

optimizer = optim.SGD(net4.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()
    
infocurvefind = {
                'trainloss':[],
                'trainacc':[],
                'testloss':[],
                'testacc':[],
                't_list':[]
                }

def evaluate_curve():
    
    log_test_acc = []
    log_train_acc = []
    log_test_loss = []
    log_train_loss = []
    log_t_list = []
    
    print('===Evaluating Curve===>')
    
    t_list = np.linspace(0, 1, 30).tolist()
    
    for t_id, t in enumerate(t_list):
        print('now evaluationg t_id = '+str(t_id))
        
        if t<=0.5:
            for param5,param3,param1 in zip(net5.parameters(),net3.parameters(),net1.parameters()):
                if param5.requires_grad:
                    param5.data = 2*(t*param3.data + (0.5 - t)*param1.data)
        else:
            for param5,param3,param2 in zip(net5.parameters(),net3.parameters(),net2.parameters()):
                if param5.requires_grad:
                    param5.data = 2*((t - 0.5)*param2.data + (1-t)*param3.data)
        
        '''
        as per section A.2 in MC paper
        need to perform a forward pass over training set to record BN params
        '''
        net5.train()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net5(inputs)
            
        net5.eval()
        
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
                outputs = net5(inputs)

                loss = criterion(outputs, targets)

                cummloss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum().item()
                
                #progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                #    % (loss/(batch_idx+1), 100.*correct/total, correct, total))

            if loaderid == 0:
                log_test_acc += [100.*correct/total]
                log_test_loss += [cummloss/(batch_idx+1)]
                
            else:
                log_train_acc += [100.*correct/total]
                log_train_loss += [cummloss/(batch_idx+1)]
            
            log_t_list += t_list 
                
                      
    infocurvefind['testacc'].append(log_test_acc)
    infocurvefind['testloss'].append(log_test_loss)
    infocurvefind['trainacc'].append(log_train_acc)
    infocurvefind['trainloss'].append(log_train_loss)
    infocurvefind['t_list'].append(log_t_list)

def train_curve_step():
        
    t = np.random.uniform(0.0,1.0)
    
    '''Only Polychain curve implemented here'''
    
    if t<= 0.5:
        for param4,param3,param1 in zip(net4.parameters(),net3.parameters(),net1.parameters()):
            if param4.requires_grad:
                param4.data = 2*(t*param3.data + (0.5 - t)*param1.data)
    else:
        for param4,param3,param2 in zip(net4.parameters(),net3.parameters(),net2.parameters()):
            if param4.requires_grad:
                param4.data = 2*((t - 0.5)*param2.data + (1-t)*param3.data)
           
        
    inputs, targets = dataiter.next()
    inputs, targets = inputs.to(device), targets.to(device)
    
    optimizer.zero_grad()

    outputs = net4(inputs)
    loss = criterion(outputs, targets)
    loss.backward()        

    for param3,param4 in zip(net3.parameters(),net4.parameters()):

        if param3.requires_grad:
            if t<=0.5:
                param3.data -= args.lr * param4.grad.data * 2*t
            else:
                param3.data -= args.lr * param4.grad.data * 2*(1.0-t)
    
    if theta_step%500 == 0:
        print('Theta step: {}/{} \n'.format(theta_step,args.numthetasteps))
        
def save_log_n_model():

    np.save('./results/curve_find/' + unique_run_str+ '/' + 
            'step_' + str(theta_step) + '_curve_find_info.npy' ,
            infocurvefind)
    torch.save(net4.state_dict(),
               './results/curve_find/' + unique_run_str + '/' + 'theta_step_' + str(theta_step) +'.t7')
    
for theta_step in range(args.numthetasteps): #500 steps correspond to 1 epoch for bsz = 100
    
#    if theta_step == 0:
        
#        evaluate_curve()
#        save_log_n_model()

    '''Scale lr by 1/10 at '''
    if theta_step in [45*500,80*500]:
        optimizer.param_groups[0]['lr'] *= 0.1
        args.lr *= 0.1
    
    if theta_step%(500) == 0:
        dataiter = iter(trainloader)
    
            
    train_curve_step()
        
    if theta_step == args.numthetasteps-1:
        print('Done finding curve, now will evaluate points on obtained curve and store logs and models')
    #theta_step%args.log_interval == 0 and theta_step > 0: # evaluate the curve every 20 epochs
        evaluate_curve()
        save_log_n_model()