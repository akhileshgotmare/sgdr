'''Storing activations for ResNet18'''
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
import ipdb

'''Critical to remove flipping and cropping '''
transform_train = transforms.Compose([
#    transforms.RandomCrop(32, padding=4),
#    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
'''Critical to set shuffle to False so we compute activations for same inputs across all models'''
trainloader = torch.utils.data.DataLoader(trainset, batch_size=500, shuffle=False, num_workers=2)    

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

parser.add_argument('--save_acts', action='store_true', help='save activations | by default we dont save everything') 
parser.add_argument('--model_index', type=int, default=0, metavar='Nepoch', help='number of epochs to train (default: 0)')
args = parser.parse_args()
'''
load_path_list = ['results/15419857841740403LB_const_largelr/epoch_init.t7',
                  'results/15419857841740403LB_const_largelr/epoch_20.t7']

'''
load_path_list = ['results/15420174015569954LB_stepdecay_largelr/epoch_init.t7',
                  'results/15420174015569954LB_stepdecay_largelr/epoch_20.t7',
                  'results/15419858381338964SB_stepdcy_small_lr/epoch_init.t7',
                  'results/15419858381338964SB_stepdcy_small_lr/iter_199.t7',
                  'results/15419858030298254LB_warmup_largelr/epoch_init.t7',
                  'results/15419858030298254LB_warmup_largelr/epoch_20.t7'] 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
save_things = args.save_acts



i = args.model_index

for path_ind,load_path in enumerate(load_path_list[i:i+1]):
    
    print('Path # :  '+ str(path_ind) + ' / ' + str(len(load_path_list))  )

    if not os.path.isfile(load_path):
        ipdb.set_trace()
        raise Exception('The load_path you provided does not exist!')
        

    '''Load net here - for which you want acts'''

    net = resnet.ResNet18()
    net.load_state_dict(torch.load(load_path ))
    net.eval()

    #dir containing activations from different models for this run (eg. results/1555253232532/activations/)
    activations_path = os.path.abspath(os.path.join(load_path, os.pardir)) + '/activations_newest' 
    model_name = os.path.basename(load_path) #eg. epoch_0.t7

    if not os.path.isdir(activations_path):
        os.mkdir(activations_path)
    if not os.path.isdir(activations_path + '/' + model_name):
        os.mkdir(activations_path + '/' + model_name)


    for batch_idx, (inputs, targets) in enumerate(trainloader,0): #the 0 in the argument resets the trainloader
        
        print('batch_id # :  ' + str(batch_idx) + ' / 20')
        
    
        if batch_idx < 20:
            
            print('Now on batch : ' + str(batch_idx))

            net = net.to(device)
            inputs = inputs.to(device)

            x = inputs.clone() #trainset.train_data[1000*batch_idx:1000*(batch_idx+1)]#inputs.clone()

            layer_count = 0

            filesavename = lambda layer_count : activations_path + '/' + model_name + '/' + 'layer_' + str(layer_count) + '_batch_' + str(batch_idx) + '.npy'


            #ipdb.set_trace()
            x = net._modules['conv1'](x);  layer_count += 1 
            if save_things: np.save(filesavename(layer_count),x.detach().cpu().numpy()) #1
            x = net._modules['bn1'](x);  layer_count += 1
            if save_things: np.save(filesavename(layer_count),x.detach().cpu().numpy()) #2
            x = F.relu(x);  layer_count += 1
            if save_things: np.save(filesavename(layer_count),x.detach().cpu().numpy()) #3


            x1 = x
            x = net._modules['layer1'][0]._modules['conv1'](x);  layer_count += 1
            if save_things: np.save(filesavename(layer_count),x.detach().cpu().numpy()) #4
            x = net._modules['layer1'][0]._modules['bn1'](x);  layer_count += 1
            if save_things: np.save(filesavename(layer_count),x.detach().cpu().numpy())
            x = F.relu(x);  layer_count += 1
            if save_things: np.save(filesavename(layer_count),x.detach().cpu().numpy())
            x = net._modules['layer1'][0]._modules['conv2'](x);  layer_count += 1
            if save_things: np.save(filesavename(layer_count),x.detach().cpu().numpy())
            x = net._modules['layer1'][0]._modules['bn2'](x);  layer_count += 1
            if save_things: np.save(filesavename(layer_count),x.detach().cpu().numpy())
            x += net._modules['layer1'][0]._modules['shortcut'](x1);  layer_count += 1
            if save_things: np.save(filesavename(layer_count),x.detach().cpu().numpy())
            x = F.relu(x);  layer_count += 1
            if save_things: np.save(filesavename(layer_count),x.detach().cpu().numpy()) #10

            x1 = x
            x = net._modules['layer1'][1]._modules['conv1'](x);  layer_count += 1
            if save_things: np.save(filesavename(layer_count),x.detach().cpu().numpy()) #11
            x = net._modules['layer1'][1]._modules['bn1'](x);  layer_count += 1
            if save_things: np.save(filesavename(layer_count),x.detach().cpu().numpy())
            x = F.relu(x);  layer_count += 1
            if save_things: np.save(filesavename(layer_count),x.detach().cpu().numpy())
            x = net._modules['layer1'][1]._modules['conv2'](x);  layer_count += 1
            if save_things: np.save(filesavename(layer_count),x.detach().cpu().numpy())
            x = net._modules['layer1'][1]._modules['bn2'](x);  layer_count += 1
            if save_things: np.save(filesavename(layer_count),x.detach().cpu().numpy())
            x += net._modules['layer1'][1]._modules['shortcut'](x1);  layer_count += 1
            if save_things: np.save(filesavename(layer_count),x.detach().cpu().numpy())
            x = F.relu(x);  layer_count += 1
            if save_things: np.save(filesavename(layer_count),x.detach().cpu().numpy()) #17

            x1 = x
            x = net._modules['layer2'][0]._modules['conv1'](x);  layer_count += 1
            if save_things: np.save(filesavename(layer_count),x.detach().cpu().numpy()) #18
            x = net._modules['layer2'][0]._modules['bn1'](x);  layer_count += 1
            if save_things: np.save(filesavename(layer_count),x.detach().cpu().numpy())
            x = F.relu(x);  layer_count += 1
            if save_things: np.save(filesavename(layer_count),x.detach().cpu().numpy())
            x = net._modules['layer2'][0]._modules['conv2'](x);  layer_count += 1
            if save_things: np.save(filesavename(layer_count),x.detach().cpu().numpy())
            x = net._modules['layer2'][0]._modules['bn2'](x);  layer_count += 1
            if save_things: np.save(filesavename(layer_count),x.detach().cpu().numpy())
            x += net._modules['layer2'][0]._modules['shortcut'](x1);  layer_count += 1
            if save_things: np.save(filesavename(layer_count),x.detach().cpu().numpy())
            x = F.relu(x);  layer_count += 1
            if save_things: np.save(filesavename(layer_count),x.detach().cpu().numpy()) #24


            #ipdb.set_trace()
            x1 = x
            x = net._modules['layer2'][1]._modules['conv1'](x);  layer_count += 1
            if save_things: np.save(filesavename(layer_count),x.detach().cpu().numpy()) #25
            x = net._modules['layer2'][1]._modules['bn1'](x);  layer_count += 1
            if save_things: np.save(filesavename(layer_count),x.detach().cpu().numpy())
            x = F.relu(x);  layer_count += 1
            if save_things: np.save(filesavename(layer_count),x.detach().cpu().numpy())
            x = net._modules['layer2'][1]._modules['conv2'](x);  layer_count += 1
            if save_things: np.save(filesavename(layer_count),x.detach().cpu().numpy())
            x = net._modules['layer2'][1]._modules['bn2'](x);  layer_count += 1
            if save_things: np.save(filesavename(layer_count),x.detach().cpu().numpy())
            x += net._modules['layer2'][1]._modules['shortcut'](x1);  layer_count += 1
            if save_things: np.save(filesavename(layer_count),x.detach().cpu().numpy())
            x = F.relu(x);  layer_count += 1
            if save_things: np.save(filesavename(layer_count),x.detach().cpu().numpy()) #31

            x1 = x
            x = net._modules['layer3'][0]._modules['conv1'](x);  layer_count += 1
            if save_things: np.save(filesavename(layer_count),x.detach().cpu().numpy()) #32
            x = net._modules['layer3'][0]._modules['bn1'](x);  layer_count += 1
            if save_things: np.save(filesavename(layer_count),x.detach().cpu().numpy())
            x = F.relu(x);  layer_count += 1
            if save_things: np.save(filesavename(layer_count),x.detach().cpu().numpy())
            x = net._modules['layer3'][0]._modules['conv2'](x);  layer_count += 1
            if save_things: np.save(filesavename(layer_count),x.detach().cpu().numpy())
            x = net._modules['layer3'][0]._modules['bn2'](x);  layer_count += 1
            if save_things: np.save(filesavename(layer_count),x.detach().cpu().numpy())
            x += net._modules['layer3'][0]._modules['shortcut'](x1);  layer_count += 1
            if save_things: np.save(filesavename(layer_count),x.detach().cpu().numpy())
            x = F.relu(x);  layer_count += 1
            if save_things: np.save(filesavename(layer_count),x.detach().cpu().numpy()) #38

            x1 = x
            x = net._modules['layer3'][1]._modules['conv1'](x);  layer_count += 1
            if save_things: np.save(filesavename(layer_count),x.detach().cpu().numpy()) #39
            x = net._modules['layer3'][1]._modules['bn1'](x);  layer_count += 1
            if save_things: np.save(filesavename(layer_count),x.detach().cpu().numpy())
            x = F.relu(x);  layer_count += 1
            if save_things: np.save(filesavename(layer_count),x.detach().cpu().numpy())
            x = net._modules['layer3'][1]._modules['conv2'](x);  layer_count += 1
            if save_things: np.save(filesavename(layer_count),x.detach().cpu().numpy())
            x = net._modules['layer3'][1]._modules['bn2'](x);  layer_count += 1
            if save_things: np.save(filesavename(layer_count),x.detach().cpu().numpy())
            x += net._modules['layer3'][1]._modules['shortcut'](x1);  layer_count += 1
            if save_things: np.save(filesavename(layer_count),x.detach().cpu().numpy())
            x = F.relu(x);  layer_count += 1
            if save_things: np.save(filesavename(layer_count),x.detach().cpu().numpy()) #45

            x1 = x
            x = net._modules['layer4'][0]._modules['conv1'](x);  layer_count += 1
            if save_things: np.save(filesavename(layer_count),x.detach().cpu().numpy()) #46
            x = net._modules['layer4'][0]._modules['bn1'](x);  layer_count += 1
            if save_things: np.save(filesavename(layer_count),x.detach().cpu().numpy())
            x = F.relu(x);  layer_count += 1
            if save_things: np.save(filesavename(layer_count),x.detach().cpu().numpy())
            x = net._modules['layer4'][0]._modules['conv2'](x);  layer_count += 1
            if save_things: np.save(filesavename(layer_count),x.detach().cpu().numpy())
            x = net._modules['layer4'][0]._modules['bn2'](x);  layer_count += 1
            if save_things: np.save(filesavename(layer_count),x.detach().cpu().numpy())
            x += net._modules['layer4'][0]._modules['shortcut'](x1);  layer_count += 1
            if save_things: np.save(filesavename(layer_count),x.detach().cpu().numpy())
            x = F.relu(x);  layer_count += 1
            if save_things: np.save(filesavename(layer_count),x.detach().cpu().numpy()) #52

            x1 = x
            x = net._modules['layer4'][1]._modules['conv1'](x);  layer_count += 1
            if save_things: np.save(filesavename(layer_count),x.detach().cpu().numpy()) #53
            x = net._modules['layer4'][1]._modules['bn1'](x);  layer_count += 1
            if save_things: np.save(filesavename(layer_count),x.detach().cpu().numpy())
            x = F.relu(x);  layer_count += 1
            if save_things: np.save(filesavename(layer_count),x.detach().cpu().numpy())
            x = net._modules['layer4'][1]._modules['conv2'](x);  layer_count += 1
            if save_things: np.save(filesavename(layer_count),x.detach().cpu().numpy())
            x = net._modules['layer4'][1]._modules['bn2'](x);  layer_count += 1
            if save_things: np.save(filesavename(layer_count),x.detach().cpu().numpy())
            x += net._modules['layer4'][1]._modules['shortcut'](x1);  layer_count += 1
            if save_things: np.save(filesavename(layer_count),x.detach().cpu().numpy())
            x = F.relu(x);  layer_count += 1
            if save_things: np.save(filesavename(layer_count),x.detach().cpu().numpy()) #59

            x = F.avg_pool2d(x, 4)

            x = x.view(x.size(0), -1)

            x = net._modules['linear'](x);  layer_count += 1
            if save_things: np.save(filesavename(layer_count),x.detach().cpu().numpy()) #60
            
        else:
            break #we need activations for only 10k samples - 10 iterations x 1k bsz