#Lightly modified from https://github.com/kuangliu/pytorch-cifar/

'''Train CIFAR10 with PyTorch.'''
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

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

parser.add_argument('--epochs', type=int, default=200, metavar='Nepoch', help='number of epochs to train (default: 20)')
parser.add_argument('--train_bsz', type=int, default=100, metavar='train_bsz', help='training batch size (default: 100)')

parser.add_argument('--batch_multiplier', type=int, default=1, metavar='btch_mul',
                    help='scaling by this factor for large batch | see link https://medium.com/\
                    @davidlmorton/increasing-mini-batch-size-without-increasing-memory-6794e10db672')

parser.add_argument('--lrscheme', type=str,  default='sgdr', help='choices: sgdr, constant, step_decay, warmup')

parser.add_argument('--lrmax', default=0.05, type=float, help='max sgdr learning rate') #should be 0.1 for resnet
parser.add_argument('--lrmin', default=0.000001, type=float, help='min sgdr learning rate')
parser.add_argument('--warmup_len', type=int, default=20, metavar='warmup_len', help='number of epochs spent in warmup')


parser.add_argument('--BN', '-BN', action='store_true', help='batch norm in vgg architectures')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--dataparallel', '-dpar', action='store_true', help='resume from checkpoint')

parser.add_argument('--donotsave', action='store_true', help='dont save models and logs | by default we save everything') 
parser.add_argument('--savestr', type=str,  default='_', help='name your experiment - used for naming logdir')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

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
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_bsz, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')

'''
Critical - CHOICE OF ARCHITECTURE 
'''
net = resnet.ResNet18()
#if args.BN: 
#    net = vgg.vgg16_bn()
#else:
#    net = vgg.vgg11()
    
# net = VGG('VGG19')
# net = vgg.vgg11_bn()
# net = resnet.ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
net = net.to(device)

if device == 'cuda' and args.dataparallel:
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lrmax, momentum=0.9, weight_decay=5e-4)

if args.lrscheme == 'sgdr': 
    ''' 
    sgdr learning rate list 
    ''' #(lr = step_max at epochs 0, 11, 32, 73, 154)
    step_max = args.lrmax #0.05
    step_min = args.lrmin #0.000001
    lrlist = []
    Ti = 10
    Tcur = 0
    Tmult = 2
    for i in range(args.epochs):
        newlr = step_min + np.multiply(0.5*(step_max - step_min),(1 + np.cos((Tcur/Ti)*np.pi)))
        lrlist += [newlr]
        Tcur += 1
        if newlr ==  step_min: #or newlr < 0.000108:
            Tcur = 0
            Ti = Ti*Tmult

elif args.lrscheme == 'warmup':

    ''' 
    Warmup learning rate list  
    '''
    #(increase from step_min to step_max in 20 epochs and step_decay at 60,120,150 epochs later
    step_max = args.lrmax
    step_min = args.lrmin
    lrlist = []
    warmup_len = args.warmup_len
    
    lrlist = [step_min + (step_max - step_min)*x/warmup_len for x in range(20)] + \
         [step_max]*40 + [step_max*0.1]*60 + [step_max * 0.01]*30 + [step_max*0.001]*150
    
    
epochs_of_interest = [0,10,12,30,33,70,74,150,152,155] + [55,65,115,125,145,155]
        
'''
Dictionaries to log results
'''

#'VGG16 without batchnorm trained using sgdr, storing models at epochs of interest'
info_epoch = {'args_dict':vars(args),
              'epoch_index':[],'epoch_test_accuracy':[],'epoch_test_loss':[],'epoch_lr':[]}
info_minibatch = {'epoch_n_batch':[],'batch_train_accuracy':[],'batch_train_loss':[]}


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    count = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        
        inputs, targets = inputs.to(device), targets.to(device)
        if count == 0:
            optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)/ args.batch_multiplier
        loss.backward()
        if clip_flag: #and args.lrscheme == 'goyal_warmup' 
            nn.utils.clip_grad_norm(net.parameters(), args.clip)
        if count == 0:
            optimizer.step()
            count = args.batch_multiplier
        count -= 1
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
    info_minibatch['epoch_n_batch'].append((epoch,batch_idx))
    info_minibatch['batch_train_accuracy'].append(100.*correct/total)
    info_minibatch['batch_train_loss'].append(train_loss/(batch_idx+1))
        

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    count = 0
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        if count == 0:
            optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        if count == 0:
            optimizer.step()
            count = args.batch_multiplier
        
        count -= 1

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
        if args.train_bsz == 100 and batch_idx == 199:
            torch.save(net.state_dict(),'./results/'+unique_run_str+ args.savestr+'/'+'iter_'+str(batch_idx)+'.t7')
            
        info_minibatch['epoch_n_batch'].append((epoch,batch_idx))
        info_minibatch['batch_train_accuracy'].append(100.*correct/total)
        info_minibatch['batch_train_loss'].append(train_loss/(batch_idx+1))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc
        
    info_epoch['epoch_index'].append(epoch)
    info_epoch['epoch_test_accuracy'].append(100.*correct/total)
    info_epoch['epoch_test_loss'].append(test_loss/(batch_idx+1))
    info_epoch['epoch_lr'].append(optimizer.param_groups[0]['lr'])

    
    

for epoch in range(start_epoch, start_epoch+args.epochs):
    
    if not args.donotsave:
        if not os.path.isdir('results'): 
            os.mkdir('results')
        if not os.path.isdir('results/'+unique_run_str + args.savestr):
            os.mkdir('results/'+unique_run_str + args.savestr)
            

        torch.save(net.state_dict(),'./results/'+unique_run_str + args.savestr +'/'+'epoch_init'+'.t7')


    
    if args.lrscheme == 'sgdr':
        optimizer.param_groups[0]['lr'] = lrlist[epoch]
    
    if args.lrscheme == 'step_decay' and epoch in [60,120,150]:
        optimizer.param_groups[0]['lr'] *= 0.1
        
    if args.lrscheme == 'warmup':
        optimizer.param_groups[0]['lr'] = lrlist[epoch]
        
    
    train(epoch)
    test(epoch)
    
    print('LR is '+str(optimizer.param_groups[0]['lr']))

        
    if not args.donotsave:
        #if epoch in epochs_of_interest or 1:
        torch.save(net.state_dict(),'./results/'+unique_run_str+ args.savestr +'/'+'epoch_'+str(epoch)+'.t7')

        np.save('./results/'+unique_run_str+ args.savestr+'/'+'infep.npy',info_epoch)
        np.save('./results/'+unique_run_str+ args.savestr+'/'+'infmn.npy',info_minibatch)