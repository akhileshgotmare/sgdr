from __future__ import print_function
from sklearn.decomposition import PCA

#import setGPU
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


'''
Vectorizing w_70, w_150 and theta
'''

net_A = resnet.ResNet18()
net_B = resnet.ResNet18()
net_C = resnet.ResNet18()

net_A.load_state_dict(torch.load('results/15417101052776022/epoch_70.t7'))
net_A.eval()
net_B.load_state_dict(torch.load('results/15417101052776022/epoch_150.t7'))
net_B.eval()
net_C.load_state_dict(torch.load('results/curve_find/15418861362468774/theta_step_50499.t7'))
net_C.eval()

# net_C = torch.load('curve_find/15417853945005875/theta_step_50499.t7')
# net_C.eval()

param_vec_A = torch.cat(
                [p.view(-1) for p in net_A.parameters() if p.requires_grad]).data.numpy()

param_vec_B = torch.cat(
                [p.view(-1) for p in net_B.parameters() if p.requires_grad]).data.numpy()

param_vec_C = torch.cat(
                [p.view(-1) for p in net_C.parameters() if p.requires_grad]).data.cpu().numpy()

'''
Fitting PCA over these three vecs
'''

Big_matrix = np.array([param_vec_A,param_vec_B,param_vec_C])

pca = PCA(n_components=2)
pca.fit(Big_matrix)

X_op = pca.transform(Big_matrix)

'''
Creating grid - manually chosen deltax and deltay, selected after plotting X_op 
(see results/sgdr_scratchpad.ipynb notebook)
'''

deltax = 1
deltay = 0.5
x= np.arange(-20, 20, deltax)
y = np.arange(-6, 8, deltay)

X, Y = np.meshgrid(x, y) #X and Y are of shape (28, 40)

'''
Saving model corresponding to each vector on the grid
'''


for i in range(X.shape[0]):
    print('Now on i '+str(i))
    for j in range(X.shape[1]):

        model_vec = pca.inverse_transform(np.array([X[i,j],Y[i,j]]))

        net_grid = resnet.ResNet18() #chose appropriate model defn here

        start_ind = 0

        for p in net_grid.parameters():
            if p.requires_grad:
                copy_vals = model_vec[start_ind:(start_ind+len(p.view(-1)))]
                start_ind += len(p.view(-1))
                copy_vals = copy_vals.reshape(p.size())
                copy_vals = torch.from_numpy(copy_vals)
                p.data = copy_vals.type(torch.FloatTensor) 

        torch.save(net_grid.state_dict(),
                    'results/curve_find/15418861362468774/net_i_'+str(i)+'_j_'+str(j) + '.t7')