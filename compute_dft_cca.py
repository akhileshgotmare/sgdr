''' Turning run_dft_cca to handle something more general - eg. layer 0 and 34 for epoch 10 and 150'''
''' Works for activations from warmup too '''
 
import numpy as np
import time
import cca_core
import dft_ccas
import os
import os.path
import ipdb

'''
'results/15420174015569954LB_stepdecay_largelr/epoch_20.t7',
'results/15419858381338964SB_stepdcy_small_lr/iter_199.t7',
'results/15419858030298254LB_warmup_largelr/epoch_init.t7',
'''


path = '/mlodata1/gotmare/sgdr/results/15419858030298254LB_warmup_largelr/activations_newest'

mode = 'diagonal' #'custom col'

os.chdir(path)
save_flag = False
epochA = 'epoch_init.t7'
epochB = 'epoch_20.t7'#str(150) #str(150)
#epochB = 'iter_199.t7'

store_results = path + '/resnet18_comparing_' + epochA + '_and_' + epochB

if not os.path.isdir(store_results):
    os.mkdir(store_results)

n_bigdata = 10000
N_layers = 61

completed_set = ['0'] #numbering starts from 1 for this set 
layer_set = [str(x) for x in range(N_layers) if str(x) not in completed_set] 

def stack_minibatches(epoch, layer):
    
    act_path_list = [epoch + '/layer_' + layer + '_batch_' + str(batch_id) + '.npy' for batch_id in range(20)]
    act_list = [np.load(act_path) for act_path in act_path_list]
    
    L = np.vstack(act_list)
    
    if int(layer) > 59:
        L = np.expand_dims(L,axis = 2)
        L = np.expand_dims(L,axis = 3)
    
    return L

for base_layer in reversed(layer_set):
    
    print('On Layer: '+ base_layer)
    
    if not os.path.isdir(store_results + '/layer_' + base_layer + '_and_X'):
        os.mkdir(store_results + '/layer_' + base_layer + '_and_X' )
            
    second_cand_list = [str(x) for x in reversed(range(int(base_layer),N_layers))]
    layers_to_compare = [(base_layer,x) for x in second_cand_list]
    
    for pair_ind, (layerA,layerB) in enumerate(layers_to_compare):
        
        L1 = stack_minibatches(epochA, layerA)
        L2 = stack_minibatches(epochB, layerB)
        
        #ipdb.set_trace()
        
        L1 = np.transpose(L1,(0,2,3,1)) ; L2 = np.transpose(L2,(0,2,3,1))
        
        df_output = dft_ccas.fourier_ccas(L1, L2) 
        save_path = store_results + '/layer_' + base_layer + '_and_X' + \
                    '/approximate_' + str(n_bigdata) + '_dftcca_models_' + \
                    layerA + '_and_' + layerB + '.df'
        
        df_output.to_pickle(save_path)
        
        print('Done computing CCA for pair: ' + str(pair_ind) + 
              'consisting of: layer ' + epochA + ' and layer ' + epochB)