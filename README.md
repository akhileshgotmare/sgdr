## Implementation for ResNet18 trained using SGDR on CIFAR10

SGDR implemented as presented in this [paper](https://arxiv.org/pdf/1608.03983.pdf), built on top of [this repo](https://github.com/kuangliu/pytorch-cifar) and vgg (with BN) defn borrowed from [here](https://github.com/keskarnitish/large-batch-training/blob/master/PyTorch/vgg.py).

After storing iterates from SGDR, we run the [mode connectivity](https://arxiv.org/abs/1802.10026) procedure on a pair of [snapshots](https://openreview.net/forum?id=BJYwwY9ll). Next, we obtain and evaluate models on the plane defined by the two snapshots and their connectivity.

Warmup: In a slightly independent direction, we apply CCA on activations iterates from LB (with and without warmup) and SB training to understand the differences in their training dynamics. Here after training ResNet18 using these 3 different setups, we store activations corresponding to the intialization model and the model after 200 iterations (equal to warmup length). Finally we apply CCA on these 3 pairs of activation sets. 


## Directory layout

    .
    ├── decoupled_backprop                   
        ├── main.py           # main file training resnet in decoupled fashion
        └── utils.py
        └── resnet.py
        └── vgg.py
        └── find_curve.py 
        └── store_grid_models.py 
        └── eval_grid_models.py
        └── store_resnet18_acts.py
        └── cca_core.py
        └── dft_ccas.py
        └── compute_dft_cca.py
        
        
## Training details

![alt text]%(https://github.com/epfml/msc-akhilesh-gotmare/blob/master/decoupled_backprop/val_acc_comparison.jpg)

### Dependencies

Anaconda, PyTorch