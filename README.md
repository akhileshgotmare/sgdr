## Implementation for VGG trained using SGDR on CIFAR10

SGDR implemented as presented in this [paper](https://arxiv.org/pdf/1608.03983.pdf), built on top of [this repo](https://github.com/kuangliu/pytorch-cifar) and vgg (with BN) defn borrowed from [here](https://github.com/keskarnitish/large-batch-training/blob/master/PyTorch/vgg.py).

## Directory layout

    .
    ├── decoupled_backprop                   
        ├── main.py           # main file training resnet in decoupled fashion
        └── utils.py
        ├── resnet.py                           
        ├── vgg.py                           

## Training details

![alt text]%(https://github.com/epfml/msc-akhilesh-gotmare/blob/master/decoupled_backprop/val_acc_comparison.jpg)

### Dependencies

Anaconda, PyTorch