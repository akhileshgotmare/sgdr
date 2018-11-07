## Implementation for VGG trained using SGDR on CIFAR10

SGDR implemented as presented in this [workshop paper](https://arxiv.org/pdf/1608.03983.pdf), built on top of [this repo](https://github.com/kuangliu/pytorch-cifar) and vgg (with BN) defn borrowed from [here](https://github.com/keskarnitish/large-batch-training/blob/master/PyTorch/vgg.py).

## Directory layout

    .
    ├── decoupled_backprop                   
        ├── main.py           # main file training resnet in decoupled fashion
        └── utils.py
        ├── resnet.py                           
        ├── vgg.py                           

## Training details

Current setup that attains ~86.3% on validation set in 50 epochs (CIFAR10)
(fixed error: B1.step() instead of B2.step())

* optimizer for B1, B2, h: adam, adam, adam
* stepsizes for B1, B2, h: 10e-4, 10e-4, 10e-3
* penalty weight: 10e+6
* Number of h-steps: 1

Results (best config for decoupled backprop so far)

![alt text](https://github.com/epfml/msc-akhilesh-gotmare/blob/master/decoupled_backprop/val_acc_comparison.jpg)

### Dependencies

Anaconda, PyTorch