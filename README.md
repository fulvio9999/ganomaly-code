# UPDATE: 
This repo is an integration of [Ganomaly](https://github.com/samet-akcay/ganomaly): training on Fashion-MNIST and training on contamination of datasets (MNIST, Fashion-MNIST and CIFAR10).

# GANomaly

This repository contains PyTorch implementation of the following paper: GANomaly: Semi-Supervised Anomaly Detection via Adversarial Training [[1]](#reference)

##  1. Table of Contents
- [GANomaly](#ganomaly)
    - [Table of Contents](#table-of-contents)
    - [Installation](#installation)
    - [Training](#training)
    - [Testing](#testing)
    - [Reference](#reference)
    

## 2. Installation
1. First clone the repository
   ```
   git clone https://github.com/samet-akcay/ganomaly.git
   ```
2. Create the virtual environment via conda
    ```
    conda create -n ganomaly python=3.7
    ```
3. Activate the virtual environment.
    ```
    conda activate ganomaly
    ```
3. Install the dependencies.
   ```
   conda install -c intel mkl_fft
   pip install --user --requirement requirements.txt
   ```

## 3. Training
To replicate the results for MNIST, FASHION-MNIST and CIFAR10 datasets, run the following commands:

``` shell
# MNIST
sh experiments/run_mnist.sh

# FASHION-MNIST
sh experiments/run_fashion_mnist.sh

# CIFAR
sh experiments/run_cifar.sh # CIFAR10
```

To replicate the results about training set contamination for MNIST, FASHION-MNIST and CIFAR10 datasets, run the following commands:

``` shell
# MNIST
sh experiments/run_mnist_pullation.sh

# FASHION-MNIST
sh experiments/run_fashion_mnist_pullation.sh

# CIFAR
sh experiments/run_cifar_pullation.sh # CIFAR10
```

## 4. Testing
To replicate the results for MNIST, FASHION-MNIST and CIFAR10 datasets, run the following commands:

``` shell
# MNIST
sh experiments/run_test_mnist.sh

# FASHION-MNIST
sh experiments/run_test_fashion_mnist.sh

# CIFAR
sh experiments/run_test_cifar.sh # CIFAR10
```

## 5. Reference
[1]  Akcay S., Atapour-Abarghouei A., Breckon T.P. (2019) GANomaly: Semi-supervised Anomaly Detection via Adversarial Training. In: Jawahar C., Li H., Mori G., Schindler K. (eds) Computer Vision – ACCV 2018. ACCV 2018. Lecture Notes in Computer Science, vol 11363. Springer, Cham
