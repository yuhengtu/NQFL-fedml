# run
Comment unnecessary parts and run install.sh

Take nqfl for example
```
cd ./fedml_experiments/standalone/fedavg_nqfl

# lr_mnist
nohup sh run_fedavg_standalone_pytorch.sh 0 10 10 10 mnist ./../../../data/mnist lr hetero 1000 5 0.03 sgd 0 6 > ./fedavg_standalone.txt 2>&1 &

# cnn_mnist
nohup sh run_fedavg_standalone_pytorch.sh 0 10 10 10 mnist ./../../../data/mnist cnn hetero 200 20 0.03 sgd 0 6 > ./fedavg_standalone.txt 2>&1 &

# cnn_cifar10
nohup sh run_fedavg_standalone_pytorch.sh 0 10 10 10 cifar10 ./../../../data/cifar10 cnn hetero 200 20 0.03 sgd 0 6 > ./fedavg_standalone.txt 2>&1 &

# cnn_femnist
nohup sh run_fedavg_standalone_pytorch.sh 0 10 10 10 femnist ./../../../data/FederatedEMNIST/datasets cnn hetero 200 20 0.03 sgd 0 6 > ./fedavg_standalone.txt 2>&1 &
```


# algorithm
fedavg_qsgd is [QSGD](https://arxiv.org/pdf/1610.02132.pdf)

fedavg_lloyd is [DFL](https://arxiv.org/pdf/2303.08423.pdf)

fedavg_sk is [AdaQuantFL](https://arxiv.org/pdf/2102.04487.pdf)

fedavg_nqfl and fedavg_sk_nqfl are ours


# result
https://wandb.ai/yuhengtu/fedml/overview?workspace=user-yuhengtu


# Reference
[Fedml](https://github.com/FedML-AI/FedML)
[lloyd-max alogrithm](https://github.com/JosephChataignon/Max-Lloyd-algorithm/blob/master/1%20dimension/max_lloyd_1D.py)

