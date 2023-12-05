#!/usr/bin/env bash

GPU=$1

CLIENT_NUM=$2

WORKER_NUM=$3

BATCH_SIZE=$4

DATASET=$5

DATA_PATH=$6

MODEL=$7

DISTRIBUTION=$8

ROUND=$9

EPOCH=${10}

LR=${11}

OPT=${12}

CI=${13}

BITS=${14}

python3 ./main_fedavg.py \
--gpu $GPU \
--dataset $DATASET \
--data_dir $DATA_PATH \
--model $MODEL \
--partition_method $DISTRIBUTION  \
--client_num_in_total $CLIENT_NUM \
--client_num_per_round $WORKER_NUM \
--comm_round $ROUND \
--epochs $EPOCH \
--batch_size $BATCH_SIZE \
--client_optimizer $OPT \
--lr $LR \
--ci $CI \
--quantized_bits $BITS

# debug
# sh run_fedavg_standalone_pytorch.sh 0 1000 10 10 mnist ./../../../data/mnist lr hetero 200 1 0.03 sgd 0 6

# train
# lr_mnist
# nohup sh run_fedavg_standalone_pytorch.sh 0 10 10 10 mnist ./../../../data/mnist lr hetero 200 20 0.03 sgd 0 6 > ./fedavg_standalone.txt 2>&1 &
# nohup sh run_fedavg_standalone_pytorch.sh 0 10 10 10 mnist ./../../../data/mnist lr hetero 1000 5 0.03 sgd 0 6 > ./fedavg_standalone.txt 2>&1 &

# cnn_mnist
# nohup sh run_fedavg_standalone_pytorch.sh 0 10 10 10 mnist ./../../../data/mnist cnn hetero 200 20 0.03 sgd 0 6 > ./fedavg_standalone.txt 2>&1 &

# cnn_femnist
# nohup sh run_fedavg_standalone_pytorch.sh 0 10 10 10 femnist ./../../../data/FederatedEMNIST/datasets cnn hetero 200 20 0.03 sgd 0 6 > ./fedavg_standalone.txt 2>&1 &