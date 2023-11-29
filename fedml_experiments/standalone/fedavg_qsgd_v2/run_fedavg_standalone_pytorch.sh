#!/usr/bin/env bash

GPU=$1

CLIENT_NUM=$2 #客户端的数量

WORKER_NUM=$3 #每轮参与训练的客户端数量

BATCH_SIZE=$4

DATASET=$5

DATA_PATH=$6

MODEL=$7

DISTRIBUTION=$8 #客户端数据的划分方式，比如"hetero"表示异构划分

ROUND=$9 #训练的总轮数

EPOCH=${10} #每个客户端在每轮训练中运行的epoch数

LR=${11} #学习率

OPT=${12}

CI=${13} #是否进行联邦学习中的交叉迭代（cross iteration）

BITS=${14} #量化位数，表示是否对梯度进行量化

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
# sh run_fedavg_standalone_pytorch.sh 0 1000 10 10 mnist ./../../../data/mnist lr hetero 200 1 0.03 sgd 0 8

# train
# nohup sh run_fedavg_standalone_pytorch.sh 0 10 10 10 mnist ./../../../data/mnist lr hetero 200 20 0.03 sgd 0 8 > ./fedavg_standalone.txt 2>&1 &