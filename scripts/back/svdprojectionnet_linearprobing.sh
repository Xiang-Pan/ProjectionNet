#!/bin/bash
# 11/28
# python main.py -m \
# mode=source_pretrain \
# training=svdprojectionnet \
# training.feature_source='[y,s]' \
# training.batch_size=32,64,128 \
# training.optimizer.lr=1e-5,1e-4,1e-3 \
# server=AMASK hydra/launcher={$server}

python main.py -m \
mode=target_finetune \
pretrain=svdprojectionnet \
pretrain.feature_source='[y,s]' \
training.batch_size=32 \
pretrain.optimizer.lr=1e-4 \
pretrain.monitor=source_val_loss \
training=svdprojectionnet \
training.feature_source='[y]' \
training.batch_size=32 \
training.optimizer.lr=1e-4 \
target_fsl=0.0,0.1,0.2,0.4,0.6,0.8,1.0 \
seed=0 \
server=AMASK hydra/launcher={$server} \
debug=false &

python main.py -m \
mode=target_finetune \
pretrain=svdprojectionnet \
pretrain.feature_source='[y,s]' \
pretrain.batch_size=32 \
pretrain.optimizer.lr=1e-4 \
pretrain.monitor=source_val_loss \
training=svdprojectionnet \
training.feature_source='[y,s]' \
training.batch_size=32 \
training.optimizer.lr=1e-4 \
target_fsl=0.0,0.1,0.2,0.4,0.6,0.8,1.0 \
server=AMASK hydra/launcher={$server} \
seed=0 \
debug=false &

python main.py -m \
mode=target_finetune \
pretrain=svdprojectionnet \
pretrain.feature_source='[y,s]' \
pretrain.batch_size=32 \
pretrain.optimizer.lr=1e-4 \
pretrain.monitor=source_val_loss \
training=svdprojectionnet \
training.feature_source='[y,s,e]' \
training.batch_size=32 \
training.optimizer.lr=1e-4 \
target_fsl=0.0,0.1,0.2,0.4,0.6,0.8,1.0 \
server=AMASK hydra/launcher={$server} \
seed=0 \
debug=false &