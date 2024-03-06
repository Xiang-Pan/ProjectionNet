#!/usr/bin/env fish
python main.py -m \
mode=source_pretrain \
dataset=vlcs \
    dataset.target_envs='[0]','[1]','[2]','[3]' \
    dataset.seed=0 \
training=projectionnet_optimization \
    training.feature_source='[y,s]' \
    training.batch_size=64 \
    training.optimizer.lr=5e-5 \
server=AMASK hydra/launcher={$server}