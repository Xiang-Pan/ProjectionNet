#!/usr/bin/env fish
python main.py -m \
mode=source_pretrain \
dataset=vlcs \
    dataset.target_envs='[0]','[1]','[2]','[3]' \
    dataset.seed=0 \
training=erm \
    training.batch_size=32 \
    training.optimizer.lr=5e-5 \
    training.seed="range(0,5)"
server=AMASK hydra/launcher={$server}