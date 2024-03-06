#!/usr/bin/env fish
python main.py -m \
mode=source_pretrain \
indomain_test=false \
dataset=officehome \
    dataset.target_envs='[0]' \
    dataset.seed=0 \
training=nuc \
    training.batch_size=32 \
    training.optimizer.lr=5e-5 \
    training.nuc_scale=1e-2,1e-1 \
server=AMASK hydra/launcher=basic