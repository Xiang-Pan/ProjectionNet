#!/usr/bin/env fish

python main.py -m \
    mode=feature_extraction \
    dataset=officehome \
        dataset.target_envs='[0]','[1]','[2]','[3]' \
        dataset.seed=0 \
    pretrain=dann \
        pretrain.monitor=source_val_acc \
    server={$server} hydra/launcher={$server}