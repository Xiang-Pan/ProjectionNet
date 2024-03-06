#!/usr/bin/env fish
python main.py -m \
    mode=feature_extraction \
    dataset=officehome \
        dataset.target_envs='[0]','[1]','[2]','[3]' \
        dataset.seed=0 \
    pretrain=diwa \
        pretrain.init_method=-1 \
        pretrain.weight_average_method=uniform \
    server={$server} hydra/launcher={$server}