#!/usr/bin/env fish
python main.py -m \
    mode=feature_extraction \
    dataset=terraincognita \
        dataset.target_envs='[0]','[1]','[2]','[3]' \
        dataset.seed=0 \
    pretrain=nuc \
        pretrain.batch_size=32 \
        pretrain.optimizer.lr=5e-5 \
        pretrain.nuc_scale=1e-2,1e-1 \
    server={$server} hydra/launcher={$server}