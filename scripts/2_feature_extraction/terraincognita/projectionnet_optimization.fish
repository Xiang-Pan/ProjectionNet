#!/usr/bin/env fish
python main.py -m \
    mode=feature_extraction \
    dataset=terraincognita \
        dataset.target_envs='[0]','[1]','[2]','[3]' \
        dataset.seed=0 \
    pretrain=projectionnet_optimization \
        pretrain.method=projectionnet \
        pretrain.feature_source='[y,s]' \
        pretrain.batch_size=64 \
        pretrain.optimizer.lr=5e-5 \
        pretrain.monitor=source_val_acc \
    server={$server} hydra/launcher={$server}