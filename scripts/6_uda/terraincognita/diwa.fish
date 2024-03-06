#!/usr/bin/env fish
python main.py -m \
    mode=target_finetune \
    indomain_test=true \
    dataset=terraincognita \
        dataset.target_envs='[0]','[1]','[2]','[3]' \
        dataset.seed=0 \
    pretrain=diwa \
        pretrain.init_method=-1 \
        pretrain.weight_average_method=uniform \
    training=diwa \
        training.batch_size=32 \
        training.optimizer.lr=5e-5 \
        training.seed=0,1,2,3,4 \
        training.optimizer.feature_lr_scale=1 \
    fsl=0.0 \
    server={$server} hydra/launcher={$server}