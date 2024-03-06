#!/usr/bin/env fish
python main.py -m \
    mode=target_finetune \
    indomain_test=true \
    dataset=vlcs \
        dataset.target_envs='[0]','[1]','[2]','[3]' \
        dataset.seed=0 \
    pretrain=dann \
        pretrain.monitor=source_val_acc \
    training=erm \
        training.batch_size=32 \
        training.optimizer.lr=5e-5 \
        training.seed=0,1,2,3,4 \
        training.optimizer.feature_lr_scale=1 \
    fsl=0.0 \
    server={$server} hydra/launcher={$server}