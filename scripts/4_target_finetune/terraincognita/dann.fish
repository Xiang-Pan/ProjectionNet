#!/usr/bin/env fish
python main.py -m \
    mode=target_finetune \
    indomain_test=true \
    dataset=terraincognita \
        dataset.target_envs='[0]','[1]','[2]','[3]' \
        dataset.seed=0 \
    pretrain=dann \
        pretrain.monitor=source_val_acc \
    training=erm \
        training.batch_size=32 \
        training.optimizer.lr=5e-5 \
        training.seed='range(0,5)' \
        training.optimizer.feature_lr_scale=1 \
    fsl=0.1,0.2,0.4,0.6,0.8,1.0 \
    server={$server} hydra/launcher={$server}