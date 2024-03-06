#!/usr/bin/env fish
python main.py -m \
    mode=target_finetune \
    indomain_test=true \
    dataset=officehome \
        dataset.target_envs='[0]','[1]','[2]','[3]' \
        dataset.seed=0 \
    pretrain=projectionnet_optimization \
        pretrain.monitor=source_val_acc \
        pretrain.batch_size=64 \
        pretrain.optimizer.lr=5e-5 \
        pretrain.feature_source='[y,s]' \
    training=projectionnet_optimization \
        training.monitor=source_val_acc \
        training.batch_size=64 \
        training.optimizer.lr=5e-5 \
        training.ye_orth_reg_scale=0 \
        training.ys_orth_reg_scale=0 \
        training.feature_source='[y]','[y,s]','[y,s,e]' \
        training.seed='range(0,5)' \
        training.optimizer.feature_lr_scale=1 \
    fsl=0.1,0.2,0.4,0.6,0.8,1.0 \
    server={$server} hydra/launcher={$server}