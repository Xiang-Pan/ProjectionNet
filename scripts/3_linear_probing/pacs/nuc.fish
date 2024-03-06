#!/usr/bin/env fish
python main.py -m \
    mode=linear_probing \
    dataset=pacs \
        dataset.target_envs='[0]','[1]','[2]','[3]' \
        dataset.seed=0 \
    pretrain=nuc \
        pretrain.batch_size=32 \
        pretrain.optimizer.lr=5e-5 \
        pretrain.nuc_scale=1e-2,1e-1 \
    training=linear_probing \
        training.feature_source='[base_feature]' \
        training.domain=target \
        training.seed='range(0,5)' \
    fsl=0.1,0.2,0.4,0.6,0.8,1.0 \
    server=AMASK hydra/launcher={$server}_cpu