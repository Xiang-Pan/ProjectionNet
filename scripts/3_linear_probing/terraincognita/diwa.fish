#!/usr/bin/env fish
python main.py -m \
    mode=linear_probing \
    dataset=terraincognita \
        dataset.target_envs='[0]','[1]','[2]','[3]' \
        dataset.seed=0 \
    pretrain=diwa \
        pretrain.init_method=-1 \
        pretrain.weight_average_method=uniform \
    training=linear_probing \
        training.feature_source='[base_feature]' \
        training.domain=target \
        training.seed='range(0,5)' \
    fsl=0.1,0.2,0.4,0.6,0.8,1.0 \
    server={$server} hydra/launcher={$server}_cpu