#!/usr/bin/env fish
python main.py -m \
    mode=linear_probing \
    dataset=terraincognita \
        dataset.target_envs='[0]','[1]','[2]','[3]' \
        dataset.seed=0 \
    pretrain=projectionnet_optimization \
        pretrain.feature_source='[y,s]' \
        pretrain.batch_size=64 \
        pretrain.optimizer.lr=5e-5 \
        pretrain.monitor=source_val_acc \
    training=linear_probing \
        training.feature_source='[base_feature]','[y_projected]','[y_projected,s_projected]','[y_projected,s_projected,e_projected]','[base_feature,y_projected,s_projected,e_projected]','[base_feature,y_projected]' \
        training.domain=target \
        training.seed='range(0,5)' \
    fsl=0.1,0.2,0.4,0.6,0.8,1.0 \
    server={$server} hydra/launcher={$server}_cpu
