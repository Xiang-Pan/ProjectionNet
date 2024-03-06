#!/usr/bin/env fish
#11/27
# set cluster=short_hostname
echo "short_hostname: $short_hostname"
# python main.py -m \
# mode=target_finetune \
# training=projectionnet_linearprobing training.batch_size=64 \
# 'training.feature_source=[y]' \
# pretrain=projectionnet pretrain.batch_size=64 'pretrain.feature_source=[y, s]' \
# target_fsl=0.0,0.1,0.2,0.4,0.6,0.8,1.0 \
# server=$short_hostname hydra/launcher=$short_hostname

# python main.py -m \
# mode=target_finetune \
# training=projectionnet_linearprobing training.batch_size=64 \
# 'training.feature_source=[y,s]' \
# pretrain=projectionnet pretrain.batch_size=64 'pretrain.feature_source=[y, s]' \
# target_fsl=0.0,0.1,0.2,0.4,0.6,0.8,1.0 \
# server=$short_hostname hydra/launcher=$short_hostname

python main.py -m \
mode=target_finetune \
training=projectionnet_linearprobing \
training.batch_size=32 'training.feature_source=[y],[y,z],[y,s,e]' training.optimizer.lr=1e-4 \
pretrain=projectionnet \
'pretrain.feature_source=[y, s]' \
pretrain.batch_size=32 pretrain.optimizer.lr=1e-4 \
target_fsl=0.0,0.1,0.2,0.4,0.6,0.8,1.0 \
server=$short_hostname hydra/launcher=$short_hostname