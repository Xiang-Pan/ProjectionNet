#!/usr/bin/env fish
python main.py -m \
mode=source_pretrain \
training=svdprojectionnet \
training.feature_source='[y,s]' \
training.batch_size=32 \
training.optimizer.lr=5e-5 \
server=AMASK hydra/launcher={$server} \

# python main.py -m \
# mode=source_pretrain \
# training=svdprojectionnet \
# training.feature_source='[y,s]' \
# training.batch_size=32 \
# training.optimizer.lr=1e-5 \
# server=AMASK hydra/launcher=basic \
# debug=true