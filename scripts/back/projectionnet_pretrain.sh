#!/usr/bin/env fish
python main.py -m \
mode=source_pretrain \
training=svdprojectionnet \
training.feature_source='[y,s]' \
training.batch_size=64 \
server={$server} hydra/launcher=basic