#!/usr/bin/env fish
# get params from command line
# set mode $argv[1]
# if test $mode = "local"
#     set hydra/launcher "basic"
# else if test $mode = "ai"
#     set hydra/launcher "ai"
# else
#     echo "wrong mode"
#     exit 1
# end
#TODO: terraincognita run
python main.py -m \
    mode=feature_extraction \
    dataset=terraincognita \
        dataset.target_envs='[0]','[1]','[2]','[3]' \
        dataset.seed=0 \
    pretrain=dann \
        pretrain.monitor=source_val_acc \
    server={$server} hydra/launcher={$server}