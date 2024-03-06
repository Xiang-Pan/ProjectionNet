#!/usr/bin/env fish
set -l algorithms "DANN"
set -l datasets "OfficeHome" "VLCS" "PACS" "TerraIncognita"

python -m domainbed.scripts.sweep delete_incomplete \
       --data_dir=AMASK/AMASK/data \
       --output_dir=./outputs/ \
       --command_launcher slurm \
       --datasets $datasets \
       --single_test_envs \
       --algorithms $algorithms \
       --n_hparams 1 \
       --n_trials 5 \
       --skip_confirmation

python -m domainbed.scripts.sweep launch \
       --data_dir=AMASK/AMASK/data \
       --output_dir=./outputs/ \
       --command_launcher slurm \
       --datasets $datasets \
       --single_test_envs \
       --algorithms $algorithms \
       --n_hparams 1 \
       --n_trials 5 \
       --skip_confirmation