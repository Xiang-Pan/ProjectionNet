#!/usr/bin/env fish
set -l data_dir AMASK/AMASK/data
set -l steps_list -1 0
set -l test_env_list 0 1 2 3
set -l dataset_list OfficeHome VLCS PACS TerraIncognita

for dataset in $dataset_list
    for steps in $steps_list
        for test_env in $test_env_list
            set -l path_for_init outputs/{$dataset}/ERM/{$test_env}/{$steps}
            set -l init_parent_path (dirname $path_for_init)
            set -l output_dir train_output/{$dataset}/ERM/{$test_env}/{$steps}
            mkdir -p $init_parent_path
            echo $path_for_init
            set -l cmd
            python -m domainbed.scripts.sweep delete_incomplete\
                --data_dir={$data_dir} \
                --output_dir={$output_dir} \
                --command_launcher slurm \
                --datasets {$dataset} \
                --test_env {$test_env} \
                --path_for_init {$path_for_init} \
                --algorithms ERM \
                --n_hparams 20 \
                --skip_confirmation \
                --n_trials 1
            python -m domainbed.scripts.sweep launch\
                --data_dir={$data_dir} \
                --output_dir={$output_dir} \
                --command_launcher slurm \
                --datasets {$dataset} \
                --test_env {$test_env} \
                --path_for_init {$path_for_init} \
                --algorithms ERM \
                --n_hparams 20 \
                --skip_confirmation \
                --n_trials 1
        end
    end
end
