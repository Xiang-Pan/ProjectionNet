#!/usr/bin/env fish
set -l steps_list -1 0
set -l test_env_list 0 1 2 3
# set -l dataset_list VLCS DomainNet PACS TerraIncognita
set -l dataset_list DomainNet
for dataset in $dataset_list
    for steps in $steps_list
        for test_env in $test_env_list
            set -l path_for_init outputs/{$dataset}/ERM/{$test_env}/{$steps}
            set -l parent_path (dirname $path_for_init)
            # check  if path_for_init exists
            if test -e $path_for_init
                echo "path_for_init exists"
                continue
            else
                echo "path_for_init does not exist"
            end
            set -l output_dir train_output/{$dataset}/ERM/{$test_env}/{$steps}
            mkdir -p $parent_path
            echo $path_for_init
            set -l cmd
            python3 -m domainbed.scripts.train \
                --data_dir=AMASK/AMASK/data \
                --algorithm ERM \
                --dataset {$dataset} \
                --test_env {$test_env} \
                --init_step \
                --path_for_init {$path_for_init} \
                --steps {$steps} \
                --output_dir {$output_dir}
        end
    end
end