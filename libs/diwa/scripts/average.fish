#!/usr/bin/env fish
set -l data_dir AMASK/AMASK/data
set -l steps_list -1 0
set -l test_env_list 0 1 2 3
set -l diwa_list restricted uniform
set -l trial_seed_list 0 1 2
# set -l dataset_list OfficeHome VLCS DomainNet PACS TerraIncognita
set -l dataset_list VLCS PACS TerraIncognita

for dataset in $dataset_list
    for steps in $steps_list
        for test_env in $test_env_list
            for diwa in $diwa_list
                for trial_seed in $trial_seed_list
                    set -l path_for_init outputs/{$dataset}/ERM/{$test_env}/{$steps}
                    set -l init_parent_path (dirname $path_for_init)
                    set -l output_dir train_output/{$dataset}/ERM/{$test_env}/{$steps}
                    set -l weight_selection $diwa
                    python -m domainbed.scripts.diwa\
                        --data_dir={$data_dir} \
                        --output_dir={$output_dir} \
                        --dataset {$dataset}\
                        --test_env {$test_env}\
                        --weight_selection {$weight_selection} \
                        --trial_seed {$trial_seed}
                end
            end
        end
    end
end
