import torch
import hydra
import pandas as pd
import os


def assert_file(path):
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    os.system(f"rsync -avzrP ai:AMASK/AMASK/AMASKPATH/{path} {os.path.dirname(path)}")
    os.system(f"rsync -avzrP MYMACHINE:/mnt/AMASK/AMASKPATH/{path} {os.path.dirname(path)}")
    return torch.load(path, map_location=map_location)



@hydra.main(config_path='configs', config_name='default', version_base="1.3.0")
def get_best_model_path_domainbed(cfg):
    output_path = "./libs/DomainBed/outputs/"
    summary_path = f"{output_path}/summary.pkl"
    assert_file(summary_path)
    
    summary = pd.read_pickle(summary_path)
    # get the model name
    algorithm = cfg.pretrain.method
    dataset = cfg.dataset.name
    test_env = cfg.dataset.target_envs
    trial_seed = cfg.dataset.seed
    print(summary)
    selected_row = summary[(summary['algorithm'] == algorithm)]
    selected_row = selected_row[(selected_row['dataset'] == dataset)]
    selected_row = selected_row[(selected_row['test_env'] == test_env)]
    selected_row = selected_row[(selected_row['trial_seed'] == trial_seed)]
    return selected_row['selection_best_model_path'].values[0]


if __name__ == '__main__':
    get_best_model_path_domainbed()