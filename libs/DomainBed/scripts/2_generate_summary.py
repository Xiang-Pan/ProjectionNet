
import os
import torch
import numpy as np
import pandas as pd
import yaml


output_path = "./outputs/"
selection = "source_val_acc"
df = pd.DataFrame()
for child in os.listdir(output_path):
    if os.path.isfile(output_path + child):
        continue
    try:
        results = pd.read_json(output_path + child + "/results.jsonl", lines=True)
    except:
        continue
    # print(results)
    args = results["args"].iloc[0]
    test_envs = args["test_envs"]
    test_env = test_envs[0]

    envs = [c for c in results.iloc[0].index if "env" in c]
    envs = [c.split("_")[0].replace("env", "") for c in envs]
    envs = list(set(envs))
    train_envs = [c for c in envs if c not in test_envs]
    algorithm = args["algorithm"]
    dataset = args["dataset"]
    test_envs = args["test_envs"]
    trial_seed = args["trial_seed"]
    if selection == "source_val_acc":
        # get the best source val acc step
        best_source_val_acc = 0
        best_step_idx = 0
        for row_idx, row in results.iterrows():
            source_val_acc = np.mean([row[f"env{env}_in_acc"] for env in train_envs])
            if source_val_acc > best_source_val_acc:
                best_source_val_acc = source_val_acc
                best_step_idx = row_idx
        step_lists = [f for f in os.listdir(output_path + child) if "step" in f]
        best_step = int(step_lists[best_step_idx].replace("model_step", "").replace(".pkl", ""))
        # output_path = os.path.abspath(output_path)
        best_model_path = output_path + child + "/model_step" + str(best_step) + ".pkl"
        # output the source_val_acc.yaml
        best_source_val_acc = float(best_source_val_acc)
        save_dict = {
            "algorithm": algorithm.lower(),
            "dataset": dataset.lower(),
            "test_env": test_env,
            "trial_seed": trial_seed,
            "step": best_step,
            "selection": {
                "method": "source_val_acc",
                "value": best_source_val_acc,
                "best_model_path": best_model_path,
            }, 
        }
        flatten_dict = {}
        for k, v in save_dict.items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    flatten_dict[k + "_" + k2] = v2
            else:
                flatten_dict[k] = v
        append_df = pd.DataFrame(flatten_dict, index=[0])
        df = pd.concat([df, append_df], ignore_index=True)
        with open(output_path + child + "/source_val_acc.yaml", "w") as f:
            yaml.dump(save_dict, f)
df.to_csv(output_path + "summary.csv", index=False)
df.to_pickle(output_path + "summary.pkl")


summary = pd.read_pickle(output_path + "summary.pkl")


