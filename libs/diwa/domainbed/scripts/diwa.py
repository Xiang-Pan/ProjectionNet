import argparse
import os
import json
import random
import numpy as np
import torch
import torch.utils.data
from domainbed import datasets, algorithms_inference
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import FastDataLoader
from domainbed.lib import misc


def _get_args():
    parser = argparse.ArgumentParser(description='Domain generalization')

    parser.add_argument('--dataset', type=str)
    parser.add_argument('--test_env', type=int)

    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--data_dir', type=str, default="default")

    # select which checkpoints
    parser.add_argument('--weight_selection', type=str, default="uniform") # or "restricted"
    parser.add_argument(
        '--trial_seed',
        type=int,
        default="-1",
        help='Trial number (used for seeding split_dataset and random_hparams).'
    )

    inf_args = parser.parse_args()
    return inf_args


def create_splits(domain, inf_args, dataset, _filter):
    splits = []

    for env_i, env in enumerate(dataset):
        if domain == "test" and env_i != inf_args.test_env:
            continue
        elif domain == "train" and env_i == inf_args.test_env:
            continue

        if _filter == "full":
            splits.append(env)
        else:
            out_, in_ = misc.split_dataset(
                env, int(len(env) * 0.2), misc.seed_hash(inf_args.trial_seed, env_i)
            )
            if _filter == "in":
                splits.append(in_)
            elif _filter == "out":
                splits.append(out_)
            else:
                raise ValueError(_filter)

    return splits


def get_dict_folder_to_score(inf_args):
    output_folders = [
        os.path.join(output_dir, path)
        for output_dir in inf_args.output_dir.split(",")
        for path in os.listdir(output_dir)
    ]
    output_folders = [
        output_folder for output_folder in output_folders
        if os.path.isdir(output_folder) and "done" in os.listdir(output_folder) and "model_best.pkl" in os.listdir(output_folder)
    ]

    dict_folder_to_score = {}
    for folder in output_folders:
        model_path = os.path.join(folder, "model_best.pkl")
        save_dict = torch.load(model_path)
        train_args = save_dict["args"]

        if train_args["dataset"] != inf_args.dataset:
            continue
        if train_args["test_envs"] != [inf_args.test_env]:
            continue
        if train_args["trial_seed"] != inf_args.trial_seed and inf_args.trial_seed != -1:
            continue
        score = misc.get_score(
            json.loads(save_dict["results"]),
            [inf_args.test_env])

        print(f"Found: {folder} with score: {score}")
        dict_folder_to_score[folder] = score

    if len(dict_folder_to_score) == 0:
        raise ValueError(f"No folders found for: {inf_args}")
    return dict_folder_to_score

def get_wa_results(
    good_checkpoints, dataset, data_names, data_splits, device, weight_selection_name
):
    wa_algorithm = algorithms_inference.DiWA(
        dataset.input_shape,
        dataset.num_classes,
        len(dataset) - 1,
    )
    for folder in good_checkpoints:
        save_dict = torch.load(os.path.join(folder, "model_best.pkl"))
        train_args = save_dict["args"]

        # load individual weights
        algorithm = algorithms_inference.ERM(
            dataset.input_shape, dataset.num_classes,
            len(dataset) - 1,
            save_dict["model_hparams"]
        )
        algorithm.load_state_dict(save_dict["model_dict"], strict=False)
        wa_algorithm.add_weights(algorithm.network)
        del algorithm

    wa_algorithm.to(device)
    wa_algorithm.eval()
    random.seed(train_args["seed"])
    np.random.seed(train_args["seed"])
    torch.manual_seed(train_args["seed"])
    torch.backenAMASKcudnn.deterministic = True
    torch.backenAMASKcudnn.benchmark = False

    data_loaders = [
        FastDataLoader(
            dataset=split,
            batch_size=64,
            num_workers=dataset.N_WORKERS
        ) for split in data_splits
    ]

    data_evals = zip(data_names, data_loaders)
    dict_results = {}

    for name, loader in data_evals:
        print(f"Inference at {name}")
        dict_results[name + "_acc"] = misc.accuracy(wa_algorithm, loader, None, device)

    dict_results["length"] = len(good_checkpoints)
    return dict_results, wa_algorithm



def print_results(dict_results):
    results_keys = sorted(list(dict_results.keys()))
    misc.print_row(results_keys, colwidth=12)
    misc.print_row([dict_results[key] for key in results_keys], colwidth=12)


def main():
    inf_args = _get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Begin DiWA for: {inf_args} with device: {device}")

    if inf_args.dataset in vars(datasets):
        dataset_class = vars(datasets)[inf_args.dataset]
        dataset = dataset_class(
            inf_args.data_dir, [inf_args.test_env], hparams={"data_augmentation": False}
        )
    else:
        raise NotImplementedError

    # load individual folders and their corresponding scores on train_out
    dict_folder_to_score = get_dict_folder_to_score(inf_args)

    # load data: test and optionally train_out for restricted weight selection
    data_splits, data_names = [], []
    dict_domain_to_filter = {"test": "full"}
    if inf_args.weight_selection == "restricted":
        assert inf_args.trial_seed != -1
        dict_domain_to_filter["train"] = "out"
    for domain in dict_domain_to_filter:
        _data_splits = create_splits(domain, inf_args, dataset, dict_domain_to_filter[domain])
        if domain == "train":
            data_splits.append(misc.MergeDataset(_data_splits))
        else:
            data_splits.append(_data_splits[0])
        data_names.append(domain)


    # compute score after weight averaging
    if inf_args.weight_selection == "restricted":
        # Restricted weight selection

        ## sort individual members by decreasing accuracy on train_out
        sorted_checkpoints = sorted(dict_folder_to_score.keys(), key=lambda x: dict_folder_to_score[x], reverse=True)
        selected_indexes = []
        best_result = -float("inf")
        dict_best_results = {}
        ## incrementally add them to the WA
        for i in range(0, len(sorted_checkpoints)):
            selected_indexes.append(i)
            selected_checkpoints = [sorted_checkpoints[index] for index in selected_indexes]

            ood_results, wa_algorithm = get_wa_results(
                selected_checkpoints, dataset, data_names, data_splits, device, weight_selection_name="restricted"
            )
            ood_results["i"] = i
            ## accept only if WA's accuracy is improved
            if ood_results["train_acc"] >= best_result:
                dict_best_results = ood_results
                ood_results["accept"] = 1
                best_result = ood_results["train_acc"]
                print(f"Accepting index {i}")
                # save the model as the best model
                best_wa_algorithm = wa_algorithm

            else:
                ood_results["accept"] = 0
                selected_indexes.pop(-1)
                print(f"Skipping index {i}")
            print_results(ood_results)

        ## print final scores
        dict_best_results["final"] = 1
        print_results(dict_best_results)
        save_dir = f"{inf_args.output_dir}/trial_seed={inf_args.trial_seed}/"
        os.makedirs(save_dir, exist_ok=True)
        torch.save(best_wa_algorithm.state_dict(), f"{save_dir}/restricted.pt")

    elif inf_args.weight_selection == "uniform":
        dict_results, wa_algorithm = get_wa_results(
            list(dict_folder_to_score.keys()), dataset, data_names, data_splits, device,
            weight_selection_name="uniform"
        )
        print_results(dict_results)
        save_dir = f"{inf_args.output_dir}/trial_seed={inf_args.trial_seed}/"
        os.makedirs(save_dir, exist_ok=True)
        torch.save(wa_algorithm.state_dict(), f"{save_dir}/uniform.pt")

    else:
        raise ValueError(inf_args.weight_selection)
    # save the model 


if __name__ == "__main__":
    main()
