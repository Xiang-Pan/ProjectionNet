
import argparse
import bisect
import hydra
from omegaconf import OmegaConf
from libs.DomainBed.domainbed.datasets import OfficeHome, DomainNet, VLCS, PACS, TerraIncognita
from libs.DomainBed.domainbed.lib import misc
from libs.DomainBed.domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader
from torch.utils.data import DataLoader, Dataset
class MultiDomainDataset(Dataset):
    def __init__(self, datasets, env_idxs=None):
        self.datasets = datasets
        self.lengths = [len(d) for d in self.datasets]
        if env_idxs is None:
            self.env_idxs = [i for i, d in enumerate(self.datasets) for _ in range(len(d))]
        else:
            self.env_idxs = env_idxs
            
    def find_dataset(self, index, lengths):
        # Create a list of cumulative sums of lengths
        cumulative_lengths = [sum(lengths[:i+1]) for i in range(len(lengths))]
        
        # Use bisect to find the position where index would fit in the cumulative list
        dataset_index = bisect.bisect(cumulative_lengths, index)
        
        # Adjust for Python's 0-based indexing
        return dataset_index if dataset_index < len(lengths) else -1


    def __len__(self):
        return sum(len(d) for d in self.datasets)

    def __getitem__(self, i):
        dataset_idx = self.find_dataset(i, self.lengths)
        data_idx = i - sum(self.lengths[:dataset_idx])
        x = self.datasets[dataset_idx][data_idx][0]
        y = self.datasets[dataset_idx][data_idx][1]
        e = self.env_idxs[dataset_idx]
        return x, y, e
    
# from libs.DomainBed.domainbed.datasets import split_dataset


def get_dataset(cfg):
    name = cfg.dataset.name
    if name == "officehome":
        C = OfficeHome
    elif name == "domainnet":
        C = DomainNet
    elif name == "vlcs":
        C = VLCS
    elif name == "pacs":
        C = PACS
    elif name == "terraincognita":
        C = TerraIncognita
    else:
        raise ValueError("Unknown dataset %s" % name)
        
    # hparams = {}
    # hparams["holdout_fraction"] = 0.2
    # hparams["data_augmentation"] = True
    # hparams["seed"] = 1
    # hparams["target_envs"] = [0]
    # hparams["uda_holdout_fraction"] = 0
    # hparams["class_balanced"] = False
    # hparams["batch_size"] = 64
    
    dataset_config_dict = OmegaConf.to_container(cfg.dataset, resolve=True)
    dataset_config_dict["seed"] = cfg.dataset.seed
    dataset = C(root=cfg.server.data_root_folder,
                        test_envs=cfg.dataset.target_envs,
                        hparams=dataset_config_dict)
    in_splits = []
    out_splits = []
    uda_splits = []
    
    for env_i, env in enumerate(dataset):
        uda = []

        out, in_ = misc.split_dataset(env,
            int(len(env)*cfg.dataset.holdout_fraction),
            misc.seed_hash(cfg.dataset.seed, env_i))

        if env_i in cfg.dataset.target_envs:
            uda, in_ = misc.split_dataset(in_,
                int(len(in_)*cfg.dataset.uda_holdout_fraction),
                misc.seed_hash(cfg.dataset.seed, env_i))

        if cfg.dataset.class_balanced:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
            if uda is not None:
                uda_weights = misc.make_weights_for_balanced_classes(uda)
        else:
            in_weights, out_weights, uda_weights = None, None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))
        if len(uda):
            uda_splits.append((uda, uda_weights))
            
    test_splits = []
    if cfg.mode == "source_pretrain":
        assert cfg.indomain_test is False
    if cfg.indomain_test:
        # logger.info("!!! In-domain test mode On !!!")
        # assert hparams["val_augment"] is False, (
        #     "indomain_test split the val set into val/test sets. "
        #     "Therefore, the val set should be not augmented."
        # )
        val_splits = []
        for env_i, (out_split, _weights) in enumerate(out_splits):
            n = len(out_split) // 2
            seed = misc.seed_hash(cfg.dataset.seed, env_i)
            val_split, test_split = misc.split_dataset(out_split, n, seed=misc.seed_hash(cfg.dataset.seed, env_i))
            val_splits.append((val_split, None))
            test_splits.append((test_split, None))
            # logger.info(
            #     "env %d: out (#%d) -> val (#%d) / test (#%d)"
            #     % (env_i, len(out_split), len(val_split), len(test_split))
            # )
        out_splits = val_splits
            

    in_splits = [s[0] for s in in_splits]
    out_splits = [s[0] for s in out_splits]
    test_splits = [s[0] for s in test_splits]
    uda_splits = [s[0] for s in uda_splits]


    target_envs = cfg.dataset.target_envs
    source_envs = [env_i for env_i, env in enumerate(dataset) if env_i not in target_envs]
    print("source_envs", source_envs)
    print("target_envs", target_envs)
    source_train_datasets = [env for env_i, env in enumerate(in_splits) if env_i not in target_envs]
    source_val_datasets = [env for env_i, env in enumerate(out_splits) if env_i not in target_envs]
    source_test_datasets = [env for env_i, env in enumerate(test_splits) if env_i not in target_envs]
    
    target_train_datasets = [env for env_i, env in enumerate(in_splits) if env_i in target_envs]
    target_val_datasets = [env for env_i, env in enumerate(out_splits) if env_i in target_envs]
    target_test_datasets = [env for env_i, env in enumerate(test_splits) if env_i in target_envs]
    
    
    source_train_dataset = MultiDomainDataset(source_train_datasets, source_envs)
    source_val_dataset = MultiDomainDataset(source_val_datasets, source_envs)
    target_train_dataset = MultiDomainDataset(target_train_datasets, target_envs)
    target_val_dataset = MultiDomainDataset(target_val_datasets, target_envs)
    source_test_dataset = MultiDomainDataset(source_test_datasets, source_envs)
    target_test_dataset = MultiDomainDataset(target_test_datasets, target_envs)
    dataset_dict = {
        "source_train_dataset": source_train_dataset,
        "source_val_dataset": source_val_dataset,
        "target_train_dataset": target_train_dataset,
        "target_val_dataset": target_val_dataset,
        "source_test_dataset": source_test_dataset,
        "target_test_dataset": target_test_dataset,
    }
    return dataset_dict

@hydra.main(config_path="configs", config_name="default", version_base="1.3.0")
def test(cfg):
    dataset_dict = get_dataset(cfg)
    print(dataset_dict.keys())
    target_envs = cfg.dataset.target_envs
    print("target_envs", target_envs)
    for k, v in dataset_dict.items():
        print(k, len(v))
        
    # for split in dataset_dict:
        # print(dataset_dict[split][0][0].shape, dataset_dict[split][0][1], dataset_dict[split].env_idxs)
        # for i in range(len(dataset_dict[split])):
        #     print(dataset_dict[split][i][0].shape, dataset_dict[split][i][1], dataset_dict[split][i][2])
        # print("====================================")
    
    
if __name__ == "__main__":
    test()
    # hparams = {}
    # hparams["holdout_fraction"] = 0.2
    # hparams["data_augmentation"] = True
    # hparams["seed"] = 1
    # hparams["target_envs"] = [0]
    # hparams["uda_holdout_fraction"] = 0
    # hparams["class_balanced"] = False
    # hparams["batch_size"] = 64
    # args = argparse.Namespace(**hparams)
    
    
    # dataset = get_dataset(hparams)




