# MKL_THREADING_LAYER=GNU
import os
import shutil

import signal
s3_prefix = ""
# import sshfs
# from sshfs import SSHFileSystem
# remote = SSHFileSystem(
#     host='AMASK.asuscomm.com',
#     port=23,
#     username='AMASK',
#     password='AMASK',
# )

from lightning.pytorch.plugins.io import AsyncCheckpointIO
from pathlib import Path
import numpy as np
import socket
import lightning as pl
import torch
import wandb
import hydra
from os import path
import yaml
import pickle as pkl
from domainbed_utils import get_best_model_path_domainbed
from torch.utils.data import random_split
import re
import seaborn as sns
import matplotlib.pyplot as plt
from omegaconf import OmegaConf, open_dict
import torchmetrics
from lightning.pytorch.callbacks import StochasticWeightAveraging
import torch.nn as nn
from itertools import product
# from pytorch_adapt.datasets import get_officehome
# from pytorch_adapt.datasets import OfficeHome
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
from typing import Optional
from torchvision.models import resnet50, resnet18
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import LightningDataModule, LightningModule
# from pytorch_adapt.datasets import get_officehome
from torch.utils.data import DataLoader, Dataset
from linear_svd import *
from domainbed_dataset import get_dataset

def assert_file(path):
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    os.system(f"rsync -avzrP ai:AMASK/AMASK/AMASKPATH/{path} {os.path.dirname(path)}")
    os.system(f"rsync -avzrP MYMACHINE:/mnt/AMASK/AMASKPATH/{path} {os.path.dirname(path)}")
    return 

def myload(path, map_location="cpu"):
    assert_file(path)
    logger.info(f"Loading from {path}")
    if "mnt" in path:
        logger.debug(f"Loading from remote: {path}")
        with remote.open(path, "rb") as f:
            return torch.load(f, map_location=map_location)
    else:
        return torch.load(path, map_location=map_location)


import logging
logger = logging.getLogger(f"./logs/{__name__}.log")
logger.setLevel(logging.DEBUG)


def myhash(cfg):
    # ignore all the default, null, none, empty
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg = {k: v for k, v in cfg.items() if v is not None and v != "null" and v != ""}
    return hash(tuple(sorted(cfg.items())))

#* TODO: fixme
def mylistdir(path):
    print(path)
    if os.path.exists(path):
        ckpt_files = [file for file in os.listdir(path) if file.endswith(".ckpt")]
        if len(ckpt_files) > 0:
            return os.listdir(path)
    
    pieces = path.split("/")
    pieces = [piece for piece in pieces if piece != ""]
    path = "/".join(pieces)
    
    # listdir = remote.ls(f"/mnt/AMASK/AMASKPATH/{path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    source_path = fr"MYMACHINE:/mnt/AMASK/AMASKPATH/{path}/".replace("[", r"\\[").replace("]", r"\\]").replace("'", r"\'").replace(" ", "' '")
    os.makedirs(path, exist_ok=True)
    target_path = path.replace("[", "").replace("]", "").replace("'", "").replace(" ", "")
    os.makedirs(target_path, exist_ok=True)
    cmd = f"rsync -azP --ignore-existing {source_path} {target_path}"
    os.system(cmd)
    logger.debug(cmd)
    source_path = fr"ai:AMASK/AMASK/AMASKPATH/{path}/".replace("[", r"\\[").replace("]", r"\\]").replace("'", r"\'").replace(" ", "' '")
    target_path = path.replace("[", "").replace("]", "").replace("'", "").replace(" ", "")
    cmd = f"rsync -azP --ignore-existing {source_path} {target_path}"
    logger.debug(cmd)
    os.system(cmd)
    for entry in os.listdir(target_path):
        if os.path.isfile(f"{target_path}/{entry}"):
            shutil.move(f"{target_path}/{entry}", f"{path}/{entry}")
        elif os.path.isdir(f"{target_path}/{entry}"):
            shutil.rmtree(f"{path}/{entry}", ignore_errors=True)
            shutil.move(f"{target_path}/{entry}", f"{path}/{entry}")
    return os.listdir(path)
    # 


def get_best_model_path(output_dir, monitor):
    # output_pieces = output_dir.split("/")
    # output_dir = "/".join(output_pieces)
    # files = os.listdir(output_dir)
    files = mylistdir(output_dir)
    files = [file for file in files if file.endswith(".ckpt")]
    # contain monitor
    logger.debug(monitor)
    files = [file for file in files if re.search(f"{monitor}=[0-9\.]+", file) is not None]
    logger.debug(files)
    # get the monitor value but not the .ckpt
    monitor_values = [re.search(f"{monitor}=([0-9\.]+)", file).group(1) for file in files]
    monitor_values = [float(value[:-1]) if value.endswith(".") else float(value) for value in monitor_values]
    if "acc" in monitor:
        best = max(monitor_values)
    else:
        best = min(monitor_values)
    # get the 
    best_file = [file for file in files if re.search(f"{monitor}={best}", file) is not None][0]
    
    # source_path = output_dir.replace("[", r"\\\[").replace("]", r"\\\]").replace("'", r"\'").replace(" ", "' '")
    # target_path = f"./outputs" + str(hash(source_path))
    # source_file = f"MYMACHINE:/mnt/AMASK/labs/{source_path}/"
    # target_file = target_path + "/" + best_file
    # cmd = f"rsync -azP {source_file} {target_path}"
    # os.system(cmd)
    # import shutil
    # shutil.move(target_file, output_dir)
    
    return f"{output_dir}/{best_file}"

    
class DataModule(LightningDataModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.training.batch_size
        # seed = cfg.dataset.seed
        # souce_domains = cfg.dataset.source_domains
        # traget_domains = cfg.dataset.target_domains
        # full_domains = cfg.dataset.full_domains
        # source_domain_idxs = [full_domains.index(d) for d in souce_domains]
        # target_domain_idxs = [full_domains.index(d) for d in traget_domains]
        # source_train_datasets = get_dataset(cfg.server.data_root_folder, source_domains=cfg.dataset.source_domains, source_domain_idxs=source_domain_idxs)
        # target_train_datasets = get_dataset(cfg.server.data_root_folder, target_domains=cfg.dataset.target_domains, target_domain_idxs=target_domain_idxs)
        # self.source_test_datasets = source_train_datasets["src_val"]
        # self.target_test_datasets = target_train_datasets["target_val"]
        # # split source datasets into train and val
        # source_train_val_datasets = source_train_datasets["src_train"]
        # # set split seed
        # torch.manual_seed(seed)
        # source_train_size = int(len(source_train_val_datasets)*0.8)
        # source_val_size = len(source_train_val_datasets) - source_train_size
        # self.source_train_datasets, self.source_val_datasets = torch.utils.data.random_split(source_train_val_datasets, [source_train_size, source_val_size])
        # target_train_val_datasets = target_train_datasets["target_train"]
        # target_train_size = int(len(target_train_val_datasets)*0.8)
        # target_val_size = len(target_train_val_datasets) - target_train_size 
        # self.target_train_datasets, self.target_val_datasets = torch.utils.data.random_split(target_train_val_datasets, [target_train_size, target_val_size])
        # self.batch_size = cfg.training.batch_size
        # self.dataset_dict = {
        #     "source_train": self.source_train_datasets,
        #     "source_val": self.source_val_datasets,
        #     "source_test": self.source_test_datasets,
        #     "target_train": self.target_train_datasets,
        #     "target_val": self.target_val_datasets,
        #     "target_test": self.target_test_datasets,
        # }
        
        # logger.info(f"Source train size: {len(self.source_train_datasets)}")
        # logger.info(f"Source val size: {len(self.source_val_datasets)}")
        # logger.info(f"Target train size: {len(self.target_train_datasets)}")
        # logger.info(f"Target val size: {len(self.target_val_datasets)}")
        # logger.info(f"Source test size: {len(self.source_test_datasets)}")
        # logger.info(f"Target test size: {len(self.target_test_datasets)}")
        # self.target_fsl = cfg.target_fsl
        self.dataset_dict = get_dataset(cfg)
        for split in self.dataset_dict:
            logger.info(f"{split} size: {len(self.dataset_dict[split])}")
            self.__setattr__(split, self.dataset_dict[split])
        self.mode = cfg.mode
    
    def train_dataloader(self):
        if self.mode == "source_pretrain":
            return DataLoader(self.source_train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        elif self.mode == "target_finetune":
            if self.cfg.fsl != -1:
                # random sample the target train datasets
                if type(self.cfg.fsl) == float:
                    self.cfg.fsl = int(self.cfg.fsl * len(self.target_train_dataset))
                self.target_train_dataset = torch.utils.data.Subset(self.target_train_dataset, np.random.choice(len(self.target_train_dataset), self.cfg.fsl, replace=False))
            return DataLoader(self.target_train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        else:
            raise NotImplementedError

    def val_dataloader(self):
        if self.mode == "target_finetune":
            # make source val dataset empty
            self.source_val_dataset = torch.utils.data.Subset(self.source_val_dataset, [])
        source_val_dataloader = DataLoader(self.source_val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        target_val_dataloader = DataLoader(self.target_val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        return [source_val_dataloader, target_val_dataloader]
    
    def test_dataloader(self):
        return DataLoader(self.target_test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
    
    def predict_dataloader(self):
        return DataLoader(self.target_test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

class ERM(nn.Module):
    def __init__(self, num_classes, mode) -> None:
        super().__init__()
        self.train_step = 0
        self.net = resnet50(weights="DEFAULT")
        fc_in_features = self.net.fc.in_features
        self.fc_in_features = fc_in_features
        self.net.fc= nn.Linear(fc_in_features, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.mode = mode
    
    def get_feature_encoder(self):
        return nn.Sequential(*list(self.net.children())[:-1])
    
    def get_feature(self, x):
        feat = self.get_feature_encoder()(x)
        feat = feat.reshape(x.shape[0], -1)
        return feat
    
    def forward_feature(self, feat):
        logit = self.net.fc(feat)
        return logit
    
    def forward(self, x):
        if self.mode == "feature_extraction":
            feat = self.get_feature(x)
            return feat
        else:
            return self.net(x)
    
class NUC(ERM):
    """
    Empirical Risk Minimization with nuclear norm (ERM_NU)
    """

    def __init__(self, num_classes, mode, nuc_scale=0.0) -> None:
        super().__init__(num_classes, mode)
        self.nuc_scale = nuc_scale
    
    def get_loss(self, x, y, step):
        batch_size = x.shape[0]
        feat = self.get_feature(x)
        y_logit = self.forward_feature(feat)
        y_loss = self.criterion(y_logit, y)
        # check step
        if self.train_step > 100:
            u,s,v = torch.svd(feat, compute_uv=True)
            nuc_loss = torch.sum(torch.abs(s)) / batch_size
        else:
            nuc_loss = torch.tensor(0.0).to("cuda")
        nuc_loss = nuc_loss * self.nuc_scale
        loss = y_loss + self.nuc_scale * nuc_loss
        loss_dict = {
            "loss": loss,
            "y_loss": y_loss,
            "nuc_loss": nuc_loss,
        }
        return loss_dict, y_logit
    
def mat_plot(mat, step):
    if len(mat.shape) == 1:
        mat = mat.unsqueeze(0)
    plt.figure(figsize=(mat.shape[1]*0.1, mat.shape[0]*0.1))
    mat = mat.detach().cpu().numpy()
    sns.heatmap(mat, vmin=-1, vmax=1, annot=True, cmap="PiYG", square=True)
    plt.text(0, 0.5, str(step), color='blue', fontsize=20)
    return wandb.Image(plt)
    
class DragonNet(nn.Module):
    def __init__(self, num_classes, use_fc_bias=False, mode="source_pretrain") -> None:
        super().__init__()
        net = resnet50(weights="DEFAULT")
        fc_in_features = net.fc.in_features
        self.feature_encoder = nn.Sequential(*list(net.children())[:-1])
        #
        # self.classifier = nn.Linear(2048, num_classes)
        self.fc_list = nn.ModuleList([nn.Linear(fc_in_features, num_classes, bias=use_fc_bias
                                                ).to("cuda") for _ in range(4)])
        self.avg_fc = nn.Linear(fc_in_features, num_classes, bias=use_fc_bias)
        self.num_classes = num_classes
        self.use_fc_bias = use_fc_bias
        self.mode = mode
        
    def set_head(self, avg_list, target_index, avg_method="mean"):
        fc_weights = torch.stack([fc.weight.data for fc in self.fc_list], dim=0)
        avg_weights = fc_weights[avg_list,:,:]
        if avg_method == "mean":
            avg_weight = torch.mean(avg_weights, dim=0)
        else:
            raise NotImplementedError
        self.fc_list[target_index].weight.data = avg_weight
    
    def update_avg_fc(self, epoch, mode=None):
        # udpate self.avg_fc
        self.fc_weights = torch.stack([fc.weight.data for fc in self.fc_list], dim=0)
        # 3 * 65 * 2048
        # 2048 -> k
        # 65 * 2048 -> (65 * 65) * (65 * 2048) * (2048 * 2048)
        # U S V
        # w0, w1, w2
        # drop the last fc layer
        if mode is None:
            mode = self.mode
        if mode == "source_pretrain" or mode == "feature_selection":
            #FIXME
            useable_list = [0, 1, 2]
        elif mode == "target_finetune":
            useable_list = [0, 1, 2, 3]
        useable_fc_weights = self.fc_weights[useable_list,:, :]
        mean_fc_weights = torch.mean(useable_fc_weights, dim=0)
        std_fc_weights = torch.std(useable_fc_weights, dim=0)
        self.avg_fc.weight.data = mean_fc_weights
        if self.use_fc_bias:
            self.fc_bias = torch.stack([fc.bias.data for fc in self.fc_list], dim=0)
            useable_fc_bias = self.fc_bias[useable_list, :]
            mean_fc_bias = torch.mean(useable_fc_bias, dim=0)
            std_fc_bias = torch.std(useable_fc_bias, dim=0)
            self.avg_fc.bias.data = mean_fc_bias
        return self.avg_fc
    
    def get_feature(self, x):
        feat = self.feature_encoder(x)
        feat = feat.reshape(x.shape[0], -1)
        return feat
    
    def forward(self, x, e):
        if self.training:
            feat = self.get_feature(x)
            batch_size = x.shape[0]
            logit_list = torch.empty((batch_size, self.num_classes)).to("cuda")
            for e_idx in torch.Tensor([0, 1, 2, 3]).to("cuda"):
                e_feat = feat[e==e_idx]
                logit = self.fc_list[int(e_idx)](e_feat)
                logit_list[e==e_idx] = logit
            return logit_list
        else:
            if self.mode == "source_pretrain":
                feat = self.get_feature(x)
                logit = self.avg_fc(feat)
                return logit
            elif self.mode == "target_finetune":
                feat = self.get_feature(x)
                batch_size = x.shape[0]
                logit_list = torch.empty((batch_size, self.num_classes)).to("cuda")
                for e_idx in torch.Tensor([3]).to("cuda"):
                    e_feat = feat[e==e_idx]
                    logit = self.fc_list[int(e_idx)](e_feat)
                    logit_list[e==e_idx] = logit
                return logit_list
            
class LinearSVDWrapper(nn.Module): 
    def __init__(self, U, D, V): 
        super(LinearSVDWrapper, self).__init__()
        self.U = U
        self.D = D
        self.V = V

    def forward(self, X):
        X = X.T
        X = self.U(X)
        X = self.D * X 
        X = self.V(X)
        X = X.T
        return X


class FeatureEncoder(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        net = resnet50(weights="DEFAULT")
        fc_in_features = net.fc.in_features
        self.feature_encoder = nn.Sequential(*list(net.children())[:-1])
        self.fc = nn.Linear(fc_in_features, num_classes)
    
    def forward(self, x):
        feat = self.feature_encoder(x)
        feat = feat.reshape(x.shape[0], -1)
        logit = self.fc(feat)
        return logit


class ProjectionNet(nn.Module):
    def __init__(self,
                 num_classes,
                 num_envs=4,
                 feature_source=["y", "s", "e"],
                 mode="source_pretrain",
                 ys_orth_reg_scale=1.0,
                 ye_orth_reg_scale=1.0,
                 projection_method="svd") -> None:
        super().__init__()
        net = resnet50(weights="DEFAULT")
        fc_in_features = net.fc.in_features
        self.fc_in_features = fc_in_features
        self.feature_encoder = nn.Sequential(*list(net.children())[:-1])
        self.feature_source = feature_source
        self.mode = mode
        if self.mode in ["source_pretrain", "feature_extraction"]:
            self.y_classifier = nn.Linear(fc_in_features * 2, num_classes)
            self.e_classifier = nn.Linear(fc_in_features * 2, num_envs)
        elif self.mode == "target_finetune":
            self.y_classifier = nn.Linear(fc_in_features * len(self.feature_source), num_classes)
            logger.debug(f"y_classifier weight shape: {self.y_classifier.weight.shape}")
            self.e_classifier = None
        

        self.num_classes = num_classes
        self.projection_method = projection_method
    
        if self.projection_method == "inverse":
            self.y_mat = nn.Parameter(torch.randn(fc_in_features, fc_in_features))
            self.e_mat = nn.Parameter(torch.randn(fc_in_features, fc_in_features))
            self.s_mat = nn.Parameter(torch.randn(fc_in_features, fc_in_features))
            self.y_proj = self.get_projection(self.y_mat)
            self.e_proj = self.get_projection(self.e_mat)
            self.s_proj = self.get_projection(self.s_mat)
        elif self.projection_method == "optimization":
            self.y_proj = nn.Linear(fc_in_features, fc_in_features, bias=False)
            self.e_proj = nn.Linear(fc_in_features, fc_in_features, bias=False)
            self.s_proj = nn.Linear(fc_in_features, fc_in_features, bias=False)
        elif self.projection_method == "svd":
            device = "cuda"
            self.U = Orthogonal(d=fc_in_features).to(device)
            # initialize D to be binary, select 683 random indices 
            y_idxs = np.random.choice(fc_in_features, 683, replace=False)
            # select 683 random indices and not in y_idxs
            e_idxs = np.random.choice(np.setdiff1d(np.arange(fc_in_features), y_idxs), 683, replace=False)
            # select 682 random indices and not in y_idxs or e_idxs
            s_idxs = np.random.choice(np.setdiff1d(np.arange(fc_in_features), np.union1d(y_idxs, e_idxs)), 682, replace=False)
            self.y_D = torch.empty(fc_in_features, 1).uniform_(0.99, 1.01).to(device)
            self.y_D[y_idxs] = 1
            self.y_D[s_idxs] = 0.01
            self.y_D[e_idxs] = 0.01
            self.s_D = torch.empty(fc_in_features, 1).uniform_(0.99, 1.01).to(device)
            self.s_D[y_idxs] = 1
            self.s_D[s_idxs] = 0.01
            self.s_D[e_idxs] = 0.01
            self.e_D = torch.empty(fc_in_features, 1).uniform_(0.99, 1.01).to(device)
            self.e_D[e_idxs] = 1
            self.e_D[s_idxs] = 0.01
            self.e_D[y_idxs] = 0.01
            self.V = Orthogonal(d=fc_in_features).to(device)
            self.y_proj = LinearSVDWrapper(self.U, self.y_D, self.V)
            self.e_proj = LinearSVDWrapper(self.U, self.e_D, self.V)
            self.s_proj = LinearSVDWrapper(self.U, self.s_D, self.V)
        else:
            raise NotImplementedError
        self.none_reduction_criterion = nn.CrossEntropyLoss(reduction="none")
        self.criterion = nn.CrossEntropyLoss()
        self.feature_source = feature_source
        self.ye_orth_reg_scale = ye_orth_reg_scale
        self.ys_orth_reg_scale = ys_orth_reg_scale
        self.mode = mode
        
    def get_feature_encoder(self):
        return self.feature_encoder

    def get_ls_project(self, A):
        P = A @ (torch.inverse(A.T @ A) @ A.T)
        return P

    def apply_projection(self, x, P):
        return (P @ x.T).T
   
    def linear_orth_reg(self, a_mat, b_mat):
        assert self.projection_method in ["optimization", "inverse"]
        a_dim = a_mat.shape[0]
        a = torch.norm(torch.trace(a_mat @ b_mat.T), p=2) + torch.norm(torch.trace(a_mat.T @ b_mat), p=2) 
        a = a / (2 * a_dim)
        return a
    
    def sigma_orth_reg(self, sigma_1, sigma_2):
        assert self.projection_method == "svd"
        return torch.norm(torch.trace(sigma_1 @ sigma_2.T), p=2) / sigma_1.shape[0]
    
    def binary_regularization(self, D, p=2):
        assert self.projection_method == "svd"
        assert p % 2 == 0
        # Calculate the difference from the nearest binary value (0 or 1)
        diff_from_binary = torch.min(D, 1 - D)
        # Square these differences and sum
        penalty = torch.sum(diff_from_binary ** p)
        return penalty
    
    def check_projection(self, P, P_name):
        diff_1 = torch.norm(P - P.T, p=2)
        logger.info(f"{P_name}/diff_1: {diff_1}")
        wandb.log({f"{P_name}/diff_1": diff_1})
        # assert torch.allclose(P, P.T, atol=1e-6)
        diff_2 = torch.norm(P @ P - P, p=2)
        logger.info(f"{P_name}/diff_2: {diff_2}")
        wandb.log({f"{P_name}/diff_2": diff_2})
        # assert torch.allclose(P @ P, P)
        P_rank = torch.linalg.matrix_rank(P)
        logger.info(f"{P_name}/rank: {P_rank}")
        wandb.log({f"{P_name}/rank": P_rank})
        
    def get_projection(self, feat):
        if self.projection_method == "inverse":
            self.y_proj = self.get_ls_project(self.y_mat)
            self.e_proj = self.get_ls_project(self.e_mat)
            self.s_proj = self.get_ls_project(self.s_mat)
            y_projected = self.apply_projection(feat, self.y_proj)
            e_projected = self.apply_projection(feat, self.e_proj)
            s_projected = self.apply_projection(feat, self.s_proj)
            return y_projected, s_projected,e_projected
        elif self.projection_method == "optimization":
            y_projected = self.y_proj(feat)
            e_projected = self.e_proj(feat)
            s_projected = self.s_proj(feat)
            return y_projected, s_projected,e_projected
        elif self.projection_method == "svd":
            y_projected = self.y_proj(feat)
            e_projected = self.e_proj(feat)
            s_projected = self.s_proj(feat)
            return y_projected, s_projected,e_projected
    
    def forward(self, x):
        if self.mode == "source_pretrain":
            feat = self.get_base_feature(x)
            y_projected, s_projected,e_projected = self.get_projection(feat)
            y_feat = torch.cat([y_projected, s_projected], dim=1)
            e_feat = torch.cat([e_projected, s_projected], dim=1)
            y_logit = self.y_classifier(y_feat)
            e_logit = self.e_classifier(e_feat)
            return y_logit, e_logit
        elif self.mode == "target_finetune":
            feat = self.get_base_feature(x)
            if self.feature_source == ["y"]:
                y_projected, s_projected,e_projected = self.get_projection(feat)
                y_feat = y_projected
                y_logit = self.y_classifier(y_feat)
                e_logit = None
                return y_logit, e_logit
            elif self.feature_source == ["y", "s"]:
                y_projected, s_projected,e_projected = self.get_projection(feat)
                y_feat = torch.cat([y_projected, s_projected], dim=1)
                y_logit = self.y_classifier(y_feat)
                e_logit = None
                return y_logit, e_logit
            elif self.feature_source == ["y", "s", "e"]: 
                y_projected, s_projected,e_projected = self.get_projection(feat)
                y_feat = torch.cat([y_projected, s_projected,e_projected], dim=1)
                y_logit = self.y_classifier(y_feat)
                e_logit = None
                return y_logit, e_logit
            else:
                raise NotImplementedError
        elif self.mode == "feature_extraction":
            feat = self.get_base_feature(x)
            y_projected, s_projected,e_projected = self.get_projection(feat)
            return feat, y_projected, s_projected,e_projected
        raise NotImplementedError
        
    def get_base_feature(self, x):
        feat = self.feature_encoder(x)
        feat = feat.reshape(x.shape[0], -1)
        return feat

    def linear_projection_reg(self, weight):
        #! Only for projection method = "optimization"
        assert self.projection_method == "optimization"
        # weight is 2028 * 10
        # weight @ weight^T
        return torch.norm(weight @ weight.T - torch.eye(weight.shape[0]).to("cuda"))
    
    def ls_projection_matrix_reg(self, P):
        assert self.projection_method == "inverse"
        return (torch.norm(P @ P - P) + torch.norm(P - P.T))/P.shape[0]
    
    def get_loss(self, x, y, e):
        y_logit, e_logit = self.forward(x)
        y_loss = self.criterion(y_logit, y)
        y_loss_unreduced = self.none_reduction_criterion(y_logit, y)
        y_loss = y_loss.mean()
        if self.projection_method == "inverse":
            if self.mode == "source_pretrain":
                # logger.debug(torch.unique(e))
                e_loss = self.criterion(e_logit, e)
                ye_orth_reg = self.linear_orth_reg(self.y_mat, self.e_mat)
                ys_orth_reg = self.linear_orth_reg(self.y_mat, self.s_mat)
                es_orth_reg = self.linear_orth_reg(self.e_mat, self.s_mat)
                y_proj_reg = self.ls_projection_matrix_reg(self.y_proj)
                e_proj_reg = self.ls_projection_matrix_reg(self.e_proj)
                s_proj_reg = self.ls_projection_matrix_reg(self.s_proj)
                loss_dict = {"y_loss": y_loss,
                            "e_loss": e_loss,
                            "ye_orth_reg": ye_orth_reg * self.ye_orth_reg_scale,
                            "ys_orth_reg": ys_orth_reg * self.ys_orth_reg_scale,
                            "es_orth_reg": es_orth_reg,
                            "y_proj_reg": y_proj_reg,
                            "e_proj_reg": e_proj_reg,
                            "s_proj_reg": s_proj_reg,
                            "loss": y_loss + e_loss + ye_orth_reg + ys_orth_reg + es_orth_reg + y_proj_reg + e_proj_reg + s_proj_reg,
                            }
            elif self.mode == "target_finetune":
                y_proj_reg = self.projection_matrix_reg(self.y_proj)
                s_proj_reg = self.projection_matrix_reg(self.s_proj)
                e_logit = None
                ys_orth_reg = self.orth_reg(self.y_mat, self.s_mat)
                loss_dict = {"y_loss": y_loss,
                            "y_proj_reg": y_proj_reg,
                            "s_proj_reg": s_proj_reg,
                            "ys_orth_reg": ys_orth_reg * self.ys_orth_reg_scale,
                            "loss": y_loss + y_proj_reg + s_proj_reg,
                            }
        elif self.projection_method == "optimization":
            if self.mode == "source_pretrain":
                e_loss = self.criterion(e_logit, e)
                y_proj_reg = self.linear_projection_reg(self.y_proj.weight)
                e_proj_reg = self.linear_projection_reg(self.e_proj.weight)
                s_proj_reg = self.linear_projection_reg(self.s_proj.weight)
                ye_orth_reg = self.linear_orth_reg(self.y_proj.weight, self.e_proj.weight)
                ys_orth_reg = self.linear_orth_reg(self.y_proj.weight, self.s_proj.weight)
                es_orth_reg = self.linear_orth_reg(self.e_proj.weight, self.s_proj.weight)
                loss_dict = {"y_loss": y_loss,
                            "e_loss": e_loss,
                            "y_proj_reg": y_proj_reg,
                            "e_proj_reg": e_proj_reg,
                            "s_proj_reg": s_proj_reg,
                            "loss": y_loss + e_loss + y_proj_reg + e_proj_reg + s_proj_reg}
            elif self.mode == "target_finetune":
                y_proj_reg = self.linear_projection_reg(self.y_proj.weight)
                s_proj_reg = self.linear_projection_reg(self.s_proj.weight)
                ys_orth_reg = self.linear_orth_reg(self.y_proj.weight, self.s_proj.weight)
                loss_dict = {"y_loss": y_loss,
                            "y_proj_reg": y_proj_reg,
                            "s_proj_reg": s_proj_reg,
                            "ys_orth_reg": ys_orth_reg * self.ys_orth_reg_scale,
                            "loss": y_loss + y_proj_reg + s_proj_reg}
        elif self.projection_method == "svd":
            if self.mode == "source_pretrain":
                y_proj_reg = self.binary_regularization(self.y_D)
                e_proj_reg = self.binary_regularization(self.e_D)
                s_proj_reg = self.binary_regularization(self.s_D)
                ye_orth_reg = self.sigma_orth_reg(self.y_D, self.e_D)
                ys_orth_reg = self.sigma_orth_reg(self.y_D, self.s_D)
                es_orth_reg = self.sigma_orth_reg(self.e_D, self.s_D)
                y_loss = self.criterion(y_logit, y)
                e_loss = self.criterion(e_logit, e)
                loss_dict = {
                                "y_loss": y_loss,
                                "e_loss": e_loss,
                                "y_proj_reg": y_proj_reg,
                                "e_proj_reg": e_proj_reg,
                                "s_proj_reg": s_proj_reg,
                                "ye_orth_reg": ye_orth_reg * self.ye_orth_reg_scale,
                                "ys_orth_reg": ys_orth_reg * self.ys_orth_reg_scale,
                                "es_orth_reg": es_orth_reg,
                                "loss": y_loss + e_loss + y_proj_reg + e_proj_reg + s_proj_reg + ye_orth_reg + ys_orth_reg + es_orth_reg,
                            }    
            elif self.mode == "target_finetune":
                y_proj_reg = self.binary_regularization(self.y_D)
                s_proj_reg = self.binary_regularization(self.s_D)
                ys_orth_reg = self.sigma_orth_reg(self.y_D, self.s_D)
                loss_dict = {
                                "y_loss": y_loss,
                                "y_proj_reg": y_proj_reg,
                                "s_proj_reg": s_proj_reg,
                                "ys_orth_reg": ys_orth_reg * self.ys_orth_reg_scale,
                                "loss": y_loss + y_proj_reg + s_proj_reg,
                            }
        return loss_dict, y_logit, y_loss_unreduced, e_logit




# class SVDNet(nn.Module):
#     def __init__(self, num_classes, num_envs, mode) -> None:
#         super().__init__()
#         net = resnet50(weights="DEFAULT")
#         fc_in_features = self.net.fc.in_features
#         self.feature_encoder = nn.Sequential(*list(self.net.children())[:-1])
#         self.U = Orthogonal(d=fc_in_features)
#         self.V = Orthogonal(d=fc_in_features)
#         self.y_D = torch.empty(fc_in_features, 1).uniform_(0.99, 1.01)
#         self.s_D = torch.empty(fc_in_features, 1).uniform_(0.99, 1.01)
#         self.e_D = torch.empty(fc_in_features, 1).uniform_(0.99, 1.01)
#         self.y_classifier = nn.Linear(fc_in_features, num_classes)
#         self.e_classifier = nn.Linear(fc_in_features, num_envs)
        
#         self.none_reduction_criterion = nn.CrossEntropyLoss(reduction="none")
#         self.criterion = nn.CrossEntropyLoss()
#         self.projection_method = "svd"
    
#     def orth_reg(self, D1, D2):
#         # D1 is 2048 * 1
#         # D2 is 2048 * 1
#         # D1.T @ D2
#         return torch.norm(torch.trace(D1.T @ D2), p=2)
    
#     def binary_regularization(D):
#         # Calculate the difference from the nearest binary value (0 or 1)
#         diff_from_binary = torch.min(D, 1 - D)
#         # Square these differences and sum
#         penalty = torch.sum(diff_from_binary ** 2)
#         return penalty
    
#     def svd_forward(self, x, D):
#         x = self.U(x)
#         x = D * x 
#         x = self.V(x)
#         return x

#     def get_base_feature(self, x):
#         feat = self.feature_encoder(x)
#         feat = feat.reshape(x.shape[0], -1)
#         return feat

#     def forward(self, x):
#         feat = self.get_base_feature(x)
#         y_feat = self.svd_forward(feat, self.y_D)
#         s_feat = self.svd_forward(feat, self.s_D)
#         e_feat = self.svd_forward(feat, self.e_D)
#         y_logit = self.y_classifier(y_feat)
#         e_logit = self.e_classifier(e_feat)
#         return y_logit, e_logit
    
#     def get_loss(self, x, y, e):
#         y_logit, e_logit = self.forward(x)
#         y_loss = self.criterion(y_logit, y)
#         y_loss_unreduced = self.none_reduction_criterion(y_logit, y)
#         y_loss = y_loss.mean()
#         if self.projection_method == "svd":
#             if self.mode == "source_pretrain":
#                 e_loss = self.criterion(e_logit, e)
#                 y_loss = self.criterion(y_logit, y)
#                 ye_orth_reg = self.orth_reg(self.y_D, self.e_D)
#                 ys_orth_reg = self.orth_reg(self.y_D, self.s_D)
#                 es_orth_reg = self.orth_reg(self.e_D, self.s_D)
#                 loss_dict = {"y_loss": y_loss,
#                             "e_loss": e_loss,
#                              "ye_orth_reg": ye_orth_reg,
#                             "ys_orth_reg": ys_orth_reg * self.ys_orth_reg_scale,
#                             }
    
class DANN(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        net = resnet50(weights="DEFAULT")
        fc_in_features = self.net.fc.in_features
        self.feature_encoder = nn.Sequential(*list(self.net.children())[:-1])
        self.y_classifier = nn.Linear(fc_in_features, num_classes)
        self.e_classifier = nn.Linear(fc_in_features, 3)
        
    def forward(self, x):
        feat = self.feature_encoder(x)
        feat = feat.reshape(x.shape[0], -1)
        y_logit = self.y_classifier(feat)
        e_logit = self.e_classifier(feat)
        return y_logit, e_logit

    def get_feature(self, x):
        feat = self.feature_encoder(x)
        feat = feat.reshape(x.shape[0], -1)
        return feat
    



class DomainAdaptationModule(LightningModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.mode = cfg.mode
        num_classes = self.cfg.dataset.num_classes
        num_envs = self.cfg.dataset.num_envs
        if cfg.training.method == "erm":
            self.model = ERM(num_classes, mode=cfg.mode)
        elif cfg.training.method == "nuc":
            self.model = NUC(num_classes, mode=cfg.mode, nuc_scale=cfg.training.nuc_scale)
        elif cfg.training.method == "dragonnet":
            self.model = DragonNet(num_classes, use_fc_bias=cfg.training.use_fc_bias, mode=cfg.mode)
        elif cfg.training.is_domainbed:
            self.model = ERM(num_classes, mode=cfg.mode)
        elif cfg.training.method == "projectionnet":
            self.model = ProjectionNet(num_classes, 
                                        num_envs=num_envs, 
                                        feature_source=cfg.training.feature_source, 
                                        mode=cfg.mode,
                                        ye_orth_reg_scale=cfg.training.ye_orth_reg_scale,
                                        ys_orth_reg_scale=cfg.training.ys_orth_reg_scale,
                                        projection_method=cfg.training.projection_method)
        elif cfg.training.method == "diwa" and cfg.training.base_method == "ERM":
            self.model = ERM(num_classes, mode=cfg.mode)
        self.criterion = nn.CrossEntropyLoss(reduction="none")
        self.y_acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.e_acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_envs)
        self.num_domains = len(cfg.dataset.full_domains)
        self.domain_list = list(range(self.num_domains))
        self.input_e = cfg.training.input_e
        self.test_prefix = ""
        self.test_outputs = []
    
    def forward(self, x, e=None):
        if self.input_e:
            return self.model(x, e)
        else:
            return self.model(x)

    def debatch(self, batch, debug=False):
        x, y, e = batch
        return x, y, e
    
    def deloss(self, loss, loss_lambda):
        return loss + loss_lambda
    

    def general_step(self, batch, batch_idx, stage, dataloader_idx=0):
        x, y, e = self.debatch(batch)
        # logger.debug(y)
        if self.cfg.training.method == "dann":
            e_loss = self.criterion(e_logit, e)
        elif self.cfg.training.method == "projectionnet":
            loss_dict, y_logit, y_loss_unreduced, e_logit = self.model.get_loss(x, y, e)
            # log the loss
            for k, v in loss_dict.items():
                self.log(f"{stage}/{k}", v)
            y_acc = self.y_acc_metric(y_logit, y)
            if self.mode == "source_pretrain":
                e_acc = self.e_acc_metric(e_logit, e)
                self.log(f"{stage}/e_acc", e_acc, prog_bar=True)
            self.log(f"{stage}/y_acc", y_acc, prog_bar=True)
            y_loss_unreduced = self.criterion(y_logit, y)
            y_pred = torch.argmax(y_logit, dim=1)
            loss = loss_dict["loss"]
        elif self.cfg.training.method == "erm":
            y_logit = self.forward(x)
            y_loss_unreduced = self.criterion(y_logit, y)
            loss = y_loss_unreduced.mean()
            y_pred = torch.argmax(y_logit, dim=1)
            self.log(f"{stage}/y_loss", loss, prog_bar=True)
            self.log(f"{stage}/y_acc", self.y_acc_metric(y_pred, y), prog_bar=True)
        elif self.cfg.training.method == "nuc":
            loss_dict, y_logit = self.model.get_loss(x, y, e)
            for k, v in loss_dict.items():
                self.log(f"{stage}/{k}", v)
            y_acc = self.y_acc_metric(y_logit, y)
            self.log(f"{stage}/y_acc", y_acc, prog_bar=True)
            y_loss_unreduced = self.criterion(y_logit, y)
            y_pred = torch.argmax(y_logit, dim=1)
            loss = loss_dict["loss"]
            self.model.train_step += 1
        else:
            if self.input_e:
                output = self.forward(x, e)
            else:
                output = self.forward(x)
            
            if type(output) == tuple:
                y_logit, e_logit = output
            if type(output) == dict:
                y_logit = output["y_logit"]
                e_logit = output["e_logit"]
            else:
                y_logit = output
            
            assert y_logit.shape[0] == y.shape[0]
            assert y_logit.shape[1] == self.cfg.dataset.num_classes
            y_loss_unreduced = self.criterion(y_logit, y)
            loss = y_loss_unreduced.mean()
            y_pred = torch.argmax(y_logit, dim=1)
            self.log(f"{stage}/y_loss", loss, prog_bar=True)
            self.log(f"{stage}/y_acc", self.y_acc_metric(y_pred, y), prog_bar=True)
        
        for domain in self.domain_list:
            e_idx = torch.where(e == domain)[0]
            if len(e_idx) == 0:
                continue
            domain_y_loss = y_loss_unreduced[e_idx].mean()
            # logger.debug("e_idx", e_idx, len(e_idx))
            # logger.debug(y[e_idx].shape, y.shape)
            domain_y_acc = self.y_acc_metric(y_pred[e_idx], y[e_idx])
            self.log(f"{stage}/e={domain}/y_loss", domain_y_loss)
            self.log(f"{stage}/e={domain}/y_acc", domain_y_acc)
        
        return loss

    def training_step(self, batch, batch_idx):
        return self.general_step(batch, batch_idx, "train")
    
    # def on_train_batch_end(self, **params):
    #     if self.cfg.training.method == "nuc":
    #         self.model.train_step += 1
        
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self.general_step(batch, batch_idx, "val", dataloader_idx=dataloader_idx)
        
    def on_test_epoch_start(self):
        self.test_outputs = []
    
    def test_step(self, batch, batch_idx):
        prefix = "test" if self.test_prefix == "" else f"test/{self.test_prefix}"
        if self.mode != "feature_extraction":
            return self.general_step(batch, batch_idx, prefix)
        else:
            x, y, e = self.debatch(batch)
            output = self.model(x)
            if type(output) == torch.Tensor:
                # erm
                output = [output]
            elif type(output) == tuple:
                # projectionnet
                output = list(output)
            append_list = output + [y, e]
            self.test_outputs.append(append_list)
            return None
    
    def on_test_epoch_end(self):
        if self.mode == "feature_extraction":
            if self.cfg.training.method == "projectionnet":
                base_feature, y_projected, s_projected,e_projected, y, e = zip(*self.test_outputs)
                base_feature = torch.cat(base_feature, dim=0)
                y_projected = torch.cat(y_projected, dim=0)
                s_projected = torch.cat(s_projected, dim=0)
                e_projected = torch.cat(e_projected, dim=0)
                y = torch.cat(y, dim=0)
                e = torch.cat(e, dim=0)
                feature_dict = {
                    "base_feature": base_feature,
                    "y_projected": y_projected,
                    "s_projected": s_projected,
                    "e_projected": e_projected,
                    "y": y,
                    "e": e,
                }
                full_path = f"outputs/{self.cfg.feature_extraction_str}/{self.checkpoint_filename}"
                os.makedirs(full_path, exist_ok=True)
                logger.info(f"Saving feature_dict to {full_path}/{self.test_prefix}.pt")
                torch.save(feature_dict, f"{full_path}/{self.test_prefix}.pt")
            elif self.cfg.training.method in ["erm","nuc"] or self.cfg.training.is_domainbed or self.cfg.training.method == "diwa":
                base_feature, y, e = zip(*self.test_outputs)
                base_feature = torch.cat(base_feature, dim=0)
                y = torch.cat(y, dim=0)
                e = torch.cat(e, dim=0)
                feature_dict = {
                    "base_feature": base_feature,
                    "y": y,
                    "e": e,
                }
                full_path = f"outputs/{self.cfg.feature_extraction_str}/{self.checkpoint_filename}"
                os.makedirs(full_path, exist_ok=True)
                logger.info(f"Saving feature_dict to {full_path}/{self.test_prefix}.pt")
                torch.save(feature_dict, f"{full_path}/{self.test_prefix}.pt")
            return self.test_outputs
        else:
            return None

    def configure_optimizers(self):
        if self.cfg.training.optimizer.feature_lr_scale == 1.0:
            if self.cfg.training.optimizer.name == "adam":
                optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.training.optimizer.lr, weight_decay=0)
        elif self.cfg.training.optimizer.feature_lr_scale == 0.0:
            if self.cfg.training.optimizer.name == "adam":
                if self.cfg.training.method == "projectionnet":
                    optimizer = torch.optim.Adam([{"params": self.model.y_classifier.parameters()},
                                                {"params": self.model.e_classifier.parameters()},
                                                ], lr=self.cfg.training.optimizer.lr, weight_decay=0)
                elif self.cfg.training.method == "erm":
                    optimizer = torch.optim.Adam([{"params": self.model.net.fc.parameters()},
                                                ], lr=self.cfg.training.optimizer.lr, weight_decay=0)
        else:
            if self.cfg.training.optimizer.name == "adam":
                encoder = self.model.get_feature_encoder() # sequential
                # encoder_parameters = encoder.parameters()
                # print("encoder_parameters", encoder_parameters)

                encoder_parameters = []
                other_parameters = []
                # not in encoder
                for name, param in self.model.named_parameters():
                    logger.debug(f"name: {name}")
                    if "feature_encoder" not in name:
                        logger.debug(f"other_parameters: {name}")
                        other_parameters.append(param)
                    else:
                        logger.debug(f"encoder_parameters: {name}")
                        encoder_parameters.append(param)
                logger.debug(f"encoder_parameters_lr: {self.cfg.training.optimizer.lr * self.cfg.training.optimizer.feature_lr_scale}")
                logger.debug(f"other_parameters_lr: {self.cfg.training.optimizer.lr}")
                optimizer = torch.optim.Adam([
                    {"params": other_parameters, "lr": self.cfg.training.optimizer.lr},
                    {"params": encoder_parameters, "lr": self.cfg.training.optimizer.lr * self.cfg.training.optimizer.feature_lr_scale},
                ], weight_decay=0)

                                               
        return optimizer

    def on_validation_epoch_start(self):
    # for dragonnet
        if self.input_e:
            self.model.update_avg_fc(self.current_epoch)
    
    def on_validation_epoch_end(self):
        metrics = self.trainer.callback_metrics
        # log meric_dict

class EBDataset(Dataset):
    def __init__(self, features: dict, feature_source=["y_projeted", "s_projected"], target="y"):
        if target not in ["y", "e"]:
            raise ValueError("target must be 'y' or 's'")
        self.features = features
        self.data = torch.cat([features[name] for name in feature_source], dim=1)
        self.targets = features[target]
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        target = self.targets[index]
        return data, target
    
class EBDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.feature_dict = self.load_feature_dict(cfg)
        
        target_val = self.feature_dict["target_val"]
        permutation = torch.randperm(target_val["base_feature"].shape[0])
        for k,v in target_val.items():
            target_val[k] = v[permutation]
        
        # split
        global_seed = torch.random.get_rng_state()
        torch.manual_seed(cfg.dataset.seed)
        new_target_val = {}
        new_target_test = {}
        for k,v in target_val.items():
            new_target_val[k] = v[:int(v.shape[0] * 0.5)]
            new_target_test[k] = v[int(v.shape[0] * 0.5):]
        target_val = new_target_val
        target_test = new_target_test
        torch.random.set_rng_state(global_seed)
        
        self.feature_dict["target_val"] = target_val
        self.feature_dict["target_test"] = target_test
        self.feature_source = cfg.training.feature_source
        self.target = cfg.training.target
        self.domain = cfg.training.domain
        self.dataset_dict = {}
        self.setup()
        
        if self.domain == "source":
            assert cfg.fsl == -1
        elif self.domain == "target":
            # do the val test split
            global_seed = torch.random.get_rng_state()
            torch.manual_seed(cfg.dataset.seed)
            # self.dataset_dict["target_val"], self.dataset_dict["target_test"] = random_split(self.dataset_dict["target_val"], [int(len(self.dataset_dict["target_val"]) * 0.5), int(len(self.dataset_dict["target_val"]) - len(self.dataset_dict["target_val"]) * 0.5)])
            if cfg.fsl != -1:
                data = self.dataset_dict["target_train"].data
                targets = self.dataset_dict["target_train"].targets
                permutation = torch.randperm(data.shape[0])
                data = data[permutation]
                targets = targets[permutation]
                if type(cfg.fsl) == float:
                    fsl = int(cfg.fsl * data.shape[0])
                else:
                    fsl = cfg.fsl
                data = data[:fsl]
                targets = targets[:fsl]
                self.dataset_dict["target_train"].data = data
                self.dataset_dict["target_train"].targets = targets
            torch.random.set_rng_state(global_seed)
        
        #!FULL DATA
        self.batch_size = cfg.training.batch_size
        assert self.batch_size >= 50000
        
        # set probing dataset
        if self.cfg.mode == "source_linear_probing":
            self.domain = "source"
        elif self.cfg.mode == "target_linear_probing":
            self.domain = "target"
    
    def load_feature_dict(self, cfg):
        name_list = ["source_train", "source_val", "target_train", "target_val"]
        if cfg.pretrain.is_domainbed:
            #! DANN
            best_model_path = get_best_model_path_domainbed(cfg)
            best_model_path = best_model_path.replace(".pkl", "")
            checkpoint_filename = best_model_path.split("/")[-1]
            best_model_dir = f"outputs/{self.cfg.feature_extraction_str}/{checkpoint_filename}"
        elif cfg.pretrain.method == "diwa":
            #! DIWA
            checkpoint_path = cfg.pretrain.weight_path
            #FIXME: change this to non hardcode 
            if "[" or "]" in checkpoint_path:
                checkpoint_path = checkpoint_path.replace("[", "").replace("]", "")
            checkpoint_path = checkpoint_path.replace("officehome", "OfficeHome")
            checkpoint_filename = checkpoint_path.split("/")[-1].replace(".pt", "")
            best_model_dir = f"outputs/{self.cfg.feature_extraction_str}/{checkpoint_filename}"
        else:
            #! ERM, NUC, Projectionnet
            logger.debug(f"outputs/{cfg.feature_extraction_str}")
            best_model_path = get_best_model_path(f"outputs/{cfg.feature_extraction_str}", cfg.pretrain.monitor)
            best_model_path = best_model_path.replace(".ckpt", "")
            best_model_dir = best_model_path
        logger.info(f"Loading feature_dict from {cfg.feature_extraction_str}")
        feature_dict = {}
        for name in name_list:
            feature_dict[name] = myload(f"{best_model_dir}/phase={name}_dataset.pt", map_location="cpu")
        return feature_dict
    
    def setup(self):
        for stage in ["train", "val", "test"]:
            self.dataset_dict[f"{self.domain}_{stage}"] = EBDataset(self.feature_dict[f"{self.domain}_{stage}"], self.feature_source, self.target)
    
    def get_dataset(self, split):
        return self.dataset_dict[f"{self.domain}_{split}"]



@hydra.main(config_path='configs', config_name='default', version_base='1.3')
def main(cfg):
    import optuna, random
    
    
    print(cfg)
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    # assert here
    # if cfg.training.max_epochs == 60:
        # assert cfg.training.optimizer.feature_lr_scale == 1.0
    # elif cfg.training.max_epochs == 10:
        # assert cfg.training.optimizer.feature_lr_scale == 0.1
        
    if cfg.mode == "source_pretrain":
        assert cfg.indomain_test == False
    elif cfg.mode == "target_finetune":
        assert cfg.indomain_test == True
    
    if type(cfg.dataset.target_envs) == int:
        cfg.dataset.target_envs = [cfg.dataset.target_envs]
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg = OmegaConf.create(cfg_dict)
    hostname = socket.gethostname()
    # import hashlib
    # cfg_hash = hashlib.md5(str(cfg).encode()).hexdigest()
    # with open(f"./outputs/wandb.pkl", "rb") as f:
    # #     runs_list = pkl.load(f)
    # hash_set = set()
    # for run in runs_list:
    #     hash_set.add(run["cfg_hash"])
    # if cfg_hash in hash_set:
    #     logger.info(f"cfg_hash: {cfg_hash} already exist")
    # else:
    #     logger.info(f"cfg_hash: {cfg_hash} not exist")
    
    logger.info(f"hostname: {hostname}")
    
    #FIXME: 
    if cfg.mode == "source_pretrain":
        cfg.str = cfg.str
    elif cfg.mode == "feature_selection":
        cfg.str = cfg.feature_selection_str
    elif cfg.mode == "target_finetune":
        #! TODO: fix this ugly code
        # if cfg.feature_lr_decay != 1.0:
            # cfg.str = cfg.target_finetune_feature_lr_decay_str
        # else:
        cfg.str = cfg.target_finetune_str
    elif cfg.mode == "linear_probing":
        cfg.str = cfg.linear_probing_str
    elif cfg.mode == "feature_extraction":
        cfg.str = cfg.feature_extraction_str
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        cfg_dict["training"] = cfg_dict["pretrain"]
        cfg = OmegaConf.create(cfg_dict)
    else:
        raise NotImplementedError
    
    #check wandb exist such run
    
    
    if cfg.debug:
        output_dir = f"outputs/debug"
        cfg.wandb.tags += ["debug"]
    else:
        output_dir = f"outputs/{cfg.str}"
        
    # linear probing
    if cfg.mode in ["linear_probing"]:
        datamodule = EBDataModule(cfg)
        
        wandb.init(project=cfg.wandb.project,
                    entity=cfg.wandb.entity,
                    name=cfg.str,
                    dir="./outputs",
                    mode="offline" if cfg.debug else "online",
                    tags=list(str(t) for t in cfg.wandb.tags),
        )
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        # fsl float makre sure two decimal
        if type(cfg_dict["fsl"]) == float:
            cfg_dict["fsl"] = round(cfg_dict["fsl"], 2)
        wandb.config.update(cfg_dict)
        train_dataset = datamodule.get_dataset("train")
        val_dataset = datamodule.get_dataset("val")
        test_dataset = datamodule.get_dataset("test")
        # using sklearn
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import PredefinedSplit
        from sklearn.model_selection import GridSearchCV
        param_grid = {
            "clf__C": [1e-5, 5e-5,
                       1e-4, 5e-4,
                       1e-3, 5e-3,
                       1e-2, 5e-2, 
                       1e-1, 5e-1],
            "clf__solver": ["lbfgs"],
            "clf__max_iter": [1000],
            "clf__random_state": [cfg.training.seed],
        }
        target = "y"
        
        val_data = val_dataset.data
        val_targets = val_dataset.targets
            
        train_data = train_dataset.data
        train_targets = train_dataset.targets
        
        train_val_sample_weight = None
        model = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
        
        train_val_data = torch.cat([train_data, val_data], dim=0)
        train_val_targets = torch.cat([train_targets, val_targets], dim=0)

        logger.info(f"train_data: {train_data.shape}")
        logger.info(f"val_data: {val_data.shape}")
        logger.info(f"train_val_data: {train_val_data.shape}")

        test_fold = [-1] * len(train_data) + [0] * len(val_data)
        ps = PredefinedSplit(test_fold=test_fold)
        
        pl.seed_everything(cfg.training.seed)
        
        grid_search = GridSearchCV(model, param_grid, cv=ps, n_jobs=-1, verbose=1)
        
        grid_search.fit(train_val_data, train_val_targets, clf__sample_weight=train_val_sample_weight)
        
        logger.info(grid_search.best_params_)
        logger.info(grid_search.best_score_)
        logger.info(grid_search.best_estimator_)
        best_model = grid_search.best_estimator_
        
        best_model.fit(train_data, train_targets, clf__sample_weight=train_val_sample_weight)
        target_pred = best_model.predict(test_dataset.data)
        test_acc = accuracy_score(test_dataset.targets, target_pred)
        logger.info(f"test_acc: {test_acc}")
        # log best hyperparameters
        train_acc = accuracy_score(train_val_targets, best_model.predict(train_val_data))
        val_acc = accuracy_score(val_targets, best_model.predict(val_data))
        wandb.log({"best_params": grid_search.best_params_})
        wandb.log({f"train/{target}_acc": train_acc})
        wandb.log({f"val/{target}_acc": val_acc})
        wandb.log({f"test/{target}_acc": test_acc})
        
        # save model
        os.makedirs(output_dir, exist_ok=True)
        torch.save(best_model, f"{output_dir}/linear_probing_best.pt")
    else:
        datamodule = DataModule(cfg)
        
        # target finetune onlu
        if cfg.mode != "source_pretrain":
            #!!!!!!!!!!!!!!!!!
            # set pretrain manually!!!!!!!!!!!!
            # pretrain_str = cfg.source_pretrain_str
            # pretrain_folder = f"./outputs/{pretrain_str}"
            # pretrain_yaml = f"{pretrain_folder}/config.yaml"
            # checkfile exist
            # #! TODO: remove the hardcode
            # if not os.path.exists(pretrain_yaml):
            #     cwd = os.getcwd()
            #     pretrain_yaml_full_path = os.path.join(cwd, pretrain_yaml)
                
            #     # rsync from remote ds
            #     parent_folder = "/".join(pretrain_folder.split("/")[:-1])
            #     os.makedirs(parent_folder, exist_ok=True)
            #     cmd = f"rsync -avzP ds:'{pretrain_yaml_full_path}' '{pretrain_yaml_full_path}'"
            #     # also rsync the checkpoint
            #     #TODO: remove the hardcode
            #     # if cfg.training.method == "erm":
            #         # checkpoint_path = f"AMASK/AMASK/AMASKPATH/outputs/name=officehome/source_domains=['art', 'clipart', 'product']/target_domains=['real']/dataset_seed=0/mode=source_pretrain/type=resnet50/method=erm/batch_size=32/lr=5e-05/training_seed=0/epoch=47-source_val_acc=0.81.ckpt"
            # cfg.pretrain = OmegaConf.load(pretrain_yaml)
            # cfg_dict = OmegaConf.to_container(cfg, resolve=False)
            # cfg_dict["pretrain"]["monitor"] = "source_val_acc"
            # cfg = OmegaConf.create(cfg_dict)
            # test only
            if cfg.test_only:
                # load finetune
                logger.info("test_only")
                logger.debug(output_dir)
                assert "target_finetune" in output_dir
                # print("output_dir", output_dir)
                logger.debug(output_dir)
                # only pretrain is domainbed but not target finetune
                checkpoint_path = get_best_model_path(output_dir, cfg.pretrain.monitor)
                logger.info(f"checkpoint_path: {checkpoint_path}")
                checkpoint_filename = checkpoint_path.split("/")[-1].replace(".ckpt", "")
                checkpoint_filename = checkpoint_path.split("/")[-1].replace(".pkl", "")
                checkpoint_filename = checkpoint_path.split("/")[-1].replace(".pt", "")
            else:
                # load pretrain
                pretrain_folder = f"./outputs/{cfg.source_pretrain_str}"
                if cfg.pretrain.is_domainbed:
                    checkpoint_path = get_best_model_path_domainbed(cfg)
                elif cfg.pretrain.method == "diwa":
                    checkpoint_path = cfg.training.weight_path
                    if "[" or "]" in checkpoint_path:
                        checkpoint_path = checkpoint_path.replace("[", "").replace("]", "")
                    checkpoint_path = checkpoint_path.replace("officehome", "OfficeHome")
                    checkpoint_path = checkpoint_path.replace("pacs", "PACS")
                    checkpoint_path = checkpoint_path.replace("domainnet", "DomainNet")
                    checkpoint_path = checkpoint_path.replace("vlcs", "VLCS")
                    checkpoint_path = checkpoint_path.replace("terraincognita", "TerraIncognita")
                    # assert os.path.exists(checkpoint_path), f"{checkpoint_path} does not exist"
                else:
                    logger.info(f"pretrain_folder: {pretrain_folder}")
                    logger.info(f"cfg.pretrain.monitor: {cfg.pretrain.monitor}")
                    checkpoint_path = get_best_model_path(pretrain_folder, cfg.pretrain.monitor)
                checkpoint_filename = checkpoint_path.split("/")[-1].replace(".ckpt", "")
                checkpoint_filename = checkpoint_path.split("/")[-1].replace(".pkl", "")
                checkpoint_filename = checkpoint_path.split("/")[-1].replace(".pt", "")

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        # save config
        
        # set wandb
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/config.yaml", "w") as f:
            OmegaConf.save(cfg, f)

        if cfg.wandb.enabled:
            # wandb.init(id=run_id)
            wandb_logger = WandbLogger(project=cfg.wandb.project, 
                                    entity=cfg.wandb.entity,
                                    name=cfg.str,
                                    dir="./outputs/wandb",
                                    offline=cfg.debug,
                                    tags=list(str(t) for t in cfg.wandb.tags),
                                    reinit=True,
                                    settings=wandb.Settings(start_method="fork")
                                    )
            
            def timeout_handler(signum, frame):
                raise TimeoutError("Execution timed out")

            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(60)  # Set the timeout to 60 seconds

            try:
                wandb_logger.log_hyperparams(cfg_dict)
            except TimeoutError:
                return
            finally:
                signal.alarm(0)  # Reset the alarm
            
            
            trainer_logger = [wandb_logger]
        else:
            trainer_logger = None

        def get_callbacks(cfg):
            if cfg.mode == "source_pretrain":
                loss_monitor = "val/y_loss/dataloader_idx_0"
                loss_file_name = "epoch={epoch}-source_val_loss={val/y_loss/dataloader_idx_0:.2f}"
                acc_monitor = "val/y_acc/dataloader_idx_0"
                acc_file_name = "epoch={epoch}-source_val_acc={val/y_acc/dataloader_idx_0:.2f}"
            elif cfg.mode == "target_finetune":
                loss_monitor = "val/y_loss/dataloader_idx_1"
                loss_file_name = "epoch={epoch}-target_val_loss={val/y_loss/dataloader_idx_1:.2f}"
                acc_monitor = "val/y_acc/dataloader_idx_1"
                acc_file_name = "epoch={epoch}-target_val_acc={val/y_acc/dataloader_idx_1:.2f}"
            
            loss_model_checkpoint = ModelCheckpoint(
                dirpath=s3_prefix+output_dir,
                monitor=loss_monitor,
                save_top_k=1,
                filename=loss_file_name,
                auto_insert_metric_name=False,
                save_last=True,
                mode="min",
                enable_version_counter=False,
            )
            acc_model_checkpoint = ModelCheckpoint(
                dirpath=s3_prefix+output_dir,
                save_top_k=1,
                monitor=acc_monitor,
                filename=acc_file_name,
                auto_insert_metric_name=False,
                save_last=True,
                mode="max",
                enable_version_counter=False,
            )
            #! For oracle selection, we do not consider it here
            # target_acc_model_checkpoint = ModelCheckpoint(
            #     dirpath=output_dir,
            #     every_n_epochs=1,
            #     monitor="val/y_acc/dataloader_idx_1",
            #     filename="epoch={epoch}-target_val_acc={val/y_acc/dataloader_idx_1:.2f}",
            #     auto_insert_metric_name=False,
            #     save_last=True,
            #     mode="max",
            #     enable_version_counter=False,
            # )
            # target_loss_model_checkpoint = ModelCheckpoint(
            #     dirpath=output_dir,
            #     every_n_epochs=1,
            #     monitor="val/y_loss/dataloader_idx_1",
            #     filename="epoch={epoch}-target_val_loss={val/y_loss/dataloader_idx_1:.2f}",
            #     auto_insert_metric_name=False,
            #     save_last=True,
            #     mode="min",
            #     enable_version_counter=False,
            # )

            callbacks = []
            callbacks += [loss_model_checkpoint, acc_model_checkpoint]
            # callbacks += [target_acc_model_checkpoint, target_loss_model_checkpoint]
            if cfg.training.optimizer.swa:
                callbacks += [StochasticWeightAveraging(swa_lrs=cfg.training.optimizer.swa_lrs,
                                                        swa_epoch_start=cfg.training.optimizer.swa_epoch_start,
                                                        annealing_epochs=cfg.training.optimizer.annealing_epochs,
                                                        annealing_strategy=cfg.training.optimizer.annealing_strategy,
                                                        )]
            return callbacks
        
        # set callbacks
        if cfg.mode in ["source_pretrain", "target_finetune"]:
            callbacks = get_callbacks(cfg)
        else:
            callbacks = None

        if cfg.mode == "feature_extraction":
            logger.debug(cfg.training)
            max_epochs = 1
        else:
            max_epochs = cfg.training.max_epochs
        # async_ckpt_io = AsyncCheckpointIO()
        trainer = pl.Trainer(   
                                accelerator="auto",
                                max_epochs=max_epochs,
                                fast_dev_run=cfg.debug,
                                logger=trainer_logger,
                                # plugins=[async_ckpt_io],
                                callbacks=callbacks,
                            )

        pl.seed_everything(cfg.training.seed)
        module = DomainAdaptationModule(cfg)
        
        #* loading part
        if cfg.test_only:
            # directly load the model
            state_dict = myload(checkpoint_path)["state_dict"]
            module.load_state_dict(state_dict, strict=False)
        # if cfg.mode == "target_finetune" or cfg.mode == "feature_selection":
        elif cfg.mode in ["target_finetune", "feature_extraction", "feature_selection"]:
            if cfg.pretrain.method == "projectionnet":
                logger.info(f"load from {checkpoint_path}")
                state_dict = myload(checkpoint_path)["state_dict"]
                y_classifier_weight = state_dict["model.y_classifier.weight"]
                e_classifier_weight = state_dict["model.e_classifier.weight"]
                state_dict.pop("model.e_classifier.weight")
                state_dict.pop("model.e_classifier.bias")
                state_dict.pop("model.y_classifier.weight")
                module.load_state_dict(state_dict, strict=False)
                logger.info(f"load from {checkpoint_path}")
                logger.debug(f"cfg.training.feature_source: {cfg.training.feature_source}")
                logger.debug(f"cfg.pretrain: {cfg.pretrain}")
                
                #FIXME: check the feature_source in pretrain config
                #TODO: check this
                source_env_num = len(cfg.dataset.full_domains) - len(cfg.dataset.target_envs)
                
                e_classifier_weight = e_classifier_weight[:source_env_num,:]
                # load the e_classifier_weight
                module.model.e_classifier = nn.Linear(module.model.fc_in_features * 2, source_env_num)
                module.model.e_classifier.weight.data[:source_env_num,:] = e_classifier_weight
                module.model.e_classifier.bias.data[:source_env_num] = 0
                
                # load the y_classifier_weight
                if cfg.training.feature_source == ["y"]:
                    y_classifier_weight_y_part = y_classifier_weight[:, :module.model.fc_in_features]
                    y_classifier_weight_s_part = y_classifier_weight[:, module.model.fc_in_features:]
                    module.model.y_classifier.weight.data = y_classifier_weight_y_part
                elif cfg.training.feature_source == ["y", "s"]:
                    module.model.y_classifier.weight.data = y_classifier_weight
                elif cfg.training.feature_source == ["y", "s", "e"]:
                    module.model.y_classifier.weight.data[:, :module.model.fc_in_features * 2] = y_classifier_weight
        
                if cfg.feature_selection.method != "none":
                    #FIXME: check this
                    print(cfg.feature_selection.method)
                    assert cfg.feature_selection.method in ["mean-std", "mean", "random"]
                    assert cfg.pretrain.method == "dragonnet"
                    # save the fc_list
                    fc_list = module.model.fc_list
                    fc_weights = torch.stack([fc.weight.data for fc in fc_list], dim=0)
                    # fc_bias = torch.stack([fc.bias.data for fc in fc_list], dim=0)
                    fc_weights = fc_weights[:-1,:,:]
                    if cfg.feature_selection.method == "mean-std":
                        value_list = []
                        for dim in range(fc_weights.shape[-1]):
                            weight = fc_weights[:,:,dim]
                            weight = torch.abs(weight)
                            mean = weight.mean(dim=0)
                            std = weight.std(dim=0)
                            value = (mean / std).mean().item()
                            value_list.append(value)
                        # rank = sorted(range(len(value_list)), key=lambda k: value_list[k], reverse=True)
                        top_k = cfg.feature_selection.top_k
                        rank = sorted(range(len(value_list)), key=lambda k: value_list[k], reverse=True)
                        rank = rank[:top_k]
                    elif cfg.feature_selection.method == "mean":
                        value_list = []
                        for dim in range(fc_weights.shape[-1]):
                            weight = fc_weights[:,:,dim]
                            weight = torch.abs(weight)
                            mean = weight.mean(dim=0)
                            value = mean.mean().item()
                            value_list.append(value)
                        # rank = sorted(range(len(value_list)), key=lambda k: value_list[k], reverse=True)
                        top_k = cfg.feature_selection.top_k
                        rank = sorted(range(len(value_list)), key=lambda k: value_list[k], reverse=True)
                        rank = rank[:top_k]
                    elif cfg.feature_selection.method == "random":
                        top_k = cfg.feature_selection.top_k
                        rank = np.random.choice(fc_weights.shape[-1], top_k, replace=False)
                    mask = torch.zeros_like(fc_weights)
                    mask[:,:,rank] = 1
                    masked_fc_weights = fc_weights * mask
                    in_feature_dim = masked_fc_weights.shape[-1]
                    out_feature_dim = masked_fc_weights.shape[1]
                    module.model.fc_list = nn.ModuleList([nn.Linear(in_feature_dim, out_feature_dim, bias=False).to("cuda") for _ in range(4)])
                    module.model.fc_list[0].weight.data = masked_fc_weights[0,:,:]
                    module.model.fc_list[1].weight.data = masked_fc_weights[1,:,:]
                    module.model.fc_list[2].weight.data = masked_fc_weights[2,:,:]
                    module.model.set_head([0, 1, 2], 3)
            elif cfg.pretrain.method == "erm":
                logger.info(f"load from {checkpoint_path}")
                state_dict = myload(checkpoint_path)["state_dict"]
                module.load_state_dict(state_dict, strict=False)
            elif cfg.pretrain.is_domainbed:
                checkpoint_path = "./libs/DomainBed/" + checkpoint_path
                state_dict = myload(checkpoint_path, map_location="cpu")["model_dict"]
                # featureizer.network -> model.net
                # classifier -> model.fc
                state_dict = {k.replace("featurizer.network.", "model.net."): v for k,v in state_dict.items()}
                state_dict = {k.replace("classifier.", "model.net.fc."): v for k,v in state_dict.items()}
                # drop others
                state_dict = {k: v for k,v in state_dict.items() if "model.net" in k}
                module.load_state_dict(state_dict, strict=True)
            elif cfg.pretrain.method == "diwa":
                #* diaw repo
                logger.info(f"load from {checkpoint_path}")
                state_dict = myload(checkpoint_path)
                state_dict = {k.replace("network_wa.0.network.", "model.net."): v for k,v in state_dict.items()}
                state_dict = {k.replace("network_wa.1.", "model.net.fc."): v for k,v in state_dict.items()}
                module.load_state_dict(state_dict, strict=True)
            elif cfg.pretrain.method == "nuc":
                logger.info(f"load from {checkpoint_path}")
                state_dict = myload(checkpoint_path)["state_dict"]
                module.load_state_dict(state_dict, strict=False)
            else:
                raise NotImplementedError
        #* fit
        if cfg.mode != "feature_extraction" and cfg.fsl != 0.0:
            if not cfg.test_only:
                logger.debug(f"cfg.mode: {cfg.mode}")
                trainer.fit(module, datamodule)
                trainer.validate(module, datamodule=datamodule)
        
                #* test
                # trainer.test(module, datamodule=datamodule)
                # based on different selection metric, load the best model state dict
                for callback in trainer.callbacks:
                    if isinstance(callback, ModelCheckpoint):
                        monitor_str= callback.monitor.replace("/", "-")
                        module.test_prefix = f"monitor={monitor_str}"
                        trainer.test(module, datamodule=datamodule, ckpt_path=callback.best_model_path)

                # log best model path
                for callback in trainer.callbacks:
                    if isinstance(callback, ModelCheckpoint):
                        wandb.log({f"{callback.monitor}/best": callback.best_model_score})
                        monitor_str= callback.monitor.replace("/", "-")
                        callback.to_yaml(f"{output_dir}/monitor={monitor_str}.yaml")
            else:
                #! test only
                monitor_str = "val/y_acc/dataloader_idx_1"
                monitor_str = monitor_str.replace("/", "-")
                module.test_prefix = f"monitor={monitor_str}"
                trainer.test(module, datamodule=datamodule, ckpt_path=checkpoint_path)
        
        elif cfg.fsl == 0.0:
            monitor_str = "val/y_acc/dataloader_idx_1"
            monitor_str = monitor_str.replace("/", "-")
            module.test_prefix = f"monitor={monitor_str}"
            trainer.test(module, datamodule=datamodule)
                    
        elif cfg.mode == "feature_extraction":
            for phase in datamodule.dataset_dict.keys():
                dataset = datamodule.dataset_dict[phase]
                dataloader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=False, num_workers=cfg.num_workers)
                module.test_prefix = f"phase={phase}"
                checkpoint_filename = checkpoint_filename.replace(".ckpt", "")
                module.checkpoint_filename = checkpoint_filename
                trainer.test(module, dataloaders=dataloader)
    
    
    # get the ""test/monitor=val-y_acc-dataloader_idx_1/y_acc" from summary
    wandb.finish()
    # optuna_value = trainer.logger.experiment.summary["test/monitor=val-y_acc-dataloader_idx_1/y_acc"]
    # return optuna_value
    

    
if __name__ == "__main__":
    main()