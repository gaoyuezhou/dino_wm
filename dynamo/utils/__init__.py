import os
import torch
import wandb
import random
import einops
import numpy as np
import torch.nn as nn
from . import inference
import torch.utils.data
from pathlib import Path
from hydra.types import RunMode
from typing import Callable, Dict
from prettytable import PrettyTable
from collections import OrderedDict
from torch.utils.data import random_split
from hydra.core.hydra_config import HydraConfig


# Modified from https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    return total_params, table


def get_split_idx(l, seed, train_fraction=0.95):
    rng = torch.Generator().manual_seed(seed)
    idx = torch.randperm(l, generator=rng).tolist()
    l_train = int(l * train_fraction)
    return idx[:l_train], idx[l_train:]


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk


def freeze_module(module: nn.Module) -> nn.Module:
    for param in module.parameters():
        param.requires_grad = False
    module.eval()
    return module


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def batch_indexing(input, idx):
    """
    Given an input with shape (*batch_shape, k, *value_shape),
    and an index with shape (*batch_shape) with values in [0, k),
    index the input on the k dimension.
    Returns: (*batch_shape, *value_shape)
    """
    batch_shape = idx.shape
    dim = len(idx.shape)
    value_shape = input.shape[dim + 1 :]
    N = batch_shape.numel()
    assert input.shape[:dim] == batch_shape, "Input batch shape must match index shape"
    assert len(value_shape) > 0, "No values left after indexing"

    # flatten the batch shape
    input_flat = input.reshape(N, *input.shape[dim:])
    idx_flat = idx.reshape(N)
    result = input_flat[np.arange(N), idx_flat]
    return result.reshape(*batch_shape, *value_shape)


def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a, idx, axis=axis)


def transpose_batch_timestep(*args):
    return (einops.rearrange(arg, "b t ... -> t b ...") for arg in args)


class TrainWithLogger:
    def reset_log(self):
        self.log_components = OrderedDict()

    def log_append(self, log_key, length, loss_components):
        for key, value in loss_components.items():
            key_name = f"{log_key}/{key}"
            count, sum = self.log_components.get(key_name, (0, 0.0))
            self.log_components[key_name] = (
                count + length,
                sum + (length * value.detach().cpu().item()),
            )

    def flush_log(self, epoch, iterator=None):
        log_components = OrderedDict()
        iterator_log_component = OrderedDict()
        for key, value in self.log_components.items():
            count, sum = value
            to_log = sum / count
            log_components[key] = to_log
            # Set the iterator status
            log_key, name_key = key.split("/")
            iterator_log_name = f"{log_key[0]}{name_key[0]}".upper()
            iterator_log_component[iterator_log_name] = to_log
        postfix = ",".join(
            "{}:{:.2e}".format(key, iterator_log_component[key])
            for key in iterator_log_component.keys()
        )
        if iterator is not None:
            iterator.set_postfix_str(postfix)
        wandb.log(log_components, step=epoch)
        self.log_components = OrderedDict()


class SaveModule(nn.Module):
    def set_snapshot_path(self, path):
        self.snapshot_path = path
        print(f"Setting snapshot path to {self.snapshot_path}")

    def save_snapshot(self):
        os.makedirs(self.snapshot_path, exist_ok=True)
        torch.save(self.state_dict(), self.snapshot_path / "snapshot.pth")

    def load_snapshot(self):
        self.load_state_dict(torch.load(self.snapshot_path / "snapshot.pth"))


def split_datasets(dataset, train_fraction=0.95, random_seed=42):
    dataset_length = len(dataset)
    lengths = [
        int(train_fraction * dataset_length),
        dataset_length - int(train_fraction * dataset_length),
    ]
    train_set, val_set = random_split(
        dataset, lengths, generator=torch.Generator().manual_seed(random_seed)
    )
    return train_set, val_set


def reduce_dict(f: Callable, d: Dict):
    return {k: reduce_dict(f, v) if isinstance(v, dict) else f(v) for k, v in d.items()}


def get_hydra_jobnum_workdir():
    if HydraConfig.get().mode == RunMode.MULTIRUN:
        job_num = HydraConfig.get().job.num
        work_dir = Path(HydraConfig.get().sweep.dir) / HydraConfig.get().sweep.subdir
    else:
        job_num = 0
        work_dir = HydraConfig.get().run.dir
    return job_num, work_dir
