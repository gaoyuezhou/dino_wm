import json
import os
import random
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from einops import rearrange
from torch.utils.data import Dataset

from .traj_dset import TrajDataset, get_train_val_sliced, TrajSlicerDataset


class RearrangeDataset(TrajDataset):
    def __init__(
        self,
        data_path: str = str(Path(os.getenv("DATASET_DIR")) / "rearrange_1k"),
        n_rollout: Optional[int] = None,
        transform: Optional[Callable] = None,
        normalize_action: bool = False,
    ):
        self.data_path = Path(data_path)
        self.transform = transform
        self.normalize_action = normalize_action

        meta_file = self.data_path / "metadata.json"
        with open(meta_file, "r") as f:
            meta = json.load(f)

        episodes = meta.get("episodes", [])
        if n_rollout is not None:
            episodes = episodes[:n_rollout]
        self.seq_lengths = [ep["n_actions"] for ep in episodes]

        self.actions = []
        self.obs_paths = []
        for ep in episodes:
            ep_id = ep["episode"]
            ep_dir = self.data_path / "episodes" / f"ep_{ep_id:04d}"
            actions = np.load(ep_dir / "actions.npy")
            self.actions.append(torch.as_tensor(actions).float())
            self.obs_paths.append(ep_dir / "obs.npy")

        if len(self.actions) > 0:
            self.action_dim = (
                self.actions[0].shape[-1] if self.actions[0].ndim > 1 else 1
            )
        else:
            self.action_dim = 0

        self.state_dim = 0
        self.proprio_dim = 1

        if normalize_action and len(self.actions) > 0:
            all_actions = torch.cat(self.actions, dim=0)
            self.action_mean = all_actions.mean(dim=0)
            self.action_std = all_actions.std(dim=0) + 1e-6
        else:
            self.action_mean = torch.zeros(self.action_dim)
            self.action_std = torch.ones(self.action_dim)
        self.state_mean = torch.zeros(self.state_dim)
        self.state_std = torch.ones(self.state_dim)
        self.proprio_mean = torch.zeros(self.proprio_dim)
        self.proprio_std = torch.ones(self.proprio_dim)

        # normalize actions in place
        for i in range(len(self.actions)):
            self.actions[i] = (self.actions[i] - self.action_mean) / self.action_std

    def get_seq_length(self, idx):
        return self.seq_lengths[idx]

    def get_all_actions(self):
        return torch.cat(self.actions, dim=0)

    def get_frames(self, idx, frames):
        obs_arr = np.load(self.obs_paths[idx])
        act = self.actions[idx][frames]
        state = torch.zeros(len(frames), self.state_dim)
        proprio = torch.zeros(len(frames), self.proprio_dim)

        image = torch.as_tensor(obs_arr[frames])  # THWC
        image = rearrange(image, "T H W C -> T C H W") / 255.0
        if self.transform:
            image = self.transform(image)
        obs = {"visual": image, "proprio": proprio}
        return obs, act, state, {}

    def __getitem__(self, idx):
        return self.get_frames(idx, range(self.get_seq_length(idx)))

    def __len__(self):
        return len(self.seq_lengths)

    def preprocess_imgs(self, imgs):
        if isinstance(imgs, np.ndarray):
            raise NotImplementedError
        elif isinstance(imgs, torch.Tensor):
            return rearrange(imgs, "b h w c -> b c h w") / 255.0


class FilteredDatasetWrapper(Dataset):
    def __init__(self, base_dataset, indices):
        self.base = base_dataset
        self.indices = indices

        # Forward metadata attributes
        for attr in [
            "proprio_dim", "state_dim", "action_dim",
            "proprio_mean", "proprio_std",
            "state_mean", "state_std",
            "action_mean", "action_std",
        ]:
            if hasattr(base_dataset, attr):
                setattr(self, attr, getattr(base_dataset, attr))

    def __getitem__(self, idx):
        return self.base[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def select_condition_then_sample_rest(dataset, n_slices, target_actions={4, 5}, seed=42, verbose=False):
    """Filter ``dataset`` based on the second-to-last action of each slice.

    All slices whose ``action[-2]`` is contained in ``target_actions`` are
    selected first.  If ``dataset`` is a :class:`TrajSlicerDataset` (or exposes
    ``slices`` and ``dataset.actions`` attributes), the action for each slice is
    reconstructed directly from ``dataset.slices[idx]`` and the underlying
    ``dataset.dataset.actions`` without invoking ``dataset[idx]``.  When the
    underlying dataset is a ``Subset``, ``dataset.dataset.indices`` is used to
    map slice indices back to the original trajectories.  Datasets without these
    attributes fall back to calling ``dataset[idx]``.  Random samples from the
    remaining slices are then added until ``n_slices`` entries are obtained.
    ``seed`` ensures deterministic sampling.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    match_indices = []
    non_match_indices = []

    for idx in range(len(dataset)):
        if isinstance(dataset, TrajSlicerDataset) or (
            hasattr(dataset, "slices")
            and hasattr(dataset, "dataset")
            and hasattr(dataset.dataset, "actions")
        ):
            traj_idx, start, end = dataset.slices[idx]
            if hasattr(dataset.dataset, "indices"):
                traj_idx = dataset.dataset.indices[traj_idx]
            act = dataset.dataset.actions[traj_idx][start:end]
            act = torch.as_tensor(act)
            num_frames = getattr(dataset, "num_frames", act.shape[0])
            frameskip = getattr(dataset, "frameskip", 1)
            if act.ndim == 1:
                act = act.reshape(num_frames, frameskip)
            else:
                act = act.reshape(num_frames, frameskip * act.shape[-1])
        else:
            _, act, _ = dataset[idx]

        if act.shape[0] >= 2:
            last_action = act[-2]
            action_val = int(last_action.item()) if last_action.ndim == 0 else int(last_action[0].item())
            if action_val in target_actions:
                match_indices.append(idx)
            else:
                non_match_indices.append(idx)
        else:
            non_match_indices.append(idx)

    if verbose:
        print(f"[Filter] Matched: {len(match_indices)} | Non-matched: {len(non_match_indices)}")

    if len(match_indices) > n_slices:
        sampled_indices = random.sample(match_indices, n_slices)
    else:
        n_remaining = n_slices - len(match_indices)
        sampled_rest = random.sample(non_match_indices, min(n_remaining, len(non_match_indices)))
        sampled_indices = match_indices + sampled_rest

    return FilteredDatasetWrapper(dataset, sampled_indices)


def load_rearrange_slice_train_val(
    transform,
    n_rollout: int = 200,
    data_path: str = str(Path(os.getenv("DATASET_DIR"), "rearrange_1k")),
    normalize_action: bool = False,
    split_ratio: float = 0.9,
    num_hist: int = 3,
    num_pred: int = 1,
    frameskip: int = 1,
    # Filtering options
    filter_train: bool = False,
    filter_val: bool = False,
    n_slices_train: Optional[int] = None,
    n_slices_val: Optional[int] = None,
    filter_actions: set = {4, 5},
    seed_train: int = 42,
    seed_val: int = 99,
    verbose: bool = False,
):
    """
    Loads RearrangeDataset, slices into train/val, and optionally filters both
    using action[-2] match logic.
    """
    dset = RearrangeDataset(
        n_rollout=n_rollout,
        transform=transform,
        data_path=data_path,
        normalize_action=normalize_action,
    )
    print("created dset")

    dset_train, dset_val, train_slices, val_slices = get_train_val_sliced(
        traj_dataset=dset,
        train_fraction=split_ratio,
        num_frames=num_hist + num_pred,
        frameskip=frameskip,
    )
    print("created dset_train, dset_val, train_slices, val_slices ")

    if filter_train:
        if n_slices_train is None:
            raise ValueError("n_slices_train must be set if filter_train is True")
        train_slices = select_condition_then_sample_rest(
            dataset=train_slices,
            n_slices=n_slices_train,
            target_actions=filter_actions,
            seed=seed_train,
            verbose=verbose,
        )
        print("filtered train_slices")

    if filter_val:
        if n_slices_val is None:
            raise ValueError("n_slices_val must be set if filter_val is True")
        val_slices = select_condition_then_sample_rest(
            dataset=val_slices,
            n_slices=n_slices_val,
            target_actions=filter_actions,
            seed=seed_val,
            verbose=verbose,
        )
        print("filtered val_slices")
    

    datasets = {"train": train_slices, "valid": val_slices}
    traj_dset = {"train": dset_train, "valid": dset_val}
    return datasets, traj_dset

