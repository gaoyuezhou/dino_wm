import json
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from einops import rearrange

from .traj_dset import TrajDataset, get_train_val_sliced, TrajSlicerDataset


class RearrangeDataset(TrajDataset):
    def __init__(
        self,
        data_path: str = "data/rearrange_1k",
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
            self.action_dim = self.actions[0].shape[-1] if self.actions[0].ndim > 1 else 1
        else:
            self.action_dim = 0
        self.state_dim = 1
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


def load_rearrange_slice_train_val(
    transform,
    n_rollout=50,
    data_path="data/rearrange_1k",
    normalize_action=False,
    split_ratio=0.8,
    num_hist=0,
    num_pred=0,
    frameskip=0,
):
    dset = RearrangeDataset(
        n_rollout=n_rollout,
        transform=transform,
        data_path=data_path,
        normalize_action=normalize_action,
    )
    dset_train, dset_val, train_slices, val_slices = get_train_val_sliced(
        traj_dataset=dset,
        train_fraction=split_ratio,
        num_frames=num_hist + num_pred,
        frameskip=frameskip,
    )

    datasets = {"train": train_slices, "valid": val_slices}
    traj_dset = {"train": dset_train, "valid": dset_val}
    return datasets, traj_dset
