import utils
import hydra
import torch
import einops

import numpy as np

from workspaces import base
from utils import get_split_idx

from accelerate import Accelerator

accelerator = Accelerator()
OBS_ELEMENT_INDICES = {
    "agent_pos": np.arange(0, 2),
    "block_pos": np.arange(2, 4),
    "block_rot": np.arange(4, 5),
}


def calc_state_dist(a, b):
    result = {}
    for k, v in OBS_ELEMENT_INDICES.items():
        idx = torch.Tensor(v).long()
        result[k] = ((a[idx] - b[idx]) ** 2).mean()
    result["total"] = ((a - b) ** 2).mean()
    return result


def mean_dicts(dicts):
    result = {}
    for k in dicts[0].keys():
        result[k] = np.mean([x[k] for x in dicts])
    return result


class PushTWorkspace(base.Workspace):
    def __init__(self, cfg, work_dir):
        super().__init__(cfg, work_dir)

    def _report_result_upon_completion(self, goal_idx=None):
        return {
            "max_coverage": max(self.env.coverage_arr),
            "final_coverage": self.env.coverage_arr[-1],
        }

    def run_offline_eval(self):
        train_idx, val_idx = get_split_idx(
            len(self.dataset),
            self.cfg.seed,
            train_fraction=self.cfg.train_fraction,
        )
        embeddings = utils.inference.embed_trajectory_dataset(
            self.encoder, self.dataset
        )
        embeddings = [
            einops.rearrange(x, "T V E -> T (V E)") for x in embeddings
        ]  # flatten views
        if self.accelerator.is_main_process:
            states = []
            actions = []
            for i in range(len(self.dataset)):
                T = self.dataset.get_seq_length(i)
                states.append(self.dataset.states[i, :T])
                actions.append(self.dataset.actions[i, :T])
            embd_state_linear_probe_results = (
                utils.inference.linear_probe_with_trajectory_split(
                    embeddings,
                    states,
                    train_idx,
                    val_idx,
                )
            )
            # add prefix to keys
            embd_state_linear_probe_results = {
                f"embd_state_{k}": v for k, v in embd_state_linear_probe_results.items()
            }
            embd_action_linear_probe_results = (
                utils.inference.linear_probe_with_trajectory_split(
                    embeddings,
                    actions,
                    train_idx,
                    val_idx,
                )
            )
            embd_action_linear_probe_results = {
                f"embd_action_{k}": v
                for k, v in embd_action_linear_probe_results.items()
            }

            state_dists = []
            N = 200
            rng = np.random.default_rng(self.cfg.seed)
            for i in range(N):
                query_traj_idx = rng.choice(len(self.dataset))
                query_frame_idx = rng.choice(
                    range(10, self.dataset.get_seq_length(query_traj_idx))
                )
                query_embedding = embeddings[query_traj_idx][query_frame_idx]
                query_frame_state = self.dataset.states[query_traj_idx, query_frame_idx]

                pool_embeddings = torch.cat(
                    [x for i, x in enumerate(embeddings) if i != query_traj_idx]
                )
                pool_states = torch.cat(
                    [x for i, x in enumerate(states) if i != query_traj_idx]
                )
                _, nn_idx = utils.inference.batch_knn(
                    query_embedding.unsqueeze(0),
                    pool_embeddings,
                    metric=utils.inference.mse,
                    k=1,
                    batch_size=1,
                )
                closest_frame_state = pool_states[nn_idx[0, 0]]
                state_dist = calc_state_dist(query_frame_state, closest_frame_state)
                state_dists.append(state_dist)
            mean_state_dist = mean_dicts(state_dists)
            return {
                **embd_state_linear_probe_results,
                **embd_action_linear_probe_results,
                **mean_state_dist,
            }
        else:
            return None
