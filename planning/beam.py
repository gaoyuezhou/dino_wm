import torch
import numpy as np
from einops import repeat
from .base_planner import BasePlanner
from utils import move_to_device


class BeamPlanner(BasePlanner):
    def __init__(
        self,
        horizon,
        beam_width,
        discrete_actions,
        wm,
        action_dim,
        objective_fn,
        preprocessor,
        evaluator,
        wandb_run,
        logging_prefix="plan_0",
        log_filename="logs.json",
        **kwargs,
    ):
        super().__init__(
            wm,
            action_dim,
            objective_fn,
            preprocessor,
            evaluator,
            wandb_run,
            log_filename,
        )
        self.horizon = horizon
        self.beam_width = beam_width
        self.logging_prefix = logging_prefix
        self.score_mode = "prefix"  # "last" or "prefix"

        actions_tensor = torch.tensor(discrete_actions, dtype=torch.float32)
        self.actions_set = self.preprocessor.normalize_actions(actions_tensor).to(self.device)
        if self.actions_set.ndim == 1:
            self.actions_set = self.actions_set.unsqueeze(-1)

        zero_action = self.preprocessor.normalize_actions(
            torch.zeros(1, 1, self.action_dim)
        ).to(self.device)
        self.zero_action = zero_action[0, 0]

    def plan(self, obs_0, obs_g, actions=None):
        trans_obs_0 = move_to_device(
            self.preprocessor.transform_obs(obs_0), self.device
        )
        trans_obs_g = move_to_device(
            self.preprocessor.transform_obs(obs_g), self.device
        )
        z_obs_g = self.wm.encode_obs(trans_obs_g)
        n_evals = trans_obs_0["visual"].shape[0]
        elite_actions = torch.zeros(
            n_evals, self.horizon, self.action_dim, device=self.device
        )
        losses_log = []

        for traj in range(n_evals):
            cur_trans_obs_0 = {
                key: value[traj : traj + 1] for key, value in trans_obs_0.items()
            }
            cur_z_obs_g = {
                key: value[traj : traj + 1] for key, value in z_obs_g.items()
            }

            sequences = self.zero_action.repeat(1, self.horizon, 1)
            for t in range(self.horizon):
                num_beams = sequences.shape[0]
                num_actions = self.actions_set.shape[0]
                # expand
                expanded = sequences.repeat_interleave(num_actions, dim=0)
                expanded[:, t] = self.actions_set.repeat(num_beams, 1)

                cur_obs_batch = {
                    key: repeat(arr, "1 ... -> n ...", n=expanded.shape[0])
                    for key, arr in cur_trans_obs_0.items()
                }
                cur_goal_batch = {
                    key: repeat(arr, "1 ... -> n ...", n=expanded.shape[0])
                    for key, arr in cur_z_obs_g.items()
                }

                with torch.no_grad():
                    i_z_obses, i_zs = self.wm.rollout(obs_0=cur_obs_batch, act=expanded)
                losses = self.objective_fn(i_z_obses, cur_goal_batch)
                topk = torch.argsort(losses)[: self.beam_width]
                sequences = expanded[topk]
                losses = losses[topk]

            elite_actions[traj] = sequences[0]
            losses_log.append(losses[0].item())

        self.wandb_run.log(
            {f"{self.logging_prefix}/loss": float(np.mean(losses_log)), "step": 1}
        )
        if self.evaluator is not None:
            logs, successes, _, _ = self.evaluator.eval_actions(
                elite_actions, filename=f"{self.logging_prefix}_output"
            )
            logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}
            logs.update({"step": 1})
            self.wandb_run.log(logs)
            self.dump_logs(logs)

        return elite_actions, np.full(n_evals, np.inf)
