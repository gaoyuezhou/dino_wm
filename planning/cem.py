import torch
import numpy as np
from einops import rearrange, repeat
import torch.nn.functional as F
from .base_planner import BasePlanner
from utils import move_to_device


class CEMPlanner(BasePlanner):
    def __init__(
        self,
        horizon,
        topk,
        num_samples,
        opt_steps,
        eval_every,
        wm,
        action_dim,
        objective_fn,
        preprocessor,
        evaluator,
        wandb_run,
        discrete_actions,
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
        self.topk = topk
        self.num_samples = num_samples
        self.opt_steps = opt_steps
        self.eval_every = eval_every
        self.logging_prefix = logging_prefix
        actions_tensor = torch.tensor(discrete_actions, dtype=torch.float32)
        self.actions_set = self.preprocessor.normalize_actions(actions_tensor).to(
            self.device
        )
        


    def plan(self, obs_0, obs_g, actions=None):
        """
        Args:
            actions: normalized
        Returns:
            actions: (B, T, action_dim) torch.Tensor, T <= self.horizon
        """
        trans_obs_0 = move_to_device(
            self.preprocessor.transform_obs(obs_0), self.device
        )
        trans_obs_g = move_to_device(
            self.preprocessor.transform_obs(obs_g), self.device
        )
        z_obs_g = self.wm.encode_obs(trans_obs_g)
        n_evals = trans_obs_0["visual"].shape[0]
        logits = torch.zeros(
            n_evals, self.horizon, len(self.actions_set), device=self.device
        )
        elite_actions = torch.zeros(
            n_evals, self.horizon, self.action_dim, device=self.device
        )

        for i in range(self.opt_steps):
            losses = []
            for traj in range(n_evals):
                cur_trans_obs_0 = {
                    key: repeat(
                        arr[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples
                    )
                    for key, arr in trans_obs_0.items()
                }
                cur_z_obs_g = {
                    key: repeat(
                        arr[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples
                    )
                    for key, arr in z_obs_g.items()
                }

                dist = torch.distributions.Categorical(logits=logits[traj])
                indices = dist.sample((self.num_samples,))
                action = self.actions_set[indices]
                if self.actions_set.ndim == 1:
                    action = action.unsqueeze(-1)
                
                
                with torch.no_grad():
                    i_z_obses, i_zs = self.wm.rollout(
                        obs_0=cur_trans_obs_0,
                        act=action,
                    )

                loss = self.objective_fn(i_z_obses, cur_z_obs_g)
                topk_idx = torch.argsort(loss)[: self.topk]
                elite_idx = indices[topk_idx]
                losses.append(loss[topk_idx[0]].item())

                counts = F.one_hot(
                    elite_idx, num_classes=len(self.actions_set)
                ).float().sum(dim=0)
                probs = counts / counts.sum(dim=-1, keepdim=True)
                logits[traj] = torch.log(probs + 1e-8)

                best_seq_idx = elite_idx[0]
                best_seq = self.actions_set[best_seq_idx]
                if self.actions_set.ndim == 1:
                    best_seq = best_seq.unsqueeze(-1)
                elite_actions[traj] = best_seq

            self.wandb_run.log(
                {f"{self.logging_prefix}/loss": np.mean(losses), "step": i + 1}
            )
            if self.evaluator is not None and i % self.eval_every == 0:
                logs, successes, _, _ = self.evaluator.eval_actions(
                    elite_actions, filename=f"{self.logging_prefix}_output_{i+1}"
                )
                logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}
                logs.update({"step": i + 1})
                self.wandb_run.log(logs)
                self.dump_logs(logs)
                if np.all(successes):
                    break

        return elite_actions, np.full(n_evals, np.inf)
