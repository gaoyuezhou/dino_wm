import torch
import numpy as np
from einops import rearrange, repeat
import torch.nn.functional as F
from .base_planner import BasePlanner
from utils import move_to_device


class CEMPlanner(BasePlanner):
    def __init__(self, horizon, topk, num_samples, opt_steps, eval_every,
                 wm, action_dim, objective_fn, preprocessor, evaluator, wandb_run,
                 discrete_actions, logging_prefix="plan_0", log_filename="logs.json",
                 alpha=0.5, p_min=0.02, temp_start=1.0, temp_end=0.5, **kwargs):
        super().__init__(wm, action_dim, objective_fn, preprocessor, evaluator, wandb_run, log_filename)
        self.horizon = horizon
        self.topk = topk
        self.num_samples = num_samples
        self.opt_steps = opt_steps
        self.eval_every = eval_every
        self.logging_prefix = logging_prefix
        self.alpha = alpha
        self.p_min = p_min
        self.temp_start = temp_start
        self.temp_end = temp_end

        actions_tensor = torch.as_tensor(discrete_actions, dtype=torch.float32)
        self.actions_set = self.preprocessor.normalize_actions(actions_tensor).to(self.device)

    def plan(self, obs_0, obs_g, actions=None):
        trans_obs_0 = move_to_device(self.preprocessor.transform_obs(obs_0), self.device)
        trans_obs_g = move_to_device(self.preprocessor.transform_obs(obs_g), self.device)
        z_obs_g = self.wm.encode_obs(trans_obs_g)
        if "visual" not in z_obs_g and "visual_tokens" in z_obs_g:
            z_obs_g["visual"] = z_obs_g["visual_tokens"]

        n_evals = trans_obs_0["visual"].shape[0]
        A = len(self.actions_set)
        H = self.horizon
        logits = torch.zeros(n_evals, H, A, device=self.device)           # uniform start

        elite_actions = torch.zeros(n_evals, H, self.action_dim, device=self.device)
        best_losses = torch.full((n_evals,), float("inf"))

        # helper to get annealed temperature per iter
        def iter_temp(it):
            if self.opt_steps <= 1: return self.temp_end
            t = it / (self.opt_steps - 1)
            return self.temp_start * (1 - t) + self.temp_end * t

        for it in range(self.opt_steps):
            losses_for_logging = []
            Tcur = iter_temp(it)

            # (Optional) vectorization across trajs is possible; we keep the simple per-traj loop
            for traj in range(n_evals):
                # Repeat the single start state for this traj to size [N]
                cur_trans_obs_0 = {k: repeat(v[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples)
                                   for k, v in trans_obs_0.items()}
                cur_z_obs_g = {k: repeat(v[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples)
                               for k, v in z_obs_g.items()}

                # Sample sequences from per-t categorical
                dist = torch.distributions.Categorical(logits=logits[traj] / Tcur)  # [H, A]
                indices = dist.sample((self.num_samples,))                           # [N, H]
                action = self.actions_set[indices]                                   # [N, H, action_dim] or [N,H]
                if self.actions_set.ndim == 1:
                    action = action.unsqueeze(-1)

                with torch.no_grad():
                    z_obses, _ = self.wm.rollout(obs_0=cur_trans_obs_0, act=action)

                loss = self.objective_fn(z_obses, cur_z_obs_g).reshape(-1)          # [N], lower is better
                k = min(self.topk, self.num_samples)
                topk_idx = torch.topk(loss, k=k, largest=False).indices             # [k]
                losses_for_logging.append(loss[topk_idx[0]].item())

                elite_idx = indices[topk_idx]                                        # [k, H]
                counts = F.one_hot(elite_idx, num_classes=A).float().sum(dim=0)     # [H, A]
                denom = counts.sum(dim=-1, keepdim=True)                             # [H, 1]
                # Safe normalization (should be k, but guard anyway)
                probs_elite = counts / torch.clamp_min(denom, 1.0)

                # Smooth + floor
                probs_old = torch.softmax(logits[traj] / Tcur, dim=-1)               # [H, A]
                probs_new = (1 - self.alpha) * probs_old + self.alpha * probs_elite  # [H, A]
                probs_new = torch.clamp(probs_new, min=self.p_min)
                probs_new = probs_new / probs_new.sum(dim=-1, keepdim=True)
                logits[traj] = torch.log(probs_new + 1e-8)

                # Keep per-traj current best (greedy on elites’ best)
                best_seq_idx = elite_idx[0]                                          # [H]
                best_seq = self.actions_set[best_seq_idx]                            # [H, action_dim] or [H]
                if self.actions_set.ndim == 1:
                    best_seq = best_seq.unsqueeze(-1)

                # Update the running best if improved
                if loss[topk_idx[0]].item() < best_losses[traj]:
                    best_losses[traj] = loss[topk_idx[0]].item()
                    elite_actions[traj] = best_seq

            # logging
            self.wandb_run.log({f"{self.logging_prefix}/loss": float(np.mean(losses_for_logging)), "step": it + 1})
            if self.evaluator is not None and (self.eval_every > 0) and (it % self.eval_every == 0):
                logs, successes, _, _ = self.evaluator.eval_actions(elite_actions, filename=f"{self.logging_prefix}_output_{it+1}")
                logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}
                logs.update({"step": it + 1})
                self.wandb_run.log(logs)
                self.dump_logs(logs)
                if np.all(successes):
                    break

        return elite_actions, best_losses.cpu().numpy()
