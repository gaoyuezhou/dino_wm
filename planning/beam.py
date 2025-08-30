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
      # Transform & encode
      trans_obs_0 = move_to_device(self.preprocessor.transform_obs(obs_0), self.device)
      trans_obs_g = move_to_device(self.preprocessor.transform_obs(obs_g), self.device)
      z_obs_g = self.wm.encode_obs(trans_obs_g)
      if "visual" not in z_obs_g and "visual_tokens" in z_obs_g:
          z_obs_g["visual"] = z_obs_g["visual_tokens"]

      n_evals = trans_obs_0["visual"].shape[0]
      elite_actions = torch.zeros(n_evals, self.horizon, self.action_dim, device=self.device)
      losses_log = []

      # allow either "score_mode" or "scoremode"; default "last"
      score_mode = getattr(self, "score_mode", getattr(self, "scoremode", "last"))
      if score_mode not in ("last", "prefix"):
          raise ValueError(f"Unknown score_mode: {score_mode}")

      def _time_tile_goal(goal_batch, L, mode):
          """
          Ensure goal has a time dimension matching the preds:
          - mode 'last': make sure time dim == 1
          - mode 'prefix': expand/match time dim to L
          Works for shapes [B, D], [B, 1, D], [B, 1, N, D], etc.
          """
          out = {}
          for k, v in goal_batch.items():
              # Ensure at least [B, ...]
              assert v.ndim >= 2, f"Goal tensor {k} must be at least [B, ...], got {v.shape}"
              if mode == "last":
                  # want time dim = 1 at dim=1
                  if v.ndim == 2:
                      # [B, D] -> [B, 1, D]
                      v = v.unsqueeze(1)
                  else:
                      # [B, T?, ...] -> take first time slice if T != 1
                      if v.shape[1] != 1:
                          v = v[:, :1]
              else:  # "prefix"
                  if v.ndim == 2:
                      # [B, D] -> [B, L, D]
                      v = v.unsqueeze(1).expand(-1, L, -1)
                  else:
                      # [B, T?, ...] -> expand time dim to L
                      if v.shape[1] == 1:
                          sizes = list(v.shape)
                          sizes[1] = L
                          v = v.expand(*sizes)
                      elif v.shape[1] != L:
                          # tile/repeat along time, then slice to L
                          reps = [1] * v.ndim
                          reps[1] = L
                          v = v.repeat(*reps)[:, :L]
              out[k] = v
          return out

      for traj in range(n_evals):
          cur_trans_obs_0 = {key: value[traj: traj + 1] for key, value in trans_obs_0.items()}
          cur_z_obs_g     = {key: value[traj: traj + 1] for key, value in z_obs_g.items()}

          sequences = self.zero_action.repeat(1, self.horizon, 1)

          for t in range(self.horizon):
              num_beams   = sequences.shape[0]
              num_actions = self.actions_set.shape[0]

              # expand beams by all actions at step t
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

              # WM rollout (full horizon)
              with torch.no_grad():
                  i_z_obses, _ = self.wm.rollout(obs_0=cur_obs_batch, act=expanded)
                  # i_z_obses[k] can be [B, H, D] or [B, H, N, D], etc.

              # Score only decided part
              L = t + 1
              if score_mode == "last":
                  preds = {k: v[:, t:t+1] for k, v in i_z_obses.items()}    # [B, 1, ...]
                  goal  = _time_tile_goal(cur_goal_batch, L=1, mode="last") # [B, 1, ...]
              else:  # "prefix"
                  preds = {k: v[:, :L] for k, v in i_z_obses.items()}       # [B, L, ...]
                  goal  = _time_tile_goal(cur_goal_batch, L=L, mode="prefix")

              # (Optional) sanity: time dims must match
              # for k in preds:
              #     assert preds[k].shape[1] == goal[k].shape[1], f"time mismatch for '{k}': {preds[k].shape} vs {goal[k].shape}"

              losses = self.objective_fn(preds, goal)  # expect shape [B]
              # If your loss could return extra dims, reduce to [B]:
              if losses.ndim > 1:
                  losses = losses.reshape(losses.shape[0], -1).mean(dim=1)

              # select top-k
              topk = torch.argsort(losses)[: self.beam_width]
              sequences = expanded[topk]
              losses    = losses[topk]

              # light progress print
              try:
                  print(f"[{self.logging_prefix}] Planning step {t+1}/{self.horizon} | min loss: {losses.min().item():.4f}")
              except Exception:
                  pass

          elite_actions[traj] = sequences[0]
          losses_log.append(losses[0].item())

      self.wandb_run.log({f"{self.logging_prefix}/loss": float(np.mean(losses_log)), "step": 1})

      if self.evaluator is not None:
          logs, successes, _, _ = self.evaluator.eval_actions(
              elite_actions, filename=f"{self.logging_prefix}_output"
          )
          logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}
          logs.update({"step": 1})
          self.wandb_run.log(logs)
          self.dump_logs(logs)

      return elite_actions, np.full(n_evals, np.inf)

