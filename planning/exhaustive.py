import torch
import numpy as np
from einops import repeat
import torch.nn.functional as F  # kept for parity; not used here
from contextlib import nullcontext

from .base_planner import BasePlanner
from utils import move_to_device


class ExhaustivePlanner(BasePlanner):
    """
    Exact full-search planner for small action/horizon:
      - Enumerates ALL A^H sequences (on-the-fly, in chunks).
      - Scores end-to-end with the world model/objective.
      - Picks the best sequence per start state.

    API compatibility:
      - Constructor signature mirrors CEM/TrajectoryCEMPlanner.
      - Keeps unused knobs (alpha/p_min/temp_start/temp_end) to avoid config breaks.
      - Returns (elite_actions, best_losses) with same shapes.
      - Uses wandb/evaluator like your other planners.

    Extra kwargs (optional):
      - chunk_size: int, how many sequences to evaluate per rollout chunk (default 2048).
      - use_autocast: bool, use AMP on CUDA if available (default False).
    """

    def __init__(self, horizon, topk, num_samples, opt_steps, eval_every,
                 wm, action_dim, objective_fn, preprocessor, evaluator, wandb_run,
                 discrete_actions, logging_prefix="plan_exhaustive", log_filename="logs.json",
                 # Unused legacy knobs kept for config compatibility:
                 alpha=0.5, p_min=0.02, temp_start=1.0, temp_end=0.5,
                 **kwargs):
        super().__init__(wm, action_dim, objective_fn, preprocessor, evaluator, wandb_run, log_filename)

        self.H = int(horizon)
        self.K = int(min(topk, max(1, num_samples)))  # kept for symmetry; not used
        self.N = int(max(1, num_samples))             # not used for enumeration, kept for symmetry
        self.T = int(max(1, opt_steps))               # we run a single pass; we still log step=1
        self.eval_every = int(eval_every)
        self.logging_prefix = logging_prefix

        # Extra (optional) controls
        self.chunk_size = int(kwargs.get("chunk_size", 2048))
        self.use_autocast = bool(kwargs.get("use_autocast", False))

        # Normalized action set (matches your planners)
        actions_tensor = torch.as_tensor(discrete_actions, dtype=torch.float32)
        self.actions_set = self.preprocessor.normalize_actions(actions_tensor).to(self.device)
        self.A = len(self.actions_set)

    # ---- Base-A enumeration for a contiguous block of indices ----
    def _make_index_block(self, start: int, count: int):
        """
        Returns a [count, H] tensor of action indices representing the block
        {start, ..., start+count-1} in base-A, without materializing all A^H.
        """
        A, H, dev = self.A, self.H, self.device
        ids = torch.arange(start, start + count, device=dev, dtype=torch.long)
        seq = torch.empty(count, H, dtype=torch.long, device=dev)
        # Fill digits from the right (least significant)
        for t in range(H - 1, -1, -1):
            seq[:, t] = ids % A
            ids //= A
        return seq  # [count, H]

    def plan(self, obs_0, obs_g, actions=None):
        # ---- Preprocess & encode goal (same as your CEM) ----
        trans_obs_0 = move_to_device(self.preprocessor.transform_obs(obs_0), self.device)
        trans_obs_g = move_to_device(self.preprocessor.transform_obs(obs_g), self.device)
        z_obs_g = self.wm.encode_obs(trans_obs_g)
        if "visual" not in z_obs_g and "visual_tokens" in z_obs_g:
            z_obs_g["visual"] = z_obs_g["visual_tokens"]

        n_evals = trans_obs_0["visual"].shape[0]
        A, H = self.A, self.H
        total = A ** H  # total sequences

        best_losses = torch.full((n_evals,), float("inf"), device=self.device)
        best_seqs = torch.zeros(n_evals, H, dtype=torch.long, device=self.device)

        # AMP context (CUDA only)
        amp_ctx = (
            torch.cuda.amp.autocast(enabled=True)
            if (self.use_autocast and torch.cuda.is_available() and self.device.type == "cuda")
            else nullcontext()
        )

        # Enumerate per start state (keeps peak RAM small)
        with torch.no_grad(), amp_ctx:
            for b in range(n_evals):
                cur_obs_0_1 = {k: trans_obs_0[k][b:b+1] for k in trans_obs_0}
                cur_goal_1  = {k: z_obs_g[k][b:b+1]      for k in z_obs_g}

                chunk_best_log = []  # for optional logging

                # Iterate over blocks of sequences
                for start in range(0, total, self.chunk_size):
                    count = min(self.chunk_size, total - start)
                    idx_chunk = self._make_index_block(start, count)  # [count, H]

                    # Map to concrete actions
                    acts = self.actions_set[idx_chunk]               # [count, H, action_dim] or [count, H]
                    if self.actions_set.ndim == 1:
                        acts = acts.unsqueeze(-1)

                    # Repeat obs/goal to match chunk
                    cur_obs_rep  = {k: repeat(v, "1 ... -> n ...", n=acts.shape[0]) for k, v in cur_obs_0_1.items()}
                    cur_goal_rep = {k: repeat(v, "1 ... -> n ...", n=acts.shape[0]) for k, v in cur_goal_1.items()}

                    # Score with world model + objective
                    z_obses, _ = self.wm.rollout(obs_0=cur_obs_rep, act=acts)
                    loss = self.objective_fn(z_obses, cur_goal_rep).reshape(-1)  # [count]

                    # Track best in this block
                    min_idx = torch.argmin(loss)
                    min_val = loss[min_idx]
                    if min_val < best_losses[b]:
                        best_losses[b] = min_val
                        best_seqs[b] = idx_chunk[min_idx]
                    chunk_best_log.append(min_val.item())

                # Optional per-eval logging
                self.wandb_run.log({f"{self.logging_prefix}/chunk_best_loss": float(np.min(chunk_best_log))})

        # Convert best sequences to concrete action vectors
        elite_actions = self.actions_set[best_seqs]
        if self.actions_set.ndim == 1:
            elite_actions = elite_actions.unsqueeze(-1)

        # One-step summary log for parity with CEM
        self.wandb_run.log({f"{self.logging_prefix}/loss": float(best_losses.mean().item()), "step": 1})

        # Optional evaluation hook, same as your planners
        if self.evaluator is not None and (self.eval_every > 0):
            logs, successes, _, _, _ = self.evaluator.eval_actions(
                elite_actions, filename=f"{self.logging_prefix}_output_final"
            )
            logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}
            self.wandb_run.log(logs)
            self.dump_logs(logs)

        return elite_actions, best_losses.detach().cpu().numpy()
