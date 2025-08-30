import torch
import numpy as np
from einops import rearrange, repeat
import torch.nn.functional as F
from .base_planner import BasePlanner
from utils import move_to_device


class CEMPlannerDiscrete(BasePlanner):
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
                logs, successes, _, _, _ = self.evaluator.eval_actions(elite_actions, filename=f"{self.logging_prefix}_output_{it+1}")
                logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}
                logs.update({"step": it + 1})
                self.wandb_run.log(logs)
                self.dump_logs(logs)
                if np.all(successes):
                    break

        return elite_actions, best_losses.cpu().numpy()




class TrajectoryCEMPlanner(BasePlanner):
    """
    Full-trajectory CEM:
      - Maintains a pool of full sequences (indices in [0..A-1]).
      - Scores end-to-end with the world model/objective.
      - Selects top-k elites.
      - Refills pool by crossover + mutation, with soft, loss-weighted parent sampling.
      - Supports annealed mutation, block (segment) mutations, duplicate control, and restarts.

    Reuses your interface: num_samples=N (population size), topk=K, opt_steps=T, etc.

    New kwargs (all optional, sensible defaults for H<10):
      - parent_temp: temperature for softmax over -elite_loss when sampling parents (default 0.5).
      - crossover_prob: probability to perform crossover when generating a child (default 0.9).
      - two_point_crossover: if True, use two-point crossover; else one-point (default True).
      - mut_rate_start: per-position mutation rate at iter 0 (default ~1/H).
      - mut_rate_end: per-position mutation rate at final iter (default ~0.2/H).
      - seg_mut_prob: probability to apply an additional block (segment) mutation to a child (default 0.25).
      - seg_len_min/max: min/max segment length for block mutation (defaults 2 / min(6, H)).
      - enforce_unique: try to reduce duplicates after refill (default True).
      - restart_patience: if best loss doesn’t improve for this many iters, partial restart (default 3).
      - restart_frac: fraction of pool to reinitialize on restart (default 0.5).
    """

    def __init__(self, horizon, topk, num_samples, opt_steps, eval_every,
                 wm, action_dim, objective_fn, preprocessor, evaluator, wandb_run,
                 discrete_actions, logging_prefix="plan_traj", log_filename="logs.json",
                 # Unused legacy knobs kept for API compatibility:
                 alpha=0.5, p_min=0.02, temp_start=1.0, temp_end=0.5,
                 **kwargs):
        super().__init__(wm, action_dim, objective_fn, preprocessor, evaluator, wandb_run, log_filename)
        self.H = horizon
        self.K = min(topk, num_samples)
        self.N = num_samples
        self.T = opt_steps
        self.eval_every = eval_every
        self.logging_prefix = logging_prefix

        # New knobs with robust defaults for short horizons
        self.parent_temp = float(kwargs.get("parent_temp", 0.5))
        self.crossover_prob = float(kwargs.get("crossover_prob", 0.9))
        self.two_point_crossover = bool(kwargs.get("two_point_crossover", True))
        self.mut_rate_start = float(kwargs.get("mut_rate_start", max(1e-3, 1.0 / max(2, self.H))))
        self.mut_rate_end = float(kwargs.get("mut_rate_end", max(5e-4, 0.2 / max(2, self.H))))
        self.seg_mut_prob = float(kwargs.get("seg_mut_prob", 0.25))
        self.seg_len_min = int(kwargs.get("seg_len_min", 2))
        self.seg_len_max = int(kwargs.get("seg_len_max", min(6, self.H)))
        self.enforce_unique = bool(kwargs.get("enforce_unique", True))
        self.restart_patience = int(kwargs.get("restart_patience", 3))
        self.restart_frac = float(kwargs.get("restart_frac", 0.5))

        # Action set normalized as in your original planner
        actions_tensor = torch.as_tensor(discrete_actions, dtype=torch.float32)
        self.actions_set = self.preprocessor.normalize_actions(actions_tensor).to(self.device)

    # ---------- Helpers ----------
    @staticmethod
    def _lin_anneal(v0, v1, it, it_max):
        if it_max <= 1: return v1
        t = it / (it_max - 1)
        return v0 * (1 - t) + v1 * t

    def _rand_actions(self, shape, A):
        return torch.randint(0, A, shape, device=self.device)

    def _mutate_pointwise(self, seqs, A, p):
        """
        Per-position mutation: with prob p, replace action with a different random action.
        seqs: [M,H] long
        """
        if p <= 0: return seqs
        M, H = seqs.shape
        mask = torch.rand((M, H), device=seqs.device) < p
        if not mask.any():
            return seqs
        # Sample new actions != current value
        # Trick: sample in [0..A-2], then add (new >= old) to skip old
        new_raw = torch.randint(0, max(1, A - 1), (M, H), device=seqs.device)
        cur = seqs
        new = new_raw + (new_raw >= cur).long()
        return torch.where(mask, new, cur)

    def _mutate_segment(self, seqs, A):
        """
        For some children, apply one block mutation: pick a random segment and resample it.
        """
        M, H = seqs.shape
        if self.seg_mut_prob <= 0 or self.seg_len_min <= 0:
            return seqs
        apply = torch.rand((M,), device=seqs.device) < self.seg_mut_prob
        if not apply.any():
            return seqs

        minL = min(self.seg_len_min, H)
        maxL = max(minL, min(self.seg_len_max, H))
        # Draw lengths
        lens = torch.randint(minL, maxL + 1, (M,), device=seqs.device)
        # Draw starts so that start+len <= H
        starts = torch.randint(0, H, (M,), device=seqs.device)
        ends = torch.minimum(starts + lens, torch.tensor(H, device=seqs.device))
        # Adjust starts where end == H to keep length inside bounds
        starts = torch.minimum(starts, ends - 1)

        # Apply resample within segment for those selected
        for i in torch.nonzero(apply, as_tuple=False).flatten():
            s, e = starts[i].item(), ends[i].item()
            if e <= s:  # guard
                continue
            seg_len = e - s
            seqs[i, s:e] = self._rand_actions((seg_len,), A)
        return seqs

    def _crossover(self, pa, pb):
        """
        pa, pb: [M,H] longs
        Returns children by one- or two-point crossover (vectorized-ish).
        """
        M, H = pa.shape
        if H <= 1:
            return pa.clone()  # nothing to crossover
        if self.two_point_crossover and H >= 3:
            # two-point: choose cut1 < cut2
            cut1 = torch.randint(1, H - 1, (M,), device=pa.device)
            # ensure cut2 > cut1
            cut2 = torch.randint(2, H, (M,), device=pa.device)
            cut2 = torch.maximum(cut2, cut1 + 1)
            idx = torch.arange(H, device=pa.device)[None, :].expand(M, H)
            child = torch.where(idx < cut1[:, None], pa,
                                torch.where(idx < cut2[:, None], pb, pa))
        else:
            # one-point
            cut = torch.randint(1, H, (M,), device=pa.device)
            idx = torch.arange(H, device=pa.device)[None, :].expand(M, H)
            child = torch.where(idx < cut[:, None], pa, pb)
        return child

    def _refill(self, elites, elite_losses, A, num_new, mut_rate):
        """
        elites: [K,H] long, elite_losses: [K] (lower is better)
        Returns children: [num_new, H]
        """
        if num_new <= 0:
            return elites.new_zeros((0, elites.shape[1]))

        # Parent selection weights (soft, sharper than uniform)
        if len(elite_losses) > 1 and self.parent_temp > 0:
            shifted = elite_losses - elite_losses.min()
            w = torch.softmax(-shifted / max(1e-8, self.parent_temp), dim=0)
        else:
            w = torch.full_like(elite_losses, 1.0 / max(1, elites.shape[0]))

        # Sample parents with replacement
        pa_idx = torch.multinomial(w, num_new, replacement=True)
        pb_idx = torch.multinomial(w, num_new, replacement=True)
        pa = elites[pa_idx]  # [num_new,H]
        pb = elites[pb_idx]  # [num_new,H]

        # Crossover with probability
        do_cross = (torch.rand(num_new, device=elites.device) < self.crossover_prob)
        children = pa.clone()
        if do_cross.any():
            children_cross = self._crossover(pa[do_cross], pb[do_cross])
            children[do_cross] = children_cross

        # Mutation: pointwise + optional segment mutation
        children = self._mutate_pointwise(children, A, mut_rate)
        children = self._mutate_segment(children, A)
        return children

    def _ensure_diversity(self, pool, A):
        """
        Reduce duplicates in the pool by randomizing some rows.
        pool: [N,H] long
        """
        if not self.enforce_unique or pool.shape[0] <= 1:
            return pool
        # Unique rows
        uniq, inv = torch.unique(pool, dim=0, return_inverse=True)
        if uniq.shape[0] == pool.shape[0]:
            return pool  # already unique
        # For duplicates (inv counts > 1), randomize those extra instances
        counts = torch.bincount(inv)
        dup_mask = torch.zeros_like(inv, dtype=torch.bool)
        seen = torch.zeros_like(counts)
        for i in range(inv.shape[0]):
            idx = inv[i].item()
            if seen[idx] == 0:
                seen[idx] = 1
            else:
                dup_mask[i] = True
        # Reinitialize duplicates with random sequences
        num_dup = dup_mask.sum().item()
        if num_dup > 0:
            pool[dup_mask] = self._rand_actions((num_dup, self.H), A)
        return pool

    def plan(self, obs_0, obs_g, actions=None):
        trans_obs_0 = move_to_device(self.preprocessor.transform_obs(obs_0), self.device)
        trans_obs_g = move_to_device(self.preprocessor.transform_obs(obs_g), self.device)
        z_obs_g = self.wm.encode_obs(trans_obs_g)
        if "visual" not in z_obs_g and "visual_tokens" in z_obs_g:
            z_obs_g["visual"] = z_obs_g["visual_tokens"]

        n_evals = trans_obs_0["visual"].shape[0]
        A = len(self.actions_set)
        H, N, K, T = self.H, self.N, self.K, self.T

        # Population (indices) per eval: [n_evals, N, H]
        pool = self._rand_actions((n_evals, N, H), A)
        best_losses = torch.full((n_evals,), float("inf"), device=self.device)
        best_seqs = torch.zeros(n_evals, H, dtype=torch.long, device=self.device)
        no_improve = torch.zeros(n_evals, dtype=torch.long, device=self.device)

        for it in range(T):
            losses_for_logging = []

            # Anneal mutation rate over iterations
            mut_rate = self._lin_anneal(self.mut_rate_start, self.mut_rate_end, it, T)

            for b in range(n_evals):
                # Prepare batch of initial obs repeated N times
                cur_trans_obs_0 = {k: repeat(v[b].unsqueeze(0), "1 ... -> n ...", n=N) for k, v in trans_obs_0.items()}
                cur_z_obs_g = {k: repeat(v[b].unsqueeze(0), "1 ... -> n ...", n=N) for k, v in z_obs_g.items()}

                # Map indices -> concrete action vectors
                acts = self.actions_set[pool[b]]               # [N,H,act_dim] or [N,H]
                if self.actions_set.ndim == 1:
                    acts = acts.unsqueeze(-1)

                with torch.no_grad():
                    z_obses, _ = self.wm.rollout(obs_0=cur_trans_obs_0, act=acts)

                loss = self.objective_fn(z_obses, cur_z_obs_g).reshape(-1)  # [N], lower is better

                # Select elites
                elite_idx = torch.topk(loss, k=K, largest=False).indices    # [K]
                elites = pool[b, elite_idx]                                 # [K,H]
                elite_losses = loss[elite_idx]                              # [K]

                # Track best
                best_cur_loss, best_cur_idx = elite_losses.min(dim=0)
                losses_for_logging.append(best_cur_loss.item())
                if best_cur_loss < best_losses[b]:
                    best_losses[b] = best_cur_loss
                    best_seqs[b] = elites[best_cur_idx]
                    no_improve[b] = 0
                else:
                    no_improve[b] += 1

                # Refill
                num_new = N - K
                children = self._refill(elites, elite_losses, A, num_new, mut_rate)

                # Next pool: elitism + offspring
                next_pool = torch.empty_like(pool[b])
                next_pool[:K] = elites
                next_pool[K:] = children

                # Diversity control (optional)
                next_pool = self._ensure_diversity(next_pool, A)

                # Optional partial restart on stall
                if self.restart_patience > 0 and no_improve[b] >= self.restart_patience:
                    r = int(max(1, round(self.restart_frac * N)))
                    next_pool[-r:] = self._rand_actions((r, H), A)
                    no_improve[b] = 0  # reset stall counter

                pool[b] = next_pool

            # Logging and optional eval
            self.wandb_run.log({f"{self.logging_prefix}/loss": float(np.mean(losses_for_logging)), "step": it + 1})

            if self.evaluator is not None and (self.eval_every > 0) and (it % self.eval_every == 0):
                elite_actions = self.actions_set[best_seqs]
                if self.actions_set.ndim == 1:
                    elite_actions = elite_actions.unsqueeze(-1)
                logs, successes, _, _, _ = self.evaluator.eval_actions(elite_actions, filename=f"{self.logging_prefix}_output_{it+1}")
                logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}
                logs.update({"step": it + 1})
                self.wandb_run.log(logs)
                self.dump_logs(logs)
                if np.all(successes):
                    break

        # Return best sequences as concrete actions
        elite_actions = self.actions_set[best_seqs]
        if self.actions_set.ndim == 1:
            elite_actions = elite_actions.unsqueeze(-1)
        return elite_actions, best_losses.detach().cpu().numpy()
