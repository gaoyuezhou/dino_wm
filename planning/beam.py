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

        n_evals = trans_obs_0["visual"].shape[0]
        elite_actions = torch.zeros(n_evals, self.horizon, self.action_dim, device=self.device)
        losses_log = []

        # allow either "score_mode" or "scoremode" attribute; default to "last"
        score_mode = getattr(self, "score_mode", getattr(self, "scoremode", "last"))
        if score_mode not in ("last", "prefix"):
            raise ValueError(f"Unknown score_mode: {score_mode}")

        for traj in range(n_evals):
            # slice per-trajectory inputs
            cur_trans_obs_0 = {key: value[traj: traj + 1] for key, value in trans_obs_0.items()}
            cur_z_obs_g     = {key: value[traj: traj + 1] for key, value in z_obs_g.items()}

            # start with all zeros (valid action in your WM)
            sequences = self.zero_action.repeat(1, self.horizon, 1)

            for t in range(self.horizon):
                num_beams   = sequences.shape[0]
                num_actions = self.actions_set.shape[0]

                # expand current beams by all actions at step t
                expanded = sequences.repeat_interleave(num_actions, dim=0)
                expanded[:, t] = self.actions_set.repeat(num_beams, 1)

                # batch current obs + goal for rollout
                cur_obs_batch = {
                    key: repeat(arr, "1 ... -> n ...", n=expanded.shape[0])
                    for key, arr in cur_trans_obs_0.items()
                }
                cur_goal_batch = {
                    key: repeat(arr, "1 ... -> n ...", n=expanded.shape[0])
                    for key, arr in cur_z_obs_g.items()
                }

                # world-model rollout for full horizon (unchanged)
                with torch.no_grad():
                    i_z_obses, i_zs = self.wm.rollout(obs_0=cur_obs_batch, act=expanded)
                    # i_z_obses[k] shape: [B, H, D_k]

                # -------- score only what we've already decided --------
                L = t + 1
                if score_mode == "last":
                    # use only the newly decided step t -> time dim = 1
                    preds = {k: v[:, t:t+1] for k, v in i_z_obses.items()}
                    goal  = {
                        k: (cur_goal_batch[k].unsqueeze(1) if cur_goal_batch[k].ndim == 2
                            else cur_goal_batch[k])
                        for k in cur_goal_batch
                    }
                else:  # "prefix"
                    # use decided prefix [0..t] -> time dim = L
                    preds = {k: v[:, :L] for k, v in i_z_obses.items()}
                    goal  = {
                        k: (
                            cur_goal_batch[k].unsqueeze(1).expand(-1, L, -1)
                            if cur_goal_batch[k].ndim == 2
                            else cur_goal_batch[k].expand(-1, L, -1)
                        )
                        for k in cur_goal_batch
                    }

                # (optional) sanity check: same time length
                # for k in preds:
                #     assert preds[k].shape[:2] == goal[k].shape[:2], f"time mismatch for key {k}"

                losses = self.objective_fn(preds, goal)  # shape [B]

                # select top-k beams
                topk = torch.argsort(losses)[: self.beam_width]
                sequences = expanded[topk]
                losses    = losses[topk]

                # lightweight progress print
                try:
                    print(f"[{self.logging_prefix}] Planning step {t+1}/{self.horizon} | min loss: {losses.min().item():.4f}")
                except Exception:
                    pass

            # take best sequence
            elite_actions[traj] = sequences[0]
            losses_log.append(losses[0].item())

        # final logs
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
