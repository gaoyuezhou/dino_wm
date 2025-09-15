import os
import torch
import imageio
import numpy as np
from einops import rearrange, repeat
from utils import (
    cfg_to_dict,
    seed,
    slice_trajdict_with_t,
    aggregate_dct,
    move_to_device,
    concat_trajdict,
)
from torchvision import utils
from models.inverse_dynamics import InverseDynamicsProjector
from models.inverse_mlp import InverseMLP

class PlanEvaluator:  # evaluator for planning
    def __init__(
        self,
        obs_0,
        obs_g,
        state_0,
        state_g,
        env,
        wm,
        frameskip,
        seed,
        preprocessor,
        n_plot_samples,
        classifier=None,
    ):
        self.obs_0 = obs_0
        self.obs_g = obs_g
        self.state_0 = state_0
        self.state_g = state_g
        self.env = env
        self.wm = wm
        self.frameskip = frameskip
        self.seed = seed
        self.preprocessor = preprocessor
        self.n_plot_samples = n_plot_samples
        self.device = next(wm.parameters()).device
        self.classifier = classifier

        self.plot_full = False  # plot all frames or frames after frameskip

    def assign_init_cond(self, obs_0, state_0):
        self.obs_0 = obs_0
        self.state_0 = state_0

    def assign_goal_cond(self, obs_g, state_g):
        self.obs_g = obs_g
        self.state_g = state_g

    def get_init_cond(self):
        return self.obs_0, self.state_0

    def _get_trajdict_last(self, dct, length):
        new_dct = {}
        for key, value in dct.items():
            new_dct[key] = self._get_traj_last(value, length)
        return new_dct

    def _get_traj_last(self, traj_data, length):
        last_index = np.where(length == np.inf, -1, length - 1)
        last_index = last_index.astype(int)
        if isinstance(traj_data, torch.Tensor):
            traj_data = traj_data[np.arange(traj_data.shape[0]), last_index].unsqueeze(
                1
            )
        else:
            traj_data = np.expand_dims(
                traj_data[np.arange(traj_data.shape[0]), last_index], axis=1
            )
        return traj_data

    def _mask_traj(self, data, length):
        """
        Zero out everything after specified indices for each trajectory in the tensor.
        data: tensor
        """
        result = data.clone()  # Clone to preserve the original tensor
        for i in range(data.shape[0]):
            if length[i] != np.inf:
                result[i, int(length[i]) :] = 0
        return result

    def eval_actions(
        self, actions, action_len=None, filename="output", save_video=False
    ):
        """
        actions: detached torch tensors on cuda
        Returns
            metrics, and feedback from env
        """
        n_evals = actions.shape[0]
        if action_len is None:
            action_len = np.full(n_evals, np.inf)

        # rollout in wm
        trans_obs_0 = move_to_device(
            self.preprocessor.transform_obs(self.obs_0), self.device
        )
        trans_obs_g = move_to_device(
            self.preprocessor.transform_obs(self.obs_g), self.device
        )
        with torch.no_grad():
            wm_actions = actions
            if isinstance(self.wm.action_encoder, (InverseDynamicsProjector, InverseMLP)):
                if wm_actions.dim() == 3:
                    wm_actions = wm_actions.unsqueeze(2)
                elif wm_actions.dim() == 4 and wm_actions.shape[2] == 1:
                    pass
                else:
                    raise ValueError(
                        "Planner actions must have shape (B,T,D) or (B,T,1,D)"
                    )
                if wm_actions.shape[-1] != self.wm.action_encoder.output_dim:
                    raise ValueError(
                        f"Planner action dim {wm_actions.shape[-1]} does not match latent dim {self.wm.action_encoder.output_dim}"
                    )
            i_z_obses, _ = self.wm.rollout(
                obs_0=trans_obs_0,
                act=wm_actions,
            )
        i_final_z_obs = self._get_trajdict_last(i_z_obses, action_len + 1)

        # rollout in env
        print(f"Eval actions first batch: {wm_actions[0]}")
        if self.classifier is not None:
            classifier_actions = (
                actions.squeeze(2)
                if actions.dim() == 4 and actions.shape[2] == 1
                else actions
            )
            logits = self.classifier(classifier_actions)
            discrete = logits.argmax(dim=-1).float()
            actions_for_env = discrete.unsqueeze(-1)
        else:
            actions_for_env = (
                actions.squeeze(2)
                if actions.dim() == 4 and actions.shape[2] == 1
                else actions
            )

        actions_for_env = (
            actions_for_env.squeeze(2)
            if actions_for_env.dim() == 4 and actions_for_env.shape[2] == 1
            else actions_for_env
        )
        if self.classifier is not None:
            exec_actions = repeat(
                actions_for_env.cpu(), "b t d -> b (t f) d", f=self.frameskip
            )
        else:
            exec_actions = rearrange(
                actions_for_env.cpu(), "b t (f d) -> b (t f) d", f=self.frameskip
            )
            exec_actions = self.preprocessor.denormalize_actions(exec_actions).numpy()
        print(f"Exec actions first batch: {exec_actions[0]}")
        e_obses, e_states = self.env.rollout(self.seed, self.state_0, exec_actions)
        e_visuals = e_obses["visual"]
        e_final_obs = self._get_trajdict_last(e_obses, action_len * self.frameskip + 1)
        e_final_state = e_states[:, :, -1, :]   # (B, 2, d)
       #print("[CHK e_final_state]", np.asarray(e_final_state).shape)  # exp

        # compute eval metrics
        logs, successes = self._compute_rollout_metrics(
            e_state=e_final_state,
            e_obs=e_final_obs,
            i_z_obs=i_final_z_obs,
        )

        # plot trajs
        i_visuals = None
        if self.wm.decoder is not None:
            i_visuals = self.wm.decode_obs(i_z_obses)[0]["visual"]
            i_visuals = self._mask_traj(
                i_visuals, action_len + 1
            )  # we have action_len + 1 states
        e_visuals = self.preprocessor.transform_obs_visual(e_visuals)
        e_visuals = self._mask_traj(e_visuals, action_len * self.frameskip + 1)
        self._plot_rollout_compare(
            e_visuals=e_visuals,
            i_visuals=i_visuals,
            successes=successes,
            save_video=save_video,
            filename=filename,
        )

        return logs, successes, e_obses, e_states, exec_actions

    def _compute_rollout_metrics(self, e_state, e_obs, i_z_obs):
        """
        Args
            e_state
            e_obs
            i_z_obs
        Return
            logs
            successes
        """
        
        eval_results = self.env.eval_state(self.state_g, e_state)
        successes = eval_results['success']

        logs = {
            f"success_rate" if key == "success" else f"mean_{key}": np.mean(value) if key != "success" else np.mean(value.astype(float))
            for key, value in eval_results.items()
        }

        visual_dists = np.linalg.norm(e_obs["visual"] - self.obs_g["visual"], axis=1)
        mean_visual_dist = np.mean(visual_dists)
        proprio_dists = np.linalg.norm(e_obs["proprio"] - self.obs_g["proprio"], axis=1)
        mean_proprio_dist = np.mean(proprio_dists)

        e_obs = move_to_device(self.preprocessor.transform_obs(e_obs), self.device)
        e_z_obs = self.wm.encode_obs(e_obs)
        if "visual" not in e_z_obs and "visual_tokens" in e_z_obs:
            e_z_obs["visual"] = e_z_obs["visual_tokens"]
        div_visual_emb = ((e_z_obs["visual"] - i_z_obs["visual"]) ** 2).mean().item()  # element-wise MSE, like training #torch.norm(e_z_obs["visual"] - i_z_obs["visual"]).item()
        div_proprio_emb = torch.norm(e_z_obs["proprio"] - i_z_obs["proprio"]).item()

        logs.update({
            "mean_visual_dist": mean_visual_dist,
            "mean_proprio_dist": mean_proprio_dist,
            "mean_div_visual_emb": div_visual_emb,
            "mean_div_proprio_emb": div_proprio_emb,
        })

        return logs, successes

    def _plot_rollout_compare(
        self, e_visuals, i_visuals, successes, save_video=False, filename=""
    ):
        """
        i_visuals may have less frames than e_visuals due to frameskip, so pad accordingly
        e_visuals: (b, t, h, w, c)
        i_visuals: (b, t, h, w, c)
        goal: (b, h, w, c)
        """
        e_visuals = e_visuals[: self.n_plot_samples]
        goal_visual = self.obs_g["visual"][: self.n_plot_samples]
        goal_visual = self.preprocessor.transform_obs_visual(goal_visual)

        correction = 0.3  # to distinguish env visuals and imagined visuals

        if i_visuals is not None:
            i_visuals = i_visuals[: self.n_plot_samples]
            i_visuals = i_visuals.unsqueeze(2)
            i_visuals = torch.cat(
                [i_visuals] + [i_visuals] * (self.frameskip - 1),
                dim=2,
            )  # pad i_visuals (due to frameskip)
            i_visuals = rearrange(i_visuals, "b t n c h w -> b (t n) c h w")
            i_visuals = i_visuals[:, : i_visuals.shape[1] - (self.frameskip - 1)]

            if save_video:
                for idx in range(e_visuals.shape[0]):
                    success_tag = "success" if successes[idx] else "failure"
                    frames = []
                    for i in range(e_visuals.shape[1]):
                        e_obs = e_visuals[idx, i, ...]
                        i_obs = i_visuals[idx, i, ...]
                        e_obs = torch.cat(
                            [e_obs.cpu(), goal_visual[idx, 0] - correction], dim=2
                        )
                        i_obs = torch.cat(
                            [i_obs.cpu(), goal_visual[idx, 0] - correction], dim=2
                        )
                        frame = torch.cat([e_obs - correction, i_obs], dim=1)
                        frame = rearrange(frame, "c w1 w2 -> w1 w2 c")
                        frame = rearrange(frame, "w1 w2 c -> (w1) w2 c")
                        frame = frame.detach().cpu().numpy()
                        frames.append(frame)
                    video_writer = imageio.get_writer(
                        f"{filename}_{idx}_{success_tag}.mp4"
                    )

                    for frame in frames:
                        frame = frame * 2 - 1 if frame.min() >= 0 else frame
                        video_writer.append_data(
                            (((np.clip(frame, -1, 1) + 1) / 2) * 255).astype(np.uint8)
                        )
                    video_writer.close()

            # pad i_visuals or subsample e_visuals
            if not self.plot_full:
                e_visuals = e_visuals[:, :: self.frameskip]
                i_visuals = i_visuals[:, :: self.frameskip]

            n_columns = e_visuals.shape[1]
            assert (
                i_visuals.shape[1] == n_columns
            ), f"Rollout lengths do not match, {e_visuals.shape[1]} and {i_visuals.shape[1]}"

            # add a goal column
            e_visuals = torch.cat([e_visuals.cpu(), goal_visual - correction], dim=1)
            i_visuals = torch.cat([i_visuals.cpu(), goal_visual - correction], dim=1)
            rollout = torch.cat([e_visuals.cpu() - correction, i_visuals.cpu()], dim=1)
            n_columns += 1

            imgs_for_plotting = rearrange(rollout, "b h c w1 w2 -> (b h) c w1 w2")
            imgs_for_plotting = (
                imgs_for_plotting * 2 - 1
                if imgs_for_plotting.min() >= 0
                else imgs_for_plotting
            )
            utils.save_image(
                imgs_for_plotting,
                f"{filename}.png",
                nrow=n_columns,  # nrow is the number of columns
                normalize=True,
                value_range=(-1, 1),
            )
        else:
            if save_video:
                for idx in range(e_visuals.shape[0]):
                    success_tag = "success" if successes[idx] else "failure"
                    frames = []
                    for i in range(e_visuals.shape[1]):
                        e_obs = e_visuals[idx, i, ...]
                        frame = torch.cat(
                            [e_obs.cpu(), goal_visual[idx, 0]], dim=2
                        )
                        frame = rearrange(frame, "c w1 w2 -> w1 w2 c")
                        frame = rearrange(frame, "w1 w2 c -> (w1) w2 c")
                        frame = frame.detach().cpu().numpy()
                        frames.append(frame)
                    video_writer = imageio.get_writer(
                        f"{filename}_{idx}_{success_tag}.mp4"
                    )

                    for frame in frames:
                        frame = frame * 2 - 1 if frame.min() >= 0 else frame
                        video_writer.append_data(
                            (((np.clip(frame, -1, 1) + 1) / 2) * 255).astype(np.uint8)
                        )
                    video_writer.close()

            if not self.plot_full:
                e_visuals = e_visuals[:, :: self.frameskip]

            n_columns = e_visuals.shape[1]
            e_visuals = torch.cat([e_visuals.cpu(), goal_visual], dim=1)
            rollout = e_visuals.cpu()
            n_columns += 1

            imgs_for_plotting = rearrange(rollout, "b h c w1 w2 -> (b h) c w1 w2")
            imgs_for_plotting = (
                imgs_for_plotting * 2 - 1
                if imgs_for_plotting.min() >= 0
                else imgs_for_plotting
            )
            utils.save_image(
                imgs_for_plotting,
                f"{filename}.png",
                nrow=n_columns,
                normalize=True,
                value_range=(-1, 1),
            )
