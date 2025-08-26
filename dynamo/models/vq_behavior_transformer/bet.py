import tqdm
import torch
import einops
import logging
import torch.nn as nn
from pathlib import Path

from .vqvae import VqVae
import torch.nn.functional as F
from accelerate import Accelerator
from typing import Dict, Optional, Tuple, Sequence
from .gpt import GPT, GPTConfig
from .utils import MLP, batch_idx


accelerator = Accelerator()


def repeat_start_to_length(x: torch.Tensor, length: int, dim: int = 0):
    """
    Pad tensor x to length along dim, repeating the first value at the start.
    """
    pad_size = length - x.shape[dim]
    if pad_size <= 0:
        return x
    first_frame = x.index_select(dim, torch.tensor(0, device=x.device))
    repeat_shape = [1] * len(x.shape)
    repeat_shape[dim] = pad_size
    pad = first_frame.repeat(*repeat_shape)
    return torch.cat([pad, x], dim=dim)


class GroupedOptimizer(torch.optim.Optimizer):
    def __init__(self, optimizers: Sequence[torch.optim.Optimizer]):
        self.optimizers = optimizers
        param_groups = []
        for optimizer in self.optimizers:
            param_groups.extend(optimizer.param_groups)
        super().__init__(param_groups, optimizers[0].defaults)

    def step(self, closure=None):
        for optimizer in self.optimizers:
            optimizer.step(closure)

    def zero_grad(self, set_to_none: bool = False):
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none)

    def state_dict(self):
        return {
            i: optimizer.state_dict() for i, optimizer in enumerate(self.optimizers)
        }

    def load_state_dict(self, state_dict):
        for i, optimizer in enumerate(self.optimizers):
            optimizer.load_state_dict(state_dict[i])


class BehaviorTransformer(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        goal_dim: int,
        views: int,
        vqvae_latent_dim: int,
        vqvae_n_embed: int,
        vqvae_groups: int,
        vqvae_fit_steps: int,
        vqvae_iters: int,
        n_layer: int,
        n_head: int,
        n_embd: int,
        dropout: float = 0.0,
        vqvae_encoder_loss_multiplier: float = 1.0,
        vqvae_batch_size: int = 1024,
        act_scale: float = 1.0,
        offset_loss_multiplier: float = 1.0e3,
        secondary_code_multiplier: float = 0.5,
        gamma: float = 2.0,
        obs_window_size: int = 10,
        act_window_size: int = 10,
    ):
        super().__init__()
        self.GOAL_SPEC = ["concat", "stack", "unconditional"]
        self._obs_dim = obs_dim * views
        self._act_dim = act_dim
        self._goal_dim = goal_dim * views
        self._obs_window_size = obs_window_size
        self._act_window_size = act_window_size
        if goal_dim <= 0:
            self._cbet_method = "unconditional"
        else:
            self._cbet_method = "stack"

        self._gpt_model = GPT(
            GPTConfig(
                block_size=obs_window_size + act_window_size,
                n_layer=n_layer,
                n_head=n_head,
                n_embd=n_embd,
                dropout=dropout,
                input_dim=self._obs_dim + self._goal_dim,
            )
        )

        # the first n batches of actions are collected for VQ training.
        self.vqvae_fit_steps = vqvae_fit_steps
        self.vqvae_iters = vqvae_iters
        self.vqvae_batch_size = vqvae_batch_size
        self.vqvae_is_fit = False
        self._vqvae_model = VqVae(
            input_dim_h=act_window_size,
            input_dim_w=act_dim,
            n_latent_dims=vqvae_latent_dim,
            vqvae_n_embed=vqvae_n_embed,
            vqvae_groups=vqvae_groups,
            encoder_loss_multiplier=vqvae_encoder_loss_multiplier,
            act_scale=act_scale,
        )
        self._vqvae_optim = torch.optim.Adam(
            self._vqvae_model.parameters(),
            lr=1e-3,
            weight_decay=1e-4,
        )

        self._G = vqvae_groups
        self._C = vqvae_n_embed  # number of code integers
        self._D = vqvae_latent_dim

        self._map_to_cbet_preds_bin = MLP(
            in_channels=self._gpt_model.config.output_dim,
            hidden_channels=[1024, 1024, self._G * self._C],
        )
        self._map_to_cbet_preds_offset = MLP(
            in_channels=self._gpt_model.config.output_dim,
            hidden_channels=[
                1024,
                1024,
                self._G * self._C * (act_dim * self._act_window_size),
            ],
        )

        self._collected_actions = []
        self._offset_loss_multiplier = offset_loss_multiplier
        self._secondary_code_multiplier = secondary_code_multiplier
        self._criterion = FocalLoss(gamma=gamma)

    def _unpack_actions(self, action_seq: torch.Tensor):
        """Unpack actions from (N, total_window, A) to (N, T, W, A)"""
        n, total_w, act_dim = action_seq.shape
        act_w = self._act_window_size
        obs_w = total_w + 1 - act_w
        result = torch.empty((n, obs_w, act_w, act_dim)).to(action_seq.device)
        for i in range(obs_w):
            result[:, i, :, :] = action_seq[:, i : i + act_w, :]
        return result

    def _maybe_fit_vq(self):
        if self.vqvae_is_fit or len(self._collected_actions) < self.vqvae_fit_steps:
            return
        all_actions = torch.cat(self._collected_actions)
        all_actions = einops.rearrange(all_actions, "N T W A -> (N T) (W A)")
        # only train on unique actions
        all_actions = torch.unique(all_actions, dim=0)
        all_actions = einops.rearrange(
            all_actions,
            "... (W A) -> ... W A",
            W=self._act_window_size,
        )
        pbar = tqdm.trange(
            self.vqvae_iters,
            desc="VQ training",
            disable=not accelerator.is_local_main_process,
        )
        for epoch in pbar:
            shuffle_idx = torch.randperm(len(all_actions))
            for i in range(0, len(all_actions), self.vqvae_batch_size):
                batch = all_actions[shuffle_idx[i : i + self.vqvae_batch_size]]
                loss, vq_code, loss_dict = self._vqvae_model(batch)
                self._vqvae_optim.zero_grad()
                loss.backward()
                self._vqvae_optim.step()
        accelerator.wait_for_everyone()
        # wrapping the the model in DDP syncs the weights from main to other processes
        self._vqvae_model = accelerator.prepare(self._vqvae_model)
        self._vqvae_model = accelerator.unwrap_model(self._vqvae_model)
        self._vqvae_model.eval()
        print("n_different_codes", len(torch.unique(vq_code)))
        print("n_different_combinations", len(torch.unique(vq_code, dim=0)))
        print("losses", loss_dict)
        self.vqvae_is_fit = True

    def train(self, mode=True):
        # if vqvae is already trained, make sure we freeze it
        super().train(mode)
        if self.vqvae_is_fit:
            self._vqvae_model.eval()

    def forward(
        self,
        obs_seq: torch.Tensor,
        goal_seq: Optional[torch.Tensor],
        action_seq: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, float]]:
        if (
            (action_seq is not None)
            and (len(self._collected_actions) < self.vqvae_fit_steps)
            and (self.training)
        ):
            action_seq_all = accelerator.gather(action_seq)
            self._collected_actions.append(self._unpack_actions(action_seq_all))
            self._maybe_fit_vq()

        if obs_seq.shape[1] < self._obs_window_size:
            obs_seq = repeat_start_to_length(obs_seq, self._obs_window_size, dim=1)
        if self._cbet_method == "unconditional":
            gpt_input = obs_seq
        elif self._cbet_method == "stack":
            gpt_input = torch.cat([goal_seq, obs_seq], dim=-1)
        else:
            raise NotImplementedError

        gpt_output = self._gpt_model(gpt_input)  # N, T, n_embd

        if self._cbet_method == "concat":
            gpt_output = gpt_output[:, goal_seq.size(1) :, :]
        else:
            gpt_output = gpt_output

        cbet_logits, cbet_offsets = self._forward_heads(gpt_output)
        predicted_action, decoded_action, sampled_centers, sampled_offsets = (
            self._sample_action(cbet_logits, cbet_offsets)
        )

        if action_seq is not None:
            loss, loss_dict = self._calc_loss(
                action_seq,
                predicted_action,
                decoded_action,
                sampled_centers,
                cbet_logits,
            )
        else:
            loss, loss_dict = None, {}
        return predicted_action, loss, loss_dict

    def _calc_loss(
        self, action_seq, predicted_action, decoded_action, sampled_centers, cbet_logits
    ):
        action_seq = self._unpack_actions(action_seq)
        _, action_bins = self._vqvae_model.get_code(action_seq)
        # flatten for cross entropy loss
        action_bins_flat = einops.rearrange(action_bins, "N T ... -> (N T) ...")
        cbet_logits_flat = einops.rearrange(cbet_logits, "N T ... -> (N T) ...")

        offset_loss = F.l1_loss(action_seq, predicted_action)

        # batch, time, windowsize (t ... t+N), action dim -> [:, -1, 0, :] is for rollout
        action_diff = F.mse_loss(action_seq[:, -1, 0], predicted_action[:, -1, 0])
        action_diff_tot = F.mse_loss(action_seq[:, -1], predicted_action[:, -1])
        action_diff_mean_res1 = (action_seq - decoded_action)[:, -1, 0].abs().mean()
        action_diff_mean_res2 = (action_seq - predicted_action)[:, -1, 0].abs().mean()
        action_diff_max = (action_seq - predicted_action)[:, -1, 0].abs().max()
        cbet_loss1 = self._criterion(cbet_logits_flat[:, 0], action_bins_flat[:, 0])
        cbet_loss2 = self._criterion(cbet_logits_flat[:, 1], action_bins_flat[:, 1])
        cbet_loss = cbet_loss1 * 5 + cbet_loss2 * self._secondary_code_multiplier

        eq_mask = action_bins == sampled_centers
        equal_total_code_rate = (eq_mask.sum(-1) == self._G).float().mean()
        equal_single_code_rate = eq_mask[..., 0].float().mean()
        equal_single_code_rate2 = eq_mask[..., 1].float().mean()

        loss = cbet_loss + self._offset_loss_multiplier * offset_loss
        loss_dict = {
            "classification_loss": cbet_loss.detach().cpu().item(),
            "offset_loss": offset_loss.detach().cpu().item(),
            "total_loss": loss.detach().cpu().item(),
            "equal_total_code_rate": equal_total_code_rate,
            "equal_single_code_rate": equal_single_code_rate,
            "equal_single_code_rate2": equal_single_code_rate2,
            "action_diff": action_diff.detach().cpu().item(),
            "action_diff_tot": action_diff_tot.detach().cpu().item(),
            "action_diff_mean_res1": action_diff_mean_res1.detach().cpu().item(),
            "action_diff_mean_res2": action_diff_mean_res2.detach().cpu().item(),
            "action_diff_max": action_diff_max.detach().cpu().item(),
        }
        if not self.vqvae_is_fit:
            loss = loss * 0.0  # do not train the model until VQ is fit
        return loss, loss_dict

    def _forward_heads(self, gpt_output):
        cbet_logits = self._map_to_cbet_preds_bin(gpt_output)
        cbet_logits = einops.rearrange(cbet_logits, "N T (G C) -> N T G C", G=self._G)
        cbet_offsets = self._map_to_cbet_preds_offset(gpt_output)
        cbet_offsets = einops.rearrange(
            cbet_offsets,
            "N T (G C W A) -> N T G C W A",
            G=self._G,
            C=self._C,
            W=self._act_window_size,
            A=self._act_dim,
        )
        return cbet_logits, cbet_offsets

    def _sample_action(self, cbet_logits, cbet_offsets):
        # W = action_window
        # flatten for downstream VQ decoding
        cbet_probs = torch.softmax(cbet_logits, dim=-1)
        sampled_centers = einops.rearrange(
            torch.multinomial(cbet_probs.view(-1, self._C), num_samples=1),
            "(N T G) 1 -> N T G",
            N=cbet_probs.shape[0],
            T=cbet_probs.shape[1],
            G=self._G,
        )
        centers = self._vqvae_model.draw_code_forward(sampled_centers).clone().detach()
        decoded_action = (
            self._vqvae_model.get_action_from_latent(centers).clone().detach()
        )  # N T W A

        sampled_offsets = batch_idx(cbet_offsets, sampled_centers)  # N T G W A
        # offset on each residual VQ group; sum on group dim
        sampled_offsets = sampled_offsets.sum(dim=2)
        predicted_action = decoded_action + sampled_offsets
        return predicted_action, decoded_action, sampled_centers, sampled_offsets

    def configure_optimizers(self, weight_decay, learning_rate, betas):
        optimizer1 = self._gpt_model.configure_optimizers(
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            betas=betas,
        )

        optimizer1.add_param_group({"params": self._map_to_cbet_preds_bin.parameters()})
        optimizer2 = torch.optim.AdamW(
            self._map_to_cbet_preds_offset.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
        )
        optim = GroupedOptimizer([optimizer1, optimizer2])
        return optim

    def load_model(self, path: Path):
        if (path / "cbet_model.pt").exists():
            self.load_state_dict(torch.load(path / "cbet_model.pt"))
        else:
            logging.warning("No model found at %s", path)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 0, reduction: str = "mean"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        if reduction not in ("mean", "sum", "none"):
            raise NotImplementedError
        self.reduction = reduction

    def forward(self, input, target):
        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1, target.view(-1, 1)).view(-1)
        pt = logpt.exp()

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
