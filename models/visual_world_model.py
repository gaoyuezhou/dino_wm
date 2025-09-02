import torch
import torch.nn as nn
from torchvision import transforms
from einops import rearrange, repeat
from .inverse_dynamics import InverseDynamicsProjector

class VWorldModel(nn.Module):
    def __init__(
        self,
        image_size,  # 224
        num_hist,
        num_pred,
        encoder,
        proprio_encoder,
        action_encoder,
        decoder,
        predictor,
        proprio_dim=0,
        action_dim=0,
        concat_dim=0,
        num_action_repeat=1,
        num_proprio_repeat=1,
        train_encoder=True,
        train_predictor=False,
        train_decoder=True,
        lambda_l2=0.0,
        var_target=1.0,
        lambda_var=0.0,
        action_dropout_p=0.0,
        action_noise_sigma=0.0,
    ):
        super().__init__()
        self.num_hist = num_hist
        self.num_pred = num_pred
        self.encoder = encoder
        self.proprio_encoder = proprio_encoder
        self.action_encoder = action_encoder
        self.decoder = decoder  # decoder could be None
        self.predictor = predictor  # predictor could be None
        self.train_encoder = train_encoder
        self.train_predictor = train_predictor
        self.train_decoder = train_decoder
        self.num_action_repeat = num_action_repeat
        self.num_proprio_repeat = num_proprio_repeat
        self.proprio_dim = proprio_dim * num_proprio_repeat 
        self.action_dim = action_dim * num_action_repeat 
        self.emb_dim = self.encoder.emb_dim + (self.action_dim + self.proprio_dim) * (concat_dim) # Not used

        print(f"num_action_repeat: {self.num_action_repeat}")
        print(f"num_proprio_repeat: {self.num_proprio_repeat}")
        print(f"proprio encoder: {proprio_encoder}")
        print(f"action encoder: {action_encoder}")
        print(f"proprio_dim: {proprio_dim}, after repeat: {self.proprio_dim}")
        print(f"action_dim: {action_dim}, after repeat: {self.action_dim}")
        print(f"emb_dim: {self.emb_dim}")

        self.concat_dim = concat_dim # 0 or 1
        assert concat_dim == 0 or concat_dim == 1, f"concat_dim {concat_dim} not supported."
        print("Model emb_dim: ", self.emb_dim)

        if "dino" in self.encoder.name:
            decoder_scale = 16  # from vqvae
            num_side_patches = image_size // decoder_scale
            self.encoder_image_size = num_side_patches * encoder.patch_size
            self.encoder_transform = transforms.Compose(
                [transforms.Resize(self.encoder_image_size)]
            )
        else:
            # set self.encoder_transform to identity transform
            self.encoder_transform = lambda x: x

        self.decoder_criterion = nn.MSELoss()
        self.decoder_latent_loss_weight = 0.25
        self.emb_criterion = nn.MSELoss()
        self.lambda_l2 = lambda_l2
        self.var_target = var_target
        self.lambda_var = lambda_var
        self.action_noise_sigma = action_noise_sigma
        self.action_dropout = nn.Dropout(p=action_dropout_p) if action_dropout_p > 0 else nn.Identity()

    def train(self, mode=True):
        super().train(mode)
        if self.train_encoder:
            self.encoder.train(mode)
        if self.predictor is not None and self.train_predictor:
            self.predictor.train(mode)
        self.proprio_encoder.train(mode)
        self.action_encoder.train(mode)
        if self.decoder is not None and self.train_decoder:
            self.decoder.train(mode)

    def eval(self):
        super().eval()
        self.encoder.eval()
        if self.predictor is not None:
            self.predictor.eval()
        self.proprio_encoder.eval()
        self.action_encoder.eval()
        if self.decoder is not None:
            self.decoder.eval()

    def encode(self, obs, act):
        """
        input :  obs: dict with "visual" (B,T,3,H,W), "proprio" (B,T,...)
        expects encode_obs to return:
            z_dct["visual_tokens"] : (B,T,F,Cv)
            z_dct["visual_frame"]  : (B,T,1,Cv)  # per-frame/CLS
            z_dct["proprio"]       : (B,T,Cp)
        """
        z_dct = self.encode_obs(obs)
        #print(f"z_dct: {z_dct} and shapes {[v.shape for v in z_dct.values()]}")

        # Get action embedding; for inverse projector we pass observations only
        act_emb, aux_act = self.encode_act(act, z_dct["visual_frame"])
        #   non-inverse (e.g., MLP on actions) -> (B,T,Za)

        visual = z_dct["visual_tokens"]  # (B,T,F,Cv)
        proprio = z_dct["proprio"]       # (B,T,Cp)
        F = visual.shape[2]

        if self.concat_dim == 0:
            # Token-level concat: add proprio + action as extra tokens
            proprio_tok = proprio.unsqueeze(2)     # (B,T,1,Cp)

            if act_emb.dim() == 4:
                act_tok = act_emb                  # already (B,T,1,Za)
            elif act_emb.dim() == 3:
                act_tok = act_emb.unsqueeze(2)     # (B,T,1,Za)
            else:
                raise ValueError(f"Unexpected act_emb shape {act_emb.shape}")

            z = torch.cat([visual, proprio_tok, act_tok], dim=2)  # (B,T,F+2, ...)

        elif self.concat_dim == 1:
            # Channel-level concat: tile per token
            proprio_tiled = repeat(proprio.unsqueeze(2), "b t 1 a -> b t f a", f=F)
            proprio_rep = proprio_tiled.repeat(1, 1, 1, getattr(self, "num_proprio_repeat", 1))

            if act_emb.dim() == 4:
                # (B,T,1,Za) -> (B,T,F,Za) without copy
                act_tiled = act_emb.expand(-1, -1, F, -1)
            elif act_emb.dim() == 3:
                # (B,T,Za) -> (B,T,1,Za) -> (B,T,F,Za)
                act_tiled = act_emb.unsqueeze(2).expand(-1, -1, F, -1)
            else:
                raise ValueError(f"Unexpected act_emb shape {act_emb.shape}")

            act_rep = act_tiled.repeat(1, 1, 1, getattr(self, "num_action_repeat", 1))

            z = torch.cat([visual, proprio_rep, act_rep], dim=3)  # (B,T,F,Cv + Cp*rep + Za*rep)
            #print(f"Shape z: {z.shape}")

        else:
            raise ValueError(f"Unknown concat_dim: {self.concat_dim}")

        return z, aux_act
    
    def encode_act(self, act, obs_emb=None):
        """
        Always return shape (B,T,1,Za).
        If using an inverse dynamics projector (predicts a_t from (o_t, o_{t+1})):
        align embeddings so index t gets a_{t+1}, pad last with a_{T-1}.
        """
        # Inverse path: obs_emb is (B,T,1,Cv) or similar per-frame embedding
        if isinstance(self.action_encoder, InverseDynamicsProjector) and obs_emb is not None:
            a = self.action_encoder(obs_emb)          # (B,T,1,Za)
            if a.dim() != 4 or a.shape[2] != 1:
                raise ValueError(f"inverse action head must return (B,T,1,Za), got {tuple(a.shape)}")

            # Shift left by 1 along T, pad last with final available action
            if a.size(1) > 1:
                a = torch.cat([a[:, 1:, :, :], a[:, -1:, :, :]], dim=1)  # (B,T,1,Za)
            a_raw = a
            a = self.action_dropout(a)
            if self.action_noise_sigma > 0 and self.training:
                a = a + torch.randn_like(a) * self.action_noise_sigma
            return a, {"a": a, "a_raw": a_raw}

        # Proprio (or other direct) path: encoder likely returns (B,T,Za)
        a = self.action_encoder(act)                  # (B,T,Za) or (B,T,1,Za)
        if a.dim() == 3:
            a = a.unsqueeze(2)                        # -> (B,T,1,Za)
        elif a.dim() == 4 and a.shape[2] == 1:
            pass
        else:
            raise ValueError(f"direct action head must return (B,T,Za) or (B,T,1,Za), got {tuple(a.shape)}")
        return a, {"a": a, "a_raw": a}
    
    def encode_proprio(self, proprio):
        proprio = self.proprio_encoder(proprio)
        return proprio
    
    def encode_obs(self, obs):
        """
        input : obs (dict): keys
            "visual":  (B,T,3,H,W)
            "proprio": (B,T,...)
        output: dict:
            - "visual_tokens": (B, T, N, C)   # patch tokens
            - "visual_frame":  (B, T, 1, C)   # per-frame embedding
            - "proprio":       (B, T, Pp)     # proprio embedding
        """
        visual = obs["visual"]   # (B,T,3,H,W)
        B, T = visual.shape[:2]

        # run preprocessing on flattened frames
        v_bt = rearrange(visual, "b t c h w -> (b t) c h w")
        v_bt = self.encoder_transform(v_bt)

        # restore (B,T,3,H,W) for DinoV2Encoder (it does its own flatten)
        visual = rearrange(v_bt, "(b t) c h w -> b t c h w", b=B, t=T)

        out = self.encoder(visual)  # returns (B,T,N,C) or tuple

        if isinstance(out, tuple):
            pt, cf = out                    # (B,T,N,C), (B,T,1,C)
        else:
            pt = out                        # (B,T,N,C)
            cf = pt.mean(dim=2, keepdim=True)  # (B,T,1,C)

        proprio = obs["proprio"]
        proprio_emb = self.encode_proprio(proprio)

        return {
            "visual_tokens": pt,
            "visual_frame": cf,
            "proprio": proprio_emb,
        }


    def predict(self, z):  # in embedding space
        """
        input : z: (b, num_hist, num_patches, emb_dim)
        output: z: (b, num_hist, num_patches, emb_dim)
        """
        T = z.shape[1]
        # reshape to a batch of windows of inputs
        z = rearrange(z, "b t p d -> b (t p) d")
        # (b, num_hist * num_patches per img, emb_dim)
        z = self.predictor(z)
        z = rearrange(z, "b (t p) d -> b t p d", t=T)
        return z

    def decode(self, z):
        """
        input :   z: (b, num_frames, num_patches, emb_dim)
        output: obs: (b, num_frames, 3, img_size, img_size)
        """
        z_obs, z_act = self.separate_emb(z)
        obs, diff = self.decode_obs(z_obs)
        return obs, diff

    def decode_obs(self, z_obs):
        """
        input :   z: (b, num_frames, num_patches, emb_dim)
        output: obs: (b, num_frames, 3, img_size, img_size)
        """
        b, num_frames, num_patches, emb_dim = z_obs["visual"].shape
        visual, diff = self.decoder(z_obs["visual"])  # (b*num_frames, 3, 224, 224)
        visual = rearrange(visual, "(b t) c h w -> b t c h w", t=num_frames)
        obs = {
            "visual": visual,
            "proprio": z_obs["proprio"], # Note: no decoder for proprio for now!
        }
        return obs, diff
    
    def separate_emb(self, z):
        """
        input: z (tensor)
        output: z_obs (dict), z_act (tensor)
        """
        if self.concat_dim == 0:
            z_visual, z_proprio, z_act = z[:, :, :-2, :], z[:, :, -2, :], z[:, :, -1, :]
        elif self.concat_dim == 1:
            z_visual, z_proprio, z_act = z[..., :-(self.proprio_dim + self.action_dim)], \
                                         z[..., -(self.proprio_dim + self.action_dim) :-self.action_dim],  \
                                         z[..., -self.action_dim:]
            # remove tiled dimensions
            z_proprio = z_proprio[:, :, 0, : self.proprio_dim // self.num_proprio_repeat]
            z_act = z_act[:, :, 0, : self.action_dim // self.num_action_repeat]
        z_obs = {"visual": z_visual, "proprio": z_proprio}
        return z_obs, z_act

    def forward(self, obs, act):
        """
        input:  obs (dict):  "visual", "proprio" (b, num_frames, 3, img_size, img_size)
                act: (b, num_frames, action_dim)
        output: z_pred: (b, num_hist, num_patches, emb_dim)
                visual_pred: (b, num_hist, 3, img_size, img_size)
                visual_reconstructed: (b, num_frames, 3, img_size, img_size)
        """
        loss = 0
        loss_components = {}
        z, aux_act = self.encode(obs, act)
        a = aux_act.get("a_raw") if isinstance(aux_act, dict) else None
        if a is not None:
            l2_pen = self.lambda_l2 * (a ** 2).mean()
            loss = loss + l2_pen
            loss_components["action_l2"] = l2_pen

            flat = a.view(-1, a.shape[-1])
            var = flat.var(dim=0, unbiased=False)
            var_pen = self.lambda_var * ((var - self.var_target) ** 2).mean()
            loss = loss + var_pen
            loss_components["action_variance"] = var_pen
        z_src = z[:, : self.num_hist, :, :]  # (b, num_hist, num_patches, dim)
        z_tgt = z[:, self.num_pred :, :, :]  # (b, num_hist, num_patches, dim)
        visual_src = obs['visual'][:, : self.num_hist, ...]  # (b, num_hist, 3, img_size, img_size)
        visual_tgt = obs['visual'][:, self.num_pred :, ...]  # (b, num_hist, 3, img_size, img_size)

        if self.predictor is not None:
            z_pred = self.predict(z_src)
            if self.decoder is not None:
                obs_pred, diff_pred = self.decode(
                    z_pred.detach()
                )  # recon loss should only affect decoder
                visual_pred = obs_pred['visual']
                recon_loss_pred = self.decoder_criterion(visual_pred, visual_tgt)
                decoder_loss_pred = (
                    recon_loss_pred + self.decoder_latent_loss_weight * diff_pred
                )
                loss_components["decoder_recon_loss_pred"] = recon_loss_pred
                loss_components["decoder_vq_loss_pred"] = diff_pred
                loss_components["decoder_loss_pred"] = decoder_loss_pred
            else:
                visual_pred = None

            # Compute loss for visual, proprio dims (i.e. exclude action dims)
            if self.concat_dim == 0:
                z_visual_loss = self.emb_criterion(z_pred[:, :, :-2, :], z_tgt[:, :, :-2, :].detach())
                z_proprio_loss = self.emb_criterion(z_pred[:, :, -2, :], z_tgt[:, :, -2, :].detach())
                z_loss = self.emb_criterion(z_pred[:, :, :-1, :], z_tgt[:, :, :-1, :].detach())
            elif self.concat_dim == 1:
                z_visual_loss = self.emb_criterion(
                    z_pred[:, :, :, :-(self.proprio_dim + self.action_dim)], \
                    z_tgt[:, :, :, :-(self.proprio_dim + self.action_dim)].detach()
                )
                z_proprio_loss = self.emb_criterion(
                    z_pred[:, :, :, -(self.proprio_dim + self.action_dim): -self.action_dim], 
                    z_tgt[:, :, :, -(self.proprio_dim + self.action_dim): -self.action_dim].detach()
                )
                z_loss = self.emb_criterion(
                    z_pred[:, :, :, :-self.action_dim], 
                    z_tgt[:, :, :, :-self.action_dim].detach()
                )

            loss = loss + z_loss
            loss_components["z_loss"] = z_loss
            loss_components["z_visual_loss"] = z_visual_loss
            loss_components["z_proprio_loss"] = z_proprio_loss
        else:
            visual_pred = None
            z_pred = None

        if self.decoder is not None:
            obs_reconstructed, diff_reconstructed = self.decode(
                z.detach()
            )  # recon loss should only affect decoder
            visual_reconstructed = obs_reconstructed["visual"]
            recon_loss_reconstructed = self.decoder_criterion(visual_reconstructed, obs['visual'])
            decoder_loss_reconstructed = (
                recon_loss_reconstructed
                + self.decoder_latent_loss_weight * diff_reconstructed
            )

            loss_components["decoder_recon_loss_reconstructed"] = (
                recon_loss_reconstructed
            )
            loss_components["decoder_vq_loss_reconstructed"] = diff_reconstructed
            loss_components["decoder_loss_reconstructed"] = (
                decoder_loss_reconstructed
            )
            loss = loss + decoder_loss_reconstructed
        else:
            visual_reconstructed = None
        loss_components["loss"] = loss
        return z_pred, visual_pred, visual_reconstructed, loss, loss_components

    def replace_actions_from_z(self, z, act):
        act_emb, _ = self.encode_act(act)
        if self.concat_dim == 0:
            z[:, :, -1, :] = act_emb
        elif self.concat_dim == 1:
            # act_emb is expected to be (B,T,1,Za). Tile across tokens without
            # introducing an extra singleton dimension.
            if act_emb.dim() == 4:
                act_tiled = repeat(act_emb, "b t 1 a -> b t f a", f=z.shape[2])
            elif act_emb.dim() == 3:
                act_tiled = repeat(act_emb.unsqueeze(2), "b t 1 a -> b t f a", f=z.shape[2])
            else:
                raise ValueError(f"Unexpected act_emb shape {act_emb.shape}")
            act_repeated = act_tiled.repeat(1, 1, 1, self.num_action_repeat)
            z[..., -self.action_dim:] = act_repeated
        return z


    def rollout(self, obs_0, act):
        """
        input:
        obs_0 (dict): {"visual": (B, n, 3, H, W), ...}
        act:
            - Raw actions:           (B, n+t, A)
            - Pre-embedded actions:  (B, n+t, 1, D)   # e.g., from inverse projector

        output:
        z_obses: (B, n+t+1, F, C)   # observation embeddings for each step
        z:       (B, n+t+1, F, C)   # full latent sequence after injection
        """
        B = obs_0["visual"].shape[0] # num_samples 
        num_obs_init = obs_0['visual'].shape[1] # 1

        # Detect whether 'act' is pre-embedded (4D) or raw (3D)
        is_embedded = (act is not None) and (act.dim() == 4) # true for wm.rollout calls 

        if is_embedded:
            # act: (B, num_obs_init + T_future, 1, D_act)
            act_0_emb  = act[:, :num_obs_init]     # (B, num_obs_init, 1, D_act)  context
            action_emb = act[:, num_obs_init:]     # (B, T_future,    1, D_act)   future

            # --- encode observations ONLY (avoid inverse/double-encoding of actions) ---
            z_dct   = self.encode_obs(obs_0)
            visual  = z_dct["visual_tokens"]       # (B, num_obs_init, F, C_v)
            proprio = z_dct["proprio"]             # (B, num_obs_init, C_p)
            B, T0, F, _ = visual.shape
            device, dtype = visual.device, visual.dtype

            # --- build initial z and inject act_0_emb like your existing concat logic ---
            if self.concat_dim == 0:
                # token-level: [visual_tokens] + [proprio_tok] + [action_tok]
                proprio_tok = proprio.unsqueeze(2)                     # (B,T0,1,C_p)
                act0_tok = act_0_emb if act_0_emb.dim() == 4 else act_0_emb.unsqueeze(2)  # -> (B,T0,1,D_act)
                z = torch.cat(
                    [visual, proprio_tok, act0_tok.to(device=device, dtype=dtype)],
                    dim=2
                )                                                      # (B,T0,F+2, *)

            elif self.concat_dim == 1:
                # channel-level: tile proprio & action across tokens, concat in channels
                proprio_tiled = proprio.unsqueeze(2).expand(-1, -1, F, -1)              # (B,T0,F,C_p)
                prop_rep = proprio_tiled.repeat(1, 1, 1, getattr(self, "num_proprio_repeat", 1))

                rep_a = getattr(self, "num_action_repeat", 1)
                act0_tiled = act_0_emb.expand(-1, -1, F, -1)                            # (B,T0,F, D_act)
                act0_rep   = act0_tiled.repeat(1, 1, 1, rep_a)                          # (B,T0,F, D_act*rep_a)

                # allocate tail channels and write action there
                tail = torch.zeros(B, T0, F, act0_rep.shape[-1], device=device, dtype=dtype)
                z = torch.cat([visual, prop_rep, tail], dim=3)                          # (B,T0,F, C_v + C_p*rep_p + D_act*rep_a)
                z[..., -act0_rep.shape[-1]:] = act0_rep
            else:
                raise ValueError(f"Unknown concat_dim: {self.concat_dim}")


        else:
            # Raw actions (B, n+t, A) — original path
            act_0  = act[:, :num_obs_init]             # (B, n, A)
            action = act[:, num_obs_init:]             # (B, t, A)
            z, _ = self.encode(obs_0, act_0)             # (B, n, F, C)

        # Autoregressive rollout
        t = 0
        inc = 1
        if is_embedded:
            T_future = action_emb.shape[1] # 5
            while t < T_future:
                z_pred = self.predict(z[:, -self.num_hist :])   # (B, h, F, C).  
                z_new  = z_pred[:, -inc:, ...]                  # (B, 1, F, C)

                # Inject pre-embedded action (skip encode_act to avoid double-encoding)
                if self.concat_dim == 0:
                    # Action occupies its own token at the end
                    z_new[:, :, -1, :] = action_emb[:, t : t + inc, :].squeeze(2)  # (B,1,D)
                elif self.concat_dim == 1:
                    # Action concatenated in channel dimension for each token
                    F = z_new.shape[2]
                    step_emb = action_emb[:, t : t + inc, :]                        # (B,1,1,D)
                    act_tiled = step_emb.expand(-1, -1, F, -1)                      # (B,1,F,D)
                    rep = getattr(self, "num_action_repeat", 1)
                    act_rep = act_tiled.repeat(1, 1, 1, rep)                         # (B,1,F,D*rep)
                    z_new[..., -act_rep.shape[-1]:] = act_rep
                else:
                    raise ValueError(f"Unknown concat_dim: {self.concat_dim}")

                z = torch.cat([z, z_new], dim=1)
                t += inc
        else:
            T_future = action.shape[1]
            while t < T_future:
                z_pred = self.predict(z[:, -self.num_hist :])   # (B, h, F, C)
                z_new  = z_pred[:, -inc:, ...]                  # (B, 1, F, C)
                z_new  = self.replace_actions_from_z(z_new, action[:, t : t + inc, :])
                z = torch.cat([z, z_new], dim=1)
                t += inc

        # Final extra prediction (no action injection) to get length n+t+1
        z_pred = self.predict(z[:, -self.num_hist :])
        z_new  = z_pred[:, -1 :, ...]
        z      = torch.cat([z, z_new], dim=1)

        z_obses, z_acts = self.separate_emb(z)
        return z_obses, z
