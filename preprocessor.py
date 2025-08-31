import torch
from einops import rearrange

class Preprocessor:
    def __init__(self, 
        action_mean,
        action_std,
        state_mean,
        state_std,
        proprio_mean,
        proprio_std,
        transform,
    ):
        self.action_mean = action_mean
        self.action_std = action_std
        self.state_mean = state_mean
        self.state_std = state_std
        self.proprio_mean = proprio_mean
        self.proprio_std = proprio_std
        self.transform = transform

    def normalize_actions(self, actions):
        '''
        actions: (b, t, action_dim)  
        '''
        return (actions - self.action_mean) / self.action_std

    def denormalize_actions(self, actions):
        '''
        actions: (b, t, action_dim)  
        '''
        return actions * self.action_std + self.action_mean
    
    def normalize_proprios(self, proprio):
        '''
        input shape (..., proprio_dim)
        '''
        return (proprio - self.proprio_mean) / self.proprio_std

    def normalize_states(self, state):
        '''
        input shape (..., state_dim)
        '''
        return (state - self.state_mean) / self.state_std

    def preprocess_obs_visual(self, obs_visual):
      # in Preprocessor.transform_obs_visual
      return rearrange(obs_visual, "b t h w c -> b t c h w") / 255.0

    def transform_obs_visual(self, obs_visual):
        v = torch.as_tensor(obs_visual)

        # Case: (B, 1, ?, ?, ?) – squeeze T so torchvision sees 4D
        if v.ndim == 5 and v.shape[1] == 1:
            v = v[:, 0]  # -> (B, ?, ?, ?)

            # Detect layout and convert to BCHW float32 in [0,1]
            if v.shape[-1] in (1, 3, 4) and v.shape[1] not in (1, 3, 4):
                # (B, H, W, C) -> (B, C, H, W)
                v = rearrange(v, "b h w c -> b c h w").to(torch.float32) / 255.0
            else:
                # assume already (B, C, H, W)
                v = v.to(torch.float32)

            v = self.transform(v) if self.transform else v  # torchvision on 4D
            return v.unsqueeze(1)  # back to (B, 1, C, H, W)

        # General cases (no time dim or different shapes)
        # Convert to BCHW then apply transform
        if v.ndim == 5:
            # (B, T, H, W, C) -> (B, T, C, H, W); (B,T,C,H,W) stays
            if v.shape[-1] in (1, 3, 4):
                v = rearrange(v, "b t h w c -> b t c h w").to(torch.float32) / 255.0
            else:
                v = v.to(torch.float32)
            # Flatten T for torchvision, then restore
            B, T, C, H, W = v.shape
            v = v.reshape(B * T, C, H, W)
            v = self.transform(v) if self.transform else v
            C_new, H_new, W_new = v.shape[1:]
            return v.reshape(B, T, C_new, H_new, W_new)
        if v.ndim == 4:
            if v.shape[-1] in (1, 3, 4) and v.shape[1] not in (1, 3, 4):
                v = rearrange(v, "b h w c -> b c h w").to(torch.float32) / 255.0
            else:
                v = v.to(torch.float32)
            return self.transform(v) if self.transform else v

        if v.ndim == 3:
            if v.shape[0] in (1, 3, 4):   # (C,H,W)
                v = v.to(torch.float32)
            else:                         # (H,W,C)
                v = rearrange(v, "h w c -> c h w").to(torch.float32) / 255.0
            return self.transform(v) if self.transform else v

        raise ValueError(f"Unexpected visual shape: {tuple(v.shape)}")
    
    def transform_obs(self, obs):
        '''
        np arrays to tensors
        '''
        transformed_obs = {}
        transformed_obs['visual'] = self.transform_obs_visual(obs['visual'])
        transformed_obs['proprio'] = self.normalize_proprios(torch.tensor(obs['proprio']))
        return transformed_obs
