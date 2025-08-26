import torch
import torch.nn as nn
# (keep your existing bypass if you need it)
torch.hub._validate_not_a_forked_repo = lambda a,b,c: True

class DinoV2Encoder(nn.Module):
    """
    feature_key:
      - "x_norm_patchtokens" -> (B, T, N, C)
      - "x_norm_clstoken"    -> (B, T, 1, C)  (dummy patch dim kept)
      - "both"               -> (patch_tokens: (B,T,N,C), cls_token: (B,T,1,C))
    """
    def __init__(self, name: str, feature_key: str = "both"):
        super().__init__()
        self.name = name
        self.base_model = torch.hub.load("facebookresearch/dinov2", name)
        self.feature_key = feature_key
        self.emb_dim = self.base_model.num_features
        self.patch_size = getattr(self.base_model, "patch_size", None)

        valid = {"x_norm_patchtokens", "x_norm_clstoken", "both"}
        if feature_key not in valid:
            raise ValueError(f"Invalid feature_key: {feature_key}. Choose one of {valid}.")

    def _reshape_bt(self, x):
        """Accept (B,T,3,H,W) or (B*T,3,H,W); return (BT,3,H,W), (B,T)."""
        if x.ndim == 5:
            B, T, C, H, W = x.shape
            return x.reshape(B*T, C, H, W), (B, T)
        elif x.ndim == 4:
            BT, C, H, W = x.shape
            return x, (1, BT)
        else:
            raise ValueError(f"expected (B,T,3,H,W) or (BT,3,H,W), got {x.shape}")

    def forward(self, x):
        """
        If feature_key != 'both': returns a single tensor
          - 'x_norm_patchtokens' -> (B,T,N,C)
          - 'x_norm_clstoken'    -> (B,T,1,C)
        If feature_key == 'both': returns (patch_tokens, cls_token)
          - patch_tokens: (B,T,N,C)
          - cls_token   : (B,T,1,C)
        """
        x, (B, T) = self._reshape_bt(x)
        feats = self.base_model.forward_features(x)  # ONE backbone pass

        # DINOv2 hub dict keys:
        #   'x_norm_patchtokens': (BT, N, C)
        #   'x_norm_clstoken'   : (BT, C)
        pt = feats.get("x_norm_patchtokens", None)
        ct = feats.get("x_norm_clstoken", None)

        if pt is None:
            raise RuntimeError("DINOv2 forward_features missing 'x_norm_patchtokens'")
        if ct is None:
            # some variants: no cls -> mean over patches as a fallback
            ct = pt.mean(dim=1)  # (BT, C)

        # reshape back to (B,T,...)
        pt = pt.view(B, T, *pt.shape[1:])         # (B,T,N,C)
        ct = ct.view(B, T, -1).unsqueeze(2)       # (B,T,1,C)

        if self.feature_key == "x_norm_patchtokens":
            return pt
        elif self.feature_key == "x_norm_clstoken":
            return ct
        else:  # "both"
            return pt, ct


# torch.hub._validate_not_a_forked_repo=lambda a,b,c: True

# class DinoV2Encoder(nn.Module):
#     def __init__(self, name, feature_key):
#         super().__init__()
#         self.name = name
#         self.base_model = torch.hub.load("facebookresearch/dinov2", name)
#         self.feature_key = feature_key
#         self.emb_dim = self.base_model.num_features
#         if feature_key == "x_norm_patchtokens":
#             self.latent_ndim = 2
#         elif feature_key == "x_norm_clstoken":
#             self.latent_ndim = 1
#         else:
#             raise ValueError(f"Invalid feature key: {feature_key}")

#         self.patch_size = self.base_model.patch_size

#     def forward(self, x):
#         emb = self.base_model.forward_features(x)[self.feature_key]
#         if self.latent_ndim == 1:
#             emb = emb.unsqueeze(1) # dummy patch dim
#         return emb