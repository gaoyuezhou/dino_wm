import torch
import torch.nn as nn

torch.hub._validate_not_a_forked_repo=lambda a,b,c: True

# Pin to a pre-2025 commit: dinov2 main now uses PEP 604 unions (X | None),
# which fail at import time on Python 3.9.
DINOV2_REPO = "facebookresearch/dinov2:e1277af2ba9496fbadf7aec6eba56e8d882d1e35"


class DinoV2Encoder(nn.Module):
    def __init__(self, name, feature_key, pretrained=True, init_seed=0, hub_model=None):
        """
        name: label used for logging and run names.
        hub_model: torch.hub entry to build (dinov2_vits14, ...). Defaults to name.
        pretrained: False builds the same architecture with random weights.
        init_seed: seeds the random weights so a frozen random encoder, which is not
            saved in the checkpoint, is rebuilt identically by train.py and plan.py.
        """
        super().__init__()
        self.name = name
        self.pretrained = pretrained
        hub_model = hub_model or name
        if pretrained:
            self.base_model = torch.hub.load(DINOV2_REPO, hub_model)
        else:
            # Only the CPU generator is touched while building, and it is restored after,
            # so the data/training random streams are unaffected.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(init_seed)
                self.base_model = torch.hub.load(DINOV2_REPO, hub_model, pretrained=False)
        self.feature_key = feature_key
        self.emb_dim = self.base_model.num_features
        if feature_key == "x_norm_patchtokens":
            self.latent_ndim = 2
        elif feature_key == "x_norm_clstoken":
            self.latent_ndim = 1
        else:
            raise ValueError(f"Invalid feature key: {feature_key}")

        self.patch_size = self.base_model.patch_size

    def forward(self, x):
        emb = self.base_model.forward_features(x)[self.feature_key]
        if self.latent_ndim == 1:
            emb = emb.unsqueeze(1) # dummy patch dim
        return emb
