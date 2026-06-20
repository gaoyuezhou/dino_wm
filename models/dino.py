import os
import subprocess
import torch
import torch.nn as nn
from typing import Optional

torch.hub._validate_not_a_forked_repo=lambda a,b,c: True

def _patch_dinov2_py39(cache_dir):
    """Patch DINOv2 source to replace PEP 604 union syntax (float | None)
    with typing.Optional for Python 3.9 compatibility."""
    import glob
    for py_file in glob.glob(os.path.join(cache_dir, "dinov2", "**", "*.py"), recursive=True):
        with open(py_file, "r") as f:
            content = f.read()
        if "float | None" in content:
            content = content.replace("float | None", "Optional[float]")
            if "from typing import Optional" not in content:
                content = "from typing import Optional\n" + content
            with open(py_file, "w") as f:
                f.write(content)

class DinoV2Encoder(nn.Module):
    def __init__(self, name, feature_key):
        super().__init__()
        self.name = name
        cache_dir = os.path.join(torch.hub.get_dir(), "facebookresearch_dinov2_main")
        if not os.path.exists(os.path.join(cache_dir, "dinov2", "hub", "backbones.py")):
            os.makedirs(torch.hub.get_dir(), exist_ok=True)
            subprocess.run(
                ["git", "clone", "https://github.com/facebookresearch/dinov2.git", cache_dir],
                check=True,
            )
        _patch_dinov2_py39(cache_dir)
        self.base_model = torch.hub.load(cache_dir, name, source="local", trust_repo=True)
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
