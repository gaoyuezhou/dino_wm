import sys

import torch
import torch.nn as nn


class VJEPAStudentEncoder(nn.Module):
    """Frozen p-salt student_train video ViT, used as a per-frame patch encoder."""

    def __init__(self, name, checkpoint_path, repo_path):
        super().__init__()
        self.name = name
        # Appended so p-salt's packages can't shadow this repo's models/ and datasets/.
        if repo_path not in sys.path:
            sys.path.append(repo_path)
        from app.eval.adapters.vjepa_backbone import create_video_encoder

        # Each frame is encoded alone, so there is one temporal token and the
        # causal mask would have nothing to block.
        self.base_model, info = create_video_encoder(
            checkpoint_path, num_frames=2, img_size=224, use_causal_attn=False
        )
        # not required becuase we have salt based model right to eval which has tube2
        # if info["tubelet_size"] != 1:
        #     raise ValueError(f"expected tubelet_size 1, got {info['tubelet_size']}")

        self.emb_dim = info["embed_dim"]
        self.patch_size = info["patch_size"]
        self.latent_ndim = 2
        # Not saved: these are constants, not checkpoint state.
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False
        )

    def forward(self, x):
        # DINO-WM images are scaled to [-1, 1]; the student trained on ImageNet-normalized input.
        x = (x * 0.5 + 0.5 - self.mean) / self.std
        return self.base_model(x.unsqueeze(2))
