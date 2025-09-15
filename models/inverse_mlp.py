import torch
import torch.nn as nn

from .vqvae import Quantize


class InverseMLP(nn.Module):
    """Simple inverse-dynamics MLP with VQ bottleneck.

    The module predicts the action that caused the transition between two
    consecutive observations.  It mirrors the training routine provided in the
    user snippet: given per-frame embeddings ``e_t`` it models

    ``a_t = f(e_{t+1} - e_t)``.

    The output of the MLP is passed through a vector-quantisation bottleneck
    (EMA update, 8 codes × 16 dimensional) and the commitment loss is returned
    alongside the embedding so that callers can include it in their loss
    function.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim1: int = 256,
        hidden_dim2: int = 128,
        output_dim: int = 16,
        dropout: float = 0.2,
        n_embed: int = 8,
        decay: float = 0.99,
        commitment_cost: float = 0.25,
    ):
        super().__init__()

        layers = [nn.Linear(input_dim, hidden_dim1), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.extend([nn.Linear(hidden_dim1, hidden_dim2), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim2, output_dim))
        self.mlp = nn.Sequential(*layers)

        self.quantize = Quantize(output_dim, n_embed, decay=decay)
        self.commitment_cost = commitment_cost
        self.output_dim = output_dim

    def forward(self, obs_enc: torch.Tensor):
        """Compute quantised action embeddings.

        Args:
            obs_enc: (B, T, 1, C) or (B, T, C) tensor of per-frame embeddings.

        Returns:
            tuple: (quantised_embedding, {"vq_loss": loss}) where the
                quantised_embedding has shape (B, T, 1, output_dim).
        """

        if obs_enc.dim() == 4:
            obs_enc = obs_enc.squeeze(2)

        if obs_enc.size(1) < 2:
            raise ValueError("Need at least two time steps for inverse MLP")

        diff = obs_enc[:, 1:] - obs_enc[:, :-1]  # (B, T-1, C)
        # pad to maintain length T
        diff = torch.cat([diff, diff[:, -1:, :]], dim=1)

        B, T, C = diff.shape
        x = diff.reshape(B * T, C)
        x = self.mlp(x)
        x = x.view(B, T, self.output_dim)
        quant, diff_vq, _ = self.quantize(x)
        quant = quant.unsqueeze(2)  # (B, T, 1, output_dim)

        vq_loss = self.commitment_cost * diff_vq

        return quant, {"vq_loss": vq_loss}

    def configure_optimizers(self, weight_decay, lr, betas):
        return [
            torch.optim.AdamW(self.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        ]

