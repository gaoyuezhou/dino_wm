import torch
import einops
import torch.nn as nn
import torch.nn.functional as F
from .vector_quantize_pytorch import ResidualVQ


def weights_init_encoder(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class EncoderMLP(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim=16,
        hidden_dim=128,
        layer_num=1,
        last_activation=None,
    ):
        super(EncoderMLP, self).__init__()
        layers = []

        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(layer_num):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        self.encoder = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_dim, output_dim)

        if last_activation is not None:
            self.last_layer = last_activation
        else:
            self.last_layer = None
        self.apply(weights_init_encoder)

    def forward(self, x):
        h = self.encoder(x)
        state = self.fc(h)
        if self.last_layer:
            state = self.last_layer(state)
        return state


class VqVae(nn.Module):
    def __init__(
        self,
        input_dim_h=10,  # length of action chunk
        input_dim_w=9,  # action dim
        n_latent_dims=512,
        vqvae_n_embed=32,
        vqvae_groups=4,
        encoder_loss_multiplier=1.0,
        act_scale=1.0,
    ):
        super(VqVae, self).__init__()
        self.n_latent_dims = n_latent_dims
        self.input_dim_h = input_dim_h
        self.input_dim_w = input_dim_w
        self.rep_dim = self.n_latent_dims
        self.vqvae_n_embed = vqvae_n_embed
        self.vqvae_groups = vqvae_groups
        self.encoder_loss_multiplier = encoder_loss_multiplier
        self.act_scale = act_scale

        discrete_cfg = {"groups": self.vqvae_groups, "n_embed": self.vqvae_n_embed}

        self.vq_layer = ResidualVQ(
            dim=self.n_latent_dims,
            num_quantizers=discrete_cfg["groups"],
            codebook_size=self.vqvae_n_embed,
        )
        self.embedding_dim = self.n_latent_dims

        self.encoder = EncoderMLP(
            input_dim=input_dim_w * self.input_dim_h, output_dim=n_latent_dims
        )
        self.decoder = EncoderMLP(
            input_dim=n_latent_dims, output_dim=input_dim_w * self.input_dim_h
        )

    def draw_logits_forward(self, encoding_logits):
        z_embed = self.vq_layer.draw_logits_forward(encoding_logits)
        return z_embed

    def draw_code_forward(self, encoding_indices):
        with torch.no_grad():
            z_embed = self.vq_layer.get_codes_from_indices(encoding_indices)
            z_embed = z_embed.sum(dim=0)
        return z_embed

    def get_action_from_latent(self, latent) -> torch.Tensor:
        output = self.decoder(latent) * self.act_scale
        return einops.rearrange(output, "... (T A) -> ... T A", A=self.input_dim_w)

    def preprocess(self, state):
        state = torch.Tensor(state)
        state = state / self.act_scale
        if self.input_dim_h == 1:
            state = state.squeeze(-2)  # state.squeeze(-1)
        else:
            state = einops.rearrange(state, "... T A -> ... (T A)")
        return state

    def get_code(self, state):
        state = self.preprocess(state)
        with torch.no_grad():
            state_rep = self.encoder(state)
            state_vq, vq_code, vq_loss_state = self.vq_layer(state_rep)
            vq_loss_state = torch.sum(vq_loss_state)
            return state_vq, vq_code

    def forward(self, state):
        state = self.preprocess(state)
        state_rep = self.encoder(state)
        state_rep_shape = state_rep.shape[:-1]
        state_rep_flat = state_rep.view(state_rep.size(0), -1, state_rep.size(1))
        state_rep_flat, vq_code, vq_loss_state = self.vq_layer(state_rep_flat)
        state_vq = state_rep_flat.view(*state_rep_shape, -1)
        vq_code = vq_code.view(*state_rep_shape, -1)
        dec_out = self.decoder(state_vq)

        vq_loss_state = vq_loss_state.sum()
        encoder_loss = (state - dec_out).abs().mean()
        vqvae_recon_loss = F.mse_loss(state, dec_out)

        loss = encoder_loss * self.encoder_loss_multiplier + (vq_loss_state * 5)
        loss_dict = {
            "loss": loss.detach().clone(),
            "vq_loss_state": vq_loss_state.detach().clone(),
            "vqvae_recon_loss": vqvae_recon_loss.detach().clone(),
            "encoder_loss": encoder_loss.detach().clone(),
        }
        return loss, vq_code, loss_dict

    def state_dict(
        self,
        destination: str = None,
        prefix: str = "",
        keep_vars: bool = False,
    ):
        if destination is None:
            destination = {}
        self.encoder.state_dict(destination, prefix + "encoder", keep_vars)
        self.decoder.state_dict(destination, prefix + "decoder", keep_vars)
        self.vq_layer.state_dict(destination, prefix + "vq_embedding", keep_vars)
        return destination

    def load_state_dict(self, state_dict):
        self.encoder.load_state_dict(state_dict["encoder"])
        self.decoder.load_state_dict(state_dict["decoder"])
        self.vq_layer.load_state_dict(state_dict["vq_embedding"])
