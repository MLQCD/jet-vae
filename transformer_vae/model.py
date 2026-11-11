"""Particle Transformer VAE over (z, eta, phi) with an EMD reconstruction loss.

This model never represents absolute energy scale: every particle is
``(z, eta, phi)`` with ``z`` a pT fraction. Both
training invariants the data preprocessing enforces --
``sum(z) == 1`` and jet axis ``(eta, phi) == (0, 0)`` -- are reproduced by the
decoder *by construction*, not merely encouraged by the loss:

- ``z`` comes from a softmax over all ``max_particles`` decoder slots, so it
  sums to exactly 1 for every generated jet, independent of what the loss
  supervises.
- ``(eta, phi)`` are recentered on the decoded jet's own axis before being
  returned, so that axis is exactly ``(0, 0)`` for every generated jet.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .shape_ot import (
    jet_axis_from_shape,
    recenter,
    shape_sinkhorn_emd_loss,
)


class ParticleTransformerShapeEMDVAE(nn.Module):
    """Transformer VAE for padded ``(z, eta, phi)`` particle sequences."""

    num_iter = 0

    def __init__(
        self,
        max_particles: int,
        latent_dim: int,
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        beta: float = 1.0,
        reconstruction_weight: float = 1.0,
        sinkhorn_epsilon: float = 0.05,
        sinkhorn_n_iters: int = 50,
        sinkhorn_p: float = 1.0,
        angular_scale: float = 0.4,
        sinkhorn_debias: bool = True,
    ) -> None:
        super().__init__()
        if max_particles <= 0:
            raise ValueError("max_particles must be positive")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        if reconstruction_weight <= 0.0:
            raise ValueError("reconstruction_weight must be positive")
        if sinkhorn_epsilon <= 0.0:
            raise ValueError("sinkhorn_epsilon must be positive")
        if sinkhorn_n_iters <= 0:
            raise ValueError("sinkhorn_n_iters must be positive")
        if sinkhorn_p <= 0.0:
            raise ValueError("sinkhorn_p must be positive")
        if angular_scale <= 0.0:
            raise ValueError("angular_scale must be positive")

        self.feature_dim = 3
        self.max_particles = int(max_particles)
        self.latent_dim = int(latent_dim)
        self.beta = float(beta)
        self.reconstruction_weight = float(reconstruction_weight)
        self.sinkhorn_epsilon = float(sinkhorn_epsilon)
        self.sinkhorn_n_iters = int(sinkhorn_n_iters)
        self.sinkhorn_p = float(sinkhorn_p)
        self.angular_scale = float(angular_scale)
        self.sinkhorn_debias = bool(sinkhorn_debias)

        self.input_projection = nn.Linear(self.feature_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * mlp_ratio,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.encoder_norm = nn.LayerNorm(embed_dim)
        self.fc_mu = nn.Linear(embed_dim, latent_dim)
        self.fc_var = nn.Linear(embed_dim, latent_dim)

        self.z_projection = nn.Linear(latent_dim, embed_dim)
        self.query_embedding = nn.Parameter(
            torch.zeros(1, self.max_particles, embed_dim)
        )
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * mlp_ratio,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=depth)
        self.decoder_norm = nn.LayerNorm(embed_dim)
        # Outputs (logit_z, eta_raw, phi_raw); decode() below turns these into
        # a (z, eta, phi) triple that satisfies sum(z)=1 and axis=(0,0) by
        # construction (see module docstring).
        self.momentum_head = nn.Linear(embed_dim, self.feature_dim)

        nn.init.trunc_normal_(self.query_embedding, std=0.02)

    def encode(
        self, particles: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if particles.shape[1:] != (self.max_particles, self.feature_dim):
            raise ValueError(
                "particles must have shape "
                f"[batch, {self.max_particles}, {self.feature_dim}]"
            )
        if mask.shape != particles.shape[:2]:
            raise ValueError("mask must have shape [batch, max_particles]")

        encoded = self.encoder(
            self.input_projection(particles), src_key_padding_mask=~mask
        )
        encoded = self.encoder_norm(encoded)
        weights = mask.unsqueeze(-1).to(encoded.dtype)
        pooled = encoded.mul(weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.fc_mu(pooled), self.fc_var(pooled)

    def decode(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(z_frac, eta, phi)`` with ``sum(z_frac) == 1`` and jet axis
        exactly ``(0, 0)`` for every event, guaranteed structurally.
        """
        z_token = self.z_projection(z).unsqueeze(1)
        queries = self.query_embedding.expand(z.shape[0], -1, -1)
        decoded = self.decoder(queries + z_token)
        params = self.momentum_head(self.decoder_norm(decoded))
        logit_z, eta_raw, phi_raw = params.unbind(dim=-1)

        z_frac = F.softmax(logit_z, dim=1)
        eta_raw = eta_raw.clamp(min=-8.0, max=8.0)
        phi_raw = torch.atan2(torch.sin(phi_raw), torch.cos(phi_raw))

        eta_axis, phi_axis = jet_axis_from_shape(z_frac, eta_raw, phi_raw)
        eta, phi = recenter(eta_raw, phi_raw, eta_axis, phi_axis)
        return z_frac, eta, phi

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(
        self,
        particles: torch.Tensor,
        mask: torch.Tensor,
        momenta: torch.Tensor | None = None,
        sample_latent: bool = True,
    ) -> list[torch.Tensor | None]:
        mu, logvar = self.encode(particles, mask)
        z = self.reparameterize(mu, logvar) if sample_latent else mu
        z_frac, eta, phi = self.decode(z)
        return [z_frac, eta, phi, momenta, mask, mu, logvar]

    def loss_function(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        if self.training:
            self.num_iter += 1

        pred_z, pred_eta, pred_phi, target, mask, mu, logvar = args[:7]
        if target is None:
            raise ValueError("target (z, eta, phi) is required for loss_function")
        kld_weight = float(kwargs.get("M_N", 1.0))
        kl_scale = float(kwargs.get("kl_scale", 1.0))
        if kld_weight < 0.0:
            raise ValueError("M_N must be nonnegative")
        if not 0.0 <= kl_scale <= 1.0:
            raise ValueError("kl_scale must lie in [0, 1]")

        true_z, true_eta, true_phi = target.unbind(dim=-1)
        emd_loss = shape_sinkhorn_emd_loss(
            true_z,
            true_eta,
            true_phi,
            mask,
            pred_z,
            pred_eta,
            pred_phi,
            epsilon=self.sinkhorn_epsilon,
            n_iters=self.sinkhorn_n_iters,
            p=self.sinkhorn_p,
            angular_scale=self.angular_scale,
            debias=self.sinkhorn_debias,
        )

        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + logvar - mu.square() - logvar.exp(), dim=1)
        )
        weighted_emd = self.reconstruction_weight * emd_loss
        weighted_kld = self.beta * kld_weight * kl_scale * kld_loss
        total = weighted_emd + weighted_kld
        return {
            "loss": total,
            "EMD_Loss": emd_loss,
            "Weighted_EMD_Loss": weighted_emd,
            "KLD": kld_loss,
            "Weighted_KLD": weighted_kld,
        }

    def sample(
        self, num_samples: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.decode(torch.randn(num_samples, self.latent_dim, device=device))
