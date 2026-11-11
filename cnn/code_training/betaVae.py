import torch
from torch import nn
from torch.nn import functional as F
from typing import List


class BetaVAE(nn.Module):

    num_iter = 0  # Global static variable to keep track of iterations

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 image_size: int = 32,
                 hidden_dims: List = None,
                 beta: float = 1.0,
                 beta_start: float = 0.0,
                 beta_anneal_iter: int = int(1e5),
                 beta_anneal_schedule: str = 'linear',
                 gamma: float = 1000.,
                 max_capacity: float = 25.0,
                 Capacity_max_iter: int = int(1e5),
                 loss_type: str = 'B',
                 recon_loss_type: str = 'mse',
                 recon_loss_weight: float = 1.0,
                 recon_log_x0: float = 1e-3,
                 output_mode: str = 'event_pt',
                 recenter_output: bool = True,
                 eta_max: float = 0.5,
                 phi_max: float = 0.5,
                 **kwargs) -> None:
        super(BetaVAE, self).__init__()

        self.latent_dim = latent_dim
        self.beta = beta
        self.beta_start = beta_start
        self.beta_anneal_iter = beta_anneal_iter
        self.beta_anneal_schedule = beta_anneal_schedule
        self.gamma = gamma
        self.loss_type = loss_type
        self.recon_loss_type = recon_loss_type
        self.recon_loss_weight = recon_loss_weight
        self.recon_log_x0 = recon_log_x0
        self.output_mode = output_mode
        self.recenter_output = recenter_output
        self.eta_max = float(eta_max)
        self.phi_max = float(phi_max)
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter

        if self.output_mode not in ('event_pt', 'tanh'):
            raise ValueError(f"Unsupported output_mode={self.output_mode!r}")
        if self.recon_loss_type not in ('mse', 'l1', 'sqrt_mse', 'log_mse'):
            raise ValueError(f"Unsupported recon_loss_type={self.recon_loss_type!r}")
        if self.recon_loss_weight < 0:
            raise ValueError(f"recon_loss_weight must be non-negative, got {self.recon_loss_weight}")
        if self.recon_log_x0 <= 0:
            raise ValueError(f"recon_log_x0 must be positive, got {self.recon_log_x0}")
        if self.eta_max <= 0:
            raise ValueError(f"eta_max must be positive, got {self.eta_max}")
        if self.phi_max <= 0:
            raise ValueError(f"phi_max must be positive, got {self.phi_max}")
        if self.beta_anneal_iter < 0:
            raise ValueError(f"beta_anneal_iter must be non-negative, got {self.beta_anneal_iter}")
        if self.beta_anneal_schedule not in ('linear', 'log_linear'):
            raise ValueError(f"Unsupported beta_anneal_schedule={self.beta_anneal_schedule!r}")
        if self.beta_anneal_schedule == 'log_linear' and (self.beta_start <= 0 or self.beta <= 0):
            raise ValueError(
                "log_linear beta annealing requires beta_start > 0 and beta > 0 "
                f"(got beta_start={self.beta_start}, beta={self.beta})"
            )

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Spatial size after encoding: image_size / 2^(num_conv_layers)
        num_conv = len(hidden_dims)
        downsample_factor = 2 ** num_conv
        if image_size % downsample_factor != 0:
            raise ValueError(
                f"image_size={image_size} must be divisible by 2^{num_conv}={downsample_factor} "
                f"for hidden_dims={hidden_dims}."
            )
        self.spatial_size = image_size // (2 ** num_conv)
        if self.spatial_size < 1:
            raise ValueError(
                f"image_size={image_size} too small for {num_conv} encoder layers. "
                f"Need image_size >= {2 ** num_conv}."
            )
        self.encoder_out_channels = hidden_dims[-1]
        flattened_dim = hidden_dims[-1] * self.spatial_size * self.spatial_size

        # Encoder
        # Each block: Conv2d(stride=2) + BN + LeakyReLU
        # Spatial dims halved each block: image_size -> image_size/2^num_conv
        modules = []
        cur_in = in_channels
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(cur_in, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            cur_in = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu  = nn.Linear(flattened_dim, latent_dim)
        self.fc_var = nn.Linear(flattened_dim, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, flattened_dim)

        hidden_dims.reverse()  # e.g. [512, 256, 128, 64, 32]

        dec_modules = []
        for i in range(len(hidden_dims) - 1):
            dec_modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*dec_modules)

        # Final layer: one more ConvTranspose2d to restore original spatial size,
        # then a Conv2d to produce in_channels output channels.
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=in_channels,
                      kernel_size=3, padding=1))

    def recenter_event_pt_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Shift each event-level pT-normalized image so its massless-pixel jet
        axis is at (Delta phi, Delta eta) = (0, 0).
        """
        if not self.recenter_output:
            return image

        N, _, H, W = image.shape
        dtype = image.dtype
        device = image.device

        eta_step = 2.0 * self.eta_max / float(W)
        phi_step = 2.0 * self.phi_max / float(H)
        eta = torch.linspace(
            -self.eta_max + 0.5 * eta_step,
            self.eta_max - 0.5 * eta_step,
            W,
            device=device,
            dtype=dtype,
        )
        phi = torch.linspace(
            -self.phi_max + 0.5 * phi_step,
            self.phi_max - 0.5 * phi_step,
            H,
            device=device,
            dtype=dtype,
        )

        total_pt = image.sum(dim=1)
        eta_grid = eta.view(1, 1, W)
        phi_grid = phi.view(1, H, 1)

        px_j = (total_pt * torch.cos(phi_grid)).sum(dim=(1, 2))
        py_j = (total_pt * torch.sin(phi_grid)).sum(dim=(1, 2))
        pz_j = (total_pt * torch.sinh(eta_grid)).sum(dim=(1, 2))
        pt_j = torch.sqrt(px_j.square() + py_j.square()).clamp_min(1e-8)

        phi_j = torch.atan2(py_j, px_j)
        eta_j = torch.asinh(pz_j / pt_j)

        eta_base = (eta / self.eta_max).view(1, 1, W).expand(N, H, W)
        phi_base = (phi / self.phi_max).view(1, H, 1).expand(N, H, W)
        grid = torch.stack(
            (
                eta_base + (eta_j / self.eta_max).view(N, 1, 1),
                phi_base + (phi_j / self.phi_max).view(N, 1, 1),
            ),
            dim=-1,
        )

        shifted = F.grid_sample(
            image,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False,
        )
        norm = shifted.flatten(1).sum(dim=1).view(N, 1, 1, 1).clamp_min(1e-8)
        return shifted / norm

    def encode(self, input: torch.Tensor) -> List[torch.Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: list of [mu, log_var]
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        mu      = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, self.encoder_out_channels,
                             self.spatial_size, self.spatial_size)
        result = self.decoder(result)
        result = self.final_layer(result)
        if self.output_mode == 'event_pt':
            result = F.softplus(result)
            norm = result.flatten(1).sum(dim=1).view(-1, 1, 1, 1).clamp_min(1e-8)
            result = result / norm
            result = self.recenter_event_pt_image(result)
        elif self.output_mode == 'tanh':
            result = torch.tanh(result)
        return result

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: torch.Tensor, **kwargs) -> List[torch.Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def current_beta(self) -> float:
        if self.loss_type != 'A' or self.beta_anneal_iter == 0:
            return self.beta
        progress = min(float(self.num_iter) / float(self.beta_anneal_iter), 1.0)
        if self.beta_anneal_schedule == 'log_linear':
            return self.beta_start * ((self.beta / self.beta_start) ** progress)
        return self.beta_start + (self.beta - self.beta_start) * progress

    def loss_function(self, *args, **kwargs) -> dict:
        if self.training:
            self.num_iter += 1

        recons   = args[0]
        input    = args[1]
        mu       = args[2]
        log_var  = args[3]
        kld_weight = kwargs['M_N']  # minibatch weight = current_batch_size / configured_batch_size

        if self.recon_loss_type == 'mse':
            recon_per_sample = F.mse_loss(recons, input, reduction='none').flatten(1).sum(1)
        elif self.recon_loss_type == 'l1':
            recon_per_sample = F.l1_loss(recons, input, reduction='none').flatten(1).sum(1)
        elif self.recon_loss_type == 'sqrt_mse':
            eps = 1e-8
            recon_per_sample = F.mse_loss(
                torch.sqrt(recons.clamp_min(0.0) + eps),
                torch.sqrt(input.clamp_min(0.0) + eps),
                reduction='none'
            ).flatten(1).sum(1)
        elif self.recon_loss_type == 'log_mse':
            x0 = self.recon_log_x0
            recon_per_sample = F.mse_loss(
                torch.log1p(recons.clamp_min(0.0) / x0),
                torch.log1p(input.clamp_min(0.0) / x0),
                reduction='none'
            ).flatten(1).sum(1)
        else:
            raise ValueError(f"Unsupported recon_loss_type={self.recon_loss_type!r}")
        recons_loss = recon_per_sample.mean()

        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1),
            dim=0
        )
        weighted_recons_loss = self.recon_loss_weight * recons_loss

        if self.loss_type == 'H':  # beta-VAE (Higgins et al.)
            loss = weighted_recons_loss + self.beta * kld_weight * kld_loss

        elif self.loss_type == 'A':  # beta-VAE with beta annealing
            beta = self.current_beta()
            loss = weighted_recons_loss + beta * kld_weight * kld_loss

        elif self.loss_type == 'B':  # beta-VAE with capacity annealing (Burgess et al.)
            self.C_max = self.C_max.to(input.device)
            C = torch.clamp(
                self.C_max / self.C_stop_iter * self.num_iter,
                0, self.C_max.data[0]
            )
            loss = weighted_recons_loss + self.gamma * kld_weight * (kld_loss - C).abs()
        else:
            raise ValueError(f'Undefined loss type: {self.loss_type}')

        return {
            'loss': loss,
            'Reconstruction_Loss': recons_loss,
            'Weighted_Reconstruction_Loss': weighted_recons_loss,
            'KLD': kld_loss,
        }

    def sample(self, num_samples: int, current_device: int, **kwargs) -> torch.Tensor:
        """
        Sample from the prior and decode to image space.
        :param num_samples: number of samples to generate
        :param current_device: device to run on
        :return: Tensor [num_samples x C x H x W]
        """
        z = torch.randn(num_samples, self.latent_dim).to(current_device)
        return self.decode(z)

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Reconstruct input through the VAE (encode then decode).
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self.forward(x)[0]
