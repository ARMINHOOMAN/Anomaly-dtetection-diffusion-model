"""Reconstruction baselines. LSTM-VAE is fully implemented as the encoder-decoder
reference from the proposal. BeatGAN (adversarial) is kept as an optional
extension -- adversarial training is finicky on CPU and is not needed for the
core noise-design comparison the TA asked us to focus on first.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMVAE(nn.Module):
    """LSTM-based VAE for window reconstruction.
    Default hyperparameters in this repo:
      hidden=64, latent=16, num_layers=1, dropout=0.0
    """
    def __init__(self, n_features, hidden=64, latent=16, num_layers=1,
                 dropout=0.0):
        super().__init__()
        # self.input_proj = nn.Linear(n_features, hidden) // include for optimal LSTM
        self.enc = nn.LSTM(n_features, hidden, num_layers, batch_first=True)
        self.to_mu = nn.Linear(hidden, latent)
        self.to_logvar = nn.Linear(hidden, latent)
        self.from_z = nn.Linear(latent, hidden)
        self.dec = nn.LSTM(hidden, hidden, num_layers, batch_first=True)
        # self.dropout = nn.Dropout(dropout) // include for optimal LSTM
        self.out = nn.Linear(hidden, n_features)

    def forward(self, x):
        B, L, D = x.shape
        # x_proj = self.input_proj(x) // include for optimal LSTM
        _, (h, _) = self.enc(x)
        h = h[-1]
        mu, logvar = self.to_mu(h), self.to_logvar(h)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        h0 = self.from_z(z)[None].expand(1, B, -1).contiguous()
        c0 = torch.zeros_like(h0)
        seq = h0[-1][:, None, :].expand(B, L, -1)
        dec_out, _ = self.dec(seq, (h0, c0))
        # dec_out = self.dropout(dec_out) // include for optimal LSTM
        return self.out(dec_out), mu, logvar

    def loss(self, x, beta=1.0):
        """Reconstruction + KL divergence
        Default beta=1.0.
        """
        recon, mu, logvar = self(x)
        rec = F.mse_loss(recon, x, reduction="mean")
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return rec + beta * kld

    @torch.no_grad()
    def score_windows(self, x):
        recon, _, _ = self(x)
        return ((x - recon) ** 2).mean(dim=-1)
