"""Central configuration for the diffusion-TSAD study.

Everything that changes across experiments lives here so the scripts stay clean.
Values are intentionally small so the whole pipeline runs on a CPU in minutes;
bump epochs / d_model / T for the real report runs.
"""
from dataclasses import dataclass, field


@dataclass
class DataConfig:
    name: str = "synthetic"        # "synthetic" or "smd"
    smd_root: str = "../data/SMD"  # only used when name == "smd"
    smd_entity: str = "machine-1-1"
    window: int = 64               # sliding window length L
    stride: int = 8                # stride for training windows (test uses stride=1)
    n_features: int = 10           # only used by the synthetic generator
    seed: int = 0


@dataclass
class ModelConfig:
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    ff_dim: int = 128
    dropout: float = 0.1
    latent_dim: int = 16           # LSTM-VAE latent size


@dataclass
class DiffusionConfig:
    T: int = 100                   # training diffusion steps
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    infer_steps: int = 10          # DDIM steps used at scoring time (cheap)
    infer_t_frac: float = 0.5      # start denoising from this fraction of T
    mask_ratio: float = 0.2        # selective-denoising: fraction of noised elements
    n_impute_masks: int = 4        # masking mode: interleaved temporal masks


@dataclass
class TrainConfig:
    epochs: int = 8
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.0
    device: str = "cpu"            # auto-set to cuda if available in run script


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    diff: DiffusionConfig = field(default_factory=DiffusionConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
