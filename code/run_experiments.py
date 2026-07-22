"""Run the core comparison and produce a results + cost table.

Models compared:
    LSTM-VAE            (encoder-decoder reconstruction baseline)
    DDPM-vanilla        (plain diffusion, partial-diffusion scoring)
    DDPM-masking        (conditional imputation)
    DDPM-selective      (selective denoising)

For every model we log detection quality (P/R/F1, point-adjusted F1, ROC-AUC,
PR-AUC) and cost (parameter count, training wall-clock, inference wall-clock) --
efficiency is one of the study's research questions.

Usage:
    python run_experiments.py                      # synthetic, full smoke
    python run_experiments.py --quick              # tiny + fast
    python run_experiments.py --dataset smd --entity machine-1-1 --epochs 20
"""
import argparse
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import Config
from data import get_data, WindowDataset
from utils import set_seed, make_windows, windows_to_series, evaluate_scores, count_params
from diffusion import build_diffusion
from baselines import LSTMVAE

SCORE_BATCH = 256


def to_markdown_table(df):
    """Small dependency-free markdown table writer (avoids needing tabulate)."""
    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [head, sep]
    for _, row in df.iterrows():
        cells = [f"{row[c]:.4f}" if isinstance(row[c], float) else str(row[c])
                 for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def train_loop(step_fn, params, loader, epochs, lr, wd, device):
    opt = torch.optim.Adam(params, lr=lr, weight_decay=wd)
    t0 = time.time()
    for ep in range(epochs):
        running = 0.0
        for xb in loader:
            xb = xb.to(device)
            loss = step_fn(xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
        print(f"    epoch {ep + 1:2d}/{epochs}  loss={running / len(loader):.4f}")
    return time.time() - t0


def score_series(scorer, series, L, device, stride=1):
    windows, starts = make_windows(series, L, stride)
    ds = torch.from_numpy(windows)
    out = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(ds), SCORE_BATCH):
            xb = ds[i:i + SCORE_BATCH].to(device)
            out.append(scorer(xb).cpu().numpy())
    infer_time = time.time() - t0
    win_scores = np.concatenate(out, axis=0)
    series_scores = windows_to_series(win_scores, starts, len(series), L)
    return series_scores, infer_time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="synthetic", choices=["synthetic", "smd"])
    ap.add_argument("--entity", default="machine-1-1")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--window", type=int, default=None)
    ap.add_argument("--infer-steps", type=int, default=None)
    ap.add_argument("--test-stride", type=int, default=1)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="../results")
    args = ap.parse_args()

    cfg = Config()
    cfg.data.name = args.dataset
    cfg.data.smd_entity = args.entity
    if args.window:
        cfg.data.window = args.window
    if args.infer_steps:
        cfg.diff.infer_steps = args.infer_steps
    if args.epochs:
        cfg.train.epochs = args.epochs
    if args.quick:
        cfg.train.epochs = 2
        cfg.data.n_features = 6
        args.test_stride = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.train.device = device
    set_seed(cfg.data.seed)

    print(f"device={device}  dataset={cfg.data.name}  window={cfg.data.window}  "
          f"epochs={cfg.train.epochs}")

    train_series, test_series, labels = get_data(cfg)
    D = train_series.shape[1]
    L = cfg.data.window
    print(f"train {train_series.shape}  test {test_series.shape}  "
          f"anomaly rate {labels.mean():.3f}")

    train_ds = WindowDataset(train_series, L, cfg.data.stride)
    loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True)

    rows = []
    score_curves = {}

    # ---- LSTM-VAE baseline -------------------------------------------------
    print("\n[LSTM-VAE]")
    vae = LSTMVAE(D, hidden=cfg.model.d_model, latent=cfg.model.latent_dim).to(device)
    tt = train_loop(lambda xb: vae.loss(xb), vae.parameters(), loader,
                    cfg.train.epochs, cfg.train.lr, cfg.train.weight_decay, device)
    vae.eval()
    scores, it = score_series(vae.score_windows, test_series, L, device, args.test_stride)
    m = evaluate_scores(scores, labels)
    rows.append(dict(model="LSTM-VAE", params=count_params(vae),
                     train_s=tt, infer_s=it, **m))
    score_curves["LSTM-VAE"] = scores

    # ---- diffusion regimes -------------------------------------------------
    for mode in ["vanilla", "masking", "selective"]:
        name = f"DDPM-{mode}"
        print(f"\n[{name}]")
        diff = build_diffusion(mode, D, cfg).to(device)
        tt = train_loop(lambda xb: diff.p_losses(xb), diff.parameters(), loader,
                        cfg.train.epochs, cfg.train.lr, cfg.train.weight_decay, device)
        diff.eval()
        scores, it = score_series(diff.score_windows, test_series, L, device,
                                  args.test_stride)
        m = evaluate_scores(scores, labels)
        rows.append(dict(model=name, params=count_params(diff),
                         train_s=tt, infer_s=it, **m))
        score_curves[name] = scores

    # ---- report ------------------------------------------------------------
    df = pd.DataFrame(rows)
    cols = ["model", "f1", "precision", "recall", "f1_pa", "roc_auc", "pr_auc",
            "params", "train_s", "infer_s"]
    df = df[cols]
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print("\n===== RESULTS =====")
    print(df.to_string(index=False))

    import os
    os.makedirs(args.out, exist_ok=True)
    tag = cfg.data.name if cfg.data.name == "synthetic" else f"smd_{args.entity}"
    df.to_csv(os.path.join(args.out, f"results_{tag}.csv"), index=False)
    with open(os.path.join(args.out, f"results_{tag}.md"), "w") as f:
        f.write(f"# Results ({tag})\n\n")
        f.write(f"device={device}, window={L}, epochs={cfg.train.epochs}, "
                f"diffusion T={cfg.diff.T}, infer_steps={cfg.diff.infer_steps}\n\n")
        f.write(to_markdown_table(df.round(4)))
        f.write("\n")
    print(f"\nsaved -> {args.out}/results_{tag}.csv / .md")

    # optional score plot for the best model by PR-AUC
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        best = df.sort_values("pr_auc", ascending=False).iloc[0]["model"]
        fig, ax = plt.subplots(figsize=(11, 3))
        ax.plot(score_curves[best], lw=0.8, label=f"{best} score")
        anom = np.where(labels > 0.5)[0]
        ax.scatter(anom, np.zeros_like(anom), c="r", s=4, label="true anomaly")
        ax.set_title(f"Anomaly score vs ground truth ({best})")
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f"scores_{tag}.png"), dpi=120)
        print(f"saved -> {args.out}/scores_{tag}.png")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
