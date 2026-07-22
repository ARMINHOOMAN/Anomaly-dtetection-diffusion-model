# Diffusion-Based Anomaly Detection for Multivariate Time Series

Course project for **ECE1508: Deep Generative Models** (Sun, Thaliath, Hooman).
This is the *simplified* version of the proposal that responds to the TA's review:

> *"Could you clarify which dataset you will use and how you will ensure the
> training data are mostly normal? ... focus on the core comparison first and
> keep the inter-variable-aware / multi-resolution extension optional. Please
> also report training and inference costs, since efficiency is one of your main
> research questions."*

Each of those points is addressed below and is baked into the code.

---

## 1. What we build (the core comparison)

The study isolates **noise design** as the single explanatory variable. One shared
Transformer denoiser is trained on **normal** windows and reused across three
diffusion regimes, plus one encoder–decoder baseline:

| Model | Family | Noise design / idea | Paper it follows |
|---|---|---|---|
| **LSTM-VAE** | encoder–decoder baseline | reconstruct window, KL-regularised latent | Park et al. 2018 |
| **DDPM-vanilla** | diffusion baseline | full Gaussian noising; partial-diffusion reconstruction | Ho et al. 2020 / AnoDDPM |
| **DDPM-masking** | conditional diffusion | observe part of the window, **impute** the rest | ImDiffusion / DiffAD / CSDI |
| **DDPM-selective** | selective denoising | mask the noise in training; **denoise the raw instance** at test time | Obata et al. 2026 (AnomalyFilter) |

Everything shares the same backbone, window size, normalisation, diffusion
schedule and DDIM sampler, so differences come from the noise design only.

**Kept optional (not in the core comparison), exactly as the TA suggested:**
BeatGAN adversarial baseline, inter-variable-aware selective denoising, and
multi-resolution decomposition. Hooks/notes for these are left in the code but
they are not needed to answer the three research questions.

## 2. Dataset & the normality assumption

We use the **Server Machine Dataset (SMD)** — 5 weeks of 38-dimensional server
telemetry from a large internet company (OmniAnomaly release). It is a standard
multivariate TSAD benchmark, it is lightweight (plain text, a few MB/entity, no
images), and — crucially — it answers *"how do you ensure training is mostly
normal?"*:

- SMD ships a **dedicated train/test split**. The **train** split is a curated
  **normal-operation** period (no labelled anomalies), and the **test** split
  carries **point-level anomaly labels**. So the normality assumption holds *by
  construction of the benchmark* — we train only on the normal split and never
  touch test labels during training.
- Normalisation statistics (z-score) are fit on the **train** split only, so no
  test information leaks in.

A **synthetic generator** (`data.py`) is also included so the whole pipeline runs
with zero downloads. Its train split is *guaranteed* anomaly-free, which
demonstrates the normality assumption in the cleanest possible way; the test
split contains injected spikes, level shifts and frequency bursts with labels.

To fetch one real SMD entity:
```bash
BASE=https://raw.githubusercontent.com/NetManAIOps/OmniAnomaly/master/ServerMachineDataset
for s in train test test_label; do
  curl -sSL --create-dirs -o ../data/SMD/$s/machine-1-1.txt $BASE/$s/machine-1-1.txt
done
```

## 3. How anomalies are scored

For every model we get a per-timestep score by folding overlapping windows back
onto the series (averaging windows that cover each timestep):

- **VAE / vanilla / masking / selective** → mean-squared **reconstruction error**
  between the input window and its (denoised / imputed) reconstruction.
- **Detection quality**: best **F1** (threshold picked on the score's own
  quantiles), plus **point-adjusted F1** (`f1_pa`, the common but score-inflating
  TSAD convention — reported *alongside*, not instead of, the raw F1), and the
  threshold-free **ROC-AUC** and **PR-AUC** (PR-AUC matters under the heavy class
  imbalance typical of TSAD).
- **Cost** (the TA's efficiency question): trainable **parameter count**,
  **training wall-clock**, and **inference wall-clock**, all logged automatically.

## 4. Running it

```bash
pip install -r requirements.txt

python run_experiments.py                    # synthetic, full smoke run
python run_experiments.py --quick            # tiny + fast sanity check
python run_experiments.py --dataset smd --entity machine-1-1 --epochs 10 --test-stride 5
```
Outputs land in `../results/`: a `results_<tag>.csv`, a `results_<tag>.md`, and a
score-vs-ground-truth plot `scores_<tag>.png`.

## 5. Results

Short CPU smoke runs (small backbone, 10 DDIM steps). Numbers are meant to show
the framework and the **relative ordering**, not final tuned scores.

### Synthetic (10 features, 20 epochs)
| model | F1 | precision | recall | F1 (PA) | ROC-AUC | PR-AUC | params | train_s | infer_s |
|---|---|---|---|---|---|---|---|---|---|
| LSTM-VAE | 0.663 | 0.956 | 0.508 | 0.929 | 0.879 | 0.636 | 56.5k | 3.9 | 0.13 |
| DDPM-vanilla | 0.409 | 0.698 | 0.289 | 0.871 | 0.821 | 0.379 | 80.8k | 21.6 | 8.6 |
| DDPM-masking | 0.461 | 0.618 | 0.367 | 0.839 | 0.815 | 0.416 | 82.1k | 21.5 | 35.5 |
| DDPM-selective | 0.531 | 0.765 | 0.406 | 0.929 | 0.837 | 0.506 | 80.8k | 22.0 | 4.0 |

*On smooth periodic synthetic data a simple LSTM-VAE reconstructs normal patterns
very well and is hard to beat; among the diffusion regimes the ordering is
**selective > masking > vanilla**, matching the proposal's hypothesis.*

### SMD `machine-1-1` (38 features, 10 epochs, test-stride 5)
| model | F1 | precision | recall | F1 (PA) | ROC-AUC | PR-AUC | params | train_s | infer_s |
|---|---|---|---|---|---|---|---|---|---|
| LSTM-VAE | 0.477 | 0.504 | 0.454 | 0.998 | 0.879 | 0.515 | 65.5k | 12.3 | 0.31 |
| DDPM-vanilla | **0.736** | 0.683 | 0.797 | 0.998 | **0.975** | **0.783** | 84.5k | 53.3 | 16.9 |
| DDPM-masking | 0.709 | 0.681 | 0.740 | 0.997 | 0.956 | 0.700 | 89.3k | 52.8 | 81.9 |
| DDPM-selective | 0.694 | 0.599 | 0.826 | 0.998 | 0.967 | 0.722 | 84.5k | 54.3 | 7.7 |

*On the real benchmark **all three diffusion regimes clearly beat the LSTM-VAE**
(F1 ≈ 0.69–0.74 / ROC-AUC ≈ 0.96–0.97 / PR-AUC ≈ 0.70–0.78 vs the VAE's
0.48 / 0.88 / 0.52) — the proposal's central claim. **Cost side:** diffusion
trains ~4× slower and infers 25–260× slower than the VAE; among the diffusion
regimes **selective is the cheapest at inference (7.7s) while masking is the most
expensive (82s, four imputation passes).** This is exactly the efficiency
trade-off the third research question asks about.*

### How the results map to the three research questions
1. **Can a diffusion model trained on normal data find anomalies?** Yes — ROC-AUC
   0.96–0.97 on SMD from training on the normal split only.
2. **How does the denoising strategy shape the normal-vs-anomaly gap?** It matters
   and is not one-sided: vanilla leads raw F1 on SMD, selective gives the best
   recall at the lowest inference cost, masking is strong but the most expensive;
   on smooth synthetic data the ordering flips to selective > masking > vanilla.
3. **Does the gain justify the cost?** The cost columns quantify it: 4× training
   and up to ~260× inference overhead versus the VAE — worth it on SMD, not on the
   easy synthetic set.

## 6. Repo layout
```
code/
  config.py          # all hyper-parameters
  data.py            # synthetic generator + SMD loader + windowing
  backbone.py        # Transformer denoiser (shared)
  diffusion.py       # DDPM + 3 noise designs + DDIM scoring
  baselines.py       # LSTM-VAE (BeatGAN = optional extension)
  utils.py           # windowing + metrics (P/R/F1, PA-F1, ROC/PR-AUC)
  run_experiments.py # trains everything, writes the results + cost table
```
