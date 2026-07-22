# Results (synthetic)

device=cpu, window=64, epochs=20, diffusion T=100, infer_steps=10

| model | f1 | precision | recall | f1_pa | roc_auc | pr_auc | params | train_s | infer_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LSTM-VAE | 0.6633 | 0.9559 | 0.5078 | 0.9289 | 0.8790 | 0.6362 | 56554 | 3.9326 | 0.1343 |
| DDPM-vanilla | 0.4088 | 0.6981 | 0.2891 | 0.8706 | 0.8214 | 0.3789 | 80842 | 21.5803 | 8.5751 |
| DDPM-masking | 0.4608 | 0.6184 | 0.3672 | 0.8393 | 0.8146 | 0.4157 | 82122 | 21.4595 | 35.4764 |
| DDPM-selective | 0.5306 | 0.7647 | 0.4062 | 0.9289 | 0.8370 | 0.5062 | 80842 | 22.0295 | 4.0356 |

