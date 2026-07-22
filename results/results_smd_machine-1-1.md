# Results (smd_machine-1-1)

device=cpu, window=64, epochs=10, diffusion T=100, infer_steps=10

| model | f1 | precision | recall | f1_pa | roc_auc | pr_auc | params | train_s | infer_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LSTM-VAE | 0.4773 | 0.5035 | 0.4536 | 0.9983 | 0.8787 | 0.5145 | 65542 | 12.2586 | 0.3131 |
| DDPM-vanilla | 0.7356 | 0.6832 | 0.7966 | 0.9976 | 0.9746 | 0.7829 | 84454 | 53.3201 | 16.8910 |
| DDPM-masking | 0.7091 | 0.6809 | 0.7398 | 0.9970 | 0.9559 | 0.7004 | 89318 | 52.7827 | 81.9034 |
| DDPM-selective | 0.6943 | 0.5991 | 0.8255 | 0.9983 | 0.9672 | 0.7224 | 84454 | 54.3169 | 7.7194 |

