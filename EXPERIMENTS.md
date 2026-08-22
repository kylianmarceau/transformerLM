# Experiment Log

All runs used an NVIDIA L4 GPU, seed 3043, batch size 32, context length 256, four Transformer layers,  d_model=512 , eight attention heads,  d_ff=1344 , AdamW, weight decay 0.1 on matrix parameters, 200 warmup steps, cosine decay, gradient clipping at 1.0, and BF16 autocast. Unless stated otherwise, the configuration used a 4,000-token vocabulary, RMSNorm, RoPE, and SwiGLU.

The final validation loss is the last validation value recorded in each run's evaluation JSONL log. Wall clock time is the last value in the corresponding per step log, and GPU hours are wall clock seconds divided by 3,600. For the failed no RMSNorm run, the last validation was recorded at step 200 and failure occurred after step 203.

| Run name | Configuration | Steps | Final validation loss | Wall-clock time | GPU-hours | Finding |
|----------|---------------|-------|-----------------------|-----------------|-----------|---------|
| lr_1e_4  | Standard; peak LR  1e-4  | 1,000 | 3.0329 | 103.22 s | 0.0287 | Stable but learned too slowly under the shortened sweep budget. |
|  lr_3e_4  | Standard; peak LR  3e-4  | 1,000 | 2.4515 | 103.39 s | 0.0287 | Improved on  1e-4 , but remained well behind the best stable rates. |
|  lr_1e_3  | Standard; peak LR  1e-3  | 1,000 | 2.0148 | 103.50 s | 0.0287 | Trained stably and approached the best region of the sweep. |
|  lr_3e_3  | Standard; peak LR  3e-3  | 1,000 | 1.9112 | 103.36 s | 0.0287 | Best final validation loss in the learning-rate sweep. |
|  lr_1e_2  | Standard; peak LR  1e-2  | 1,000 | 1.9316 | 103.51 s | 0.0288 | Second-best sweep result, only 0.0204 nats behind  3e-3 . |
|  lr_3e_2  | Standard; peak LR  3e-2  | 1,000 | 2.0100 | 103.38 s | 0.0287 | Stable, but performance worsened beyond the best learning-rate region. |
|  lr_1e_1  | Standard; peak LR  1e-1  | 1,000 | 2.1732 | 103.39 s | 0.0287 | The high learning rate degraded optimisation and validation loss. |
|  lr_3e_1  | Standard; peak LR  3e-1  | 1,000 | 2.6490 | 103.50 s | 0.0287 | Strong instability made this rate clearly unsuitable. |
|  lr_1e0  | Standard; peak LR  1.0  | 1,000 | 4.9104 | 102.95 s | 0.0286 | Diverged during the sweep and was retained as the required failed learning-rate run. |
|  baseline_5000  | Standard; peak LR  3e-3  | 5,000 | 1.5357 | 520.03 s | 0.1445 | Confirmed  3e-3  as a strong and stable baseline over the standard budget |
|  no_rmsnorm_5000  | No RMSNorm; peak LR  3e-3  | 203 | 1.1091e16 | 15.93 s | 0.0044 | Failed with non-finite gradients, showing that RMSNorm was essential for stability at this rate. |
|  no_rmsnorm_lr_1e_3_5000  | No RMSNorm; peak LR  1e-3  | 5,000 | 1.6108 | 370.95 s | 0.1030 | Retuning stabilised training, but finished 0.0752 nats behind the baseline. |
|  nope_5000  | No RoPE; peak LR  3e-3  | 5,000 | 1.6510 | 487.15 s | 0.1353 | Removing positional information produced the largest stable ablation penalty. |
|  vocab_1000_5000  | 1,000-token vocabulary; peak LR  3e-3  | 5,000 | 1.3460 | 617.90 s | 0.1716 | Lower token loss was not directly comparable; BPC showed the 4,000-token model was better. |
|  final_vocab4000_20000  | Standard; peak LR  3e-3 ; extended run | 20,000 | 1.3623 | 2,098.52 s | 0.5829 | Extended training continued to improve validation performance and produced the final model. |