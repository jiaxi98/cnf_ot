# VCNF inference benchmark

## Protocol

- Repository commit: `064a69bd308613d9c93d34b637a445601387acb4`
- Date: 2026-08-22
- Hardware: Apple M5 (10 CPU cores), 24 GB unified memory
- Software: Python 3.11.16, JAX/JAXlib 0.10.2, CPU backend, float64
- Flow: 2 autoregressive layers, two 16-unit conditioner layers, 5 RQS bins
- Network state: initialized parameters only; no training was performed
- Batch size: 10,000 samples at the fixed time `t=0.5`
- Timing: JIT compilation excluded, 5 warm-up calls, median of 30 synchronized calls

The benchmark command was:

```bash
env PYTHONPATH=. .venv/bin/python benchmarks/benchmark_inference.py \
  --dimensions 2 10 20 --batch-size 10000 --warmups 5 --repeats 30 \
  --output benchmark_results/inference_m5_cpu.json
```

## Experimental dimensions

These dimensions correspond to the dimensions used by the OT, RWPO, and FP
experiments in the paper.

| Dimension | Parameters | Sampling, 10k (ms) | Log density, 10k (ms) |
|---:|---:|---:|---:|
| 2 | 1,200 | 1.859 | 1.989 |
| 10 | 11,824 | 11.089 | 14.446 |
| 20 | 27,984 | 26.014 | 32.330 |

## Additional inference-only dimensions

These rows test only the initialized architecture. They are not trained 50D or
100D MFC experiments and do not provide evidence about solution accuracy.

| Dimension | Parameters | Sampling, 10k (ms) | Log density, 10k (ms) |
|---:|---:|---:|---:|
| 50 | 95,664 | 95.525 | 108.294 |
| 100 | 272,464 | 248.572 | 283.735 |

For `d >= 2`, the parameter count of the current implementation is
`16 d^2 + 1136 d - 1136`, so its parameter count is quadratic in dimension.
The timing results show increasing forward cost without a tensor-grid
exponential dependence.

## Limitations

These are local Apple M5 CPU measurements and must not be labeled as NVIDIA
A100 timings. The existing historical training times therefore need their
original hardware metadata, and an A100 run of the inference script is required
if training and inference timing are to appear in one hardware-controlled
table. A separate dry-step host-memory diagnostic is reported below.

## Dry-step training memory

Training memory was measured with `XLA_PYTHON_CLIENT_PREALLOCATE=false`. Each
configuration ran in a fresh process. The reported peak is the process RSS
during a complete JIT-compiled loss, reverse-mode differentiation, and Adam
update after one compilation/warm-up call. The output of the measured update
was discarded, so this does not train a model beyond the dry step.

For example:

```bash
env PYTHONPATH=. XLA_PYTHON_CLIENT_PREALLOCATE=false \
  .venv/bin/python benchmarks/benchmark_training_memory.py \
  --problem rwpo --dimension 10 \
  --output benchmark_results/memory_rwpo_d10.json
```

The boundary and interior batch sizes were 2,048 and 64, respectively. The 2D
rows use the paper's usual `N_t=20`, while the higher-dimensional rows use the
randomized `N_t=1` setting reported in the scalability experiment.

The OT dry-step uses the arbitrary-dimensional Gaussian endpoint construction
from `cnf_ot/mfc/ot.py`. The FP dry-step uses the paper's Ornstein-Uhlenbeck
drift `-a x`. These paths are implemented in the benchmark because the unified
`applications.py` currently contains a 2D-only OT mixture sampler and overwrites
the general FP drift with a 2D-only diagnostic drift.

| Problem | Dimension | `N_t` | Peak RSS (GiB) |
|---|---:|---:|---:|
| OT | 2 | 20 | 5.161 |
| OT | 20 | 1 | 4.906 |
| RWPO | 2 | 20 | 7.774 |
| RWPO | 10 | 1 | 6.552 |
| FP | 2 | 20 | 7.253 |
| FP | 10 | 1 | 6.293 |

For a controlled `N_t=1` comparison, the 2D peaks were 1.016 GiB for OT,
1.394 GiB for RWPO, and 1.317 GiB for FP. Thus the high-dimensional dry steps
do require more memory at fixed `N_t`, but randomized temporal quadrature
substantially reduces memory relative to unrolling 20 time samples in one JIT
graph. No tested configuration exhausted the 24 GB host memory.

These values are host-process RSS, not GPU device-memory measurements. Raw JSON
files also retain the process maximum including compilation and XLA's static
temporary-buffer estimate. The same script should be run on the paper's GPU
before describing these values as GPU peak training memory.

## Two-dimensional OU accuracy

The OU experiment uses the paper's two-layer, width-16, five-bin architecture
with `N_t=20`, `N_k=64`, `N_b=2048`, `lambda=500`, and 30,000 Adam steps. The
equation parameters are `a=1`, `gamma=0.5`, `T=1`, and the initial distribution
is `N(0, 4 I)`. Density, score, current velocity, and FP drift are evaluated
with 131,072 independent samples from the exact density at each reported time.
Both finite-difference steps are 0.01.

```bash
env PYTHONPATH=. XLA_PYTHON_CLIENT_PREALLOCATE=false \
  .venv/bin/python benchmarks/train_eval_ou.py \
  --epochs 30000 --time-batch-size 20 \
  --boundary-batch-size 2048 --interior-batch-size 64 \
  --eval-batch-size 131072 --lambda 500 --seed 42 \
  --output benchmark_results/ou_d2_nt20_lambda500_accuracy.json \
  --checkpoint benchmark_results/ou_d2_nt20_lambda500_params.pkl
```

| Time | Relative density L2 | Relative score RMSE | Relative current-velocity RMSE | Relative FP-drift RMSE |
|---:|---:|---:|---:|---:|
| 0.25 | 0.01103 | 0.02451 | 0.03958 | 0.03330 |
| 0.50 | 0.01586 | 0.02894 | 0.05239 | 0.04065 |
| 0.75 | 0.02082 | 0.03259 | 0.06303 | 0.04509 |
| 1.00 | 0.02482 | 0.03520 | 0.08114 | 0.04815 |

The density metric is the relative spatial L2 error. Each vector-field metric
is the square root of the exact-density-weighted mean squared error divided by
the corresponding weighted squared norm of the analytic field. A second
evaluation with 262,144 samples and an independent random seed changed the
reported errors by at most `4.9e-4`.
