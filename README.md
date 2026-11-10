# SLED — the frozen batch of second-order PINN training

Code for the paper on how to build the collocation batch of the second-order
(quasi-Newton) phase of physics-informed neural network (PINN) training: the
training library (`pipeline/`), six benchmarks, the paper's studies and the
phase-1 checkpoints the reported runs started from.

This repository is prepared for double-blind review. Please do not try to infer
author identities from it; report identifying material through the review
process.

## Protocol

1. **Phase 1.** SOAP on 8,192 uniform residual points (Rayleigh-Taylor: LAS),
   GradNorm loss weights, 2,000 steps (Ginzburg-Landau, Rayleigh-Taylor:
   20,000). Trained once per (benchmark, seed) and saved as a hand-over
   checkpoint.
2. **Phase 2.** 40,000 steps of the self-scaled Broyden method SSBroyden2
   (`pipeline/broyden.py`, `broyden`) on **one batch of n points**, built at the
   hand-over by a sampling rule, with the loss weights frozen. The batch stays
   fixed unless the rule redraws it.

All rules of a seed start from the same checkpoint, so they differ only in the
batch.

| study | script | rules | n | seeds |
|---|---|---|---|---|
| batch size | `batch_size_study.py` | `uniform` `sobol` `rad` `lev` `sled` | 2^11 … 2^15 (KS to 2^16, RT to 2^17) | 2, 3, 5, 7, 11, 13 |
| sampler comparison | `sampler_comparison.py` | the ten rules of the table below | 2^15 | 2, 3, 5, 7, 11, 13 |
| ablation a1 | `a1.py` | `sled_fl0` `sled_fl0.25` `sled_fl0.5` `sled_fl0.75` | 2^14 | 2, 3, 5, 7, 11, 13 |
| supplementary | `supplementary.py` | `pinnacle_s` `rad_r500_reset` `vrba_r100_reset` (KS: + `pinnacle_k_r500`); `--large`: `uniform` `sobol` `rad_r500` `sled` | 2^14; `--large` 2^15, 2^16 | 2, 3, 5, 7, 11 |

The scored rules draw from a pool of M uniform points: M = 65,536 for
n ≤ 16,384 (the same pool for every n), M = 4n above.

## Installation

Use a fresh environment (Python ≥ 3.10); SLED ships a patched copy of SOAP. From
the repository root:

```bash
python -m venv sled-env && source sled-env/bin/activate
python -m pip install "jax[cuda13]"          # or: python -m pip install jax   (CPU)
python -m pip install -r requirements.txt    # dependencies + third_party/soap_jax
python -m pip install -e .
```

Install JAX first, then `requirements.txt`, then the package (`setup.py`
requires the vendored `soap-jax==0.1.0+sled`). Tested with Python 3.13, JAX
0.10.2, Flax 0.12.7 and Optax 0.2.8.

## Quick start

```bash
python checkpoints/verify_checkpoints.py      # check the shared hand-overs
cd benchmarks/burgers
python sampler_comparison.py --smoke --seeds 0 --rules sled sobol          # ~2 min test
python sampler_comparison.py --shared_phase1 --seeds 2 --rules sled sobol  # real runs
```

## Running the studies

Run the scripts from their benchmark directory (`benchmarks/<benchmark>/`, one
of `allen_cahn burgers ginzburg_landau kdv ks rayleigh_taylor`). Without options a
script runs its full grid (`configs/<study>.py`).

```bash
cd benchmarks/ks
python batch_size_study.py --seeds 2 3 --sizes 2048 32768 --rules sled sobol
python sampler_comparison.py --seeds 2 --rules sled lev kd
python a1.py --seeds 2 --rules sled_fl0 sled_fl0.5
python supplementary.py --shared_phase1               # the protocols
python supplementary.py --shared_phase1 --large       # the larger batches
```

Options: `--seeds`, `--rules`, `--sizes` (batch size, supplementary), `--n`
(sampler comparison, a1), `--shared_phase1` (start from `checkpoints/`), `--phase1_dir DIR` (read or
train hand-overs in `DIR/seed<s>/`), `--smoke` (30 + 40 steps, tests the
pipeline only), `--force` (rerun finished runs).

**Outputs.** Phase 1 goes to `runs/phase1/seed<s>/`, each run to
`runs/<rule>/n<n>/seed<s>/`: `config.yaml`, `metrics.jsonl` (every 50 steps:
losses on the batch and on 65,536 held-out points, their ratio `gap`, the
relative L2 error), `params_end.pkl` and `summary.json` (status, hand-over /
final / best error, batch statistics, hashes). `--shared_phase1` writes to
`runs_shared/`, `--smoke` to `runs_smoke/`. Finished runs are skipped, so an
interrupted study resumes; a directory holding a different run is refused,
never overwritten. Runs that two studies share (e.g. the five batch-size rules
at 2^15) are trained once.

**Environment.** The scripts set `JAX_ENABLE_X64=True`,
`XLA_FLAGS=--xla_gpu_deterministic_ops=true` and the highest matmul
precision: reruns on the same hardware and software are bitwise identical,
across GPUs or versions they agree statistically. W&B runs offline unless
`WANDB_MODE` is set; use `WANDB_MODE=disabled` during review.

## Sampling rules

`pipeline/domain_importance_adapt.py` (`QN_RULES`). ω = 1/(Mπ) is the
importance weight of a draw from the proposal π.

| rule | batch of n points | point weights | redraw |
|---|---|---|---|
| `uniform` | i.i.d. uniform | 1 | — |
| `sobol` | scrambled Sobol | 1 | — |
| `rad` | RAD, π ∝ \|r\| / mean\|r\| + 1 | 1 | — |
| `kd` | π ∝ kernel diagonal ‖∇θ r(x)‖² (Hutchinson), floor 0.2 | ω | — |
| `lev` | π ∝ Nyström ridge leverage, floor 0.2 | ω | — |
| `sled` | **SLED** (below) | ω; 1 on the Sobol points | — |
| `uniform_r500`, `rad_r500` | as `uniform`, `rad` | 1 | every 500 steps |
| `vrba_r100` | uniform, vRBA attention weights | λ | every 100 steps |
| `pinnacle_k` | PINNACLE-K (Lau et al., ICLR 2024): greedy K-means++ on the eNTK embedding | 1 | — |
| `pinnacle_s` | PINNACLE-S: drawn without replacement ∝ PINNACLE's convergence degree | 1 | — |
| `pinnacle_k_r500` | `pinnacle_k`, reselected from a fresh pool | 1 | every 500 steps |
| `rad_r500_reset`, `vrba_r100_reset` | as `rad_r500`, `vrba_r100`, quasi-Newton memory reset at every redraw | 1, λ | every 500 / 100 steps |
| `sled_fl<φ>` | `sled` with the floor mass fixed at φ ∈ {0, 0.25, 0.5, 0.75} | as `sled` | — |

**SLED** scores the pool with the stacked residual of all equations (scaled by
the frozen loss weights): it estimates the ridge leverage and the kernel
diagonal and takes their Spearman rank correlation ρ ∈ [0, 1] as the floor mass.
The batch is round(ρn) scrambled-Sobol points (weight 1) plus n − round(ρn)
draws from the leverage proposal (weight ω). Nothing is tuned per benchmark;
every run records ρ (`rho_floor`).

The other redraw rules keep the optimizer's curvature memory across a redraw;
the `_reset` rules re-initialise it. A run with seed s draws its pool and batch
from seeds derived from s; a redraw at step i uses s + 100,003 · (i // p).

## Optimizer

`Soap_SSBroyden2` (`configs/base.py`) selects SOAP for phase 1 and
`broyden.broyden` for phase 2: SSBroyden2 with scale-free safeguards. A failed
line search restarts the curvature memory instead of storing a NaN inverse
Hessian, and the phase stops only after three steps without decrease. In the
healthy regime its steps agree with the plain update to round-off.
`Soap_SSBroyden2_original` selects the plain update (`broyden.broyden_original`).

## Benchmarks

| benchmark | PDE | reference (`data/`) | window | phase 1 |
|---|---|---|---|---|
| `burgers` | viscous Burgers, ν = 0.01/π | `burgers_refined.mat`, 1.9e-11 | t ∈ [0, 1] | 2,000 |
| `allen_cahn` | Allen-Cahn, u0 = x² sin(2πx) | `allen_cahn_sin.mat`, 5.9e-11 | t ∈ [0, 1] | 2,000 |
| `kdv` | Korteweg-de Vries | `kdv.mat` (jaxpi), ≈ 5e-12 | t ∈ [0, 1] | 2,000 |
| `ks` | Kuramoto-Sivashinsky | `ks_chaotic.mat` (jaxpi), ≈ 3e-9 | first 35 % of t | 2,000 |
| `ginzburg_landau` | complex Ginzburg-Landau, 2-D | `ginzburg_landau_square.mat` (jaxpi), ≈ 3e-8 | first 80 % of t | 20,000 |
| `rayleigh_taylor` | Boussinesq Rayleigh-Taylor, Ra = 10⁶ | `rayleigh_taylor_spectral.npy`, 6.3e-9 | t ∈ [0, 4.75] | 20,000 (LAS) |

The reported error is the relative L2 error of u (Rayleigh-Taylor: the mean
over u, v, T). `REFERENCE_SOLUTIONS.md` documents how each reference was made
and checked; `python verify_reference_solutions.py --fast` re-derives them on a
CPU (~7 min).

## Shared phase-1 checkpoints

`checkpoints/<benchmark>/seed<k>/phase1.pkl` are the hand-overs the paper's runs
started from: six benchmarks × seeds 2, 3, 5, 7, 11, 13, 17, 19, 23, 27 (60 files,
10 MB), so every study seed has one. Each holds the parameters, the frozen loss
weights, the seed, the first-order budget and the parameter hash, which is
re-checked on every restore. `phase1.json` next to it records the hand-over error,
the loss weights and the GPU model; `MANIFEST.json` lists all of them.

```bash
python checkpoints/verify_checkpoints.py            # SHA-256 and parameter hashes
python checkpoints/verify_checkpoints.py --errors   # + re-evaluate every hand-over error (CPU, minutes)
```

With `--shared_phase1` a study starts phase 2 from these files (read-only)
instead of training phase 1; phase 2 then agrees with the reported runs
statistically, not bit for bit (other GPUs and versions).

## Results and readings

```bash
cd benchmarks
python collect_results.py                                   # benchmarks/*/runs -> results/
python collect_results.py --runs_dir_name runs_shared --out_dir results/shared
python handover_readings.py --phase1_dir "../checkpoints/{benchmark}" --seeds 2 3
```

`collect_results.py` writes `runs.csv` (one row per run) and median / min / max
tables: `batch_size.csv`, `sampler_comparison.csv`, `a1.csv`,
`supplementary.csv`. `handover_readings.py` measures each hand-over's
leverage / kernel-diagonal statistics without the reference, here on the shared
checkpoints (default: `<benchmark>/runs/phase1`; output in
`benchmarks/runs/readings/`).

## Layout

```text
pipeline/          training loops, samplers and rules, broyden, models, evaluator
benchmarks/<b>/    models.py, utils.py, data/, configs/, and the four study scripts
benchmarks/        collect_results.py, handover_readings.py
checkpoints/       shared phase-1 hand-overs, MANIFEST.json, verify_checkpoints.py
third_party/       soap_jax (MIT, patched; see its README)
```

## Compute

On one H100, a 40,000-step run takes ~30–55 min with a ~10–12 GiB peak, set by
SSBroyden's three P × P float64 matrices. Rayleigh-Taylor grows about linearly
above n = 16,384: ~2–2.6 h at 2^16, ~6.3 h and ~36 GiB at 2^17, which needs an
80 GB GPU. The batch-size study is 990 runs; the sampler comparison adds 180,
a1 144.

## License

Apache License 2.0 (`LICENSE`, `NOTICE`). `third_party/soap_jax` is MIT. The
reference data of KdV, KS and Ginzburg-Landau come from jaxpi under their own
terms (`NOTICE`). The copyright holder is listed as "Anonymous Authors" for the
review.
