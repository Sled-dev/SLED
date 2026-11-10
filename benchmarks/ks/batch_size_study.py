"""Kuramoto-Sivashinsky: batch-size study (configs/batch_size_study.py).

    python batch_size_study.py                                  # the full grid
    python batch_size_study.py --seeds 2 3 --sizes 2048 32768 --rules sled sobol
    python batch_size_study.py --phase1_dir /path/to/phase1     # reuse hand-overs
    python batch_size_study.py --shared_phase1                  # the shared hand-overs
    python batch_size_study.py --smoke --seeds 0 --sizes 2048 --rules sled

Seeds are the outer loop. Phase 1 is trained once per seed with configs/base.py
into runs/phase1/seed<k>/phase1.pkl (shared with sampler_comparison.py; skipped
if present) and every run restores it. Each run writes runs/<rule>/n<n>/seed<k>/
(config.yaml, metrics.jsonl with the off-batch loss, params_end.pkl,
summary.json); finished runs are skipped, so the script resumes where it
stopped, and the n = 2^14 runs are shared with sampler_comparison.py. Run from
this directory.
"""
import os

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
# deterministic XLA GPU kernels: without them the same run is not reproducible
# bit for bit (reductions differ in the last bits and the quasi-Newton phase
# amplifies that); added to any XLA_FLAGS already set
_flags = os.environ.get("XLA_FLAGS", "")
if "--xla_gpu_deterministic_ops" not in _flags:
    os.environ["XLA_FLAGS"] = (_flags + " --xla_gpu_deterministic_ops=true").strip()
os.environ["JAX_ENABLE_X64"] = "True"          # float64 for the second-order phase
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("WANDB_MODE", "offline")  # export WANDB_MODE=online to sync

import argparse

import jax
jax.config.update("jax_default_matmul_precision", "highest")
# the PRNG every pool, batch and initialisation draws from (the default since
# JAX 0.5; older JAX would draw different points from the same seeds)
jax.config.update("jax_threefry_partitionable", True)

import pipeline.train as train
from pipeline.domain_importance_adapt import QN_RULES

import models
from utils import get_dataset
from configs import base, batch_size_study as study

BENCHMARK = "ks"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="default: the config's seeds (with --shared_phase1, those "
                         "that have a shared hand-over)")
    ap.add_argument("--sizes", type=int, nargs="+", default=study.SIZES)
    ap.add_argument("--rules", nargs="+", default=study.RULES, choices=list(QN_RULES))
    ap.add_argument("--phase1_dir", default=None,
                    help="directory with seed<k>/phase1.pkl (default runs/phase1)")
    ap.add_argument("--shared_phase1", action="store_true",
                    help="start from the shared hand-overs in "
                         "checkpoints/<benchmark>/ (runs go to runs_shared/)")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny budgets into runs_smoke/, for testing the pipeline")
    ap.add_argument("--force", action="store_true", help="rerun finished runs")
    args = ap.parse_args()
    seeds = args.seeds or (train.shared_seeds(study.SEEDS) if args.shared_phase1
                           else study.SEEDS)
    runs = [dict(seed=seed, rule=rule, n=n, name=f"{rule}/n{n}",
                 config=study.get_config(rule, n, seed))
            for seed in seeds for n in args.sizes for rule in args.rules]
    train.run_study("batch_size", BENCHMARK, runs, base, get_dataset,
                    models.KS, models.KSEvaluator,
                    phase1_dir=args.phase1_dir, smoke=args.smoke, force=args.force,
                    shared_phase1=args.shared_phase1)


if __name__ == "__main__":
    main()
