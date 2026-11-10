"""Burgers: supplementary study (configs/supplementary.py).

    python supplementary.py                                # the protocols at n = 2^14, all seeds
    python supplementary.py --large                        # uniform / sobol / rad_r500 / sled at 2^15, 2^16
    python supplementary.py --rules pinnacle_k rad_r500 vrba_r100 sled   # the comparators at 2^14
    python supplementary.py --seeds 2 --rules pinnacle_s pinnacle_k_r500
    python supplementary.py --shared_phase1                # the shared hand-overs
    python supplementary.py --smoke --seeds 0 --rules pinnacle_s rad_r500_reset

Seeds are the outer loop. Phase 1 is trained once per seed with configs/base.py
into runs/phase1/seed<k>/phase1.pkl (shared with the other studies; skipped if
present) and every run restores it; --shared_phase1 restores the shared
hand-over in checkpoints/burgers/seed<k>/ instead and writes under runs_shared/.
Each run writes <root>/<rule>/n<n>/seed<k>/ (config.yaml, metrics.jsonl with the
off-batch loss, params_end.pkl, summary.json); finished runs are skipped, so the
script resumes where it stopped. Run from this directory.
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
from configs import base, supplementary as study

BENCHMARK = "burgers"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="default: the config's seeds (with --shared_phase1, those "
                         "that have a shared hand-over)")
    ap.add_argument("--rules", nargs="+", default=None, choices=list(QN_RULES),
                    help="default: RULES (with --large: LARGE_RULES)")
    ap.add_argument("--sizes", type=int, nargs="+", default=None,
                    help="batch sizes (default 2^14; with --large: LARGE_SIZES)")
    ap.add_argument("--large", action="store_true",
                    help="the larger batches: LARGE_RULES at LARGE_SIZES")
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
    rules = args.rules or (study.LARGE_RULES if args.large else study.RULES)
    sizes = args.sizes or (study.LARGE_SIZES if args.large else [study.N])
    runs = [dict(seed=seed, rule=rule, n=n, name=f"{rule}/n{n}",
                 config=study.get_config(rule, seed, n=n))
            for seed in seeds for n in sizes for rule in rules]
    train.run_study("supplementary", BENCHMARK, runs, base, get_dataset,
                    models.Burgers, models.BurgersEvaluator,
                    phase1_dir=args.phase1_dir, smoke=args.smoke, force=args.force,
                    shared_phase1=args.shared_phase1)


if __name__ == "__main__":
    main()
