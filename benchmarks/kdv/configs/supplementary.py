"""Supplementary study for KdV: further sampling protocols and larger batches.

Phase 1 is configs/base.py, trained once per seed (or, with --shared_phase1, the
shared hand-over in checkpoints/kdv/); every run restores that checkpoint and
runs QN_STEPS quasi-Newton steps (pipeline.domain_importance_adapt.QN_RULES).

Protocols at n = N = 2^14 = 16,384 (RULES):

    pinnacle_s       PINNACLE-S: n pool points drawn without replacement with
                     probability ~ alpha (the eNTK convergence degree), frozen
    pinnacle_k_r500  PINNACLE-K reselected from a fresh pool every 500 steps, the
                     quasi-Newton memory kept (its 79 reselections add
                     hours to a run)
    rad_r500_reset   RAD (k = 1, c = 1) redrawn every 500 steps, the quasi-Newton
                     memory reset at every redraw
    vrba_r100_reset  vRBA redrawn every 100 steps, the memory reset at every redraw

Their comparators at the same n are rules of the other studies: pinnacle_k,
rad_r500, vrba_r100 and sled (COMPARATORS; --rules).

Larger batches (--large): LARGE_RULES at n = 2^15 = 32,768 and 2^16 = 65,536
(LARGE_SIZES); RAD frozen, as in the batch-size study.

The candidate pool follows the batch-size study's rule
(configs/batch_size_study.pool_mult): M = 4 n for every n here. A (rule, n) that
another study also runs builds the same run and shares its directory.
"""
from pipeline.domain_importance_adapt import configure_qn_batch

from configs import base, batch_size_study

SEEDS = [2, 3, 5, 7, 11]            # the seeds of the paper's supplementary runs
N = 2**14
RULES = ["pinnacle_s", "pinnacle_k_r500", "rad_r500_reset", "vrba_r100_reset"]
COMPARATORS = ["pinnacle_k", "rad_r500", "vrba_r100", "sled"]
LARGE_SIZES = [2**15, 2**16]
LARGE_RULES = ["uniform", "sobol", "rad", "sled"]
QN_STEPS = 40_000
LOG_EVERY = 50              # off-batch loss / error series cadence (QN steps)


def get_config(rule, seed, n=N):
    config = base.get_config(seed=seed)
    configure_qn_batch(config, rule, n, pool_mult=batch_size_study.pool_mult(n))
    config.optim.second_order_steps = QN_STEPS
    config.logging.log_every_steps = LOG_EVERY
    config.wandb.name = "supplementary_{rule}_n{n}_{seed}".format(rule=rule, n=n, seed=seed)
    return config
