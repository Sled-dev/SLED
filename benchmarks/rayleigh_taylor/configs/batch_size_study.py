"""Batch-size study for Rayleigh-Taylor: one frozen second-order batch of n points.

Phase 1 is configs/base.py, trained once per seed; every run restores that
checkpoint and optimizes ONE frozen batch of n points, drawn by the sampling
rule (pipeline.domain_importance_adapt.QN_RULES), for QN_STEPS quasi-Newton
steps. Only n and the rule vary; n = 2^11 ... 2^16 (RT extends the grid by two points).

Candidate pool of the scored rules (rad, lev, sled): M = max(POOL_SIZE, 4 n) points.
Up to n = 16,384 the pool is pinned at 65,536, so every score estimate is
identical across n and only the number of draws changes; above 16,384 it keeps
the ratio M = 4 n (a pool pinned at 65,536 would leave pool_mult = 2 at 32,768
and 1 at 65,536, where a scored rule can only draw duplicates of the pool). The
two conventions coincide at n = 16,384. uniform and sobol draw n points
directly and have no pool.
"""
from pipeline.domain_importance_adapt import configure_qn_batch

from configs import base

SEEDS = [2, 3, 5, 7, 11, 13]
SIZES = [2**11, 2**12, 2**13, 2**14, 2**15, 2**16]
RULES = ["uniform", "sobol", "rad", "lev", "sled"]
QN_STEPS = 40_000
POOL_SIZE = 65_536          # candidate pool up to n = 16,384; 4 n above
LOG_EVERY = 50              # off-batch loss / error series cadence (QN steps)


def pool_mult(n):
    if n < POOL_SIZE and POOL_SIZE % n:
        raise ValueError(f"n = {n} must divide the pinned pool {POOL_SIZE}")
    return max(POOL_SIZE // n, 4)


def get_config(rule, n, seed):
    config = base.get_config(seed=seed)
    configure_qn_batch(config, rule, n, pool_mult=pool_mult(n))
    config.optim.second_order_steps = QN_STEPS
    config.logging.log_every_steps = LOG_EVERY
    config.wandb.name = "batch_size_{rule}_n{n}_{seed}".format(rule=rule, n=n, seed=seed)
    return config
