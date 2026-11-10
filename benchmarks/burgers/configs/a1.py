"""Ablation a1 for Burgers: SLED with a fixed floor mass, n = 2^14 = 16,384.

Phase 1 is configs/base.py, trained once per seed; every run restores that
checkpoint and runs QN_STEPS quasi-Newton steps on a batch of N points built by
SLED with its measured floor mass rho (the Spearman correlation of leverage and
kernel diagonal) replaced by a fixed phi
(pipeline.domain_importance_adapt.QN_RULES, sampler type sled_fixed):

    sled_fl0       phi = 0: N leverage draws, no floor
    sled_fl0.25    phi = 0.25: 4,096 Sobol points + 12,288 leverage draws
    sled_fl0.5     phi = 0.5: 8,192 + 8,192
    sled_fl0.75    phi = 0.75: 12,288 Sobol points + 4,096 leverage draws

phi = 1 is the sobol rule. Everything else is SLED's: stacked estimator,
scrambled-Sobol floor, unfloored leverage proposal, weights omega = 1/(M pi).
rho is still measured and recorded (rho_measured). The candidate pool follows
the batch-size study's rule (configs/batch_size_study.pool_mult): M = 65,536 at
N. Its references, sled (phi = rho) and sobol (phi = 1) at N, are runs of the
batch-size study: asked for here, they build the same run and share its
directory.
"""
from pipeline.domain_importance_adapt import configure_qn_batch

from configs import base, batch_size_study

SEEDS = [2, 3, 5, 7, 11, 13]
N = 2**14
RULES = ["sled_fl0", "sled_fl0.25", "sled_fl0.5", "sled_fl0.75"]
QN_STEPS = 40_000
LOG_EVERY = 50              # off-batch loss / error series cadence (QN steps)


def get_config(rule, seed, n=N):
    config = base.get_config(seed=seed)
    configure_qn_batch(config, rule, n, pool_mult=batch_size_study.pool_mult(n))
    config.optim.second_order_steps = QN_STEPS
    config.logging.log_every_steps = LOG_EVERY
    config.wandb.name = "a1_{rule}_{seed}".format(rule=rule, seed=seed)
    return config
