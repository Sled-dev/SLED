"""Sampler comparison for Rayleigh-Taylor: every sampling rule at n = 2^15 = 32,768.

Phase 1 is configs/base.py, trained once per seed; every run restores that
checkpoint and runs QN_STEPS quasi-Newton steps on a batch of N points built by
the rule (pipeline.domain_importance_adapt.QN_RULES):

    uniform        i.i.d. uniform points, frozen
    sobol          scrambled Sobol points, frozen
    rad            RAD (k = 1, c = 1), frozen
    lev            Nystrom ridge leverage, importance-weighted, frozen
    kd             kernel diagonal ||grad_theta r||^2 ("Kd-w"), importance-weighted, frozen
    sled           SLED: leverage draws on a Sobol floor of measured mass, frozen
    uniform_r500   uniform points, redrawn every 500 steps
    rad_r500       RAD (k = 1, c = 1), redrawn every 500 steps
    vrba_r100      vRBA attention weights, redrawn every 100 steps
    pinnacle_k     PINNACLE-K: K-means++ on the eNTK convergence-degree embedding, frozen

The scored rules and pinnacle_k draw from a candidate pool of 4 N = 131,072 points. At any
other n the pool follows the batch-size study's rule
(configs/batch_size_study.pool_mult), so both studies build the same run for
the same (rule, n, seed) and share its directory.
"""
from pipeline.domain_importance_adapt import configure_qn_batch

from configs import base, batch_size_study

SEEDS = [2, 3, 5, 7, 11, 13]
N = 2**15
RULES = ["uniform", "sobol", "rad", "lev", "kd", "sled",
         "uniform_r500", "rad_r500", "vrba_r100", "pinnacle_k"]
QN_STEPS = 40_000
LOG_EVERY = 50              # off-batch loss / error series cadence (QN steps)


def get_config(rule, seed, n=N):
    config = base.get_config(seed=seed)
    configure_qn_batch(config, rule, n, pool_mult=batch_size_study.pool_mult(n))
    config.optim.second_order_steps = QN_STEPS
    config.logging.log_every_steps = LOG_EVERY
    config.wandb.name = "sampler_comparison_{rule}_{seed}".format(rule=rule, seed=seed)
    return config
