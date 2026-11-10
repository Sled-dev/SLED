"""Handover readings: reference-free statistics of a phase-1 hand-over.

At a phase-1 checkpoint (the hand-over from the first-order optimizer to the
quasi-Newton phase) this script measures, on a uniform candidate pool and
without the reference solution, how the residual-gradient kernel of the
stacked residual is spread over the domain.  The central number is

    spearman_lev_K@1e-04 = Spearman(l_hat, kappa_hat) at lam_rel = 1e-4

the rank correlation over the pool between the Nystrom ridge leverage l_hat(x)
(ridge lam = lam_rel * top eigenvalue) and the Hutchinson kernel diagonal
kappa_hat(x) = sum_e ||d/dtheta sqrt(w_e) r_e(x)||^2 of the stacked residual
(every residual family, scaled by the frozen loss weights w_e).  It is the
quantity SLED's floor is made of: the SLED batch
(pipeline.domain_importance_adapt.SLEDSampler) measures it on its own pool and
puts round(rho * n) of its n points on a scrambled Sobol set.  Its rung on the
fixed ladder (thresholds set in advance, never tuned)

    rho <= 0.75 leverage  |  0.75 < rho < 0.85 hedge  |  rho >= 0.85 cover

says whether the leverage carries information beyond the kernel diagonal
(leverage) or the kernel is flat and coverage is what counts (cover).  Every
JSON also holds, at lam_rel in (1e-2, 1e-4, 1e-6) where it depends on lam,

    d_lam, lev_spread = sup l / d_lam, ess_lev     effective dimension, leverage
                                                   concentration, ESS of the
                                                   floored leverage pmf
    flat_64, flat_128, flat_256, eig_slope_16_256  eigenvalue ratio at a FIXED
                                                   index; log-log spectrum slope
                                                   over indices 16-256
    ess_res2, r2_top1pct_mass, K_max_over_median,  residual / kernel-diagonal
    spearman_r2_K, res_rms_family, K_family_share  statistics per point and family
    lev_batch_*  the `lev` batch (stacked estimator, uniform floor 0.2) that a
                 run would draw at this checkpoint and the fraction of grid
                 cells it occupies against a Sobol batch of the same size;
                 --lev_batch_unstacked adds the sampler comparison's `lev` rule
                 (first equation only) as lev_batch_unstacked_*
    sled_*       the SLED batch at n = run_n: rho_floor, n_floor, n_scored,
                 batch_distinct, ess (and its grid coverage)

Pools (one JSON each)
    diag   M = 32,768 uniform points, pool seed 12345, probe key 7: the same
           pool and probes at every checkpoint
    run    M = 4 * run_n, pool seed = config.seed, probe key = seed + 1: the
           pool from which the frozen-batch samplers draw a batch of run_n
           points (default run_n = 2^14, the sampler comparison's n), so the run
           reading IS the floor SLED draws there (asserted against the sled_*
           block)

Nystrom: L = 1,024 landmarks (the first L pool points) and rank k = 256 on
every checkpoint and benchmark.  The eigenvalue ratio lambda_j / lambda_1 at a
fixed index j depends on k, so one (L, k) keeps the spectral statistics
comparable across checkpoints and benchmarks (the tables keep k = 256 rows
only).  As a check of the rank-k approximation, the Spearman correlation
between the rank-k reconstruction of the kernel diagonal and kappa_hat on
2,000 pool points is logged as `recon_spearman`, with `recon_trusted` =
(recon_spearman >= 0.95); it does not change the estimator.

Batch-size marks (--marks)
    The a-priori marks of the batch-size figure, computed instead of the
    readings.  On the pool the batch-size study's scored rules draw from for
    n <= 16,384 (configs/batch_size_study.POOL_SIZE = 65,536 uniform points,
    pool seed = config.seed) and with the UNSTACKED estimator of the `lev`
    rule (nystrom_leverage and the Hutchinson ntk_diag of the first equation,
    8 probes, key seed + 1), at lam_rel in (1e-2, 1e-4, 1e-6):

        d_lam            effective dimension sum_i ev_i / (ev_i + lam)
        d_lam_log        d_lam * log(d_lam), the leverage sample-complexity mark
        sup_lev          sup of the ridge leverage over the pool (the uniform
                         threshold, without constants)
        spread           sup_lev / d_lam, the advantage of a leverage batch
        ess_lev          normalised ESS of the floored (0.2) leverage proposal
        ess_rad1         the same for the frozen RAD k = 1, c = 1 proposal
                         (floor c / (1 + c) = 1/2)
        spearman_lev_K   Spearman(leverage, Hutchinson kernel diagonal)
        n_lev_thm        the leverage threshold of the embedding theorem with
                         its constants, n_Lev = 4 eps^-2 d_lam log(16 d_lam / zeta),
                         eps = 1/2, zeta = 0.1

    At lam_rel = 1e-4, d_lam, sup_lev, spearman_lev_K and ess_lev are the
    numbers the `lev` rule's own diagnostics report when it builds a batch of
    n <= 16,384 points, and ess_rad1 is the `rad` rule's ESS there.  The
    figure draws sup_lev and n_lev_thm at lam_rel = 1e-4, medians over seeds.
    Outputs: marks_all.csv (one row per benchmark x seed x lam_rel) and
    marks_summary.csv (median over seeds per benchmark and lam_rel; n_lev_thm
    is the median of the per-seed thresholds and n_lev_thm_at_median_d_lam the
    threshold at the median d_lam - the two coincide for an odd number of
    seeds).

Every pool is drawn with jax.random, so the readings and marks of a seed
depend on JAX's PRNG implementation (jax_threefry_partitionable, True by
default since JAX 0.5): compute them with the JAX the runs use.

Usage (from this directory):

    python handover_readings.py                                # six benchmarks, every seed
    python handover_readings.py --benchmarks rayleigh_taylor --seeds 2 3
    python handover_readings.py --phase1_dir ../checkpoints/{benchmark}   # the shared hand-overs
    python handover_readings.py --marks                        # batch-size marks
    python handover_readings.py --export_only                  # rebuild the tables
    python handover_readings.py --benchmarks burgers --seeds 0 --smoke \\
        --phase1_dir burgers/runs_smoke/phase1

Checkpoints are <phase1_dir>/seed<k>/phase1.pkl (+ phase1.json), phase1_dir
defaulting to <benchmark>/runs/phase1 (written by batch_size_study.py and
sampler_comparison.py); a {benchmark} in --phase1_dir is replaced by the
benchmark's name.  Outputs record the checkpoint's parameter hash and its path
relative to the benchmark directory (seed<k>/phase1.pkl when it lies outside).

Outputs (runs/readings/ unless --out):
    <benchmark>_seed<s>_step<t>_<pool>_M<M>.json   one per checkpoint and pool
    readings_<benchmark>.csv, readings_all.csv      one row per checkpoint, pool, lam_rel
    readings_all.md, ladder_verdict.md              lam_rel = 1e-4 at k = 256
    marks_<benchmark>_seed<s>_step<t>_M<M>.json     --marks: one per checkpoint
    marks_all.csv, marks_summary.csv                --marks: see above
"""
import os

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
_flags = os.environ.get("XLA_FLAGS", "")
if "--xla_gpu_deterministic_ops" not in _flags:
    os.environ["XLA_FLAGS"] = (_flags + " --xla_gpu_deterministic_ops=true").strip()
os.environ["JAX_ENABLE_X64"] = "True"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("WANDB_MODE", "offline")

import argparse
import csv
import glob
import importlib
import json
import statistics
import sys
import time
import traceback

import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from jax.flatten_util import ravel_pytree

jax.config.update("jax_default_matmul_precision", "highest")
# the PRNG every pool, batch and initialisation draws from (the default since
# JAX 0.5; older JAX would draw different points from the same seeds)
jax.config.update("jax_threefry_partitionable", True)

from pipeline.domain_importance_adapt import (SobolSampler, UniformSampler,
                                              _clean_scores, _proposal,
                                              leverage_from, make_qn_batch,
                                              ntk_diag, nystrom_decomposition,
                                              nystrom_leverage, per_point, qn_ess,
                                              residual_families,
                                              residual_matrix_fn,
                                              stacked_residual_fn)
from pipeline.train import (build_from_checkpoint, l2re_of, load_phase1,
                            now_iso, params_hash, read_json)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "runs", "readings")

# benchmark -> (model class, evaluator class) in <benchmark>/models.py
BENCHMARKS = {
    "burgers":         ("Burgers", "BurgersEvaluator"),
    "allen_cahn":      ("AllenCahn", "AllenCahnEvaluator"),
    "ginzburg_landau": ("GinzburgLandau", "GinzburgLandauEvaluator"),
    "kdv":             ("KDV", "KDVEvaluator"),
    "ks":              ("KS", "KSEvaluator"),
    "rayleigh_taylor": ("NavierStokes", "NavierStokesEvaluator"),
}

# ------------------------------------------------------------ fixed settings
# The ladder and the estimator sizes are fixed in advance; never tune them.
RUNGS = ((0.75, "leverage"), (0.85, "hedge"))     # <= 0.75 | (0.75, 0.85) | >= 0.85
LADDER_READING = "spearman_lev_K@1e-04"
LAM_RELS = (1e-2, 1e-4, 1e-6)
NYS_L, NYS_K = 1024, 256                 # one (L, k) for every checkpoint
PROBES = 8
DIAG_POOL_M, DIAG_POOL_SEED, DIAG_PROBE_KEY = 32768, 12345, 7   # the frozen diagnostics pool
RUN_POOL_MULT = 4                        # the run's own pool is 4n
RUN_N = 2**14                            # the sampler comparison's batch size
FLOOR = 0.2                              # lam_fl of ess_lev and of the lev batch
TRUST_SPEARMAN = 0.95                    # rank-k reconstruction check (logged only)
COV_GRID = {2: 64, 3: 32}                # coverage cells per dimension, by input dimension
SMOKE = {"pool_M": 1024, "run_n": 256, "L": 64, "k": 16, "probes": 2}

# batch-size marks (--marks)
MARKS_POOL_M = 65_536                    # fallback if configs/batch_size_study has no POOL_SIZE
MARKS_PROBE_OFFSET = 1                   # Hutchinson key PRNGKey(seed + 1), the lev rule's
MARKS_RAD_FLOOR = 0.5                    # RAD k = 1, c = 1: floor c / (1 + c)
N_LEV_EPS, N_LEV_ZETA = 0.5, 0.1         # constants of the leverage threshold n_lev_thm


# ------------------------------------------------------------------- helpers

def floored_pmf(score, lam_fl=FLOOR):
    s = np.maximum(np.nan_to_num(np.asarray(score, dtype=float)), 0.0)
    s = s / s.sum() if s.sum() > 0 else np.full(len(s), 1.0 / len(s))
    return (1 - lam_fl) * s + lam_fl / len(s)


def ess(pi):
    pi = np.asarray(pi)
    return float(1.0 / (len(pi) * (pi ** 2).sum()))


def _spearman(a, b):
    try:
        from scipy.stats import spearmanr
        return float(spearmanr(np.asarray(a), np.asarray(b)).statistic)
    except Exception:
        return None


def _rms(v):
    v = np.asarray(v, dtype=float)
    return float(np.sqrt(np.mean(v ** 2))) if v.size else float("nan")


def pmean(v, n_points):
    """Pool inner product: sum over ROWS divided by the number of POINTS."""
    return jnp.sum(v) / float(n_points)


def residual_rms(model, params, X, families=None, weights=None, weighted=True,
                 chunk=32768):
    """{family: RMS of r_e} plus "_total" over the stacked vector.

    `weighted=False` gives the PHYSICAL residual (no sqrt(w_e) scaling), which
    is the reference-free error proxy; `weighted=True` gives the rows of the
    stacked residual the spectrum is built from.
    """
    families = tuple(families or residual_families(model))
    r_mat = residual_matrix_fn(model)
    w = ({f: float((weights or model.state.weights)[f]) for f in families}
         if weighted else {f: 1.0 for f in families})
    tot, n = {f: 0.0 for f in families}, 0
    for i in range(0, X.shape[0], chunk):
        R = np.asarray(r_mat(params, X[i:i + chunk]))       # (c, E)
        for j, f in enumerate(families):
            tot[f] += float(np.sum(w[f] * R[:, j] ** 2))
        n += R.shape[0]
    out = {f: float(np.sqrt(tot[f] / max(n, 1))) for f in families}
    out["_total"] = float(np.sqrt(sum(tot.values()) / max(n, 1)))
    return out


def occupied_fraction(batch, dom, grid):
    """Fraction of grid^d cells of the domain box that the batch occupies."""
    X = np.asarray(batch, dtype=float)
    lo = np.asarray(dom, dtype=float)[:, 0]
    hi = np.asarray(dom, dtype=float)[:, 1]
    u = (X - lo) / np.maximum(hi - lo, 1e-300)
    idx = np.clip((u * grid).astype(np.int64), 0, grid - 1)
    flat = np.zeros(X.shape[0], dtype=np.int64)
    for d in range(X.shape[1]):
        flat = flat * grid + idx[:, d]
    return float(np.unique(flat).size) / float(grid ** X.shape[1])


def _cov_grid(config):
    return COV_GRID.get(int(config.arch.input_dim), 32)


def ntk_diag_families(stacked, params, pool, n_fam, n_probes=PROBES, key=None):
    """Per-FAMILY Hutchinson kernel diagonal, (E, M).

    hutchinson_ntk_diag sums the point's E rows; the per-family split shows how
    the kernel diagonal is shared between the residual families.  Same probes,
    same key -> the column sum is that function's output to float noise.
    """
    key = random.PRNGKey(DIAG_PROBE_KEY) if key is None else key
    m = int(pool.shape[0])
    flat, unravel = ravel_pytree(params)

    @jax.jit
    def probe_sq(v_flat):
        _, t = jax.jvp(lambda p: stacked(p, pool), (params,), (unravel(v_flat),))
        return (t ** 2).reshape(n_fam, m)          # family-major rows

    acc = jnp.zeros((n_fam, m))
    for _ in range(int(n_probes)):
        key, k = random.split(key)
        v = random.rademacher(k, flat.shape, dtype=flat.dtype).astype(flat.dtype)
        acc = acc + probe_sq(v)
    return np.asarray(acc / float(n_probes))


def ess_unfloored(score):
    s = np.maximum(np.nan_to_num(np.asarray(score, dtype=float)), 0.0)
    if s.sum() <= 0:
        return 1.0
    return ess(s / s.sum())


def eig_slope(evals, lo=16, hi=256):
    """Slope of log(eval) against log(index) over indices [lo, hi] (1-based)."""
    hi = min(int(hi), len(evals))
    if hi <= lo + 1:
        return None
    i = np.arange(lo, hi + 1, dtype=float)
    y = np.asarray(evals[lo - 1:hi], dtype=float)
    ok = y > 0
    if ok.sum() < 3:
        return None
    return float(np.polyfit(np.log(i[ok]), np.log(y[ok]), 1)[0])


def rung_of(spearman):
    if spearman is None:
        return None
    if spearman <= RUNGS[0][0]:
        return "leverage"
    if spearman >= RUNGS[1][0]:
        return "cover"
    return "hedge"


def n_lev_thm(d_lam, eps=N_LEV_EPS, zeta=N_LEV_ZETA):
    """Leverage threshold of the embedding theorem, with its constants:
    n_Lev = 4 eps^-2 d_lam log(16 d_lam / zeta)."""
    return float(4.0 / eps ** 2 * d_lam * np.log(16.0 * d_lam / zeta))


# --------------------------------------------------------------- the reading

def readings_one(model, params, dom, weights, pool_M, pool_seed, probe_key,
                 L=NYS_L, k=NYS_K, probes=PROBES):
    """Kernel-diagonal and Nystrom statistics at ONE pool and ONE Nystrom
    size (L, k)."""
    t_all = time.perf_counter()
    fams = residual_families(model)
    stacked = stacked_residual_fn(model, weights, fams)
    pool = next(iter(UniformSampler(dom, int(pool_M), int(pool_seed))))
    m = int(pool.shape[0])
    out = {"pool_M": m, "pool_seed": int(pool_seed), "probe_key": int(probe_key),
           "probes": int(probes), "nys_L": int(L), "nys_k": int(k),
           "families": list(fams),
           "family_weights": {f: float(weights[f]) for f in fams}}

    # ------------------------------------ residuals and the kernel diagonal
    t0 = time.perf_counter()
    r_pool = np.asarray(stacked(params, pool))                 # (E*M,)
    K_fam = ntk_diag_families(stacked, params, pool, len(fams), probes,
                              random.PRNGKey(int(probe_key)))
    K_pool = K_fam.sum(axis=0)
    out["probe_wall_s"] = round(time.perf_counter() - t0, 3)
    r_mat = r_pool.reshape(len(fams), m)
    r2_point = np.asarray(per_point(jnp.asarray(r_pool ** 2), m))
    out.update(
        ess_res2=ess(floored_pmf(r2_point)),
        ess_res2_unfloored=ess_unfloored(r2_point),
        ess_kd=ess(floored_pmf(K_pool)),
        r2_top1pct_mass=float(np.sort(r2_point)[::-1][:m // 100].sum()
                              / max(r2_point.sum(), 1e-300)),
        r2_max_over_median=float(r2_point.max() / max(np.median(r2_point), 1e-300)),
        K_max_over_median=float(K_pool.max() / max(np.median(K_pool), 1e-300)),
        spearman_r2_K=_spearman(r2_point, K_pool),
        res_loss_handover=float(pmean(jnp.asarray(r_pool ** 2), m)),
        res_rms_total=float(np.sqrt(pmean(jnp.asarray(r_pool ** 2), m))),
        res_rms_family={f: _rms(r_mat[i]) for i, f in enumerate(fams)},
        res_rms_family_raw=residual_rms(model, params, pool, fams,
                                        weighted=False),
        # how the kernel diagonal is shared between the residual families
        K_family_mean={f: float(K_fam[i].mean()) for i, f in enumerate(fams)},
        K_family_share={f: float(K_fam[i].sum() / max(K_pool.sum(), 1e-300))
                        for i, f in enumerate(fams)},
        K_family_max_over_median={
            f: float(K_fam[i].max() / max(np.median(K_fam[i]), 1e-300))
            for i, f in enumerate(fams)},
        K_pool_sum=float(K_pool.sum()), K_pool_mean=float(K_pool.mean()))

    # ------------------------------------------------------------- Nystrom
    t0 = time.perf_counter()
    evals, U, A, scale, nys = nystrom_decomposition(stacked, params, pool,
                                                    L=int(L), k=int(k))
    out["nystrom_wall_s"] = round(time.perf_counter() - t0, 3)
    out.update({f"nys_{key}": val for key, val in nys.items()})
    # check of the rank-k approximation: logged, never acted on
    sub = np.arange(min(2000, m))
    recon = np.asarray(per_point(
        jnp.asarray((evals[:, None] * U ** 2).sum(0)), m))[sub]
    rho = _spearman(recon, K_pool[sub])
    out["recon_spearman"] = rho
    out["recon_trusted"] = bool(rho is not None and rho >= TRUST_SPEARMAN)

    # flatness at a FIXED index - comparable across benchmarks at one k
    for j in (64, 128, 256):
        out[f"flat_{j}"] = (float(evals[j - 1] / evals[0])
                            if len(evals) >= j else None)
    out["flat_last"] = float(evals[-1] / evals[0])       # lambda_k / lambda_1
    out["eig_slope_16_256"] = eig_slope(evals)
    out["spectrum_top"] = evals[:32].tolist()
    out["k_used"] = int(len(evals))

    for lr in LAM_RELS:
        lam = float(lr) * float(evals[0])
        lev, d_lam = leverage_from(evals, U, lam, m)
        lev = np.maximum(np.nan_to_num(lev), 0.0)
        tag = f"{lr:.0e}"
        out[f"lam@{tag}"] = lam
        out[f"d_lam@{tag}"] = d_lam
        out[f"d_lam_over_k@{tag}"] = float(d_lam / max(len(evals), 1))
        out[f"sup_lev@{tag}"] = float(lev.max())
        out[f"mean_lev@{tag}"] = float(lev.mean())
        out[f"sum_lev@{tag}"] = float(lev.sum())
        out[f"lev_spread@{tag}"] = float(lev.max() / max(d_lam, 1e-300))
        out[f"spearman_lev_K@{tag}"] = _spearman(lev, K_pool)
        out[f"ess_lev@{tag}"] = ess(floored_pmf(lev, FLOOR))
        out[f"ess_lev_unfloored@{tag}"] = ess_unfloored(lev)
    out["reading_wall_s"] = round(time.perf_counter() - t_all, 1)
    return out


def lev_batch_block(model, params, dom, weights, n, seed, grid, stacked_rule=True,
                    L=NYS_L, k=NYS_K, probes=PROBES):
    """The `lev` batch a run would draw at this checkpoint (make_qn_batch, pool
    4n, uniform floor 0.2) and the grid cells it occupies against a Sobol
    batch of the same size.  stacked_rule=False is the sampler comparison's
    `lev` rule (first equation of a multi-residual model)."""
    t0 = time.perf_counter()
    batch, _, info = make_qn_batch("lev", model, params, dom, int(n),
                                   weighted=True, seed=int(seed), floor=FLOOR,
                                   loss_weights=weights, stacked=bool(stacked_rule),
                                   pool_mult=RUN_POOL_MULT, probes=int(probes),
                                   lam_rel=1e-4, nys_L=int(L), nys_k=int(k))
    nysinfo = info.get("nystrom", {})
    batch_sob = SobolSampler(dom, int(n), seed=int(seed)).sampled
    prefix = "lev_batch_" if stacked_rule else "lev_batch_unstacked_"
    return {prefix + key: val for key, val in {
        "stacked": bool(stacked_rule), "n": int(n),
        "pool_size": info.get("pool_size"), "pool_mult": RUN_POOL_MULT,
        "d_lam": nysinfo.get("d_lam"), "sup_lev": nysinfo.get("sup_lev"),
        "mean_lev": nysinfo.get("mean_lev"),
        "lev_spread": nysinfo.get("lev_spread"),
        "spearman_lev_K": nysinfo.get("spearman_lev_K"),
        "k_used": nysinfo.get("k_used"), "lam_rel": nysinfo.get("lam_rel"),
        "spectral_flatness": nysinfo.get("spectral_flatness"),
        "cov_frac_lev": occupied_fraction(batch, dom, int(grid)),
        "cov_frac_sobol": occupied_fraction(batch_sob, dom, int(grid)),
        "cov_grid": int(grid),
        "batch_distinct_lev": info.get("batch_distinct"),
        "ess_lev_draw": info.get("ess"),
        "score_max_over_median_lev": info.get("score_max_over_median"),
        "wall_s": round(time.perf_counter() - t0, 1)}.items()}


def sled_block(model, params, dom, weights, n, seed, grid, L=NYS_L, k=NYS_K,
               probes=PROBES):
    """The SLED batch at n points (pool 4n, the sampler comparison's
    settings): the measured floor rho and how the batch splits into Sobol
    floor and leverage draws."""
    t0 = time.perf_counter()
    batch, _, info = make_qn_batch("sled", model, params, dom, int(n),
                                   weighted=True, seed=int(seed), floor=FLOOR,
                                   loss_weights=weights, pool_mult=RUN_POOL_MULT,
                                   probes=int(probes), lam_rel=1e-4,
                                   nys_L=int(L), nys_k=int(k))
    return {"sled_" + key: val for key, val in {
        "n": int(n), "pool_size": info.get("pool_size"), "pool_mult": RUN_POOL_MULT,
        "rho_floor": info.get("rho_floor"), "floor": info.get("floor"),
        "n_floor": info.get("n_floor"), "n_scored": info.get("n_scored"),
        "batch_distinct": info.get("batch_distinct"), "ess": info.get("ess"),
        "cov_frac": occupied_fraction(batch, dom, int(grid)),
        "cov_grid": int(grid),
        "wall_s": round(time.perf_counter() - t0, 1)}.items()}


# ------------------------------------------------------------ batch-size marks

def marks_block(model, evaluator, params, dom, weights, pool_M, seed,
                L=NYS_L, k=NYS_K, probes=PROBES):
    """The batch-size marks at one checkpoint: one row per lam_rel.

    Pool UniformSampler(dom, pool_M, seed) - the batch-size study's pinned
    pool - and the UNSTACKED estimator of the `lev` rule: nystrom_leverage
    (first equation) for the leverage and ntk_diag with key PRNGKey(seed + 1)
    for the kernel diagonal.  ess_rad1 scores |r| (k = 1) from the
    loss-weighted squared residual r_losses_pp at RAD's floor c / (1 + c).
    """
    M = int(pool_M)
    handover, _ = l2re_of(evaluator, params)
    pool = next(iter(UniformSampler(dom, M, int(seed))))
    Kdiag = np.asarray(ntk_diag(model, params, pool, int(probes),
                                random.PRNGKey(int(seed) + MARKS_PROBE_OFFSET)))
    r2 = np.asarray(_clean_scores(model.r_losses_pp(params, pool, weights)))
    rad1 = np.sqrt(np.maximum(r2, 0.0))                 # rad1 scores |r|^1
    ess_rad1 = qn_ess(_proposal(jnp.asarray(rad1), MARKS_RAD_FLOOR, M))
    n_params = int(sum(x.size for x in jax.tree_util.tree_leaves(params)))

    rows = []
    for lam_rel in LAM_RELS:
        t0 = time.perf_counter()
        lev, d_lam, nys = nystrom_leverage(model, params, pool, L=int(L), k=int(k),
                                           lam_rel=lam_rel)
        wall = round(time.perf_counter() - t0, 3)
        lev = np.maximum(np.nan_to_num(lev, nan=0.0, posinf=0.0), 0.0)
        rows.append({
            "lam_rel": lam_rel, "handover_l2re": handover, "pool_size": M,
            "pool_seed": int(seed), "probe_key": int(seed) + MARKS_PROBE_OFFSET,
            "probes": int(probes), "nys_L": int(L), "nys_k": int(k),
            "n_params": n_params,
            "d_lam": d_lam, "d_lam_log": d_lam * np.log(max(d_lam, 1.0000001)),
            "sup_lev": float(lev.max()), "mean_lev": float(lev.mean()),
            "spread": float(lev.max() / max(d_lam, 1e-300)),
            "top_eval": nys.get("top_eval"),
            "spectral_flatness": nys.get("spectral_flatness"),
            "k_used": nys.get("k_used"),
            "ess_lev": qn_ess(_proposal(jnp.asarray(lev), FLOOR, M)),
            "ess_rad1": ess_rad1, "spearman_lev_K": _spearman(lev, Kdiag),
            "n_lev_thm": n_lev_thm(d_lam),
            "nystrom_wall_s": wall})
    return rows


# ------------------------------------------------------------------ per ckpt

def load(benchmark):
    """(models, get_dataset, configs.base, configs.batch_size_study) of a
    benchmark, imported from its directory with cwd set there
    (utils.get_dataset reads data/ relative to cwd).  Returns them and the
    previous cwd."""
    path = os.path.join(HERE, benchmark)
    others = {os.path.join(HERE, b) for b in BENCHMARKS}
    sys.path[:] = [p for p in sys.path if p not in others]
    sys.path.insert(0, path)
    for name in [m for m in sys.modules
                 if m in ("models", "utils", "configs") or m.startswith("configs.")]:
        sys.modules.pop(name)                   # the benchmarks share module names
    importlib.invalidate_caches()
    prev = os.getcwd()
    os.chdir(path)
    models = importlib.import_module("models")
    utils = importlib.import_module("utils")
    base = importlib.import_module("configs.base")
    study = importlib.import_module("configs.batch_size_study")
    return (models, utils.get_dataset, base, study), prev


def phase1_dir_of(benchmark, template):
    if template is None:
        return os.path.join(HERE, benchmark, "runs", "phase1")
    return template.replace("{benchmark}", benchmark)


def ckpt_relpath(benchmark, ck_path):
    """The checkpoint's path relative to the benchmark directory, or
    seed<k>/phase1.pkl when it lies outside (no absolute path in outputs)."""
    bench_dir = os.path.join(HERE, benchmark)
    ck = os.path.abspath(ck_path)
    if ck.startswith(bench_dir + os.sep):
        return os.path.relpath(ck, bench_dir)
    return os.path.join(os.path.basename(os.path.dirname(ck)), os.path.basename(ck))


def open_checkpoint(benchmark, mods, seed, ck_path):
    """Load and hash-check a phase-1 checkpoint and rebuild its model.
    Returns (ck, meta, config, model, evaluator, dom, step)."""
    models, get_dataset, base, _ = mods
    if not os.path.exists(ck_path):
        raise FileNotFoundError(ck_path)
    meta = read_json(os.path.join(os.path.dirname(ck_path), "phase1.json"), {}) or {}
    ck = load_phase1(ck_path)
    h = params_hash(ck["params"])
    if h != ck["phase1_ckpt_hash"]:
        raise RuntimeError(f"checkpoint {ck_path} is corrupt: params hash {h} != "
                           f"stored {ck['phase1_ckpt_hash']}")
    config = base.get_config(seed=seed)
    mcls, ecls = BENCHMARKS[benchmark]
    model, evaluator, dom = build_from_checkpoint(
        config, ck, get_dataset, getattr(models, mcls), getattr(models, ecls))
    step = next((int(s) for s in (meta.get("first_order_steps"),
                                  ck.get("first_order_steps"))
                 if s is not None), int(config.optim.first_order_steps))
    return ck, meta, config, model, evaluator, dom, step


def checkpoint_rows(benchmark, mods, seed, ck_path, args):
    """Both pools at one checkpoint; writes one JSON per pool, returns them."""
    ck, meta, config, model, _, dom, step = open_checkpoint(benchmark, mods, seed,
                                                            ck_path)
    params = model.state.params
    weights = ck["weights"]
    grid = _cov_grid(config)
    run_n = SMOKE["run_n"] if args.smoke else int(args.run_n)
    L, k = (SMOKE["L"], SMOKE["k"]) if args.smoke else (args.L, args.k)
    probes = SMOKE["probes"] if args.smoke else PROBES
    diag_M = SMOKE["pool_M"] if args.smoke else args.pool_M

    common = {
        "benchmark": benchmark, "seed": int(seed), "step": step,
        "phase1_ckpt_hash": ck["phase1_ckpt_hash"],
        "phase1_ckpt_path": ckpt_relpath(benchmark, ck_path),
        "dia_type": meta.get("dia_type") or ck.get("dia_type") or str(config.dia.type),
        "dia_batch_size": int(config.dia.batch_size), "run_n": run_n,
        "hidden_dim": int(config.arch.hidden_dim),
        "input_dim": int(config.arch.input_dim),
        "handover_l2re": meta.get("handover_l2re", ck.get("handover_l2re")),
        "handover_errors": meta.get("handover_errors"),
        "config_hash": meta.get("config_hash") or ck.get("config_hash"),
        "finished": now_iso()}

    pools = [("diag", diag_M, DIAG_POOL_SEED, DIAG_PROBE_KEY),
             ("run", RUN_POOL_MULT * run_n, int(config.seed), int(config.seed) + 1)]
    rows = []
    for kind, pool_M, pool_seed, probe_key in pools:
        ktag = "" if int(k) == NYS_K else f"_k{int(k)}"
        out_path = os.path.join(
            OUT_DIR, f"{benchmark}_seed{seed}_step{step}_{kind}_M{pool_M}{ktag}.json")
        if os.path.exists(out_path) and not args.force:
            old = read_json(out_path, {}) or {}
            if old.get("phase1_ckpt_hash") == ck["phase1_ckpt_hash"]:
                print(f"[readings] {os.path.basename(out_path)} exists, reusing")
                rows.append(old)
            else:
                print(f"[readings] {out_path} holds another checkpoint "
                      f"({old.get('phase1_ckpt_hash')} != {ck['phase1_ckpt_hash']}); "
                      "use --out or --force, skipping")
            continue
        row = {**common, "pool_kind": kind}
        row.update(readings_one(model, params, dom, weights, pool_M, pool_seed,
                                probe_key, L=L, k=k, probes=probes))
        if kind == "run":
            if not args.no_lev_batch:
                row.update(lev_batch_block(model, params, dom, weights, run_n,
                                           config.seed, grid, stacked_rule=True,
                                           L=L, k=k, probes=probes))
                if args.lev_batch_unstacked:
                    row.update(lev_batch_block(model, params, dom, weights, run_n,
                                               config.seed, grid, stacked_rule=False,
                                               L=L, k=k, probes=probes))
            row.update(sled_block(model, params, dom, weights, run_n, config.seed,
                                  grid, L=L, k=k, probes=probes))
            # the SLED floor IS the run pool's reading: same pool, probes and
            # estimator, so the two numbers must agree bit for bit
            if row["sled_rho_floor"] != row[LADDER_READING]:
                raise AssertionError(
                    f"sled_rho_floor {row['sled_rho_floor']!r} != "
                    f"{LADDER_READING} {row[LADDER_READING]!r} of the run pool")
        row["rung"] = rung_of(row.get(LADDER_READING))
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(out_path + ".tmp", "w") as fh:
            json.dump(row, fh, indent=2, default=float)
        os.replace(out_path + ".tmp", out_path)
        print(f"[readings] {benchmark} seed {seed} step {step} pool {kind} "
              f"M={pool_M}: {LADDER_READING}={row.get(LADDER_READING)} "
              f"-> rung {row['rung']}  d_lam={row.get('d_lam@1e-04'):.4g} "
              f"flat_256={row.get('flat_256')}  {row['reading_wall_s']}s")
        rows.append(row)
    return rows


def checkpoint_marks(benchmark, mods, seed, ck_path, args):
    """The batch-size marks at one checkpoint; writes one JSON, returns it."""
    study = mods[3]
    ck, meta, config, model, evaluator, dom, step = open_checkpoint(
        benchmark, mods, seed, ck_path)
    pool_M = (SMOKE["pool_M"] if args.smoke
              else int(getattr(study, "POOL_SIZE", MARKS_POOL_M)))
    L, k = (SMOKE["L"], SMOKE["k"]) if args.smoke else (args.L, args.k)
    probes = SMOKE["probes"] if args.smoke else PROBES
    ktag = "" if int(k) == NYS_K else f"_k{int(k)}"
    out_path = os.path.join(
        OUT_DIR, f"marks_{benchmark}_seed{seed}_step{step}_M{pool_M}{ktag}.json")
    if os.path.exists(out_path) and not args.force:
        old = read_json(out_path, {}) or {}
        if old.get("phase1_ckpt_hash") == ck["phase1_ckpt_hash"]:
            print(f"[marks] {os.path.basename(out_path)} exists, reusing")
            return old
        print(f"[marks] {out_path} holds another checkpoint "
              f"({old.get('phase1_ckpt_hash')} != {ck['phase1_ckpt_hash']}); "
              "use --out or --force, skipping")
        return None
    t0 = time.perf_counter()
    out = {"benchmark": benchmark, "seed": int(seed), "step": step,
           "phase1_ckpt_hash": ck["phase1_ckpt_hash"],
           "phase1_ckpt_path": ckpt_relpath(benchmark, ck_path),
           "dia_type": meta.get("dia_type") or ck.get("dia_type") or str(config.dia.type),
           "finished": None,
           "rows": marks_block(model, evaluator, model.state.params, dom,
                               model.state.weights, pool_M, int(config.seed),
                               L=L, k=k, probes=probes)}
    out["finished"] = now_iso()
    out["wall_s"] = round(time.perf_counter() - t0, 1)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(out_path + ".tmp", "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    os.replace(out_path + ".tmp", out_path)
    for r in out["rows"]:
        print("[marks] %-16s seed %-3s lam_rel=%-7.0e d_lam=%-8.4g sup_lev=%-9.4g "
              "spread=%-8.4g d_lam*log=%-8.4g n_lev_thm=%-8.4g ess_lev=%.4f "
              "spearman=%s" % (benchmark, seed, r["lam_rel"], r["d_lam"],
                               r["sup_lev"], r["spread"], r["d_lam_log"],
                               r["n_lev_thm"], r["ess_lev"], r["spearman_lev_K"]))
    return out


# --------------------------------------------------------------------- export

FLAT_KEYS = ["benchmark", "seed", "step", "phase1_ckpt_hash", "dia_type",
             "hidden_dim", "handover_l2re", "run_n",
             "pool_kind", "pool_M", "pool_seed", "probe_key", "nys_L", "nys_k",
             "k_used", "lam_rel", "spearman_lev_K", "d_lam", "d_lam_over_k",
             "sup_lev", "mean_lev", "lev_spread", "ess_lev", "ess_lev_unfloored",
             "rung", "flat_64", "flat_128", "flat_256", "flat_last",
             "eig_slope_16_256", "ess_res2", "ess_res2_unfloored", "ess_kd",
             "K_max_over_median", "r2_top1pct_mass", "r2_max_over_median",
             "spearman_r2_K", "res_rms_total", "res_loss_handover",
             "recon_trusted", "recon_spearman", "nystrom_wall_s",
             "probe_wall_s", "reading_wall_s",
             "lev_batch_d_lam", "lev_batch_sup_lev", "lev_batch_lev_spread",
             "lev_batch_spearman_lev_K", "lev_batch_cov_frac_lev",
             "lev_batch_cov_frac_sobol", "lev_batch_batch_distinct_lev",
             "lev_batch_ess_lev_draw", "lev_batch_pool_size", "lev_batch_wall_s",
             "sled_rho_floor", "sled_n_floor", "sled_n_scored",
             "sled_batch_distinct", "sled_ess", "sled_cov_frac", "sled_wall_s",
             "res_rms_family", "K_family_share"]

MARKS_KEYS = ["benchmark", "seed", "lam_rel", "step", "phase1_ckpt_hash",
              "handover_l2re", "pool_size", "pool_seed", "probe_key", "probes",
              "nys_L", "nys_k", "n_params", "d_lam", "d_lam_log", "sup_lev",
              "mean_lev", "spread", "top_eval", "spectral_flatness", "k_used",
              "ess_lev", "ess_rad1", "spearman_lev_K", "n_lev_thm",
              "nystrom_wall_s"]
# summary: median over seeds of these, per (benchmark, lam_rel)
MARKS_MEDIANS = ["handover_l2re", "d_lam", "d_lam_log", "sup_lev", "mean_lev",
                 "spread", "top_eval", "spectral_flatness", "ess_lev", "ess_rad1",
                 "spearman_lev_K", "n_lev_thm"]


def long_rows(rows):
    """One CSV row per (checkpoint, pool, k, lam_rel)."""
    out = []
    for r in rows:
        if not r or "benchmark" not in r:
            continue
        for lr in LAM_RELS:
            tag = f"{lr:.0e}"
            rec = {"lam_rel": lr}
            for key in FLAT_KEYS:
                if key == "lam_rel":
                    continue
                v = r.get(f"{key}@{tag}", r.get(key))
                if isinstance(v, dict):
                    v = json.dumps({a: (round(b, 6) if isinstance(b, float) else b)
                                    for a, b in v.items()})
                rec[key] = v
            rec["rung"] = rung_of(r.get(LADDER_READING))   # always the 1e-4 rung
            out.append(rec)
    return out


def _fmt(v, p=4):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"%.{p}g" % v
    return str(v)


def _med_range(vals, p=3):
    vals = [v for v in vals if isinstance(v, (int, float))]
    if not vals:
        return "-"
    if len(vals) == 1:
        return _fmt(vals[0], p)
    return (f"{_fmt(statistics.median(vals), p)} "
            f"({_fmt(min(vals), p)}-{_fmt(max(vals), p)})")


def _order(names):
    """Benchmarks in BENCHMARKS order, anything else after them."""
    known = [b for b in BENCHMARKS if b in names]
    return known + sorted(n for n in names if n not in BENCHMARKS)


def export(rows, out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    rows = [r for r in rows if r and "benchmark" in r]
    rank = {b: i for i, b in enumerate(_order({r["benchmark"] for r in rows}))}
    rows.sort(key=lambda r: (rank[r["benchmark"]], r.get("seed", -1),
                             str(r.get("pool_kind")), r.get("pool_M", 0)))
    longs = long_rows(rows)
    # the CSVs keep every k (the column says which); the tables below are the
    # k = NYS_K reading only: the flatness at a fixed index depends on k, so
    # rows of different k are not comparable
    longs_all, longs = longs, [r for r in longs if int(r.get("nys_k") or 0) == NYS_K]
    with open(os.path.join(out_dir, "readings_all.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FLAT_KEYS if "lam_rel" in FLAT_KEYS
                           else ["lam_rel"] + FLAT_KEYS, extrasaction="ignore")
        w.writeheader()
        for r in longs_all:
            w.writerow(r)
    for bench in _order({r["benchmark"] for r in longs_all}):
        sub = [r for r in longs_all if r["benchmark"] == bench]
        with open(os.path.join(out_dir, f"readings_{bench}.csv"), "w",
                  newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(sub[0].keys()),
                               extrasaction="ignore")
            w.writeheader()
            for r in sub:
                w.writerow(r)

    # ---- readings_all.md
    at4 = [r for r in longs if r["lam_rel"] == 1e-4]
    benches = _order({r["benchmark"] for r in at4})
    lines = ["# Handover readings — lam_rel = 1e-4, L = 1,024, k = 256", "",
             "Median (min–max) over seeds, per pool.  `rungs` counts the fixed "
             "ladder: <= 0.75 leverage · 0.75–0.85 hedge · >= 0.85 cover.", ""]
    hdr = ["benchmark", "pool", "seeds", "Spearman(l,K)", "d_lam", "lev_spread",
           "ESS_lev", "flat_256", "eig slope", "SLED n_floor/n", "rungs"]
    lines += ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for bench in benches:
        pools = sorted({(r["pool_kind"], r["pool_M"]) for r in at4
                        if r["benchmark"] == bench})
        for pool, pool_M in pools:
            sub = [r for r in at4 if r["benchmark"] == bench
                   and r["pool_kind"] == pool and r["pool_M"] == pool_M]
            rungs = {}
            for r in sub:
                rungs[r["rung"]] = rungs.get(r["rung"], 0) + 1
            frac = " ".join(f"{key} {val}/{len(sub)}"
                            for key, val in sorted(rungs.items(),
                                                   key=lambda kv: -kv[1]))
            sled = [r["sled_n_floor"] / r["run_n"] for r in sub
                    if r.get("sled_n_floor") is not None and r.get("run_n")]
            lines.append("| " + " | ".join([
                bench, f"{pool} (M={pool_M})", str(len(sub)),
                _med_range([r["spearman_lev_K"] for r in sub]),
                _med_range([r["d_lam"] for r in sub]),
                _med_range([r["lev_spread"] for r in sub]),
                _med_range([r["ess_lev"] for r in sub]),
                _med_range([r["flat_256"] for r in sub]),
                _med_range([r["eig_slope_16_256"] for r in sub]),
                _med_range(sled), frac]) + " |")
    lines += ["", "## Per checkpoint (lam_rel = 1e-4)", ""]
    hdr2 = ["benchmark", "seed", "step", "ckpt", "pool", "Spearman", "rung",
            "d_lam", "sup_lev", "spread", "ESS_lev", "flat_256", "handover l2re",
            "wall s"]
    lines += ["| " + " | ".join(hdr2) + " |", "|" + "---|" * len(hdr2)]
    for r in at4:
        lines.append("| " + " | ".join([
            r["benchmark"], str(r["seed"]), str(r["step"]),
            str(r["phase1_ckpt_hash"])[:12], f"{r['pool_kind']} (M={r['pool_M']})",
            _fmt(r["spearman_lev_K"]), str(r["rung"]), _fmt(r["d_lam"]),
            _fmt(r["sup_lev"]), _fmt(r["lev_spread"]), _fmt(r["ess_lev"]),
            _fmt(r["flat_256"]), _fmt(r["handover_l2re"]),
            _fmt(r["reading_wall_s"])]) + " |")
    with open(os.path.join(out_dir, "readings_all.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    # ---- ladder_verdict.md
    v = ["# Ladder — the rung of every checkpoint", "",
         "> The reading is **Spearman(l̂, κ̂) at λ_rel = 1e-4**.  Rungs (fixed "
         "thresholds): **≤ 0.75 leverage** · **0.75–0.85 hedge** · **≥ 0.85 "
         "cover**.  SLED's Sobol floor holds round(clamp(ρ, 0, 1) · n) of the "
         "n points.", ""]
    v += ["## Per checkpoint", "",
          "| benchmark | seed | step | pool | Spearman(l̂,κ̂)@1e-4 | rung |",
          "|---|---|---|---|---|---|"]
    for bench in benches:
        for r in [x for x in at4 if x["benchmark"] == bench]:
            v.append(f"| {bench} | {r['seed']} | {r['step']} | "
                     f"{r['pool_kind']} (M={r['pool_M']}) | "
                     f"{_fmt(r['spearman_lev_K'])} | {r['rung']} |")
    v += ["", "### Rung counts per benchmark (all pools)", "",
          "| benchmark | seeds×pools | leverage | hedge | cover |",
          "|---|---|---|---|---|"]
    for bench in benches:
        sub = [r for r in at4 if r["benchmark"] == bench]
        c = {key: sum(1 for r in sub if r["rung"] == key)
             for key in ("leverage", "hedge", "cover")}
        v.append(f"| {bench} | {len(sub)} | {c['leverage']} | {c['hedge']} | "
                 f"{c['cover']} |")
    with open(os.path.join(out_dir, "ladder_verdict.md"), "w") as fh:
        fh.write("\n".join(v) + "\n")
    print(f"\nwrote {out_dir}/readings_all.csv ({len(longs_all)} rows, "
          f"{len(longs)} at k={NYS_K}), readings_all.md, ladder_verdict.md")


def marks_long_rows(marks):
    """One row per (benchmark, seed, lam_rel) from the per-checkpoint JSONs."""
    out = []
    for m in marks:
        if not m or "rows" not in m:
            continue
        for r in m["rows"]:
            out.append({**{key: m.get(key) for key in
                           ("benchmark", "seed", "step", "phase1_ckpt_hash")}, **r})
    return out


def export_marks(marks, out_dir=OUT_DIR):
    """marks_all.csv and marks_summary.csv (median over seeds)."""
    os.makedirs(out_dir, exist_ok=True)
    longs = marks_long_rows(marks)
    rank = {b: i for i, b in enumerate(_order({r["benchmark"] for r in longs}))}
    longs.sort(key=lambda r: (rank[r["benchmark"]], r["seed"], -r["lam_rel"]))
    with open(os.path.join(out_dir, "marks_all.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MARKS_KEYS, extrasaction="ignore")
        w.writeheader()
        for r in longs:
            w.writerow(r)
    # one summary row per benchmark, estimator size and lam_rel: rows of
    # another pool or (L, k) (a smoke run) are never pooled with the real ones
    groups = {}
    for r in longs:
        groups.setdefault((r["benchmark"], r["pool_size"], r["nys_L"], r["nys_k"],
                           r["lam_rel"]), []).append(r)
    summary = []
    for (bench, pool_size, L, k, lam_rel), sub in groups.items():
        rec = {"benchmark": bench, "lam_rel": lam_rel, "pool_size": pool_size,
               "nys_L": L, "nys_k": k, "n_seeds": len(sub),
               "seeds": " ".join(str(r["seed"]) for r in sub)}
        for key in MARKS_MEDIANS:
            vals = [r[key] for r in sub if isinstance(r.get(key), (int, float))]
            rec[key] = statistics.median(vals) if vals else None
        rec["n_lev_thm_at_median_d_lam"] = (n_lev_thm(rec["d_lam"])
                                            if rec["d_lam"] is not None else None)
        summary.append(rec)
    cols = (["benchmark", "lam_rel", "pool_size", "nys_L", "nys_k", "n_seeds",
             "seeds"] + MARKS_MEDIANS + ["n_lev_thm_at_median_d_lam"])
    with open(os.path.join(out_dir, "marks_summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in summary:
            w.writerow(r)
    print(f"\nwrote {out_dir}/marks_all.csv ({len(longs)} rows), "
          f"marks_summary.csv ({len(summary)} rows)")
    for r in summary:
        if r["lam_rel"] == 1e-4 and r["nys_k"] == NYS_K:
            print(f"  {r['benchmark']:<16} lam_rel=1e-4  seeds {r['n_seeds']}  "
                  f"sup_lev {_fmt(r['sup_lev'])}  n_lev_thm {_fmt(r['n_lev_thm'])}  "
                  f"d_lam*log {_fmt(r['d_lam_log'])}")


def collect(out_dir=OUT_DIR):
    return [read_json(p, {})
            for p in sorted(glob.glob(os.path.join(out_dir, "*_seed*_M*.json")))
            if not os.path.basename(p).startswith("marks_")]


def collect_marks(out_dir=OUT_DIR):
    return [read_json(p, {})
            for p in sorted(glob.glob(os.path.join(out_dir, "marks_*_seed*_M*.json")))]


def _set_out_dir(path):
    global OUT_DIR
    OUT_DIR = path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS),
                    choices=list(BENCHMARKS))
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="default: every seed with a phase-1 checkpoint")
    ap.add_argument("--phase1_dir", default=None,
                    help="directory with seed<k>/phase1.pkl; {benchmark} is replaced "
                         "by the benchmark name (default <benchmark>/runs/phase1)")
    ap.add_argument("--run_n", type=int, default=RUN_N,
                    help="batch size of the run pool (M = 4 run_n) and of the "
                         "lev_batch / sled batches (default 2^14, the sampler "
                         "comparison's n)")
    ap.add_argument("--out", default=OUT_DIR,
                    help="output directory (default runs/readings)")
    ap.add_argument("--pool_M", type=int, default=DIAG_POOL_M,
                    help="size of the diagnostics pool")
    ap.add_argument("--L", type=int, default=NYS_L, help="Nystrom landmarks")
    ap.add_argument("--k", type=int, default=NYS_K,
                    help="Nystrom rank (the tables keep k = 256 only)")
    ap.add_argument("--no_lev_batch", action="store_true",
                    help="skip the lev batch (coverage block)")
    ap.add_argument("--lev_batch_unstacked", action="store_true",
                    help="also draw the lev batch with the first-equation `lev` rule "
                         "of the sampler comparison")
    ap.add_argument("--marks", action="store_true",
                    help="compute the batch-size marks (pinned 65,536-point pool, "
                         "unstacked estimator) instead of the readings")
    ap.add_argument("--force", action="store_true", help="recompute existing JSONs")
    ap.add_argument("--export_only", action="store_true",
                    help="only rebuild the tables (readings and marks) from the "
                         "JSONs in --out")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny pools and Nystrom sizes, for testing the script")
    args = ap.parse_args()
    # resolve paths before load() changes the working directory
    _set_out_dir(os.path.abspath(args.out))
    if args.phase1_dir is not None:
        args.phase1_dir = os.path.abspath(args.phase1_dir)

    if args.export_only:
        rows, marks = collect(OUT_DIR), collect_marks(OUT_DIR)
        if rows:
            export(rows, OUT_DIR)
        if marks:
            export_marks(marks, OUT_DIR)
        if not rows and not marks:
            print(f"[readings] no JSONs under {OUT_DIR}")
        return

    what = "marks" if args.marks else "readings"
    for benchmark in args.benchmarks:
        p1_dir = phase1_dir_of(benchmark, args.phase1_dir)
        seeds = args.seeds
        if seeds is None:
            seeds = sorted(int(os.path.basename(d)[4:]) for d in
                           glob.glob(os.path.join(p1_dir, "seed*"))
                           if os.path.exists(os.path.join(d, "phase1.pkl")))
        if not seeds:
            print(f"[{what}] {benchmark}: no checkpoint under {p1_dir}")
            continue
        mods, prev = load(benchmark)
        try:
            for seed in seeds:
                print(f"\n{'=' * 70}\n=== {what} {benchmark} seed {seed} "
                      f"===\n{'=' * 70}")
                ck_path = os.path.join(p1_dir, f"seed{seed}", "phase1.pkl")
                try:
                    if args.marks:
                        checkpoint_marks(benchmark, mods, seed, ck_path, args)
                    else:
                        checkpoint_rows(benchmark, mods, seed, ck_path, args)
                except FileNotFoundError as exc:
                    print(f"[{what}] no checkpoint for {benchmark} seed {seed}: {exc}")
                except Exception:
                    traceback.print_exc()
        finally:
            os.chdir(prev)
    if args.marks:
        export_marks(collect_marks(OUT_DIR), OUT_DIR)
    else:
        export(collect(OUT_DIR), OUT_DIR)


if __name__ == "__main__":
    main()
