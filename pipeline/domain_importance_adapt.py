import time
from functools import partial

import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from jax import lax, jit, grad, random
from jax.flatten_util import ravel_pytree

class BaseSampler:
    def __init__(self, batch_size, seed):
        self.batch_size = batch_size
        self.key = random.PRNGKey(seed)

    def __iter__(self):
        return self

    def __next__(self):
        "Generate one batch of data"
        self.key, subkey = random.split(self.key)
        #keys = random.split(subkey)
        return self.data_generation(self.key)

    def __getitem__(self, index):
        return next(self)

    def data_generation(self, key):
        raise NotImplementedError("Subclasses should implement this!")

    def resample(self, *args, **kwargs):
        return None

class UniformSampler(BaseSampler):
    def __init__(self, dom, batch_size, seed):
        super().__init__(batch_size, seed)
        self.dom = dom
        self.dim = dom.shape[0]

    def data_generation(self, key):
        "Generates data containing batch_size samples"
        batch = random.uniform(
            key,
            shape=(self.batch_size, self.dim),
            minval=self.dom[:, 0],
            maxval=self.dom[:, 1],
        )
        return batch

class IC_Axis_Sampler(BaseSampler):
    def __init__(
        self,
        init_dict: dict,        # e.g. {"u": u0, "v": v0}
        axes_dict: dict,        # must contain "coords"
        batch_size: int,
        mode: str = "random",   # "random" or "full"
        seed: int = 1234,
    ):
        super().__init__(batch_size, seed)
        self.mode = mode

        self.coords = axes_dict["coords"]

        self.fields = init_dict   # dict[str, array]

    def __iter__(self):
        return self

    def __next__(self):
        if self.mode == "full":
            return self._full_batch()
        else:
            return self._random_batch()

    # -------------------------
    # Random batch
    # -------------------------
    def _random_batch(self, key):
        idx = random.choice(
            key,
            self.coords.shape[0],
            shape=(self.batch_size,),
        )

        batch = {
            "coords": self.coords[idx, :]
        }

        for name, field in self.fields.items():
            batch[name] = field[idx]

        return batch

    def _random_batch(self):
        self.key, subkey = random.split(self.key)
        keys = random.split(subkey, self.num_devices)
        return self._random_batch(keys)

    def _full_batch(self):
        batch = {"coords": self.coords}
        for name, field in self.fields.items():
            batch[name] = field
        return batch
     
class GridSampler(BaseSampler):
    def __init__(self, dom, n_points):
        """
        Initialize GridSampler for creating uniform grids over domain.
        
        Args:
            dom: Domain boundaries, shape (dim, 2) where dom[:, 0] is lower bounds
                 and dom[:, 1] is upper bounds
            n_points: List or array of number of points per dimension
            seed: Random seed for grid generation
        """
        self.dom = dom
        self.dim = dom.shape[0]
        self.n_points = jnp.array(n_points)
        
        # Calculate total batch size from grid dimensions
        batch_size = int(jnp.prod(self.n_points))
        super().__init__(batch_size, 1234)
        
        # Pre-generate the grid (immutable in JAX philosophy)
        self._grid_points = self._generate_grid()
    
    def _generate_grid(self):
        """Generate the complete grid of points."""
        # Create 1D arrays for each dimension
        grids_1d = [
            jnp.linspace(self.dom[i, 0], self.dom[i, 1], self.n_points[i])
            for i in range(self.dim)
        ]
        
        # Create meshgrid - JAX uses 'ij' indexing by default in meshgrid
        grids = jnp.meshgrid(*grids_1d, indexing='ij')
        
        # Stack and reshape to (n_total_points, dim)
        points = jnp.stack(grids, axis=-1).reshape(-1, self.dim)
        
        return points
    
    def __iter__(self):
        """Make the sampler iterable."""
        return self
    
    def __next__(self):
        return self._grid_points
    
    def data_generation(self, key):
        return self._grid_points

class SobolSampler:
    """Static scrambled-Sobol point set over dom (d-dim box); resample() is a
    no-op. Static like GridSampler, but low-discrepancy instead of tensorized."""

    def __init__(self, dom, batch_size, seed=0):
        from scipy.stats import qmc

        dom_np = np.asarray(dom)
        self.dom = dom
        self.batch_size = int(batch_size)
        eng = qmc.Sobol(d=dom_np.shape[0], scramble=True, seed=seed)
        # round UP to the next power of two for balance, then truncate
        m = int(np.ceil(np.log2(max(self.batch_size, 2))))
        pts = eng.random_base2(m=m)[: self.batch_size]
        self.sampled = jnp.asarray(qmc.scale(pts, dom_np[:, 0], dom_np[:, 1]))
        self.lambdas_sampled = jnp.ones(self.batch_size)

    def __iter__(self):
        return self

    def __next__(self):
        return self.sampled

    def get_lambdas(self):
        return self.lambdas_sampled

    def prime(self, *args, **kwargs):
        pass

    def resample(self, *args, **kwargs):
        return self.sampled


@partial(jit, static_argnums=(0,))
def _las_resample_core(
    model, params, weights, samples, key, beta, tau, lower_bound, upper_bound
):
    def total_sampler_loss(coords):
        return jnp.sum(model.r_losses_pp(params, coords, weights))

    gradient_coords = grad(total_sampler_loss)(samples)

    scaler = jnp.sqrt(
        jnp.sum((gradient_coords + 1e-16) ** 2, axis=-1, keepdims=True)
    )
    gradient_coords = gradient_coords / scaler

    key, subkey = random.split(key)
    noise = random.normal(subkey, shape=samples.shape)

    samples = (
        samples
        + tau * gradient_coords
        + beta * jnp.sqrt(2.0 * tau) * noise
    )

    samples = jnp.clip(samples, lower_bound, upper_bound)

    return samples, key


@partial(jit, static_argnums=(0,))
def _r3_resample_core(model, params, weights, samples, fresh, key):
    losses = model.r_losses_pp(params, samples, weights)

    mask = losses > jnp.mean(losses)
    keep_count = jnp.sum(mask)

    n = samples.shape[0]
    idx = jnp.arange(n)

    # Put kept samples first, preserving their original order.
    order = jnp.argsort(jnp.where(mask, idx, n + idx))
    kept_first = jnp.take(samples, order, axis=0)

    fresh_idx = jnp.maximum(idx - keep_count, 0)
    refill = jnp.take(fresh, fresh_idx, axis=0)
    keep_selector = (idx < keep_count).reshape((n,) + (1,) * (samples.ndim - 1))
    samples = jnp.where(keep_selector, kept_first, refill)

    key, _ = random.split(key)
    return samples, key
      
class LASSampler:
    def __init__(
        self,
        base_sampler,
        beta=0.2,
        tau=0.002,
        seed: int = 1234,
    ):
        """
        JAX version of the PyTorch LAS sampler.

        Args:
            base_sample: Initial coordinates, shape (N, D)
            lower_bound: Lower domain bound, shape (D,) or scalar
            upper_bound: Upper domain bound, shape (D,) or scalar
            beta: Langevin noise coefficient
            tau: Step size
            L_iter: Number of LAS iterations
            seed: Random seed
        """
        self.base_sampler = base_sampler
        self.base_sample = next(iter(base_sampler))  # (n_devices, batch_per_device, dim)
        self.beta = beta
        self.tau = tau
        self.key = random.PRNGKey(seed)
        self.iter = 0

    
    def __iter__(self):
        """Make the sampler iterable."""
        return self
    
    def __next__(self):
        """
        Returns current sample and increments counter.
        Automatically triggers resampling when counter hits interval.
        
        Returns:
            Current sample points, shape (n_devices, batch_per_device, dim)
        """        
        return self.base_sample

    def resample(self, model, params, weights=None):
        self.base_sample, self.key = _las_resample_core(
            model,
            params,
            weights,
            self.base_sample,
            self.key,
            self.beta,
            self.tau,
            self.base_sampler.dom[:, 0],
            self.base_sampler.dom[:, 1],
        )

        self.iter += 1
        return self.base_sample

class R3Sampler:
    def __init__(
        self,
        base_sampler,
        seed: int = 1234,
    ):
        """
        R3 sampler matching the PyTorch implementation.

        Args:
            base_sample: Initial sample, shape (1, N, D)
            base_sampler: Sampler that can generate random points in-domain
            seed: Random seed
        """
        self.base_sample = next(base_sampler)  # (1, N, D)
        self.base_sampler = base_sampler
        self.key = random.PRNGKey(seed)

    def __iter__(self):
        return self

    def __next__(self):
        return self.base_sample

    def resample(self, model, params, weights=None):
        """
        Keep points with loss > mean(loss), refill the rest with fresh random points.

        Assumes:
            model.sampler_loss(params, batch, weights) returns shape (N,)
            for batch shape (1, N, D).

        Returns:
            Updated sample of shape (1, N, D)
        """
        samples = self.base_sample
        fresh = next(self.base_sampler)

        if fresh.shape[0] < samples.shape[0]:
            raise ValueError(
                f"base_sampler returned only {fresh.shape[0]} fresh points, "
                f"but {samples.shape[0]} may be needed"
            )

        self.base_sample, self.key = _r3_resample_core(
            model,
            params,
            weights,
            samples,
            fresh[: samples.shape[0], ...],
            self.key,
        )
        return self.base_sample

class vRBA:
    def __init__(
        self,
        eval_sampler,
        batch_size,
        max_iter,
        k=0.0,
        c=1.0,
        eta=0.01,
        gamma=0.999,
        max_RBA0=None,
        cap_RBA=20.0,
        seed: int = 1234,
    ):
        self.eval_sample = next(eval_sampler)
        self.batch_size = batch_size
        self.k = float(k)
        self.c = float(c)
        self.eta = float(eta)
        self.key = random.PRNGKey(seed)
        self.N_total = self.eval_sample.shape[0]

        if max_RBA0 is None:
            max_RBA0 = eta / (1.0 - gamma)
        self.max_RBA0 = float(max_RBA0)

        denom = 50_000.0 if max_iter > 50_000 else max_iter/2
        self.step_RBA = float(
            (cap_RBA - self.max_RBA0) / (max_iter / denom - 1.0)
        )
        self.step = 0

        self.lambdas = jnp.zeros((self.N_total,))

        # Placeholders. prime() or resample() will set these before the first
        # model.step ever sees them.
        self.train_idx = jnp.arange(self.batch_size)
        self.sampled = self.eval_sample[self.train_idx]
        self.lambdas_sampled = self.lambdas[self.train_idx]

    def __iter__(self):
        return self

    def __next__(self):
        return self.sampled

    def get_lambdas(self):
        return self.lambdas_sampled

    def prime(self, model, params, weights=None):
        """Call once before the training loop.
        After this, next(sampler) and get_lambdas() are valid for the first step."""
        self._draw_and_update(model, params, weights)

    def resample(self, model, params, weights=None):
        """Call after each model.step.
        After this, next(sampler) and get_lambdas() are valid for the next step."""
        self._draw_and_update(model, params, weights)
        return self.sampled

    # --- internals --------------------------------------------------------

    def _draw_and_update(self, model, params, weights):
        # (1) Draw the batch we're about to serve from current lambdas.
        self.train_idx, self.key = self._draw_indices(self.lambdas, self.key)
        self.sampled = self.eval_sample[self.train_idx]

        # (2) Compute residuals on that batch at current params.
        residuals = model.r_losses_pp(params, self.sampled, weights)

        # (3) Update lambdas at those indices; cache fresh values for the loss.
        self.lambdas, self.lambdas_sampled, self.step = self._apply_lambda_update(
            self.lambdas, self.train_idx, residuals, self.step
        )

    @partial(jit, static_argnums=(0,))
    def _apply_lambda_update(self, lambdas, train_idx, residuals, step):
        # residuals is the weighted squared-norm per point, shape (B,)
        r_mag = jnp.sqrt(lax.stop_gradient(residuals.reshape(-1)) + 1e-12)

        max_rba = self.max_RBA0 + (self.step_RBA * step) // 50000.0
        gamma = 1.0 - self.eta / max_rba

        q_it = r_mag / (jnp.max(r_mag) + 1e-12)

        lambdas_old = lambdas[train_idx]
        new_lambdas = gamma * lambdas_old + self.eta * q_it
        lambdas = lambdas.at[train_idx].set(new_lambdas)

        return lambdas, new_lambdas, step + 1

    @partial(jit, static_argnums=(0,))
    def _draw_indices(self, lambdas, key):
        lambdas_k = jnp.power(lambdas, self.k)
        lambdas_k = lambdas_k / (jnp.mean(lambdas_k) + 1e-12) + self.c
        probs = lambdas_k / (jnp.sum(lambdas_k) + 1e-12)

        key, subkey = random.split(key)
        train_idx = random.choice(
            subkey,
            a=self.N_total,
            shape=(self.batch_size,),
            replace=True,
            p=probs,
        )
        return train_idx, key
        
class RADSampler:
    """
    Residual-based Adaptive Distribution (RAD) sampler.

    Expected behavior:
    - next(sampler) returns the current sampled batch
    - sampler.resample(model, params, weights) updates the sampled batch

    Assumptions:
    - eval_sampler yields candidate samples with shape (1, N, d) or (N, d)
    - self.sampled is stored with shape (1, K, d) or (K, d)
    - model.sampler_loss(params, candidate_batch, weights) returns shape (N,)
      where candidate_batch has shape (N, d)
    """

    def __init__(
        self,
        eval_sampler,
        sampled,
        seed: int = 1234,
        k=1.0,
        c=0.0,
    ):
        self.eval_sampler = eval_sampler
        self.sampled = sampled
        self.key = random.PRNGKey(seed)
        self.k = k
        self.c = c

        if sampled.ndim == 3:
            self.sample_size = sampled.shape[1]
            self._return_with_leading_axis = True
        elif sampled.ndim == 2:
            self.sample_size = sampled.shape[0]
            self._return_with_leading_axis = False
        else:
            raise ValueError(
                f"`sampled` must have shape (1, K, d) or (K, d), got shape {sampled.shape}"
            )

    def __iter__(self):
        return self

    def __next__(self):
        return self.sampled

    @partial(jit, static_argnums=(0, 1))
    def _resample_core(self, model, params, candidate_samples, weights, key):
        """
        Heavy resampling kernel.

        Args:
            model: training model object; must provide sampler_loss(...)
            params: current model params
            candidate_samples: shape (1, N, d) or (N, d)
            weights: passed through to model.sampler_loss
            key: JAX PRNG key

        Returns:
            new_sampled: shape matched to self.sampled layout
            new_key
        """
        had_leading_axis = (candidate_samples.ndim == 3)
        cand = candidate_samples[0] if had_leading_axis else candidate_samples  # (N, d)

        # Residual-based pointwise losses on the candidate pool.
        losses = model.r_losses_pp(params, cand, weights)

        # Error weighting.
        loss_pow = jnp.power(losses, self.k)
        err_eq = loss_pow / (jnp.mean(loss_pow) + 1e-12) + self.c
        probs = err_eq / (jnp.sum(err_eq) + 1e-12)
        # Sanitize probabilities.
        n = probs.shape[0]
        probs = jnp.nan_to_num(
            probs,
            nan=1.0 / n,
            posinf=0.0,
            neginf=0.0,
        )
        probs = probs / (jnp.sum(probs) + 1e-12)

        # Sample indices.
        new_key, subkey = random.split(key)
        idx = random.choice(
            subkey,
            a=n,
            shape=(self.sample_size,),
            replace=True,
            p=probs,
        )

        # Gather selected points.
        new_sampled = jnp.take(cand, idx, axis=0)  # (K, d)

        # Restore expected output shape.
        if self._return_with_leading_axis:
            new_sampled = new_sampled[None, :, :]  # (1, K, d)

        return new_sampled, new_key

    def resample(self, model, params, weights=None):
        """
        Refresh the internal sampled batch using RAD.

        This fetches a fresh candidate pool from eval_sampler, scores it with
        model.sampler_loss(...), builds sampling probabilities, and resamples
        self.sample_size points with replacement.
        """
        candidate_samples = next(self.eval_sampler)

        new_sampled, new_key = self._resample_core(
            model,
            params,
            candidate_samples,
            weights,
            self.key,
        )

        self.sampled = new_sampled
        self.key = new_key
        return self.sampled
    


def get_sampler(config, dom):
    """Construct the DIA sampler specified by config.dia.type."""
    if config.dia.type == "uniform":
        return iter(UniformSampler(dom, config.dia.batch_size, config.seed))

    if config.dia.type == "grid":
        return GridSampler(dom, config.dia.batch_sizes)

    if config.dia.type == "sobol":
        return SobolSampler(dom, config.dia.batch_size, seed=config.seed)

    if config.dia.type == "rad":
        base_sample  = next(iter(UniformSampler(dom, config.dia.batch_size, config.seed)))
        eval_sampler = iter(UniformSampler(dom, config.dia.batch_size_eval, config.seed))
        return RADSampler(sampled=base_sample, eval_sampler=eval_sampler, seed=config.seed)

    if config.dia.type == "las":
        base_sampler = UniformSampler(dom, config.dia.batch_size, seed=config.seed)
        return LASSampler(base_sampler=base_sampler, beta=config.dia.beta, tau=config.dia.tau, seed=config.seed)

    if config.dia.type == "R3":
        base_sampler = iter(UniformSampler(dom, config.dia.batch_size, seed=config.seed))
        return R3Sampler(base_sampler=base_sampler, seed=config.seed)

    if config.dia.type == "vRBA":
        eval_sampler = iter(UniformSampler(dom, config.dia.batch_size_eval, seed=config.seed))
        first_order_steps = config.optim.first_order_steps
        if isinstance(first_order_steps, (list, tuple)):
            first_order_steps = first_order_steps[0]
        return vRBA(eval_sampler=eval_sampler, batch_size=config.dia.batch_size, max_iter=first_order_steps, seed=config.seed)

    raise ValueError(f"Unknown DIA type: {config.dia.type}")


# =============================================================================
# Frozen second-order (quasi-Newton) batch samplers
# =============================================================================
# The second-order phase optimizes ONE frozen batch. These samplers build it and
# speak the same protocol as the first-order samplers above: `resample(model,
# params, weights)` constructs the batch at the current parameters,
# `next(sampler)` returns it and `get_lambdas()` returns the served per-point
# weights. They never move points during the quasi-Newton phase; a refreshing
# protocol (config.qn_batch.refresh_every) simply calls resample() again, see
# train.second_order_step.
#
# Samplers (config.qn_batch.type):
#     uniform  fresh i.i.d. uniform points, unit weights
#     sobol    scrambled-Sobol point set (low discrepancy), unit weights
#     kd       kernel diagonal: pi ~ K(x) = ||grad_theta r(x)||^2 (the NTK
#              diagonal, Hutchinson probes); the paper's "Kd-w"
#     lev      Nystrom ridge leverage l_lambda(x) of the residual-gradient kernel
#              (L uniform landmarks, top-k eigenpairs, one JVP per eigenpair over
#              the pool, lambda = lam_rel * top eigenvalue)
#     sled     SLED: leverage draws next to a scrambled-Sobol floor whose mass is
#              the measured Spearman(l_hat, kappa_hat) of the checkpoint the
#              batch is built at (stacked estimator over all residual families)
#     rad      Residual-based Adaptive Distribution, pi ~ |r|^k/mean(|r|^k) + c,
#              k = 1, c = 1 (Wu et al. 2023); the frozen counterpart of RADSampler
#     vrba     vRBA attention: uniform draw, served point weights = the EMA steady
#              state lambda = max_rba(k_anneal) * |r|/max|r| (vRBA's convention,
#              NOT sqrt(omega); `weighted` is ignored)
#     pinnacle_k  PINNACLE-K (Lau et al., ICLR 2024): greedy K-means++ seeding on
#              the convergence-degree embedding lt^{-1/2} V^T Theta(Z_ref, x) r(x)
#              of the pool (Nystrom landmarks as lev, ALL eigenpairs, stacked
#              residual); n distinct pool points, unit weights, no floor
#     pinnacle_s  PINNACLE-S: the same embedding; n pool points drawn without
#              replacement with probability ~ alpha(x) = ||P(x)||^2, unit weights
#
# Refresh rules (config.qn_batch.refresh_every = p) redraw the batch every p
# quasi-Newton steps; the optimizer's memory is kept unless
# config.qn_batch.reset_memory is set (then it is re-initialised at every redraw).
#
# Pool: M = pool_mult * n fresh uniform points; floored proposal
# pi = (1-floor) s/sum(s) + floor/M; n draws with replacement; importance
# weights omega = 1/(M pi). A Sobol floor (sled) realises the floor mass as
# round(floor*n) scrambled-Sobol points with weight 1 next to n - n_fl draws from
# the UNFLOORED proposal carrying omega = 1/(M pi_s).
#
# WEIGHT CONVENTION: the loss computes mean((r_i * point_weights_i)**2), so the
# served weights are sqrt(omega_i) and the effective linear weight on r_i**2 is
# omega_i. The raw omega is kept in `sampler.info["omega"]`.

WEIGHT_CONVENTION = "squared"          # loss squares the served point weights
NYS_L, NYS_K, NYS_LAM_REL = 1024, 256, 1e-4   # Nystrom defaults (lev / sled)
PROBES, PROBE_KEY = 8, 7               # Hutchinson probes; default key of the readings pool
GRAD_CHUNK = 32                        # landmarks per per-sample-Jacobian block
# vRBA EMA constants — the defaults of vRBA, so the one-shot steady state
# lambda = max_rba * |r|/max|r| matches what the running sampler converges to
# under a constant residual field.
VRBA_ETA, VRBA_GAMMA, VRBA_CAP = 0.01, 0.999, 20.0
# RAD defaults (Wu et al. 2023): k=1, c=1 is the configuration the paper
# reports and the one Kiyani et al. 2025 resample with.
RAD_K, RAD_C = 1.0, 1.0
# PINNACLE (Lau et al., ICLR 2024; reference code github.com/apivich-h/pinnacle,
# deepxde_al_patch/al/eigval.py with weight_method="alignment", the setting of
# every PINNACLE run in its al_pinn.sh). The code keeps every eigenpair of the
# reference Gram with lambda > 1e-8 (absolute, float32); here the cut is relative
# to the top eigenvalue, in float64. It seeds min(pool, 5 * budget) K-means++
# centres, which fixes sklearn's n_local_trials = 2 + int(log(n_clusters)).
PIN_EIG_TOL = 1e-10
PIN_CANDIDATE_MULT = 5
# PINNACLE-S: the reference code's do_sampling draws successively WITHOUT
# replacement with probability ~ ||P(z)||^2 + 1e-9 (random.choices with the drawn
# weight set to 0) and keeps the first n draws. Here: Gumbel-top-n, which has
# exactly that distribution; the 1e-9 guard is taken relative to the mean alpha
# (alpha is float64 here and scaled differently from the code's float32 values).
PIN_P_EPS = 1e-9


# ------------------------------------------------------------------ kernels

def vrba_max_rba(k_anneal, anneal_max_iter):
    """vRBA's annealed weight cap at global step `k_anneal`
    (mirrors vRBA.__init__ + _apply_lambda_update).

    Clamped at cap_RBA: the running sampler derives step_RBA from the cap but
    never applies it, so an uncapped value would make the served weight scale
    depend on the phase-1 budget (10 -> 18 after 2,000 SOAP steps but 30 -> 70
    after 20,000), which would confound any comparison across budgets.
    """
    max_rba0 = VRBA_ETA / (1.0 - VRBA_GAMMA)
    mi = float(anneal_max_iter if anneal_max_iter else 50_000)
    denom = 50_000.0 if mi > 50_000 else mi / 2.0
    step_rba = (VRBA_CAP - max_rba0) / (mi / denom - 1.0)
    return min(VRBA_CAP, max_rba0 + (step_rba * float(k_anneal)) // 50_000.0)

def qn_ess(pi):
    """Normalized effective sample size of a proposal, in (0, 1]; 1 = uniform."""
    pi = jnp.asarray(pi)
    return float(1.0 / (pi.shape[0] * jnp.sum(pi ** 2)))

def raw_residual_fn(model):
    """f(params, X) -> signed per-point residual, shape (N,).

    Prefers an explicit `model.r_pp`; otherwise vmaps `model.r_net` over the
    batch columns. For multi-equation models (r_net returning a tuple/vector)
    the FIRST component is used — the interior PDE residual.
    """
    if hasattr(model, "r_pp"):
        return lambda p, X: jnp.asarray(model.r_pp(p, X))

    r_net = model.r_net

    def f(params, X):
        d = X.shape[1]
        cols = tuple(X[:, i] for i in range(d))
        out = jax.vmap(r_net, in_axes=(None,) + (0,) * d)(params, *cols)
        if isinstance(out, (tuple, list)):
            out = out[0]
        out = jnp.asarray(out)
        return out if out.ndim == 1 else out[:, 0]

    return f

def tree_rademacher(params, key):
    """Rademacher (+-1) pytree with the same structure/dtypes as `params`."""
    leaves, treedef = jax.tree_util.tree_flatten(params)
    keys = random.split(key, len(leaves))
    return jax.tree_util.tree_unflatten(
        treedef,
        [random.rademacher(k, leaf.shape, dtype=leaf.dtype)
         for k, leaf in zip(keys, leaves)],
    )

@partial(jit, static_argnums=(0,))
def _probe_sq(model, params, X, v):
    """One Hutchinson probe: ((d_theta r) . v)^2 per point."""
    f = raw_residual_fn(model)
    _, t = jax.jvp(lambda p: f(p, X), (params,), (v,))
    return t ** 2

def ntk_diag(model, params, X, n_probes, key):
    """Unbiased estimate of K(x_i) = ||grad_theta r(x_i)||^2, shape (N,)."""
    acc = jnp.zeros(X.shape[0])
    for _ in range(int(n_probes)):
        key, k = random.split(key)
        acc = acc + _probe_sq(model, params, X, tree_rademacher(params, k))
    return acc / float(n_probes)

def _timed_ntk_diag(model, params, X, n_probes, key):
    t0 = time.perf_counter()
    K = jax.block_until_ready(ntk_diag(model, params, X, n_probes, key))
    return K, time.perf_counter() - t0

def _clean_scores(s):
    """Non-negative, finite scores (NaN/inf -> 0)."""
    return jnp.maximum(jnp.nan_to_num(jnp.asarray(s), nan=0.0, posinf=0.0,
                                      neginf=0.0), 0.0)

def _proposal(scores, floor, M):
    """Floor-mixed proposal pi over the pool; degenerate scores -> uniform."""
    scores = jnp.maximum(jnp.nan_to_num(scores, nan=0.0, posinf=0.0), 0.0)
    total = jnp.sum(scores)
    base = jnp.where(total > 0, scores / jnp.where(total > 0, total, 1.0),
                     jnp.full(M, 1.0 / M))
    pi = (1.0 - floor) * base + floor / M
    return pi / jnp.sum(pi)

def nystrom_leverage(model, params, pool, L=NYS_L, k=NYS_K, lam_rel=NYS_LAM_REL,
                     chunk=64, eig_tol=1e-10):
    """Ridge leverage scores l_lambda(x) over the pool from a Nystrom
    approximation of the residual-gradient kernel K(x,x') = <grad r(x), grad r(x')>.

    The first L pool points (i.i.d. uniform -> uniform landmarks), per-sample
    gradients G (L,P) in chunks, K_LL = G G^T / L, top-k eigenpairs, parameter
    directions A = G^T V, and ONE JVP per eigenpair over the whole pool to get
    the eigenfunctions U (k,M); U is normalised to unit norm in the pool
    measure. lambda = lam_rel * top eigenvalue. Returns (lev (M,), d_lam, info).
    """
    r_fn = raw_residual_fn(model)
    _, unravel = ravel_pytree(params)
    pool = jnp.asarray(pool)
    L = int(min(L, pool.shape[0]))

    def r_scalar(p, z):
        return r_fn(p, z[None, :])[0]

    grad_flat = jax.jit(jax.vmap(
        lambda p, z: ravel_pytree(jax.grad(r_scalar)(p, z))[0], in_axes=(None, 0)))
    jvp_dir = jax.jit(lambda p, X, v: jax.jvp(lambda q: r_fn(q, X), (p,),
                                              (unravel(v),))[1])

    Z = pool[:L]
    G = np.concatenate([np.asarray(grad_flat(params, Z[i:i + chunk]))
                        for i in range(0, L, chunk)], axis=0)          # (L, P)
    K_LL = (G @ G.T) / L
    evals, V = np.linalg.eigh(K_LL)
    evals, V = evals[::-1][:k].copy(), V[:, ::-1][:, :k].copy()        # descending
    keep = evals > eig_tol * max(float(evals[0]), 0.0)                  # numerically-zero dirs
    evals, V = evals[keep], V[:, keep]
    A = G.T @ V                                                         # (P, k)
    U = np.stack([np.asarray(jvp_dir(params, pool, jnp.asarray(A[:, j])))
                  for j in range(len(evals))])                          # (k, M)
    U = U / np.sqrt(L * evals[:, None])
    U = U / np.sqrt((U ** 2).mean(axis=1, keepdims=True))               # unit norm, pool measure
    lam = float(lam_rel) * float(evals[0])
    filt = evals / (evals + lam)
    lev = (filt[:, None] * U ** 2).sum(0)                               # (M,) l_lambda(x)
    d_lam = float(filt.sum())
    return lev, d_lam, {"lam": lam, "lam_rel": float(lam_rel),
                        "k_used": int(len(evals)), "L": L, "k": int(k),
                        "top_eval": float(evals[0]),
                        "spectral_flatness": float(evals[-1] / evals[0])}


# --------------------------------------------- stacked residual (all families)
# The phase-2 objective is sum_e w_e * mean_i(omega_i r_e(x_i)^2) with the
# FROZEN loss-balancer weights w_e. The stacked, weight-scaled residual
#
#     stacked_residual(theta, X)  ->  (E*N,), row (e, x) = sqrt(w_e) r_e(x)
#
# (family-major: rows [e*N : (e+1)*N] belong to family e) is what the SLED
# sampler and the handover readings score, so the Gram is the Hessian-relevant
# G = sum_e w_e E[phi_e phi_e^T]; the per-POINT leverage is the sum over the
# point's E rows. `raw_residual_fn` above instead collapses a multi-equation
# model to its FIRST equation (the convention of the lev and kd samplers).
#
# Pool inner product: pmean(f) = f.sum() / N_points -- the sum over ROWS divided
# by the number of POINTS; the Nystrom normalisation uses it.

_IC_BC_HINTS = ("ic", "bc", "initial", "boundary")


def residual_families(model):
    """Ordered residual-family keys of `model` (keys of model.state.weights).

    A model may declare them as `residual_families` (in the order r_net returns
    them); otherwise they are the loss keys that are not initial / boundary
    conditions, in loss_fns order. Raises if the count does not match the
    number of residual components r_net returns -- a silent mismatch would
    scale the wrong family.
    """
    name = type(model).__name__
    fams = getattr(model, "residual_families", None)
    if fams is None:
        fams = tuple(k for k in model.loss_fns
                     if not any(h in k.lower() for h in _IC_BC_HINTS))
    e = _n_families(model)
    if len(fams) != e:
        raise ValueError(
            f"{name}: r_net returns {e} residual component(s) but "
            f"{len(fams)} family key(s) {tuple(fams)} were resolved; declare "
            f"`residual_families` on the model")
    missing = [k for k in fams if k not in model.state.weights]
    if missing:
        raise ValueError(f"{name}: family keys {missing} are not loss weights "
                         f"({sorted(model.state.weights)})")
    return tuple(fams)


def residual_matrix_fn(model):
    """f(params, X) -> (N, E) residual matrix, families in r_net order."""
    r_net = model.r_net

    def f(params, X):
        cols = tuple(X[:, i] for i in range(X.shape[1]))
        out = jax.vmap(r_net, in_axes=(None,) + (0,) * X.shape[1])(params, *cols)
        if isinstance(out, (tuple, list)):
            out = jnp.stack(out, axis=-1)
        out = jnp.asarray(out)
        return out[:, None] if out.ndim == 1 else out

    return f


def _n_families(model):
    d = int(model.config.arch.input_dim)
    probe = jnp.zeros((1, d), dtype=jnp.asarray(model.state.point_weights).dtype)
    return int(residual_matrix_fn(model)(model.state.params, probe).shape[1])


def stacked_residual_fn(model, weights=None, families=None):
    """f(params, X) -> (E*N,), row (e, x) = sqrt(w_e) * r_e(x; params).

    Family-major: rows [e*N : (e+1)*N] are family e, so the vector is laid out
    the same way for the pool, for a chunk of the pool and for the landmarks,
    and `per_point` can fold it back with a reshape.  `weights` defaults to the
    model's current (frozen) loss weights.
    """
    families = tuple(families or residual_families(model))
    weights = model.state.weights if weights is None else weights
    w = jnp.asarray([float(weights[k]) for k in families])
    sw = jnp.sqrt(jnp.maximum(w, 0.0))
    r_mat = residual_matrix_fn(model)

    def f(params, X):
        return (sw[None, :] * r_mat(params, X)).T.reshape(-1)

    return f


def per_point(v, n_points):
    """(E*N,) -> (N,): sum a stacked quantity over each point's E rows."""
    v = jnp.asarray(v)
    return v.reshape(-1, n_points).sum(0)


def hutchinson_ntk_diag(stacked, params, pool, n_probes=PROBES, key=None):
    """Per-POINT kernel diagonal K(x) = sum_e ||d/dtheta sqrt(w_e) r_e(x)||^2.

    One Rademacher probe on the STACKED residual gives all E families at once:
    E[(J v)^2] = ||J||^2 row-wise, then summed over the point's E rows.
    """
    key = random.PRNGKey(PROBE_KEY) if key is None else key
    m = pool.shape[0]
    flat, unravel = ravel_pytree(params)

    @jax.jit
    def probe_sq(v_flat):
        _, t = jax.jvp(lambda p: stacked(p, pool), (params,), (unravel(v_flat),))
        return per_point(t ** 2, m)

    acc = jnp.zeros(m)
    for _ in range(int(n_probes)):
        key, k = random.split(key)
        v = random.rademacher(k, flat.shape, dtype=flat.dtype).astype(flat.dtype)
        acc = acc + probe_sq(v)
    return np.asarray(acc / float(n_probes))


def nystrom_decomposition(stacked, params, pool, L=NYS_L, k=NYS_K,
                          eig_tol=1e-10, chunk=GRAD_CHUNK):
    """Nystrom decomposition of the stacked residual-gradient kernel.

    Landmarks are the FIRST L pool points (i.i.d. uniform -> uniform
    landmarks).  For E families the landmark Gram has L*E rows and is
    normalised by L (points), matching pmean:

        G      (L*E, P)   per-row parameter gradients at the landmarks
        K_LL = G G^T / L
        evals, V          top-k eigenpairs, numerically-null directions dropped
        A    = G^T V      (P, k) parameter directions
        U[j] = jvp(stacked(., pool), A[:, j])          one JVP per eigenpair
        U[j] /= sqrt(L * evals[j])                     Nystrom scaling
        U[j] /= sqrt(pmean(U[j]**2))                   unit norm, pool measure

    Returns (evals, U, A, scale, info) with U of shape (k_used, E*M) and
    `scale` the per-row product of the two scalings (A[:, j] evaluated at
    another theta needs the same scale to be comparable).
    """
    pool = jnp.asarray(pool)
    m = int(pool.shape[0])
    L = int(min(L, m))
    flat, unravel = ravel_pytree(params)

    def r_point(p, z):
        return stacked(p, z[None, :])                       # (E,)

    jac = jax.jit(jax.vmap(
        lambda p, z: jax.jacrev(lambda q: r_point(q, z))(p), in_axes=(None, 0)))
    jvp_dir = jax.jit(lambda p, X, v: jax.jvp(lambda q: stacked(q, X),
                                              (p,), (unravel(v),))[1])

    Z = pool[:L]
    rows = []
    for i in range(0, L, chunk):
        j = jac(params, Z[i:i + chunk])                     # pytree, leaves (c, E, ...)
        leaves = jax.tree_util.tree_leaves(j)
        c, e = leaves[0].shape[0], leaves[0].shape[1]
        rows.append(np.concatenate(
            [np.asarray(l).reshape(c * e, -1) for l in leaves], axis=1))
    G = np.concatenate(rows, axis=0)                        # (L*E, P)

    K_LL = (G @ G.T) / float(L)
    evals, V = np.linalg.eigh(K_LL)
    evals, V = evals[::-1][:k].copy(), V[:, ::-1][:, :k].copy()
    keep = evals > eig_tol * max(float(evals[0]), 0.0)
    evals, V = evals[keep], V[:, keep]
    A = G.T @ V                                             # (P, k_used)

    U = np.stack([np.asarray(jvp_dir(params, pool, jnp.asarray(A[:, j])))
                  for j in range(len(evals))])              # (k_used, E*M)
    s1 = 1.0 / np.sqrt(float(L) * evals[:, None])
    U = U * s1
    s2 = 1.0 / np.sqrt(np.maximum((U ** 2).sum(axis=1, keepdims=True) / m, 1e-300))
    U = U * s2
    scale = (s1 * s2).ravel()                               # (k_used,)

    info = {"L": L, "k": int(k), "k_used": int(len(evals)),
            "top_eval": float(evals[0]),
            "spectral_flatness": float(evals[-1] / evals[0]),
            "n_params": int(flat.size), "n_families": int(G.shape[0] // L)}
    return evals, U, A, scale, info


def leverage_from(evals, U, lam, n_points):
    """(per-point ridge leverage, d_lam) at ridge level `lam`."""
    filt = evals / (evals + lam)
    lev_rows = (filt[:, None] * U ** 2).sum(0)              # (E*M,)
    lev = np.asarray(per_point(jnp.asarray(lev_rows), n_points))
    return lev, float(filt.sum())


def pinnacle_embedding(stacked, params, pool, L=NYS_L, eig_tol=PIN_EIG_TOL,
                       chunk=GRAD_CHUNK, chunk_elems=2 ** 25):
    """PINNACLE's convergence-degree embedding of every pool point.

    `stacked(params, X) -> (E*N,)` is the stacked residual (family-major, E = 1
    for a single-residual model). Landmarks Z_ref are the FIRST L pool points
    (i.i.d. uniform, the lev rule's landmarks); G (L*E, P) holds their per-row
    parameter gradients, Theta_ref = G G^T is the landmark eNTK Gram with
    eigenpairs (lt_i, v_i), and every eigenpair with lt_i > eig_tol * lt_1 is
    kept (the reference code truncates nothing either). A pool point z embeds as

        P_i(z) = lt_i^{-1/2} v_i^T Theta(Z_ref, z) r(z)
               = lt_i^{-1/2} v_i^T G g(z),   g(z) = grad_theta 1/2 ||r(z)||^2

    i.e. the sum of the embeddings of the point's E residual rows, because a
    selected point brings all E rows into the loss. ||P(z)||^2 is the pointwise
    convergence degree alpha(z), ||sum_{z in Z} P(z)||^2 the set's alpha(Z).
    One per-point gradient per pool point (not one JVP per eigenpair) is what
    makes keeping all eigenpairs affordable. Returns (P (M, k_used) on device,
    evals, info).
    """
    pool = jnp.asarray(pool)
    m = int(pool.shape[0])
    L = int(min(L, m))
    flat, unravel = ravel_pytree(params)
    n_par = int(flat.size)

    jac_rows = jax.jit(jax.vmap(
        lambda q, z: jax.jacrev(lambda q_: stacked(unravel(q_), z[None, :]))(q),
        in_axes=(None, 0)))                                        # (c, E, P)
    loss_grad = jax.jit(jax.vmap(
        lambda q, z: jax.grad(
            lambda q_: 0.5 * jnp.sum(stacked(unravel(q_), z[None, :]) ** 2))(q),
        in_axes=(None, 0)))                                        # (c, P)

    Z = pool[:L]
    G = np.concatenate([np.asarray(jac_rows(flat, Z[i:i + chunk])).reshape(-1, n_par)
                        for i in range(0, L, chunk)], axis=0)       # (L*E, P)
    evals, V = np.linalg.eigh(G @ G.T)
    evals, V = evals[::-1].copy(), V[:, ::-1].copy()               # descending
    keep = evals > eig_tol * max(float(evals[0]), 0.0)
    evals, V = evals[keep], V[:, keep]
    W = jnp.asarray((V.T @ G) / np.sqrt(evals)[:, None])           # (k, P)

    c = max(1, min(m, int(chunk_elems) // n_par))
    pad = (-m) % c                                                 # one compiled shape
    X = jnp.concatenate([pool, pool[:pad]], axis=0) if pad else pool
    P = jnp.concatenate([loss_grad(flat, X[i:i + c]) @ W.T
                         for i in range(0, m + pad, c)], axis=0)[:m]
    P = jnp.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    return P, evals, {"L": L, "k_used": int(len(evals)),
                      "n_families": int(G.shape[0] // L), "n_params": n_par,
                      "eig_tol": float(eig_tol), "top_eval": float(evals[0]),
                      "spectral_flatness": float(evals[-1] / evals[0])}


@partial(jit, static_argnums=(1, 2))
def kmeanspp_greedy(X, n, n_trials, key):
    """Greedy K-means++ seeding of the rows of X on device: the algorithm of
    sklearn.cluster.kmeans_plusplus, which the PINNACLE code calls.

    First centre uniform; every further step draws `n_trials` candidates with
    probability ~ D^2 (squared distance to the nearest centre chosen so far) and
    keeps the one that lowers the potential sum D^2 the most. Returns the n row
    indices in the order they were chosen. A chosen row's D is set to exactly 0,
    so it is never drawn again; should every remaining row coincide with a
    centre, the draw continues uniformly over the rows not yet taken.
    """
    M = X.shape[0]
    sq = jnp.sum(X * X, axis=1)

    def dist_to(rows):                                  # (len(rows), M)
        return jnp.maximum(sq[None, :] + sq[rows][:, None] - 2.0 * (X[rows] @ X.T),
                           0.0)

    key, k0 = random.split(key)
    c0 = random.randint(k0, (1,), 0, M)
    D = dist_to(c0)[0].at[c0[0]].set(0.0)
    taken = jnp.zeros(M, dtype=bool).at[c0[0]].set(True)
    idx = jnp.zeros(n, dtype=jnp.int32).at[0].set(c0[0].astype(jnp.int32))

    def body(t, carry):
        D, taken, idx, key = carry
        key, sub = random.split(key)
        w = jnp.where(jnp.sum(D) > 0, D, jnp.where(taken, 0.0, 1.0))
        cdf = jnp.cumsum(w)
        u = random.uniform(sub, (n_trials,), dtype=cdf.dtype) * cdf[-1]
        cand = jnp.clip(jnp.searchsorted(cdf, u), 0, M - 1)
        Dc = jnp.minimum(D[None, :], dist_to(cand))
        Dc = Dc.at[jnp.arange(n_trials), cand].set(0.0)
        best = jnp.argmin(jnp.sum(Dc, axis=1))
        c = cand[best]
        return Dc[best], taken.at[c].set(True), idx.at[t].set(c.astype(jnp.int32)), key

    _, _, idx, _ = lax.fori_loop(1, n, body, (D, taken, idx, key))
    return idx


# ----------------------------------------------------------------- samplers

class FrozenBatchSampler:
    """Base of the frozen QN batch samplers: candidate pool, per-sampler
    scores, floored proposal, draw with replacement, importance weights.

    Subclasses implement `scores(model, params, weights, pool, seed)` and
    return (scores, probe_wall_s, extra_info); `_build` returns
    (batch, served_weights, omega, info). `resample(model, params, weights)`
    builds the batch (`seed`/`k_anneal` may be overridden per call, which is
    what the refresh loop does); next(sampler) returns it; get_lambdas()
    returns the served point weights (sqrt(omega)); `info` holds the
    construction statistics of the last build.
    """
    rule = None
    needs_loss_weights = False     # score uses model.r_losses_pp
    uses_probes = False            # score spends time in ntk_diag
    reports_probe_time = True      # False -> probe_wall_s reported as null
    sobol_floor_allowed = True
    # True -> the sampler measures Spearman(lev, K) at construction, so the
    # floor mass can be set from it (floor_rho; lev / sled only)
    provides_rho = False
    default_rad_k, default_rad_c = RAD_K, RAD_C
    # construction defaults a subclass may fix (sled: Sobol floor of mass rho,
    # stacked estimator)
    default_floor_source = "uniform"
    default_stacked = False
    default_floor_rho = False

    def __init__(self, dom, batch_size, seed=0, *, pool_mult=4, probes=PROBES,
                 floor=0.2, weighted=True, floor_source=None,
                 k_anneal=0, anneal_max_iter=None, lam_rel=NYS_LAM_REL,
                 nys_L=NYS_L, nys_k=NYS_K, rad_k=None, rad_c=None,
                 stacked=None, floor_rho=None):
        floor_source = self.default_floor_source if floor_source is None else floor_source
        if floor_source not in ("uniform", "sobol"):
            raise ValueError(f"unknown floor_source: {floor_source}")
        if not 0.0 <= float(floor) <= 1.0:
            raise ValueError(f"floor must lie in [0, 1], got {floor}")
        if floor_source == "sobol" and not self.sobol_floor_allowed:
            raise ValueError(f"floor_source='sobol' is undefined for rule {self.rule!r}")
        self.dom = dom
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.pool_mult = int(pool_mult)
        self.probes = int(probes)
        self.floor = float(floor)
        # floor_rho: the floor mass is not a constant but THE MEASURED
        # Spearman(l_hat, kappa_hat) of this very checkpoint, clamped to [0, 1].
        # rho ~ 1 (a flat, cover-type problem) -> an almost pure Sobol batch;
        # rho ~ 0 (concentrated leverage) -> almost pure leverage.
        self.floor_rho = bool(self.default_floor_rho if floor_rho is None else floor_rho)
        if self.floor_rho and not self.provides_rho:
            raise ValueError(
                f"floor_rho needs a sampler that measures Spearman(lev, K) when "
                f"it builds the batch; rule {self.rule!r} does not (use lev or sled)")
        self.weighted = bool(weighted)
        self.floor_source = floor_source
        self.k_anneal = int(k_anneal)
        self.anneal_max_iter = anneal_max_iter
        self.lam_rel, self.nys_L, self.nys_k = float(lam_rel), int(nys_L), int(nys_k)
        # per-rule knob defaults (class attributes) unless given explicitly
        self.rad_k = float(self.default_rad_k if rad_k is None else rad_k)
        self.rad_c = float(self.default_rad_c if rad_c is None else rad_c)
        # stacked=True scores the weight-scaled stacked residual of all
        # families instead of raw_residual_fn (the FIRST equation of a
        # multi-residual model)
        self.stacked = bool(self.default_stacked if stacked is None else stacked)
        self.sampled = None
        self.lambdas_sampled = None
        self.info = None
        self._step = None          # QN step of the current (re)build; None = the hand-over batch

    # -- sampler protocol ------------------------------------------------
    def __iter__(self):
        return self

    def __next__(self):
        if self.sampled is None:
            raise RuntimeError("call resample(model, params, weights) first")
        return self.sampled

    def get_lambdas(self):
        return self.lambdas_sampled

    def prime(self, model, params, weights=None):
        return self.resample(model, params, weights)

    def resample(self, model, params, weights=None, *, seed=None, k_anneal=None, step=None):
        seed = self.seed if seed is None else int(seed)
        k_anneal = self.k_anneal if k_anneal is None else int(k_anneal)
        # the QN step a refresh / reselection happens at (None or 0: the
        # hand-over batch); only the PINNACLE selectors use it (their key)
        self._step = None if step is None else int(step)
        if (self.needs_loss_weights or self.stacked) and weights is None:
            raise ValueError(f"{self.rule} needs loss_weights (model.state.weights)")
        batch, served, omega, info = self._build(model, params, weights, seed, k_anneal)
        info["omega"] = np.asarray(omega)
        self.sampled = batch
        self.lambdas_sampled = served
        self.info = info
        return batch

    # -- construction ----------------------------------------------------
    def _base_info(self):
        # ess / pi_omega_sum default to the uniform-draw values so that every
        # sampler — including the ones that override _build — reports the same
        # fields; the scored path in _build overwrites them with measurements.
        n = self.batch_size
        return {"rule": self.rule, "weighted": bool(self.weighted), "n_batch": int(n),
                "pool_mult": None, "probes": None, "floor": None,
                "floor_source": self.floor_source,
                "probe_wall_s": 0.0, "batch_distinct": int(n),
                # pool_size stays None for uniform / sobol: they generate n
                # points directly and never draw from a candidate pool
                "pool_size": None,
                "ess": 1.0, "pi_omega_sum": 1.0, "batch_wall_s": None,
                # floor_rho only: the measured rho and whether it set the floor
                "floor_rho": bool(getattr(self, "floor_rho", False)),
                "rho_floor": None}

    def scores(self, model, params, weights, pool, seed):
        raise NotImplementedError

    def _build(self, model, params, weights, seed, k_anneal):
        n, M = self.batch_size, self.pool_mult * self.batch_size
        info = self._base_info()
        t_batch = time.perf_counter()
        pool = next(iter(UniformSampler(self.dom, M, seed)))
        probe_wall_s = 0.0 if self.reports_probe_time else None
        scores, pw, extra = self.scores(model, params, weights, pool, seed)
        if pw is not None:
            probe_wall_s = pw
        info.update(extra)
        floor_eff = info.get("rad_floor", self.floor)   # rad carries its own floor
        if self.floor_rho:
            # the sampler reads its own hardness off this checkpoint: rho is
            # already computed by the leverage diagnostics, so this costs nothing
            rho = (info.get("nystrom") or {}).get("spearman_lev_K")
            if rho is None:
                raise RuntimeError(
                    f"rule {self.rule!r} did not report spearman_lev_K, so "
                    "floor_rho has no floor to use (is scipy installed?)")
            floor_eff = float(min(max(float(rho), 0.0), 1.0))
            info.update(floor_rho=True, rho_floor=float(rho))
        # every sampler WITHOUT floor_rho keeps its static floor byte for byte
        fl_used = floor_eff if self.floor_rho else float(self.floor)

        def build_pi(fl):
            """The sampler's proposal at floor level fl (rad carries its own
            c-floor)."""
            if "rad_floor" in info:
                return _proposal(scores, info["rad_floor"], M)
            return _proposal(jnp.asarray(scores), fl, M)

        weighted = self.weighted
        if self.floor_source == "uniform":
            pi = build_pi(fl_used)
            idx = random.choice(random.PRNGKey(seed + 2), M, shape=(n,),
                                replace=True, p=pi)
            batch = pool[idx]
            omega = 1.0 / (M * pi[idx]) if weighted else jnp.ones(n)
            n_distinct = int(np.unique(np.asarray(idx)).size)
        else:                                          # "sobol" floor
            n_fl = int(round(fl_used * n))
            n_sc = int(n) - n_fl
            pi = build_pi(0.0)                         # UNFLOORED scored pmf
            if n_sc > 0:
                idx = random.choice(random.PRNGKey(seed + 2), M, shape=(n_sc,),
                                    replace=True, p=pi)
                omega_sc = (1.0 / (M * pi[idx])) if weighted else jnp.ones(n_sc)
                scored_pts = pool[idx]
            else:
                idx = jnp.zeros((0,), dtype=jnp.int32)
                omega_sc = jnp.zeros((0,))
                scored_pts = pool[:0]
            s_pts = (SobolSampler(self.dom, n_fl, seed=seed + 3).sampled
                     if n_fl > 0 else pool[:0])
            batch = jnp.concatenate([scored_pts, s_pts], axis=0)
            omega = jnp.concatenate([omega_sc, jnp.ones(n_fl)], axis=0)
            n_distinct = int(np.unique(np.asarray(idx)).size) + n_fl
            info.update(n_floor=n_fl, n_scored=n_sc)

        s = np.asarray(scores)
        nys_stats = ({k_: info["nystrom"][k_] for k_ in
                      ("d_lam", "sup_lev", "lev_spread", "spearman_lev_K",
                       "k_used", "lam", "lam_rel", "nystrom_wall_s")}
                     if "nystrom" in info else {})
        info.update(pool_mult=int(self.pool_mult), pool_size=int(M),
                    floor=float(floor_eff),
                    probes=int(self.probes) if self.uses_probes else None,
                    probe_wall_s=probe_wall_s,
                    batch_wall_s=time.perf_counter() - t_batch,
                    # how many of the n draws are distinct pool points: the
                    # direct read-out of how far a proposal has collapsed
                    batch_distinct=n_distinct,
                    score_stats={"min": float(s.min()), "median": float(np.median(s)),
                                 "max": float(s.max()), **nys_stats},
                    score_max_over_median=float(s.max() / (np.median(s) + 1e-300)),
                    ess=qn_ess(pi),
                    # identity: sum_i pi_i * omega_i(pool) == 1 over the support
                    pi_omega_sum=float(jnp.sum(jnp.where(pi > 0, 1.0 / M, 0.0))))
        return batch, jnp.sqrt(omega), omega, info


class FrozenUniformSampler(FrozenBatchSampler):
    """Fresh i.i.d. uniform points, unit weights (the status-quo control)."""
    rule = "uniform"
    sobol_floor_allowed = False

    def _build(self, model, params, weights, seed, k_anneal):
        t0 = time.perf_counter()
        batch = next(iter(UniformSampler(self.dom, self.batch_size, seed)))
        ones = jnp.ones(self.batch_size)
        info = self._base_info()
        info["batch_wall_s"] = time.perf_counter() - t0
        return batch, ones, ones, info


class FrozenSobolSampler(FrozenBatchSampler):
    """Static scrambled-Sobol point set, unit weights."""
    rule = "sobol"
    sobol_floor_allowed = False

    def _build(self, model, params, weights, seed, k_anneal):
        t0 = time.perf_counter()
        batch = SobolSampler(self.dom, self.batch_size, seed=seed).sampled
        ones = jnp.ones(self.batch_size)
        info = self._base_info()
        info["batch_wall_s"] = time.perf_counter() - t0
        return batch, ones, ones, info


class KernelDiagonalSampler(FrozenBatchSampler):
    """kd: pi ~ K(x) = ||grad_theta r(x)||^2 via Hutchinson probes (the NTK
    diagonal of the first equation)."""
    rule = "kd"
    uses_probes = True

    def scores(self, model, params, weights, pool, seed):
        K, pw = _timed_ntk_diag(model, params, pool, self.probes,
                                random.PRNGKey(seed + 1))
        return K, pw, {}


class FrozenRADSampler(FrozenBatchSampler):
    """rad: pi ~ |r|^k/mean(|r|^k) + c == floor-mixed proposal on |r|^k with
    floor = c/(1+c); k = 1, c = 1 by default (Wu et al. 2023).

    The exponent acts on the residual MAGNITUDE, as RAD is defined (Wu et al.
    2023, eq. 2), so k = 1 is a residual-proportional density. `r_losses_pp`
    returns the loss-weighted SQUARED residual, hence the sqrt; the constant
    loss weight cancels in the normalised proposal for a single-residual model
    and makes the score the weighted per-point residual norm otherwise.
    The floor is derived from c, not taken from the `floor` knob.
    """
    rule = "rad"
    needs_loss_weights = True
    reports_probe_time = False
    sobol_floor_allowed = False

    def scores(self, model, params, weights, pool, seed):
        r_abs = jnp.sqrt(_clean_scores(model.r_losses_pp(params, pool, weights)))
        return jnp.power(r_abs, self.rad_k), None, {
            "rad_k": self.rad_k, "rad_c": self.rad_c,
            "rad_floor": self.rad_c / (1.0 + self.rad_c)}


class FrozenVRBASampler(FrozenBatchSampler):
    """vrba: uniform draw (the sampler's k=0 habit) + steady-state attention
    weights lambda = max_rba(k_anneal) * |r|/max|r|, served directly."""
    rule = "vrba"
    needs_loss_weights = True
    reports_probe_time = False
    sobol_floor_allowed = False

    def _build(self, model, params, weights, seed, k_anneal):
        n, M = self.batch_size, self.pool_mult * self.batch_size
        info = self._base_info()
        t_batch = time.perf_counter()
        pool = next(iter(UniformSampler(self.dom, M, seed)))
        r2 = _clean_scores(model.r_losses_pp(params, pool, weights))
        idx = random.choice(random.PRNGKey(seed + 2), M, shape=(n,),
                            replace=True)              # k=0 habit: uniform
        batch = pool[idx]
        r_mag = jnp.sqrt(r2[idx] + 1e-12)
        q = r_mag / (jnp.max(r_mag) + 1e-12)
        max_rba = vrba_max_rba(k_anneal, self.anneal_max_iter)
        lam = max_rba * q                              # served directly
        info.update(pool_mult=int(self.pool_mult), pool_size=int(M),
                    probes=None, probe_wall_s=None,
                    batch_distinct=int(np.unique(np.asarray(idx)).size),
                    k_anneal=int(k_anneal), max_rba=float(max_rba),
                    served_convention="vrba_lambda", weighted=False,
                    score_stats={"min": float(lam.min()),
                                 "median": float(np.median(np.asarray(lam))),
                                 "max": float(lam.max())},
                    batch_wall_s=time.perf_counter() - t_batch,
                    ess=1.0)                           # draw IS uniform
        return batch, lam, lam ** 2, info              # served lam; omega = lam^2


class LeverageSampler(FrozenBatchSampler):
    """lev: pi ~ ridge leverage l_lambda(x) (Nystrom). The Hutchinson K is
    computed only for the Spearman diagnostic and is what probe_wall_s times.
    Unstacked by default (first equation of a multi-residual model)."""
    rule = "lev"
    uses_probes = True
    provides_rho = True            # floor_rho reads spearman_lev_K from _diagnostics

    def _stacked_fn(self, model, weights):
        return stacked_residual_fn(model, weights)

    def _leverage(self, model, params, pool, weights=None):
        t0 = time.perf_counter()
        if self.stacked:
            stacked = self._stacked_fn(model, weights)
            m = int(pool.shape[0])
            evals, U, _, _, nys = nystrom_decomposition(
                stacked, params, pool, L=self.nys_L, k=self.nys_k)
            lam = float(self.lam_rel) * float(evals[0])
            lev, d_lam = leverage_from(evals, U, lam, m)
            nys = {**nys, "lam": lam, "lam_rel": float(self.lam_rel),
                   "stacked": True}
        else:
            lev, d_lam, nys = nystrom_leverage(model, params, pool, L=self.nys_L,
                                               k=self.nys_k, lam_rel=self.lam_rel)
        lev = np.maximum(np.nan_to_num(lev, nan=0.0, posinf=0.0), 0.0)
        return lev, d_lam, nys, time.perf_counter() - t0

    def _diagnostics(self, model, params, pool, seed, lev, d_lam, nys, wall,
                     weights=None):
        if self.stacked:
            t0 = time.perf_counter()
            K = hutchinson_ntk_diag(self._stacked_fn(model, weights), params,
                                    pool, self.probes, random.PRNGKey(seed + 1))
            pw = time.perf_counter() - t0
        else:
            K, pw = _timed_ntk_diag(model, params, pool, self.probes,
                                    random.PRNGKey(seed + 1))
        try:
            from scipy.stats import spearmanr
            rho = float(spearmanr(lev, np.asarray(K)).statistic)
        except Exception:
            rho = None
        return pw, {"nystrom": {**nys, "d_lam": d_lam, "sup_lev": float(lev.max()),
                                "lev_spread": float(lev.max() / max(d_lam, 1e-300)),
                                "mean_lev": float(lev.mean()),
                                "spearman_lev_K": rho, "nystrom_wall_s": wall}}

    def scores(self, model, params, weights, pool, seed):
        lev, d_lam, nys, wall = self._leverage(model, params, pool, weights)
        pw, extra = self._diagnostics(model, params, pool, seed, lev, d_lam, nys,
                                      wall, weights)
        return jnp.asarray(lev), pw, extra


class SLEDSampler(LeverageSampler):
    """sled: the leverage batch on a Sobol floor whose mass is measured.

    At the checkpoint the batch is built at, the ridge leverage l_hat and the
    kernel diagonal kappa_hat are estimated on the pool with the STACKED
    estimator (all residual families, sqrt(w_e)-scaled). Their Spearman rank
    correlation rho, clamped to [0, 1], is the floor mass: round(rho * n)
    scrambled-Sobol points at weight 1, the remaining n - round(rho * n) drawn
    from the unfloored leverage proposal with importance weights
    omega = 1/(M pi). rho -> 1 (leverage carries no information beyond the
    kernel diagonal; a flat problem) gives an almost pure Sobol batch, rho -> 0
    an almost pure leverage batch. No knob is tuned per benchmark.
    """
    rule = "sled"
    default_floor_source = "sobol"
    default_stacked = True
    default_floor_rho = True


class SLEDFixedFloorSampler(SLEDSampler):
    """sled_fixed: SLED with a FIXED Sobol-floor mass (the ablation a1).

    Everything is SLED's (stacked estimator, scrambled-Sobol floor, draws from
    the unfloored leverage proposal, omega = 1/(M pi)) except the floor mass:
    the `floor` knob phi instead of the measured rho. round(phi * n) Sobol
    points at weight 1, the other n - round(phi * n) leverage draws; phi = 0 is
    a pure leverage batch without any floor, phi = 1 a pure Sobol batch (the
    sobol rule in distribution; its scramble seed differs). rho is still
    measured (the leverage diagnostics compute it anyway) and recorded as
    `rho_measured`, the raw Spearman(l_hat, kappa_hat) the SLED rule would have
    used (before its clamp to [0, 1]).
    """
    rule = "sled_fixed"
    default_floor_rho = False

    def _build(self, model, params, weights, seed, k_anneal):
        batch, served, omega, info = super()._build(model, params, weights, seed,
                                                    k_anneal)
        info["rho_measured"] = (info.get("nystrom") or {}).get("spearman_lev_K")
        return batch, served, omega, info


class PinnacleKSampler(FrozenBatchSampler):
    """pinnacle_k: PINNACLE-K (Lau et al., ICLR 2024) as a frozen batch.

    PINNACLE picks training points that maximise the convergence degree
    alpha(Z) = sum_i lambda_i a_i(Z)^2 of the eNTK training dynamics. Every
    point of a fresh uniform pool (M = pool_mult * n, the other samplers' pool)
    gets the embedding P(z) of `pinnacle_embedding`; greedy K-means++ seeding on
    those vectors (`kmeanspp_greedy`, sklearn's algorithm, which the reference
    code calls) selects n DISTINCT pool points, favouring points that are both
    large (high alpha) and different from each other in eNTK feature space. The
    batch is served with unit weights (PINNACLE trains on an unweighted mean;
    `weighted` is ignored) and there is no floor.

    Faithful to the reference code: all eigenpairs (not top-k, so `nys_k` is
    ignored), eigenpairs of the unnormalised Gram, `alignment` scaling
    lambda^{-1/2}, the joint eNTK over ALL residual families (the stacked,
    sqrt(w_e)-scaled residual), the n_local_trials of seeding min(M, 5n)
    centres. What this study's setting reduces: the initial/boundary terms of
    these benchmarks are fixed grids or hard constraints, not sampled points, so
    PINNACLE's joint PDE/IC/BC budget allocation degenerates to collocation
    points only; the reference set Z_ref is the lev rule's L uniform landmarks
    instead of the current training set; the batch is selected once at the
    hand-over (a refresh reselects the whole batch from a fresh pool), not
    periodically with a random part of the old set retained.

    pinnacle_k_r500 reselects every 500 quasi-Newton steps (the optimizer's
    memory is kept): each reselection draws a fresh pool (the refresh seed) and
    seeds K-means++ with fold_in(PRNGKey(seed + 2), step), so its first batch is
    exactly the frozen pinnacle_k batch.
    """
    rule = "pinnacle_k"
    variant = "K"
    with_replacement = False        # K-means++ seeding: n DISTINCT pool points
    needs_loss_weights = True       # the stacked residual carries sqrt(w_e)
    reports_probe_time = False
    sobol_floor_allowed = False
    default_stacked = True

    def __init__(self, dom, batch_size, seed=0, *, floor=0.0, **kw):
        if float(floor) != 0.0:
            raise ValueError(f"rule {self.rule!r} selects its batch without a "
                             f"proposal floor; got floor={floor}")
        super().__init__(dom, batch_size, seed, floor=0.0, **kw)
        self.stacked = True

    def _build(self, model, params, weights, seed, k_anneal):
        n, M = self.batch_size, self.pool_mult * self.batch_size
        info = self._base_info()
        t_batch = time.perf_counter()
        pool = next(iter(UniformSampler(self.dom, M, seed)))
        t0 = time.perf_counter()
        P, evals, pin = pinnacle_embedding(stacked_residual_fn(model, weights),
                                           params, pool, L=self.nys_L)
        alpha = jax.block_until_ready(jnp.sum(P * P, axis=1))   # alpha(z), (M,)
        embed_wall = time.perf_counter() - t0
        t0 = time.perf_counter()
        key, key_rule = self._select_key(seed)
        idx, extra = self._select(P, alpha, int(n), key)
        idx = jax.block_until_ready(idx)
        select_wall = time.perf_counter() - t0
        batch = pool[idx]
        a = np.asarray(alpha)
        # set-level convergence degree ||sum_{z in Z} P(z)||^2 of the batch and
        # of an i.i.d. uniform n-subset of the same pool (its first n points)
        pin.update(variant=self.variant, **extra, select_key=key_rule,
                   with_replacement=self.with_replacement,
                   alpha_set_batch=float(jnp.sum(jnp.sum(P[idx], axis=0) ** 2)),
                   alpha_set_uniform=float(jnp.sum(jnp.sum(P[:n], axis=0) ** 2)),
                   alpha_pp_batch_mean=float(a[np.asarray(idx)].mean()),
                   alpha_pp_pool_mean=float(a.mean()),
                   embed_wall_s=embed_wall, select_wall_s=select_wall)
        ones = jnp.ones(n)
        info.update(pool_mult=int(self.pool_mult), pool_size=int(M), floor=0.0,
                    probe_wall_s=None, weighted=False, served_convention="unit",
                    batch_distinct=int(np.unique(np.asarray(idx)).size),
                    pinnacle=pin,
                    score_stats={"min": float(a.min()), "median": float(np.median(a)),
                                 "max": float(a.max())},
                    score_max_over_median=float(a.max() / (np.median(a) + 1e-300)),
                    # ESS of the alpha pmf over the pool: a concentration read-out
                    ess=qn_ess(_proposal(jnp.asarray(a), 0.0, M)),
                    batch_wall_s=time.perf_counter() - t_batch)
        return batch, ones, ones, info

    def _select_key(self, seed):
        """(key, how it was made) of the selection. The hand-over batch (step
        None or 0) keeps PRNGKey(seed + 2), so the first batch of a reselecting
        run IS the frozen batch; a reselection at QN step s > 0 uses
        fold_in(PRNGKey(base seed + 2), s)."""
        if not self._step:
            return random.PRNGKey(seed + 2), "PRNGKey(seed+2)"
        return (random.fold_in(random.PRNGKey(self.seed + 2), self._step),
                f"fold_in(PRNGKey(base_seed+2), {self._step})")

    def _select(self, P, alpha, n, key):
        """(pool indices of the n selected points, extra info): greedy
        K-means++ seeding."""
        M = int(P.shape[0])
        n_trials = 2 + int(np.log(min(M, PIN_CANDIDATE_MULT * n)))
        return kmeanspp_greedy(P, int(n), n_trials, key), {"kmeans_trials": n_trials}


class PinnacleSSampler(PinnacleKSampler):
    """pinnacle_s: PINNACLE-S (Lau et al., ICLR 2024), the sampling variant, as
    a frozen batch. Same embedding, pool, landmarks, eigenvalue cut and unit
    weights as pinnacle_k; the n points are drawn WITHOUT replacement with
    probability ~ alpha(z) = ||P(z)||^2 (+ PIN_P_EPS * mean alpha), as the
    reference code's do_sampling (successive random.choices draws with the drawn
    weight zeroed, the first n kept). Realised as Gumbel-top-n, which has
    exactly the distribution of successive sampling without replacement."""
    rule = "pinnacle_s"
    variant = "S"
    with_replacement = False

    def _select(self, P, alpha, n, key):
        a = _clean_scores(alpha)
        p = a + PIN_P_EPS * jnp.maximum(jnp.mean(a), 1e-300)
        g = jnp.log(p) + random.gumbel(key, p.shape, dtype=p.dtype)
        return lax.top_k(g, int(n))[1], {"p_rule": "alpha + 1e-9 * mean(alpha)",
                                         "draw": "successive, without replacement (Gumbel-top-n)"}


QN_SAMPLERS = {cls.rule: cls for cls in (
    FrozenUniformSampler, FrozenSobolSampler, KernelDiagonalSampler,
    FrozenRADSampler, FrozenVRBASampler, LeverageSampler, SLEDSampler,
    SLEDFixedFloorSampler, PinnacleKSampler, PinnacleSSampler)}
QN_TYPES = tuple(QN_SAMPLERS)
# knobs get_qn_sampler forwards from config.qn_batch to the sampler
QN_SAMPLER_KWARGS = ("pool_mult", "probes", "floor", "weighted", "lam_rel",
                     "nys_L", "nys_k", "rad_k", "rad_c")


def get_qn_sampler(config, dom, k_anneal=0, anneal_max_iter=None, n=None):
    """Construct the frozen-batch sampler specified by config.qn_batch
    (type, n, pool_mult, probes, floor, weighted, lam_rel, nys_L, nys_k,
    rad_k, rad_c). Returns None when there is no qn_batch section (or its type
    is None): the second-order phase then keeps the batch of the first-order
    sampler untouched. `n` is the batch size when
    config.qn_batch.n is absent."""
    qn = config.qn_batch if "qn_batch" in config else None
    qn_type = qn.type if qn is not None and "type" in qn else None
    if qn_type is None:
        return None
    if qn_type not in QN_SAMPLERS:
        raise ValueError(f"unknown QN batch type: {qn_type} (expected one of {QN_TYPES})")
    if dom is None:
        raise ValueError(f"qn_batch.type={qn_type!r} needs the sampling domain box; "
                         "pass train_and_evaluate(dom=...)")
    if "n" in qn and qn.n is not None:
        n = int(qn.n)
    elif n is None:
        n = int(config.dia.batch_size)
    kw = {k: qn[k] for k in QN_SAMPLER_KWARGS if k in qn and qn[k] is not None}
    return QN_SAMPLERS[qn_type](dom, int(n), seed=config.seed, k_anneal=k_anneal,
                                anneal_max_iter=anneal_max_iter, **kw)


def make_qn_batch(qn_type, model, params, dom, n, *, weighted=True, seed=0,
                  loss_weights=None, k_anneal=0, anneal_max_iter=None, **kw):
    """One-shot convenience for notebooks, tests and diagnostics: build the
    frozen batch of `qn_type` at `params` without going through a config.
    Returns (batch (n,d), point_weights (n,) = sqrt(omega), info)."""
    if qn_type not in QN_SAMPLERS:
        raise ValueError(f"unknown QN batch type: {qn_type} (expected one of {QN_TYPES})")
    sampler = QN_SAMPLERS[qn_type](dom, n, seed=seed, weighted=weighted,
                                   k_anneal=k_anneal, anneal_max_iter=anneal_max_iter, **kw)
    batch = sampler.resample(model, params, loss_weights)
    return batch, sampler.get_lambdas(), sampler.info


# ------------------------------------------------------ the study's rules
# The sampling rules compared in the paper, by name: the frozen-batch sampler
# (config.qn_batch.type), whether its importance weights are served, and the
# refresh period of the protocols that redraw the batch during the
# quasi-Newton phase (None = one frozen batch for the whole phase).
QN_RULES = {
    "uniform":      dict(type="uniform", weighted=False, refresh_every=None),
    "sobol":        dict(type="sobol",   weighted=False, refresh_every=None),
    "rad":          dict(type="rad",     weighted=False, refresh_every=None),
    "kd":           dict(type="kd",      weighted=True,  refresh_every=None),
    "lev":          dict(type="lev",     weighted=True,  refresh_every=None),
    "sled":         dict(type="sled",    weighted=True,  refresh_every=None),
    "uniform_r500": dict(type="uniform", weighted=False, refresh_every=500),
    "rad_r500":     dict(type="rad",     weighted=False, refresh_every=500),
    "vrba_r100":    dict(type="vrba",    weighted=False, refresh_every=100),
    # PINNACLE-K selects its batch: no proposal, hence no floor
    "pinnacle_k":   dict(type="pinnacle_k", weighted=False, refresh_every=None,
                         floor=0.0),
    # ablation a1: SLED with its measured floor mass rho replaced by a fixed
    # phi, carried by the floor knob (phi = 1 would be a Sobol batch, as sobol)
    "sled_fl0":     dict(type="sled_fixed", weighted=True, refresh_every=None,
                         floor=0.0),
    "sled_fl0.25":  dict(type="sled_fixed", weighted=True, refresh_every=None,
                         floor=0.25),
    "sled_fl0.5":   dict(type="sled_fixed", weighted=True, refresh_every=None,
                         floor=0.5),
    "sled_fl0.75":  dict(type="sled_fixed", weighted=True, refresh_every=None,
                         floor=0.75),
    # supplementary study: the refresh rules with the quasi-Newton memory reset
    # at every redraw (the optimizer state re-initialised at the current point)
    "rad_r500_reset":  dict(type="rad", weighted=False, refresh_every=500,
                            reset_memory=True),
    "vrba_r100_reset": dict(type="vrba", weighted=False, refresh_every=100,
                            reset_memory=True),
    # PINNACLE-S frozen; PINNACLE-K reselected every 500 steps (memory kept)
    "pinnacle_s":      dict(type="pinnacle_s", weighted=False, refresh_every=None,
                            floor=0.0),
    "pinnacle_k_r500": dict(type="pinnacle_k", weighted=False, refresh_every=500,
                            floor=0.0),
}


def configure_qn_batch(config, rule, n, pool_mult=4, probes=PROBES, floor=0.2,
                       lam_rel=NYS_LAM_REL, nys_L=NYS_L, nys_k=NYS_K,
                       refresh_every="rule"):
    """Set config.qn_batch to the named sampling rule (see QN_RULES) with a
    frozen batch of n points drawn from a candidate pool of pool_mult * n.

    The second-order phase of such a config starts from a phase-1 checkpoint,
    so the first-order sampler only has to exist: dia.type is set to uniform
    (RT trains its phase 1 with LAS). `refresh_every` overrides the rule's
    refresh period (None = frozen). Returns the config.
    """
    if rule not in QN_RULES:
        raise ValueError(f"unknown sampling rule {rule!r} (expected one of {tuple(QN_RULES)})")
    preset = QN_RULES[rule]
    config.dia.type = "uniform"
    config.qn_batch = qn = ml_collections.ConfigDict()
    qn.rule = rule
    qn.type = preset["type"]
    qn.weighted = bool(preset["weighted"])
    qn.n = int(n)
    qn.pool_mult = int(pool_mult)
    qn.probes = int(probes)
    qn.floor = float(preset.get("floor", floor))   # a preset may fix its own
    qn.lam_rel = float(lam_rel)
    qn.nys_L = int(nys_L)
    qn.nys_k = int(nys_k)
    period = preset["refresh_every"] if refresh_every == "rule" else refresh_every
    qn.refresh_every = None if period is None else int(period)
    # written only by the rules that set it, so every other rule's qn_batch (and
    # with it its run_key) is unchanged
    if preset.get("reset_memory"):
        qn.reset_memory = True
    return config
