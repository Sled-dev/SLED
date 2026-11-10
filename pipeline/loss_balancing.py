"""
Loss-balancing classes for PINN training.

Structure mirrors samplers.py:
  - module-level JIT kernels for heavy compute  (model is a static arg)
  - lightweight classes that own their mutable state
  - factory function: create_loss_balancer

Class hierarchy:
  FixedBalancer             — weights never change
  AdaptiveBalancer          — abstract base; owns momentum smoothing
    GradNormBalancer
    LRAnnealingBalancer
    ReLoBaLoBalancer        — also covers SoftAdapt
    GradNormMTLBalancer
"""
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, lax
from jax.tree_util import tree_map

import optax

def _grad_norm(loss_fn, params, batch, point_weights):
    """L2 norm of the gradient of a single loss w.r.t. params."""
    g = jax.grad(loss_fn)(params, batch, point_weights)
    return jnp.sqrt(
        jax.tree_util.tree_reduce(
            lambda acc, x: acc + jnp.sum(x * x),
            g,
            initializer=jnp.array(0.0),
        )
    )


@partial(jit, static_argnums=(0,))
def _gradnorm_core(model, params, batch, point_weights):
    loss_keys = list(model.loss_fns.keys())
    norms = jnp.stack([
        _grad_norm(fn, params, batch, point_weights)
        for fn in model.loss_fns.values()
    ])
    mean_norm = jnp.mean(norms)
    return {
        k: mean_norm / (norms[i] + 1e-5 * mean_norm)
        for i, k in enumerate(loss_keys)
    }


@partial(jit, static_argnums=(0,))
def _lrannealing_core(model, params, batch, point_weights):
    """
    Learning-rate annealing: weights ∝ max|∇L_res| / |∇L_i|.
    Requires 'res' to be a key in model.loss_fns.

    Does not work for methods with multiple residual losses
    """
    loss_keys = list(model.loss_fns.keys())
    if "res" not in loss_keys:
        raise ValueError("LR Annealing requires a 'res' loss in model.loss_fns")
    norms = {
        k: _grad_norm(fn, params, batch, point_weights)
        for k, fn in model.loss_fns.items()
    }
    max_res = norms["res"]
    return {k: max_res / (norms[k] + 1e-8) for k in loss_keys}


@partial(jit, static_argnums=(0,))
def _relobralo_core(model, params, batch, point_weights,
                    current_weights, l_dict, l0_dict,
                    T, rho_sample, alpha):
    losses    = model.losses(params, batch, point_weights)
    loss_keys = list(losses.keys())
    N         = len(loss_keys)

    loss_vec      = jnp.stack([losses[k]          for k in loss_keys])
    loss_last_vec = jnp.stack([l_dict[k]          for k in loss_keys])
    loss0_vec     = jnp.stack([l0_dict[k]         for k in loss_keys])
    w_vec         = jnp.stack([current_weights[k] for k in loss_keys])

    eps = 1e-12

    ratio_last = loss_vec / (loss_last_vec * T + eps)
    ratio_init = loss_vec / (loss0_vec     * T + eps)

    lambs_hat  = jax.lax.stop_gradient(jax.nn.softmax(ratio_last) * N)
    lambs0_hat = jax.lax.stop_gradient(jax.nn.softmax(ratio_init) * N)

    new_w_vec = jax.lax.stop_gradient(
        rho_sample       * alpha       * w_vec
        + (1 - rho_sample) * alpha       * lambs0_hat
        + (1 - alpha)                    * lambs_hat
    )

    new_weights = {k: new_w_vec[i] for i, k in enumerate(loss_keys)}
    new_l       = {k: loss_vec[i]  for i, k in enumerate(loss_keys)}
    return new_weights, new_l

@partial(jit, static_argnums=(0,))
def _softadapt_core(model, params, batch, point_weights,
                    current_weights, l_dict,
                    T, alpha):
    losses    = model.losses(params, batch, point_weights)
    loss_keys = list(losses.keys())
    N         = len(loss_keys)

    loss_vec      = jnp.stack([losses[k]          for k in loss_keys])
    loss_last_vec = jnp.stack([l_dict[k]          for k in loss_keys])
    w_vec         = jnp.stack([current_weights[k] for k in loss_keys])

    diff      = (loss_vec - loss_last_vec) * T
    lambs_hat = jax.lax.stop_gradient(jax.nn.softmax(diff) * N)

    new_w_vec = jax.lax.stop_gradient(
        alpha * w_vec + (1 - alpha) * lambs_hat
    )

    new_weights = {k: new_w_vec[i] for i, k in enumerate(loss_keys)}
    new_l       = {k: loss_vec[i]  for i, k in enumerate(loss_keys)}
    return new_weights, new_l


@partial(jit, static_argnums=(0,))
def _gradnorm_mtl_core(model, params, batch, point_weights,
                        current_weights, l0_dict, alpha):
    loss_keys = list(model.loss_fns.keys())

    g_vec = jnp.stack([
        _grad_norm(fn, params, batch, point_weights)
        for fn in model.loss_fns.values()
    ])

    losses = model.losses(params, batch, point_weights)
    L_vec  = jnp.stack([losses[k]          for k in loss_keys])
    L0_vec = jnp.stack([l0_dict[k]         for k in loss_keys])
    w_vec  = jnp.stack([current_weights[k] for k in loss_keys])

    eps = 1e-6

    G_vec = w_vec * g_vec
    G_avg = jnp.mean(G_vec)

    rel_L = L_vec / (L0_vec + eps)
    R_vec = rel_L / (jnp.mean(rel_L) + eps)
    G_hat = G_avg * (R_vec ** alpha)

    dLdw = jnp.sign(G_vec - G_hat) * g_vec
    return {k: dLdw[i] for i, k in enumerate(loss_keys)}

class FixedBalancer:
    """Default schedule: weights are constant throughout training."""

    def init(self, *_):
        pass

    def update(self, _model, _params, _batch, _point_weights, current_weights):
        return current_weights

class AdaptiveBalancer:
    """
    Base for all adaptive balancers.

    Subclasses implement _compute_raw_weights; this base applies the
    running-average momentum smoothing before returning, matching the
    apply_weights logic that used to live in TrainState.
    """

    def __init__(self, momentum):
        self.momentum = momentum

    def init(self, *_):
        pass

    def _compute_raw_weights(self, model, params, batch, point_weights, current_weights):
        raise NotImplementedError

    def update(self, model, params, batch, point_weights, current_weights):
        raw = self._compute_raw_weights(model, params, batch, point_weights, current_weights)
        smoothed = tree_map(
            lambda old_w, new_w: old_w * self.momentum + (1 - self.momentum) * new_w,
            current_weights,
            raw,
        )
        return lax.stop_gradient(smoothed)

class GradNormBalancer(AdaptiveBalancer):
    """Equal gradient-norm balancing (stateless beyond momentum)."""

    def _compute_raw_weights(self, model, params, batch, point_weights, _current_weights):
        return _gradnorm_core(model, params, batch, point_weights)

class LRAnnealingBalancer(AdaptiveBalancer):
    """Learning-rate annealing (stateless beyond momentum)."""

    def _compute_raw_weights(self, model, params, batch, point_weights, _current_weights):
        return _lrannealing_core(model, params, batch, point_weights)


class ReLoBaLoBalancer(AdaptiveBalancer):
    """Relative Loss Balancing with Residual Losses (ReLoBRaLo).

    rho is drawn as a Bernoulli sample each step with mean `rho_mean`,
    matching the published algorithm. Use rho_mean close to 1 (e.g. 0.999)
    so that the residual look-back to the initial loss happens only
    occasionally.
    """

    def __init__(self, momentum, T, rho_mean, alpha, seed=0):
        super().__init__(momentum)
        self.T        = T
        self.rho_mean = rho_mean
        self.alpha    = alpha
        self._key     = jax.random.PRNGKey(seed)
        self._l       = {}
        self._l0      = {}

    def init(self, model, params, batch, point_weights):
        losses = model.losses(params, batch, point_weights)
        for k, v in losses.items():
            self._l0[k] = v
            self._l[k]  = v

    def _compute_raw_weights(self, model, params, batch, point_weights, current_weights):
        # Bernoulli draw outside the jit; pass the scalar in.
        self._key, subkey = jax.random.split(self._key)
        rho_sample = jax.random.bernoulli(subkey, p=self.rho_mean).astype(jnp.float32)

        new_weights, new_l = _relobralo_core(
            model, params, batch, point_weights,
            current_weights, self._l, self._l0,
            self.T, rho_sample, self.alpha,
        )
        self._l = new_l
        return new_weights

class SoftAdaptBalancer(AdaptiveBalancer):
    """SoftAdapt loss balancing.

    Weights are an EMA between the previous step's weights and a
    softmax over (loss_i - loss_i_prev) * T.
    """

    def __init__(self, momentum, T, alpha):
        super().__init__(momentum)
        self.T     = T
        self.alpha = alpha
        self._l    = {}

    def init(self, model, params, batch, point_weights):
        losses = model.losses(params, batch, point_weights)
        for k, v in losses.items():
            self._l[k] = v

    def _compute_raw_weights(self, model, params, batch, point_weights, current_weights):
        new_weights, new_l = _softadapt_core(
            model, params, batch, point_weights,
            current_weights, self._l,
            self.T, self.alpha,
        )
        self._l = new_l
        return new_weights

class GradNormMTLBalancer(AdaptiveBalancer):
    """GradNorm-MTL: adapts weights via the gradient of the GradNorm loss."""

    def __init__(self, momentum, alpha, lr, init_weights):
        super().__init__(momentum)
        self.alpha     = alpha
        self.optimizer = optax.adam(learning_rate=lr)
        self.opt_state = self.optimizer.init(init_weights)
        self._l0       = {}

    def init(self, model, params, batch, point_weights):
        losses   = model.losses(params, batch, point_weights)
        self._l0 = dict(losses)

    def _compute_raw_weights(self, model, params, batch, point_weights, current_weights):
        g_dir = _gradnorm_mtl_core(
            model, params, batch, point_weights,
            current_weights, self._l0, self.alpha,
        )
        updates, self.opt_state = self.optimizer.update(g_dir, self.opt_state)
        return optax.apply_updates(current_weights, updates)


def create_loss_balancer(config, init_weights):
    """Construct the appropriate balancer from a weighting config."""
    scheme   = config.weighting.scheme
    momentum = config.weighting.momentum

    if scheme in ["gradnorm", "grad_norm"]:
        return GradNormBalancer(momentum)

    if scheme == "lrannealing":
        return LRAnnealingBalancer(momentum)

    if scheme == "relobralo":
        return ReLoBaLoBalancer(
            momentum=momentum,
            T=config.weighting.T,
            rho_mean=config.weighting.rho,
            alpha=config.weighting.alpha,
        )
    if scheme == "softadapt":
        return SoftAdaptBalancer(
            momentum=momentum,
            T=config.weighting.T,
            alpha=config.weighting.alpha,
        )

    if scheme == "gradnorm_mtl":
        return GradNormMTLBalancer(
            momentum=momentum,
            alpha=config.weighting.gradnorm_mtl_alpha,
            lr=config.weighting.gradnorm_mtl_lr,
            init_weights=init_weights,
        )

    return FixedBalancer()
