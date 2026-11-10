from functools import partial
from typing import Dict

from flax.training import train_state
import jax.numpy as jnp
from jax import jit, grad, random

import optax
from pipeline import archs
from soap_jax import soap


class TrainState(train_state.TrainState):
    weights: Dict
    point_weights: jnp.ndarray

    def update_weights(self, *, weights, **kwargs):
        """Replace weights in the state (no momentum — balancer owns that)."""
        return self.replace(weights=weights, **kwargs)

    def update_point_weights(self, *, point_weights, **kwargs):
        """Replace per-point weights in the state."""
        return self.replace(point_weights=point_weights, **kwargs)


def _create_arch(config):
    if config.arch_name == "Mlp":
        arch = archs.Mlp(**config)
    elif config.arch_name == "ModifiedMlp":
        arch = archs.ModifiedMlp(**config)
    elif config.arch_name == "PirateNet":
        arch = archs.PirateNet(**config)
    else:
        raise NotImplementedError(f"Arch {config.arch_name} not supported yet!")
    return arch


def _create_optimizer(config):
    lr = optax.exponential_decay(
        init_value=config.learning_rate,
        transition_steps=config.decay_steps,
        decay_rate=config.decay_rate,
    )

    if config.warmup_steps > 0:
        warmup = optax.linear_schedule(
            init_value=0.0,
            end_value=config.learning_rate,
            transition_steps=config.warmup_steps,
        )
        lr = optax.join_schedules([warmup, lr], [config.warmup_steps])

    if config.optimizer == "Adam":
        tx = optax.adam(learning_rate=lr, b1=config.beta1, b2=config.beta2, eps=config.eps)
    elif config.optimizer == "Soap":
        tx = soap(learning_rate=lr, b1=config.beta1, b2=config.beta2, weight_decay=0.0, precondition_frequency=2)
    elif config.optimizer == "Muon":
        tx = optax.contrib.muon(learning_rate=lr, ns_coeffs=(2, -1.5, 0.5), ns_steps=10, beta=0.99, adam_b1=0.99)
    elif config.optimizer == "Adagrad":
        tx = optax.adagrad(learning_rate=lr, eps=config.eps)
    elif "Adam" in config.optimizer:
        tx = optax.adam(learning_rate=lr, b1=config.beta1, b2=config.beta2, eps=config.eps)
    elif "Soap" in config.optimizer:
        tx = soap(learning_rate=lr, b1=config.beta1, b2=config.beta2, weight_decay=0.01, precondition_frequency=2)
    else:
        raise NotImplementedError(f"Optimizer {config.optimizer} not supported yet!")
    
    if config.schedule_free:
        tx = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.contrib.schedule_free(tx, lr, b1=config.beta1),
        )
    return tx


def _create_train_state(config, params=None, weights=None):
    arch = _create_arch(config.arch)
    x = jnp.ones(config.arch.input_dim)
    tx = _create_optimizer(config.optim)

    if params is None:
        params = arch.init(random.PRNGKey(config.seed), x)
    if weights is None:
        weights = dict(config.weighting.init_weights)

    point_weights = jnp.ones(config.dia.batch_size)

    state = TrainState.create(
        apply_fn=arch.apply,
        params=params,
        tx=tx,
        weights=weights,
        point_weights=point_weights,
    )
    return state


class PINN:
    def __init__(self, config, initial_dict, axes_dict, pde_params_dict):
        self.config = config
        self.state = _create_train_state(config)
        self.initial_dict = initial_dict
        self.axes_dict = axes_dict
        self.pde_params_dict = pde_params_dict

    def losses(self, params, batch, point_weights):
        """
        Return all individual losses as a dict, e.g. {"ics": ..., "res": ..., "bcs": ...}
        with point_weights applied but not the per-task weights.
        """
        raise NotImplementedError("Subclasses should implement this!")

    def loss(self, params, weights, batch, point_weights):
        """
        Compute the total scalar loss (used by the optimizer and second-order methods).
        """
        raise NotImplementedError("Subclasses should implement this!")

    def r_losses_pp(self, params, batch, weights):
        """
        PDE residual loss per point (no point_weights), used by domain-adaptive samplers.
        """
        raise NotImplementedError("Subclasses should implement this!")

    @property
    def loss_fns(self):
        """
        Return individual loss functions as a dict, e.g. {"ics": self.loss_ics, ...}.
        Each fn must accept (params, batch, point_weights).
        Required by loss-balancing methods.
        """
        raise NotImplementedError("Subclasses should implement this!")

    def compute_l2_error(self, params, *args):
        """Return a dict of relative L2 errors vs. the reference solution."""
        raise NotImplementedError("Subclasses should implement this!")

    @partial(jit, static_argnums=(0,))
    def step(self, state, batch, point_weights):
        grads = grad(self.loss)(state.params, state.weights, batch, point_weights)
        return state.apply_gradients(grads=grads)
