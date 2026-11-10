from itertools import chain
from typing import List, NamedTuple, Optional, Union

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
import optax.tree_utils as otu
from chex import Numeric
from jaxtyping import Array
from optax import GradientTransformation, Updates


class SOAPState(NamedTuple):
    count: jnp.ndarray  # type: ignore
    exp_avg: Updates
    exp_avg_sq: Updates
    GG: Updates
    Q: Updates


def soap(
    learning_rate: optax.ScalarOrSchedule = 3e-3,
    b1: float = 0.95,
    b2: float = 0.95,
    shampoo_beta: float = -1,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    precondition_frequency: int = 10,
    max_precond_dim: int = 10000,
    precision: jax.lax.PrecisionLike = jax.lax.Precision.HIGHEST,
) -> optax.GradientTransformationExtraArgs:
    """
    Implements SOAP algorithm (https://arxiv.org/abs/2409.11321).

    Bug-fixes relative to v0.1.0:
      - exp_avg is tracked in the projected (Kronecker-eigenbasis) space, consistent with
        exp_avg_sq, so both moments are in the same basis.
      - exp_avg is rotated into the new basis whenever Q is refreshed.
      - Bias-correction uses effective_step = count - 1 (step 1 is the init step, not a
        real update step).
      - Weight decay is applied after LR scaling (AdamW-style decoupled decay).
    """
    return optax.chain(
        scale_by_soap(
            b1=b1,
            b2=b2,
            shampoo_beta=shampoo_beta,
            eps=eps,
            precondition_frequency=precondition_frequency,
            max_precond_dim=max_precond_dim,
            precision=precision,
        ),
        optax.scale_by_learning_rate(learning_rate),
        _add_decayed_weights_post(weight_decay, learning_rate),
    )


def _add_decayed_weights_post(
    weight_decay: float,
    learning_rate: optax.ScalarOrSchedule,
) -> GradientTransformation:
    """AdamW-style decoupled weight decay applied after LR scaling.

    Effective update contribution: -lr * weight_decay * params
    """
    if weight_decay == 0.0:
        return optax.identity()

    class _WDState(NamedTuple):
        count: jnp.ndarray

    def init_fn(params):
        del params
        return _WDState(count=jnp.zeros([], jnp.int32))

    def update_fn(updates, state, params=None):
        if params is None:
            raise ValueError("_add_decayed_weights_post requires params.")
        count_inc = jnp.asarray(optax.safe_int32_increment(state.count))
        lr = learning_rate(count_inc) if callable(learning_rate) else jnp.asarray(learning_rate)
        # No weight decay on the very first step (init_step returns zero update)
        eff_decay = jnp.where(count_inc == 1, jnp.zeros_like(lr), lr * weight_decay)
        updates = jtu.tree_map(lambda u, p: u - eff_decay * p, updates, params)
        return updates, _WDState(count=count_inc)

    return optax.GradientTransformation(init_fn, update_fn)  # type: ignore


def scale_by_soap(
    b1: float = 0.95,
    b2: float = 0.95,
    shampoo_beta: float = -1,
    eps: float = 1e-8,
    precondition_frequency: int = 10,
    max_precond_dim: int = 10000,
    precision: jax.lax.PrecisionLike = jax.lax.Precision.HIGHEST,
) -> GradientTransformation:
    shampoo_beta = shampoo_beta if shampoo_beta >= 0 else b2

    def init_fn(params: Updates) -> SOAPState:
        exp_avg = otu.tree_zeros_like(params)
        exp_avg_sq = otu.tree_zeros_like(params)
        GG = jtu.tree_map(lambda p: init_conditioner(p, max_precond_dim), params)
        Q  = jtu.tree_map(lambda p: init_conditioner(p, max_precond_dim), params)
        return SOAPState(
            count=jnp.zeros([], jnp.int32),
            exp_avg=exp_avg,
            exp_avg_sq=exp_avg_sq,
            GG=GG,
            Q=Q,
        )

    def init_step(updates: Updates, state: SOAPState) -> tuple[Updates, SOAPState]:
        new_GG = jtu.tree_map(
            lambda grad, gg: update_preconditioner(grad, gg, shampoo_beta, precision),
            updates,
            state.GG,
        )
        new_Q = jtu.tree_map(lambda gg: get_orthogonal_matrix(gg), new_GG)
        return otu.tree_zeros_like(updates), state._replace(GG=new_GG, Q=new_Q)

    def update_step(updates: Updates, state: SOAPState) -> tuple[Updates, SOAPState]:
        # Project gradient into the Kronecker-eigenbasis
        grad_projected = jtu.tree_map(
            lambda grad, q: project(grad, q, precision),
            updates,
            state.Q,
        )

        # FIX 1: both moments tracked in the SAME projected space
        exp_avg    = otu.tree_update_moment(grad_projected, state.exp_avg, b1, 1)
        exp_avg_sq = otu.tree_update_moment_per_elem_norm(grad_projected, state.exp_avg_sq, b2, 2)

        # exp_avg is already in projected space — project back directly
        norm_updates = jtu.tree_map(
            lambda e_avg, e_sq, q: project_back(e_avg / (jnp.sqrt(e_sq) + eps), q, precision),
            exp_avg,
            exp_avg_sq,
            state.Q,
        )

        # FIX 3: step 1 is init_step (no moment), real steps start at count - 1
        effective_step = state.count - 1
        bc1  = 1 - b1 ** effective_step
        bc2  = 1 - b2 ** effective_step
        corr = jnp.sqrt(bc2) / bc1
        norm_updates = jtu.tree_map(lambda p: p * corr, norm_updates)

        # Update preconditioner GG
        new_GG = jtu.tree_map(
            lambda grad, gg: update_preconditioner(grad, gg, shampoo_beta, precision),
            updates,
            state.GG,
        )

        should_update_precond = effective_step % precondition_frequency == 0

        # Refresh Q and exp_avg_sq — same 2-tuple pattern as v0.1.0 (confirmed working)
        new_Q_and_exp_avg_sq = jax.lax.cond(
            should_update_precond,
            lambda: jtu.tree_map(
                lambda e, gg, q: get_orthogonal_matrix_QR(gg, q, e, precision),
                exp_avg_sq, new_GG, state.Q,
            ),
            lambda: jtu.tree_map(
                lambda e, q: (q, e),
                exp_avg_sq, state.Q,
            ),
        )
        new_Q      = jtu.tree_map(lambda _, x: x[0], updates, new_Q_and_exp_avg_sq)
        exp_avg_sq = jtu.tree_map(lambda _, x: x[1], updates, new_Q_and_exp_avg_sq)

        # FIX 2: rotate exp_avg into the new basis (separate cond, same condition)
        exp_avg = jax.lax.cond(
            should_update_precond,
            lambda: jtu.tree_map(
                lambda e_avg, old_q, new_q: project(project_back(e_avg, old_q, precision), new_q, precision),
                exp_avg, state.Q, new_Q,
            ),
            lambda: exp_avg,
        )

        new_state = SOAPState(
            count=state.count,
            exp_avg=exp_avg,
            exp_avg_sq=exp_avg_sq,
            GG=new_GG,
            Q=new_Q,
        )
        return norm_updates, new_state

    def update_fn(
        updates: Updates, state: SOAPState, params: Optional[Updates] = None
    ) -> tuple[Updates, SOAPState]:
        del params
        count_inc = jnp.asarray(optax.safe_int32_increment(state.count))
        state = state._replace(count=count_inc)
        updates, new_state = jax.lax.cond(
            count_inc == 1,
            lambda: init_step(updates, state),
            lambda: update_step(updates, state),
        )
        return updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)  # type: ignore


def update_preconditioner(
    grad: Array,
    GG: List[Union[Array, None]],
    beta: float,
    precision: jax.lax.PrecisionLike = jax.lax.Precision.HIGHEST,
) -> List[Union[Array, None]]:
    if grad.ndim == 1:
        return [lerp(GG[0], jnp.matmul(grad[:, None], grad[None, :], precision=precision), 1 - beta)]  # type: ignore

    new_GG = []
    for idx, gg in enumerate(GG):
        if gg is None:
            new_GG.append(None)
            continue
        outer_product = jnp.tensordot(
            grad, grad,
            axes=[[*chain(range(idx), range(idx + 1, len(grad.shape)))]] * 2,
            precision=precision,
        )
        new_GG.append(lerp(gg, outer_product, 1 - beta))
    return new_GG


def project(
    grad: Array,
    Q: List[Union[Array, None]],
    precision: jax.lax.PrecisionLike = jax.lax.Precision.HIGHEST,
) -> Array:
    for mat in Q:
        if mat is not None:
            grad = jnp.tensordot(grad, mat, axes=((0,), (0,)), precision=precision)
        else:
            permute_order = list(range(1, len(grad.shape))) + [0]
            grad = jnp.transpose(grad, permute_order)
    return grad


def project_back(
    grad: Array,
    Q: List[Union[Array, None]],
    precision: jax.lax.PrecisionLike = jax.lax.Precision.HIGHEST,
) -> Array:
    for mat in Q:
        if mat is not None:
            grad = jnp.tensordot(grad, mat, axes=((0,), (1,)), precision=precision)
        else:
            grad = jnp.moveaxis(grad, 0, -1)
    return grad


def get_orthogonal_matrix(gg: Array) -> Union[Array, None]:
    if gg is None:
        return None
    _, eigh = jnp.linalg.eigh(gg + 1e-30 * jnp.eye(gg.shape[0]))
    return jnp.flip(eigh, axis=1)


def get_orthogonal_matrix_QR(
    GG: List[Union[Array, None]],
    Q: List[Union[Array, None]],
    exp_avg_sq: Array,
    precision: jax.lax.PrecisionLike = jax.lax.Precision.HIGHEST,
) -> tuple[List[Union[Array, None]], Array]:
    final_Q = []
    for ind, (m, o) in enumerate(zip(GG, Q)):
        if m is None or o is None:
            final_Q.append(None)
            continue
        est_eig = jnp.diag(
            jnp.matmul(jnp.matmul(o.T, m, precision=precision), o, precision=precision)
        )
        sort_idx = jnp.argsort(est_eig, descending=True)
        exp_avg_sq = jnp.take(exp_avg_sq, sort_idx, axis=ind)
        o = o[:, sort_idx]
        power_iter = jnp.matmul(m, o, precision=precision)
        Q_new, _ = jnp.linalg.qr(power_iter)
        final_Q.append(Q_new)
    return final_Q, exp_avg_sq


def lerp(start: Array, end: Array, weight: Numeric):
    return start + weight * (end - start)


def init_conditioner(p: Array, max_precond_dim: int) -> List[Union[Array, None]]:
    if p.ndim == 1:
        return [jnp.zeros((p.shape[0], p.shape[0]))]
    return [jnp.zeros((s, s)) if s <= max_precond_dim else None for s in p.shape]
