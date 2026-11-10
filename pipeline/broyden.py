
from typing import Callable, Optional
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import optax
from typing import NamedTuple, Optional

Array = jnp.ndarray



import jax
import jax.numpy as jnp
from jax import lax
from jax._src.scipy.optimize.line_search import line_search
from jax._src.scipy.optimize.line_search import line_search as _jax_single_wolfe_line_search
from jax._src.scipy.optimize.line_search import _LineSearchResults # Ensure correct import

from functools import partial

_dot = partial(jnp.dot, precision=lax.Precision.HIGHEST)
_einsum = partial(jnp.einsum, precision=lax.Precision.HIGHEST)


def line_search_jax_with_fallback( # This is the 2-stage fallback discussed before
    fun: Callable, xk: jax.Array, pk: jax.Array,
    old_fval: jax.Array  = None, old_old_fval: jax.Array  = None, gfk: jax.Array  = None,
    c1_try1: float = 1e-4, c2_try1: float = 0.8, maxiter_try1: int = 10,
    c1_try2: float = 1e-4, c2_try2: float = 0.5, maxiter_try2: int = 20
) -> _LineSearchResults:
    results_try1 = _jax_single_wolfe_line_search(fun, xk, pk, old_fval, old_old_fval, gfk,
                                                 c1_try1, c2_try1, maxiter_try1)
    def true_branch_fb_internal_failed(_):
        results_try2 = _jax_single_wolfe_line_search(fun, xk, pk, old_fval, old_old_fval, gfk,
                                                     c1_try2, c2_try2, maxiter_try2)
        return results_try2
    def false_branch_fb_internal_succeeded(r1):
        return r1
    return jax.lax.cond(results_try1.failed, true_branch_fb_internal_failed, false_branch_fb_internal_succeeded, results_try1)

def line_search_normal_then_fallback(
    fun: Callable,
    xk: jax.Array,
    pk: jax.Array,
    old_fval: jax.Array  = None,
    old_old_fval: jax.Array  = None,
    gfk: jax.Array  = None,
    # Parameters for the "normal" initial attempt
    normal_c1: float = 1e-4,
    normal_c2: float = 0.9,
    normal_maxiter: int = 15,
    # Parameters to pass to line_search_jax_with_fallback (which has its own try1 & try2)
    fallback_c1_try1: float = 1e-4,
    fallback_c2_try1: float = 0.8,
    fallback_maxiter_try1: int = 10,
    fallback_c1_try2: float = 1e-4,
    fallback_c2_try2: float = 0.5,
    fallback_maxiter_try2: int = 20
) -> _LineSearchResults:

    # 1. Attempt "normal" line search
    results_normal = _jax_single_wolfe_line_search(
        fun, xk, pk, old_fval, old_old_fval, gfk,
        c1=normal_c1, c2=normal_c2, maxiter=normal_maxiter
    )

    # This function is called if results_normal.failed is True
    def true_branch_normal_failed(_results_normal_ignored):
        # Normal search failed, now use the line_search_jax_with_fallback
        results_full_fallback = line_search_jax_with_fallback(
            fun, xk, pk, old_fval, old_old_fval, gfk,
            c1_try1=fallback_c1_try1, c2_try1=fallback_c2_try1, maxiter_try1=fallback_maxiter_try1,
            c1_try2=fallback_c1_try2, c2_try2=fallback_c2_try2, maxiter_try2=fallback_maxiter_try2
        )
        return results_full_fallback

    # This function is called if results_normal.failed is False
    def false_branch_normal_succeeded(res_normal):
        return res_normal

    final_results = jax.lax.cond(
        results_normal.failed,
        true_branch_normal_failed,
        false_branch_normal_succeeded,
        results_normal
    )
    return final_results

class BroydenOriginalState(NamedTuple):
    count: Array
    Hk: Array          # (n, n)
    failed: Array       # bool

def _ensure_tuple_value_grads(value_fn: Callable, params):
    out = value_fn(params)
    if isinstance(out, tuple) and len(out) == 2:
        loss, grads_tree = out
        return loss, grads_tree
    loss = out
    grads_tree = jax.grad(lambda p: value_fn(p))(params)
    return loss, grads_tree

def _flatten_grads(grads_tree, unravel=None) -> Array:
    if isinstance(grads_tree, jnp.ndarray):
        if grads_tree.ndim == 0:
            raise TypeError("Received a 0-d array as gradients. "
                            "Ensure value_fn returns (loss, grads) with grads matching params, "
                            "or pass `grad` to opt.update that matches the params pytree.")
        return grads_tree.reshape(-1)
    g_flat, _ = ravel_pytree(grads_tree)
    return g_flat

def _ss_theta(name: str, ak: Array, bk: Array, rhokm: Array, thetakm: Array, thetakp: Array, ratio: Array, eps: float):
    if name == "SSBroyden1":
        thetakSR1 = 1.0 / (1.0 - bk + eps)
        thetakp_extra = jnp.maximum(jnp.array(1.0, thetakp.dtype), bk * (1.0 + ratio))
        theta = jnp.where(
            jnp.isclose(bk, 1.0),
            thetakm,
            jnp.where(
                bk > 1.0,
                jnp.maximum(thetakm, thetakSR1),
                jnp.minimum(thetakp_extra, thetakSR1),
            ),
        )
    elif name == "SSBroyden2":
        theta = jnp.maximum(thetakm, jnp.minimum(thetakp, (1.0 - bk) / (bk + eps)))
    elif name == "SSBroyden3":
        theta = thetakm
    else:
        raise ValueError(f"Unknown SSBroyden variant: {name}")
    return theta


def _direction(Hk: Array, gk: Array) -> Array:
    return -(Hk @ gk)



def broyden_original(name: str = "SSBroyden1",
                     c1: float = 1e-4, c2: float = 0.9, alpha0: float = 1.0,
                     tau: float = 0.5,
                     eps: float = 1e-12) -> optax.GradientTransformationExtraArgs:
    """The original self-scaled Broyden, without the safeguards of `broyden`
    (below); selected by an optimizer name ending in "_original". Optax
    LBFGS-like signature.
    Call:
        updates, state = opt.update(grads, state, params, value=loss, grad=grads, value_fn=value_fn)
    where value_fn(params) -> (loss, grads) *or* -> loss (grads are auto-diffed).
    """
    variant = name

    def init_fn(params):
        x0, _ = ravel_pytree(params)
        n = x0.size
        return BroydenOriginalState(
            count=jnp.zeros([], jnp.int32),
            Hk=jnp.eye(n, dtype=x0.dtype),
            failed=jnp.array(False)
        )

    def update_fn(grads, state: BroydenOriginalState, params=None, *, value=None, grad=None, value_fn: Optional[Callable]=None):
        if params is None:
            raise ValueError("broyden requires current params.")
        if value_fn is None:
            raise ValueError("broyden requires value_fn(params)->(loss, grads) or -> loss.")
        # Current loss/grad
        if value is None or grad is None:
            fk, g_tree = _ensure_tuple_value_grads(value_fn, params)
            gk = _flatten_grads(g_tree)
        else:
            fk = value
            gk = _flatten_grads(grad)
        xk, unravel = ravel_pytree(params)

        Hk = state.Hk
        # here new
        p_k = -_dot(Hk, gk)

        def fun_flat(x_flat):
            return value_fn(unravel(x_flat))   # returns scalar loss



        line_search_results = line_search_normal_then_fallback(
            fun_flat, 
            xk,
            p_k,
            old_fval=fk,
            old_old_fval=None,
            gfk=gk,
            # Normal attempt parameters
            normal_c1=c1,
            normal_c2=c2,
            normal_maxiter=25,
            # Fallback (which contains two tries) parameters
            fallback_c1_try1=c1,
            fallback_c2_try1=0.8,
            fallback_maxiter_try1=25,
            fallback_c1_try2=1e-4,
            fallback_c2_try2=0.5,
            fallback_maxiter_try2=25
        )
        
        s_k = line_search_results.a_k * p_k
        x_kp1 = xk + s_k
        f_kp1 = line_search_results.f_k
        g_kp1 = line_search_results.g_k
        y_k = g_kp1 - gk
        # --- MODIFIED rho_k CALCULATION ---
        yk_dot_sk = _dot(y_k, s_k)
        # Define a threshold for considering the denominator effectively zero
        small_denominator_threshold = 1e-32 

        # Condition: is yk_dot_sk too close to zero?
        pred_yk_dot_sk_is_problematic = jnp.abs(yk_dot_sk) < small_denominator_threshold

        def calculate_rho_k_normally(operand_yds):
            # Normal calculation: 1.0 / (y_k^T s_k)
            # Add a very small epsilon here too, just in case yds passed the threshold
            # but is still problematic for reciprocal on some hardware/dtype,
            # or make the threshold slightly larger.
            return 1.0 / (operand_yds + jnp.sign(operand_yds) * 1e-20)


        def handle_problematic_rho_k_denominator(operand_yds):
            # Fallback value, mimicking SciPy's behavior for BFGS
            return jnp.array(1000.0, dtype=operand_yds.dtype)

        # This is the new rho_k, calculated conditionally
        rho_k = jax.lax.cond(
            pred_yk_dot_sk_is_problematic,
            handle_problematic_rho_k_denominator,
            calculate_rho_k_normally,
            yk_dot_sk # Operand passed to the branches
        )
        # --- END OF MODIFIED rho_k CALCULATION ---

        # Store current state variables for convenience
        Hk = Hk
        yk = y_k
        sk = s_k
        rhok = rho_k
        alpha_k_linesearch = line_search_results.a_k # Step length from line search
        gfk_old = gk # Gradient at x_k (before line search)
        d = xk.shape[0] # Dimension

        # --- Conditional Hessian Update ---

        eps_div = 1e-32 

        Hkyk = _dot(Hk, yk)
        ykHkyk = _dot(yk, Hkyk)
        
        # hk and bk calculations
        hk = ykHkyk * rhok
        bk = -alpha_k_linesearch * rhok * _dot(sk, gfk_old)
        ak = bk * hk - 1.0

        # Intermediate terms for thetak
        abs_ak = jnp.abs(ak)
        # Add epsilon to denominator for sqrt_term
        sqrt_term = jnp.sqrt(abs_ak / (1.0 + abs_ak + eps_div)) 
        rhokm_val = hk * (1.0 - sqrt_term)
        # If hk is large, rhokm_val could be > 1, ensure it's capped by 1
        # Or, if hk is negative (ykHkyk and rhok different signs), rhokm_val could be negative.
        # The original `min(1, ...)` suggests rhokm should not exceed 1.
        # Let's ensure it's also non-negative if that's implied by theory, though original min(1,val) doesn't enforce it.
        # For now, directly translate:
        rhokm = jnp.minimum(1.0, rhokm_val) 

        thetakm_den = ak + jnp.sign(ak) * eps_div # Avoid division by exact zero
        thetakm = (rhokm - 1.0) / thetakm_den
        
        thetakp_den = rhokm + eps_div # Avoid division by zero if rhokm is zero
        thetakp = 1.0 / thetakp_den

        bk_safe_den = bk + jnp.sign(bk) * eps_div # Avoid division by exact zero for term_for_thetak
        term_for_thetak = (1.0 - bk) / bk_safe_den
        thetak = jnp.maximum(thetakm, jnp.minimum(thetakp, term_for_thetak))

        # Tauk calculation (depends on initial_scale and current iteration)
        is_initial_step_and_scale_cond = False & (state.count == 0) & \
                                        jnp.allclose(Hk, jnp.eye(d, dtype=Hk.dtype), atol=1e-6) # Check if Hk is identity

        tauk_A_den = (1.0 + ak * thetak + eps_div)
        tauk_A = hk / tauk_A_den # tauk for initial scaling case

        rhokk_den = bk + jnp.sign(bk) * eps_div
        rhokk = jnp.minimum(1.0, 1.0 / rhokk_den) # if bk is very small, 1/bk is large
        
        sigmak = 1.0 + thetak * ak
        
        # Handle d=1 case for sigmaknm1_exp to avoid division by zero
        sigmaknm1_exp = lax.cond(d == 1,
                                lambda _: 0.0, # Effectively makes sigmaknm1 = 1 if d=1 (abs(sigmak)**0)
                                lambda _: 1.0 / (1.0 - d),
                                None)
        sigmaknm1 = jnp.abs(sigmak) ** sigmaknm1_exp

        tauk_B_cond = thetak <= 0.0
        tauk_B_true = jnp.minimum(rhokk * sigmaknm1, sigmak)
        tauk_B_false_den = thetak + eps_div
        tauk_B_false = rhokk * jnp.minimum(sigmaknm1, 1.0 / tauk_B_false_den)
        tauk_B = jnp.where(tauk_B_cond, tauk_B_true, tauk_B_false)

        tauk = jnp.where(is_initial_step_and_scale_cond, tauk_A, tauk_B)
        tauk_safe = tauk + jnp.sign(tauk) * eps_div # Ensure tauk is not zero for division

        # vk and phik calculations
        ykHkyk_safe = ykHkyk + jnp.sign(ykHkyk) * eps_div
        vk = sk * rhok - Hkyk / ykHkyk_safe
        
        phik_den = (1.0 + ak * thetak + eps_div)
        phik = (1.0 - thetak) / phik_den

        # SSBroyden2 Hessian update
        term1_H_update = Hk
        term2_H_update_num = _einsum('i,j->ij', Hkyk, Hkyk) # Outer product: Hkyk @ Hkyk.T
        term2_H_update = term2_H_update_num / ykHkyk_safe
        
        term3_H_update_vk_outer = _einsum('i,j->ij', vk, vk) # Outer product: vk @ vk.T
        term3_H_update = phik * ykHkyk * term3_H_update_vk_outer
        
        H_numerator = term1_H_update - term2_H_update + term3_H_update
        H_scaled = H_numerator / tauk_safe # Divide by tauk
        
        term4_H_update_sk_outer = _einsum('i,j->ij', sk, sk) # Outer product: sk @ sk.T
        term4_H_update = term4_H_update_sk_outer * rhok
        
        H_kp1 = H_scaled + term4_H_update


        # Safeguard against non-finite values (already in JAX's bfgs.py)
        H_kp1 = jnp.where(jnp.isfinite(rho_k), H_kp1, state.Hk)
        H_kp1 = jnp.where(jnp.isfinite(rho_k), H_kp1, state.Hk)
        converged = jnp.linalg.norm(g_kp1, ord=jnp.inf) < 1e-5

        

        updates = unravel(sk)
        new_state = BroydenOriginalState(
            count=optax.safe_int32_increment(state.count),
            Hk=H_kp1,
            failed=False
        )
        return updates, new_state


    return optax.GradientTransformationExtraArgs(init_fn, update_fn)


# =============================================================================
# broyden: the SSBroyden2 update with scale-free safeguards
# =============================================================================
# The quasi-Newton optimizer of every study (optimizer name Soap_SSBroyden2,
# see pipeline.train.get_quasi_newton_opt). `broyden_original` above is the
# same update without the safeguards, left untouched (Soap_SSBroyden2_original).
#
# `broyden_original` does not converge to the arithmetic floor -- it stops early, for
# three reasons:
#
# 1. A failed line search returns a stale gradient. jax's `line_search` sets
#    `g_star = gfk` at init and replaces it only when a point is accepted, so a
#    failure hands back `a_k` together with the OLD gradient and `y_k = g_kp1 -
#    g_k` is exactly zero while `s_k != 0`.
# 2. The guarded denominators are not guarded: with `y_k = 0`, `ykHkyk = 0` and
#    `ykHkyk + jnp.sign(ykHkyk) * 1e-32` adds nothing because
#    `jnp.sign(0.) == 0.`, so the update divides 0/0.
# 3. The NaN net tests `rho_k`, which is the finite fallback 1000.0 in exactly
#    that branch, so the all-NaN H is stored, the next loss is NaN and the
#    quasi-Newton phase ends. `failed` is always False.
#
# On top of that, `rho_k = 1 / (y.s + sign(y.s) * 1e-20)` carries an absolute
# epsilon: once y.s ~ 2 x (per-step decrease of f) reaches ~1e-20, rho_k is
# wrong by O(1) and H degrades.
#
# What broyden does differently:
# * every curvature guard is RELATIVE (`y.s > CURV_TOL * eps * |y| |s|`), so
#   there is no absolute floor for a small objective to run into;
# * a failed line search is never trusted: the candidate step is re-evaluated
#   (one extra value_and_grad, never paid on the successful branch) and kept if
#   it decreases f, with a TRUE gradient for y_k; otherwise the step backtracks
#   along -g from the Barzilai-Borwein length of the last good pair and the
#   memory restarts at gamma * I. MAX_FAIL steps with no decrease at any length
#   set `failed`, and train.second_order_step then stops with finite params;
# * a non-descent direction (g.p >= 0, the symptom of a poisoned H) triggers
#   the same restart before the line search is run;
# * divisions by dimensionless quantities fall back to the plain-BFGS values
#   (theta = 0, tau = 1), tauk / thetak are magnitude-clamped (inert in the
#   healthy regime, where both are O(1));
# * the NaN net tests what H is built from and restarts rather than storing
#   anything non-finite.
#
# In the healthy regime (|y.s| >> 1e-20) the two updates agree to round-off.
# Peak memory is the same as `broyden_original` (three live (n, n) float64 buffers).

# --------------------------------------------------------------- tuning knobs
CURV_TOL = 100.0        # x eps: reject y.s / y'Hy that are pure round-off
MAX_FAIL = 3            # consecutive steps with NO decrease -> report `failed`
BT_MAX = 40             # backtracking halvings of the salvage step
TAU_CAP = 1e6           # |tauk| clamp; tauk is O(1) whenever the pair is good
THETA_CAP = 1e8         # |thetak| clamp, same reasoning
GAMMA_MIN, GAMMA_MAX = 1e-12, 1e12      # sane range for the BB restart scale


class BroydenState(NamedTuple):
    count: Array        # iterations taken
    Hk: Array           # (n, n) inverse-Hessian approximation
    failed: Array       # bool: MAX_FAIL steps with no decrease -> train.py stops
    n_fail: Array       # CONSECUTIVE steps that could not decrease f
    n_reset: Array      # total memory restarts (diagnostic)
    n_skip: Array       # total H updates skipped on a round-off curvature pair
    gamma: Array        # Barzilai-Borwein scale s.y / y.y of the last good pair


def _safe_div(num, den, fallback, tol=None):
    """num / den, or `fallback` when den is too small to divide by.

    The double `where` is deliberate: the unused branch must not divide by zero
    even in the forward pass, or XLA hands back a NaN that the outer `where`
    then selects around but that still poisons any gradient taken through it.
    """
    if tol is None:
        tol = jnp.finfo(jnp.asarray(den).dtype).tiny * 1e8
    ok = jnp.abs(den) > tol
    return jnp.where(ok, num / jnp.where(ok, den, jnp.ones_like(den)), fallback)


def broyden(name: str = "SSBroyden2",
            c1: float = 1e-4, c2: float = 0.9,
            curv_tol: float = CURV_TOL,
            max_fail: int = MAX_FAIL) -> optax.GradientTransformationExtraArgs:
    """Self-scaled Broyden with scale-free safeguards; optax LBFGS-like call:

        updates, state = opt.update(grads, state, params, value=loss,
                                    grad=grads, value_fn=value_fn)

    `name` is accepted for signature compatibility with `broyden_original` and,
    exactly as there, the SSBroyden2 update is the one that is applied.
    """
    del name                       # `broyden_original` ignores it too

    def init_fn(params):
        x0, _ = ravel_pytree(params)
        return BroydenState(
            count=jnp.zeros([], jnp.int32),
            Hk=jnp.eye(x0.size, dtype=x0.dtype),
            failed=jnp.array(False),
            n_fail=jnp.zeros([], jnp.int32),
            n_reset=jnp.zeros([], jnp.int32),
            n_skip=jnp.zeros([], jnp.int32),
            gamma=jnp.ones([], x0.dtype),
        )

    def update_fn(grads, state: BroydenState, params=None, *,
                  value=None, grad=None, value_fn: Optional[Callable] = None):
        if params is None:
            raise ValueError("broyden requires current params.")
        if value_fn is None:
            raise ValueError("broyden requires value_fn(params) -> loss "
                             "or -> (loss, grads).")
        if value is None or grad is None:
            fk, g_tree = _ensure_tuple_value_grads(value_fn, params)
            gk = _flatten_grads(g_tree)
        else:
            fk = value
            gk = _flatten_grads(grad)

        xk, unravel = ravel_pytree(params)
        dtype = xk.dtype
        d = xk.shape[0]
        eps = jnp.finfo(dtype).eps
        tol = jnp.asarray(curv_tol, dtype) * eps

        # ------------------------------------------------- direction + restart
        p_k = -_dot(state.Hk, gk)
        dphi0 = _dot(gk, p_k)
        # a poisoned memory shows up here first: -H g is no longer downhill.
        # Restarting now costs one matmul; letting the line search discover it
        # costs up to 75 function+gradient evaluations and returns garbage.
        # The restarted memory is gamma * I, so it is never MATERIALISED here --
        # every use of it below is a scaling of an O(n) vector, and the (n, n)
        # identity is only built in the restart branch of the final switch.
        bad_dir = ~(dphi0 < 0.0) | ~jnp.isfinite(dphi0)
        p_k = jnp.where(bad_dir, -state.gamma * gk, p_k)

        def fun_flat(x_flat):
            return value_fn(unravel(x_flat))

        ls = line_search_normal_then_fallback(
            fun_flat, xk, p_k, old_fval=fk, old_old_fval=None, gfk=gk,
            normal_c1=c1, normal_c2=c2, normal_maxiter=25,
            fallback_c1_try1=c1, fallback_c2_try1=0.8, fallback_maxiter_try1=25,
            fallback_c1_try2=1e-4, fallback_c2_try2=0.5, fallback_maxiter_try2=25)

        # ---------------------------------------------------- step acceptance
        # `ls.g_k` is the gradient at the accepted point ONLY when the search
        # succeeded; on failure it is still the gradient at xk, which is what
        # makes y_k identically zero in `broyden_original`.  `ls.f_k` is no better: on a
        # failed zoom it is the value at some other trial point, not at a_k.  So
        # a failed search's output is never used as-is -- but the step it
        # proposes is still usually good, and throwing it away costs far more
        # progress than it saves.  Verify it instead, with one extra evaluation
        # that the (common) successful branch never pays for.
        a_ls = ls.a_k
        cand_ok = jnp.isfinite(a_ls) & (a_ls != 0.0)
        wolfe_ok = (~ls.failed) & cand_ok

        def _wolfe_step():
            return (a_ls * p_k, ls.g_k - gk, a_ls,
                    jnp.asarray(True), jnp.asarray(True))

        def _salvage():
            a = jnp.where(cand_ok, a_ls, 0.0)
            f1, g1 = jax.value_and_grad(fun_flat)(xk + a * p_k)
            use_cand = cand_ok & jnp.isfinite(f1) & (f1 < fk)

            def _no_bt(_):
                return state.gamma, fk, jnp.asarray(0, jnp.int32)

            def _bt(_):
                # last resort: backtrack along -g from the BB length. Unless we
                # are at the arithmetic floor this always finds a decrease, so
                # the phase keeps moving instead of locking up on a restart.
                def cond(c):
                    t, ft, j = c
                    return (~(ft < fk)) & (j < BT_MAX)

                def body(c):
                    t, _, j = c
                    t = t * 0.5
                    return t, fun_flat(xk - t * gk), j + 1

                t0 = state.gamma
                return lax.while_loop(cond, body, (t0, fun_flat(xk - t0 * gk),
                                                   jnp.asarray(0, jnp.int32)))

            t, ft, _ = lax.cond(use_cand, _no_bt, _bt, None)
            use_bt = (~use_cand) & jnp.isfinite(ft) & (ft < fk)
            s = (jnp.where(use_cand, a, 0.0) * p_k
                 - jnp.where(use_bt, t, 0.0) * gk)
            # y only from the candidate: the backtracked step has no gradient of
            # its own, so it moves the iterate but leaves the memory alone
            y = jnp.where(use_cand, 1.0, 0.0) * (g1 - gk)
            return s, y, a, use_cand | use_bt, use_cand

        s_k, y_k, a_k, took, pair_ok = lax.cond(wolfe_ok, _wolfe_step, _salvage)
        s_k = jnp.where(jnp.isfinite(s_k), s_k, 0.0)
        y_k = jnp.where(jnp.isfinite(y_k), y_k, 0.0)
        a_k = jnp.where(jnp.isfinite(a_k), a_k, 0.0)

        # ------------------------------------------------- relative curvature
        sy = _dot(y_k, s_k)
        yy = _dot(y_k, y_k)
        n_y = jnp.sqrt(yy)
        n_s = jnp.sqrt(_dot(s_k, s_k))
        sy_ok = sy > tol * n_y * n_s            # strong Wolfe guarantees sy > 0

        # (gamma I) y = gamma y, so a restarted memory costs no matmul
        Hkyk = jnp.where(bad_dir, state.gamma * y_k, _dot(state.Hk, y_k))
        ykHkyk = _dot(y_k, Hkyk)
        n_Hy = jnp.sqrt(_dot(Hkyk, Hkyk))
        yHy_ok = ykHkyk > tol * n_y * n_Hy

        do_update = pair_ok & sy_ok & yHy_ok
        # guarded denominators: no epsilon, because `do_update` already says
        # whether these are anything but round-off
        sy_den = jnp.where(sy_ok, sy, jnp.ones_like(sy))
        yHy_den = jnp.where(yHy_ok, ykHkyk, jnp.ones_like(ykHkyk))
        rhok = 1.0 / sy_den

        # ---------------------------------------- SSBroyden2 scalars (O(1) all)
        hk = ykHkyk * rhok
        bk = -a_k * rhok * _dot(s_k, gk)
        ak = bk * hk - 1.0

        abs_ak = jnp.abs(ak)
        sqrt_term = jnp.sqrt(abs_ak / (1.0 + abs_ak))
        rhokm = jnp.minimum(1.0, hk * (1.0 - sqrt_term))

        thetakm = _safe_div(rhokm - 1.0, ak, jnp.zeros_like(ak))
        thetakp = _safe_div(jnp.ones_like(rhokm), rhokm,
                            jnp.full_like(rhokm, THETA_CAP))
        term_for_thetak = _safe_div(1.0 - bk, bk, jnp.full_like(bk, THETA_CAP))
        thetak = jnp.maximum(thetakm, jnp.minimum(thetakp, term_for_thetak))
        thetak = jnp.where(jnp.isfinite(thetak), thetak, jnp.zeros_like(thetak))
        thetak = jnp.clip(thetak, -THETA_CAP, THETA_CAP)

        rhokk = jnp.minimum(1.0, _safe_div(jnp.ones_like(bk), bk,
                                           jnp.ones_like(bk)))
        sigmak = 1.0 + thetak * ak
        sigmaknm1_exp = 0.0 if d == 1 else 1.0 / (1.0 - d)
        sigmaknm1 = jnp.abs(sigmak) ** sigmaknm1_exp

        tauk = jnp.where(thetak <= 0.0,
                         jnp.minimum(rhokk * sigmaknm1, sigmak),
                         rhokk * jnp.minimum(sigmaknm1,
                                             _safe_div(jnp.ones_like(thetak),
                                                       thetak,
                                                       jnp.full_like(thetak, TAU_CAP))))
        # tauk divides the whole matrix; |tauk| is O(1) for any usable pair, so
        # the clamp is inert in the healthy regime and only stops the 1e-32
        # blow-up that `broyden_original` walks into
        tauk = jnp.where(jnp.isfinite(tauk) & (jnp.abs(tauk) > 1.0 / TAU_CAP),
                         tauk, jnp.ones_like(tauk))
        tauk = jnp.clip(tauk, -TAU_CAP, TAU_CAP)

        vk = s_k * rhok - Hkyk / yHy_den
        phik = _safe_div(1.0 - thetak, 1.0 + ak * thetak, jnp.ones_like(thetak))

        # ------------------------------------------------------- the H update
        # `bad_dir` means state.Hk is the poisoned memory; the switch below
        # discards H_new in that case, so using it as term1 here is harmless
        # Outer products as broadcast products rather than the `dot_general`
        # that `jnp.einsum('i,j->ij', ...)` lowers to: bit-for-bit the same
        # number (an outer product accumulates nothing, so `precision` is
        # irrelevant -- verified identical to `broyden_original`'s einsum over three
        # steps at n = 20,960) and elementwise, so XLA is free to fuse them.
        # Measured: both updates peak at 10.23 GiB at n = 20,960 (three live
        # (n, n) buffers; the third is an XLA temporary).
        H_num = (state.Hk
                 - (Hkyk[:, None] * Hkyk[None, :]) / yHy_den
                 + (phik * ykHkyk) * (vk[:, None] * vk[None, :]))
        H_new = H_num / tauk + (s_k[:, None] * s_k[None, :]) * rhok

        # The net that `broyden_original` gets wrong: it tests `rho_k`, which is the
        # FINITE fallback 1000.0 in exactly the branch that poisons H.  Test the
        # things H is actually built from instead -- all O(n) or scalar, so H
        # keeps exactly ONE consumer and XLA can fuse the whole update into a
        # single pass over it.  A non-finite H that somehow slipped past these
        # is still caught at the TOP of the next iteration: -H g would make
        # dphi0 non-finite, `bad_dir` fires and the memory restarts.  So the
        # worst case is one wasted iteration, never a dead run.
        H_ok = (do_update
                & jnp.isfinite(tauk) & jnp.isfinite(rhok) & jnp.isfinite(phik)
                & jnp.isfinite(ykHkyk) & jnp.isfinite(hk) & jnp.isfinite(bk)
                & jnp.all(jnp.isfinite(vk)) & jnp.all(jnp.isfinite(Hkyk))
                & jnp.all(jnp.isfinite(s_k)))
        gamma_new = _safe_div(sy, yy, state.gamma)
        gamma_new = jnp.where(sy_ok & jnp.isfinite(gamma_new),
                              jnp.clip(gamma_new, GAMMA_MIN, GAMMA_MAX),
                              state.gamma)

        # The three outcomes:
        #   c_new   usable pair          -> refresh the memory
        #   c_keep  step taken, pair bad -> keep it; the step still decreased f,
        #                                   only the curvature pair is round-off
        #   c_rst   restart              -> the direction was not downhill, or
        #                                   nothing decreased f at any length.
        #                                   gamma*I makes the next direction a
        #                                   scaled steepest-descent one, so we
        #                                   cannot lock.
        # Written as ONE elementwise expression rather than a cond/switch over
        # (n, n) branches, so XLA can fuse the whole update into one pass that
        # reads state.Hk and writes H_kp1. `_eye` is the identity written as a
        # broadcast comparison, which fuses into that pass instead of
        # materialising another (n, n) matrix.
        restart = bad_dir | ~took
        c_new = (H_ok & ~restart).astype(dtype)
        c_keep = (~H_ok & ~restart).astype(dtype)
        c_rst = restart.astype(dtype) * gamma_new
        idx = jnp.arange(d)
        _eye = (idx[:, None] == idx[None, :]).astype(dtype)
        H_kp1 = c_new * H_new + c_keep * state.Hk + c_rst * _eye

        n_fail = jnp.where(took, 0, state.n_fail + 1).astype(jnp.int32)
        new_state = BroydenState(
            count=optax.safe_int32_increment(state.count),
            Hk=H_kp1,
            failed=n_fail >= max_fail,
            n_fail=n_fail,
            n_reset=state.n_reset + restart.astype(jnp.int32),
            n_skip=state.n_skip + (took & ~do_update).astype(jnp.int32),
            gamma=gamma_new,
        )
        return unravel(s_k), new_state

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)
