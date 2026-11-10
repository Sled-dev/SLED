import datetime
import hashlib
import json
import os
import pickle
import time
from jax import vmap
import optax
import jax
import jax.numpy as jnp
import numpy as np

from functools import partial

import ml_collections
import wandb
import pipeline
from pipeline.domain_importance_adapt import (IC_Axis_Sampler, get_sampler,
                                             get_qn_sampler, UniformSampler,
                                             raw_residual_fn, WEIGHT_CONVENTION)

from pipeline.log_utils import Logger
from pipeline.loss_balancing import create_loss_balancer, FixedBalancer

from flax.training import checkpoints



def save_checkpoint(state, workdir, keep=5):
    # Create the workdir if it doesn't exist.
    if not os.path.isdir(workdir):
        os.makedirs(workdir)
    # Save the checkpoint.
    if jax.process_index() == 0:
        step = int(state.step)
        checkpoints.save_checkpoint(workdir, state, step=step, keep=keep, overwrite=True)


def resample_samplers(model, samplers):
    if isinstance(samplers, dict):
        sampler = samplers["res"]
    else:
        sampler = samplers

    sampler.resample(model, model.state.params, model.state.weights)

    return model

def first_order_step(model, samplers, config, step, first_order_steps, balancer):
    # Handle potentially multiple samplers
    res_sampler = samplers["res"] if isinstance(samplers, dict) else samplers

    # Samplers that provide per-point weights (vRBA, ...) expose
    # prime()/get_lambdas(); everything else keeps uniform point weights.
    if step == 1 and hasattr(res_sampler, "prime"):
        res_sampler.prime(model, model.state.params, model.state.weights)

    if isinstance(samplers, dict):
        batch = {}
        for key, sampler in samplers.items():
            batch[key] = next(sampler)
    else:
        batch = next(samplers)

    if step == 1:
        balancer.init(model, model.state.params, batch, model.state.point_weights)


    if hasattr(res_sampler, "get_lambdas"):
            model.state = model.state.update_point_weights(
                point_weights=res_sampler.get_lambdas()
            )

    model.state = model.step(model.state, batch, model.state.point_weights)

    # Update weights at the correct interval (skip last step to avoid a wasted update)
    if step % config.weighting.update_every_steps == 0 and step < first_order_steps - 1:
        new_weights = balancer.update(model, model.state.params, batch, model.state.point_weights, model.state.weights)
        model.state = model.state.replace(weights=new_weights)

    model = resample_samplers(model, samplers)
    
    return model, batch, samplers

def logging_and_saving(model, config, step, batch, evaluator, logger, start_time):
    if jax.process_index() == 0:
        if step % config.logging.log_every_steps == 0:
            # Get the first replica of the state and batch
            log_dict = evaluator(model.state, batch)
            wandb.log(log_dict, step)
            end_time = time.time()
            logger.log_iter(step, start_time, end_time, log_dict)

    # Saving
    if config.saving.save_every_steps is not None:
        if (step + 1) % config.saving.save_every_steps == 0:
            ckpt_path = os.path.join(os.getcwd(), config.wandb.name, "ckpt")
            save_checkpoint(model.state, ckpt_path, keep=config.saving.num_keep_ckpts)
    
    return logger

def qn_refresh_every(config):
    """Refresh period of the second-order batch (config.qn_batch.refresh_every),
    None for a batch that stays frozen for the whole phase."""
    qn = config.qn_batch if "qn_batch" in config else None
    if qn is None or "refresh_every" not in qn or qn.refresh_every is None:
        return None
    return int(qn.refresh_every)


def qn_reset_memory(config):
    """Whether every batch refresh also re-initialises the quasi-Newton
    optimizer at the current parameters (config.qn_batch.reset_memory; the
    *_reset rules). Default False: the memory is kept across refreshes."""
    qn = config.qn_batch if "qn_batch" in config else None
    return bool(qn is not None and "reset_memory" in qn and qn.reset_memory)


def second_order_step(model, config, batch, step_offset, evaluator,
                      logger, log, max_iter, refresh_sampler=None):

    def lbfgs_log(params, i, logger, log, batch, start_time):
        if jax.process_index() == 0 and log:
            # Get the first replica of the state and batch
            state = model_state.replace(params=params)
            log_dict = evaluator(state, batch)
            wandb.log(log_dict, i)
            end_time = time.time()
            logger.log_iter(i, start_time, end_time, log_dict)
        return logger


    @partial(jax.jit,) 
    def lbfgs_update(params, opt_state, loss, grads, weights, batch, point_weights):

        value_fn = lambda p: model.loss(p, weights, batch, point_weights)

        updates, opt_state = opt.update(
            grads, opt_state, params,
            value=loss, grad=grads, value_fn=value_fn
        )
        params_new = optax.apply_updates(params, updates)

        return params_new, opt_state
    
    model_state = model.state
    weights = model.state.weights
    params = model.state.params
    point_weights = model.state.point_weights

    opt = get_quasi_newton_opt(config.optim.optimizer)
    opt_state= opt.init(params)

    # Optional mid-phase batch refresh (config.qn_batch.refresh_every = p):
    # every p steps `refresh_sampler` re-draws the batch at the current
    # parameters. The optimizer is UNMODIFIED — its curvature memory persists
    # across refreshes — unless config.qn_batch.reset_memory is set: then the
    # optimizer state is re-initialised at the current parameters after every
    # refresh (the *_reset rules).
    rf_period = qn_refresh_every(config)
    rf_enabled = rf_period is not None
    reset_memory = qn_reset_memory(config)
    if reset_memory and not rf_enabled:
        raise ValueError("qn_batch.reset_memory needs a refresh (qn_batch.refresh_every)")
    n_restarts = n_skips = 0        # broyden's counters over the reset windows
    if rf_enabled:
        if refresh_sampler is None:
            raise ValueError("qn_batch.refresh_every requires second_order_step(refresh_sampler=...)")
        if isinstance(batch, dict):
            raise NotImplementedError("qn_batch.refresh_every only supports a single residual batch")

    if hasattr(evaluator, "on_qn_start"):
        evaluator.on_qn_start(params, batch, point_weights)

    # refreshes fire at i = period, 2*period, ... < max_iter (never at the
    # first or the last step), so a run of T steps sees (T - 1) // period of them
    for i in range(max_iter):
        if rf_enabled and i > 0 and i % rf_period == 0:
            t_rf = time.time()
            batch = refresh_sampler.resample(
                model, params, weights,
                seed=config.seed + 100_003 * (i // rf_period),
                k_anneal=step_offset + i,           # global step for the anneal
                step=i)                             # the reselection key (PINNACLE)
            point_weights = refresh_sampler.get_lambdas()
            # the logged per-loss series reads model_state, so the served
            # weights must follow the batch that is actually being optimized
            model_state = model_state.update_point_weights(point_weights=point_weights)
            if reset_memory:
                n_restarts += int(getattr(opt_state, "n_reset", 0))
                n_skips += int(getattr(opt_state, "n_skip", 0))
                opt_state = opt.init(params)
            if hasattr(evaluator, "on_qn_refresh"):
                info = {k: v for k, v in refresh_sampler.info.items() if k != "omega"}
                evaluator.on_qn_refresh(step_offset + i,
                                        {**info, "qn_step": i,
                                         **({"reset": True} if reset_memory else {}),
                                         "resample_s": time.time() - t_rf})

        start_time = time.time()
        loss, grads = jax.value_and_grad(model.loss)(params, weights, batch, point_weights)

        params_new, opt_state_new = lbfgs_update(params, opt_state, loss, grads, weights, batch, point_weights)
        # check if the update failed
        # compute the loss after the update
        loss_new = model.loss(params_new, weights, batch, point_weights)
        # if NaN then failed
        if not jnp.isfinite(loss_new):
            # no fallback.
            print(f"NaN loss detected at iteration {i}. Stopping optimization.")
            return model.state.replace(params=params), logger, i
        else:
            params = params_new
            opt_state = opt_state_new
        #logging the state
        if (i + 1) % config.logging.log_every_steps == 0:
            lbfgs_log(params, i+step_offset, logger, log, batch, start_time)
        if hasattr(opt_state, 'failed') and opt_state.failed:
            print(f"Terminating at iteration {i} due to failure in line search.")
            break

    # broyden counts how often its safeguards fired; broyden_original has no
    # such fields, so this prints only for broyden
    if hasattr(opt_state, "n_reset"):
        print(f"[qn] broyden: {n_restarts + int(opt_state.n_reset)} memory restarts, "
              f"{n_skips + int(opt_state.n_skip)} skipped H updates in {i + 1} steps"
              + (" (memory reset at every refresh)" if reset_memory else ""))

    # steps_done counts COMPLETED iterations, so a full run returns max_iter and
    # anything less marks the phase as truncated (see OffBatchTracker.on_qn_end)
    return model_state.replace(params=params), logger, i + 1

def get_quasi_newton_opt(optimizer_name):
    # The quasi-Newton half of the optimizer name ("Soap_SSBroyden2": SOAP, then
    # SSBroyden2). SSBroyden1/2/3 select broyden.broyden, the SSBroyden update with
    # scale-free safeguards: a failed line search restarts the memory instead of
    # writing a NaN Hessian and ending the phase. The suffix "_original" selects
    # broyden.broyden_original, the same update without the safeguards.
    if "Lbfgs" in optimizer_name:
        return optax.lbfgs(learning_rate=1e-1, memory_size=500)
    variant = next((f"SSBroyden{k}" for k in "123" if f"Broyden{k}" in optimizer_name), None)
    if variant is None:
        raise ValueError(f"Unknown optimizer: {optimizer_name} (expected e.g. Soap_SSBroyden2)")
    if optimizer_name.endswith("_original"):
        from pipeline.broyden import broyden_original
        return broyden_original(name=variant, c1=1e-4, c2=0.9, alpha0=1.0, tau=0.5, eps=1e-12)
    from pipeline.broyden import broyden
    return broyden(name=variant, c1=1e-4, c2=0.9)

def make_domain(axes, axis_order):
    """
    Construct domain array [[min, max], ...] in fixed axis order.

    Parameters
    ----------
    axes : dict
        Mapping axis name -> coordinate array
    axis_order : tuple
        Desired axis ordering, e.g. ("t", "x", "y")
    time_extension : float
        Extend t_max by time_extension * dt

    Returns
    -------
    domain : jnp.ndarray, shape e.g. (ndim, 2)
    """
    bounds = []

    eps = 0
    for axis in axis_order:
        if axis not in axes:
            continue
        
        arr = axes[axis]
        if axis == 'coords':
            for i in range(arr.shape[1]):
                min_axis = jnp.min(arr[:, i])
                max_axis = jnp.max(arr[:, i])
                bounds.append(jnp.array([min_axis+eps, max_axis-eps]))
        else:        
            a0 = arr[0]
            a1 = arr[-1]
            bounds.append(jnp.array([a0, a1]))

    return jnp.stack(bounds)

def make_initial_dict(sol_dict: dict[str, jnp.ndarray], axes_dict: dict[str, jnp.ndarray] = None):
    """
    Construct initial condition dictionary from solution dictionary.
    If no time axis is present, return None.
    """
    if "t" not in axes_dict:
        return None
    
    make_initial_dict = {}
    for key, val in sol_dict.items():
        make_initial_dict[key + "0"] = val[0, ...]
    return make_initial_dict

def get_time_window_bounds(num_time_points, num_time_windows):
    if num_time_windows < 1:
        raise ValueError("num_time_windows must be at least 1.")
    if num_time_points < 2:
        raise ValueError("At least two time points are required for time-window training.")

    num_intervals = num_time_points - 1
    if num_time_windows > num_intervals:
        raise ValueError(
            "num_time_windows cannot exceed the number of time intervals "
            f"({num_intervals})."
        )

    base_steps = num_intervals // num_time_windows
    remainder = num_intervals % num_time_windows

    bounds = []
    start = 0
    for idx in range(num_time_windows):
        interval_len = base_steps + (1 if idx < remainder else 0)
        stop = start + interval_len + 1
        bounds.append((start, stop))
        start = stop - 1

    return bounds


def get_sol_dict_window_by_indices(sol_dict, start, stop):
    sol_dict_window = {}
    for key, val in sol_dict.items():
        if key == 't':
            sol_dict_window[key] = val[start:stop]
            #print("Time window:", sol_dict_window[key][0], "to", sol_dict_window[key][-1])
            # shift to start to 0
            sol_dict_window['t'] = sol_dict_window['t'] - sol_dict_window['t'][0]
        else:
            sol_dict_window[key] = val[start:stop, ...]
    return sol_dict_window


def get_sol_dict_window(sol_dict, num_time_steps, idx):
    start = num_time_steps * idx
    stop = num_time_steps * (idx + 1) + 1
    return get_sol_dict_window_by_indices(sol_dict, start, stop)

def find_pi_init(tree):
    """The `pi_init` kernel inside a (restored) param tree, if the arch has one:
    use_pi_init archs grow an extra output kernel, so a model restored from a
    checkpoint must be rebuilt WITH it (config.arch.pi_init) or the tree will
    not match apply_fn."""
    if isinstance(tree, dict):
        for k, v in tree.items():
            if k == "pi_init" and not isinstance(v, dict):
                return v
            hit = find_pi_init(v)
            if hit is not None:
                return hit
    return None


def physics_informed_init(config, axes_dict, initial_dict, pde_params_dict, model_cls):
    """
    Perform physics-informed initialization for the given model.
    Parameters
    ----------
    config : ml_collections.ConfigDict
        Configuration dictionary.
    axes_dict : dict
        Dictionary of axes arrays.
    initial_dict : dict
        Dictionary of initial condition arrays.
    pde_params_dict : dict
        Dictionary of PDE parameters.
    model_cls : class
        Model class to instantiate.
    time_stride : int
        Stride for time axis when building inputs.
    """
    print("Use physics-informed initialization...")
    # if hatattr config.pi_init_stride then time_stride = config.pi_init_stride else time_stride = 10
    if hasattr(config, "pi_init_stride"):
        time_stride = config.pi_init_stride
    else:
        time_stride = 10

    config.arch.pi_init = None  # reset any existing pi_init
    # Build the model
    model = model_cls(config=config, axes_dict=axes_dict, initial_dict=initial_dict, pde_params_dict=pde_params_dict)
    params = model.state.params

    # Build the inputs for the feature matrix
    t = axes_dict["t"][::time_stride]
    t_scaled = t / t[-1]

    # Spatial axes (everything except time)
    spatial_axes = [k for k in axes_dict.keys() if k != "t"]
    # only do if coords is not in spatial_axes
    spatial_grids = [axes_dict[k] for k in spatial_axes]
    
    # build inputs
    if 'coords' in spatial_axes:
        coords = axes_dict['coords']                # (N_coords, d)
        N_t = t_scaled.shape[0]
        N_coords = coords.shape[0]

        t_rep = jnp.repeat(t_scaled, repeats=N_coords)     # (N_t*N_coords,)
        coords_rep = jnp.tile(coords, (N_t, 1))            # (N_t*N_coords, d)

        inputs = jnp.concatenate([t_rep[:, None], coords_rep], axis=1)  # (N_t*N_coords, 1+d)
    else:
        # Meshgrid over all axes (independent spatial axes)
        mesh = jnp.meshgrid(t_scaled, *spatial_grids, indexing="ij")
        inputs = jnp.stack([m.reshape(-1) for m in mesh], axis=1)       # (N_samples, 1+ndim)

    N_samples = inputs.shape[0]

    # build targets
    field_names = list(initial_dict.keys())
    targets_cols = []

    if 'coords' in spatial_axes:
        # fields should be defined on coords: (N_coords,) or (N_coords, k)
        for name in field_names:
            field = jnp.asarray(initial_dict[name])

            if field.shape[0] != N_coords:
                raise ValueError(
                    f"Field '{name}' first dim must be N_coords={N_coords}, got {field.shape}."
                )

            # replicate across time in time-major order: [t0 all coords, t1 all coords, ...]
            if field.ndim == 1:
                col = jnp.tile(field, (N_t,))              # (N_t*N_coords,)
                targets_cols.append(col)
            else:
                # vector-valued per coord: (N_coords, k) -> (N_t*N_coords, k)
                col = jnp.tile(field, (N_t, 1))            # (N_t*N_coords, k)
                # If you truly want one column per field, you must decide how to handle k>1.
                # Here we flatten k into multiple target columns:
                targets_cols.append(col)

        # stack targets: scalar fields -> (N_samples, n_fields)
        # vector fields -> concatenate their columns
        stacked = []
        for col in targets_cols:
            if col.ndim == 1:
                stacked.append(col[:, None])
            else:
                stacked.append(col)  # (N_samples, k)
        targets = jnp.concatenate(stacked, axis=1)

    else:
        # grid case: field shaped (Nx, Ny, ...)
        for name in field_names:
            field = jnp.asarray(initial_dict[name])
            tiled = jnp.tile(field, (t_scaled.shape[0],) + (1,) * field.ndim)
            targets_cols.append(tiled.reshape(-1))          # (N_samples,)

        targets = jnp.stack(targets_cols, axis=1)           # (N_samples, n_fields)

    # final sanity check
    if targets.shape[0] != N_samples:
        raise ValueError(f"targets has {targets.shape[0]} rows, but inputs has {N_samples} rows.")

    # Compute feature matrix
    feat_matrix, _ = vmap(model.state.apply_fn, (None, 0))(params, inputs)




    # Solve for coefficients using least squares
    coeffs = []
    for i in range(targets.shape[1]):
        c, res, rank, s = jnp.linalg.lstsq(
            feat_matrix,
            targets[:, i],
            rcond=None,
        )
        coeffs.append(c)

        print(
            f"PI-init residual field {i}:",
            res if res.size > 0 else "rank-deficient",
        )

    coeffs = jnp.stack(coeffs, axis=1)
    # Set the pi_init in the config
    config.arch.pi_init = coeffs 

    del model, params 

    model = model_cls(config=config, axes_dict=axes_dict, initial_dict=initial_dict, pde_params_dict=pde_params_dict)
    return model

def _sampler_domain(sampler):
    """Domain box of a first-order sampler (UniformSampler/SobolSampler carry
    `.dom`; wrappers expose their base/eval sampler); None if unknown."""
    if hasattr(sampler, "dom"):
        return sampler.dom
    for attr in ("base_sampler", "eval_sampler"):
        if hasattr(sampler, attr):
            d = _sampler_domain(getattr(sampler, attr))
            if d is not None:
                return d
    return None


def train_and_evaluate(config: ml_collections.ConfigDict,
                        first_order_steps: int,
                        second_order_steps: int,
                        model: pipeline.models.PINN,
                        samplers: dict,
                        evaluator: pipeline.evaluator.BaseEvaluator,
                        step_offset: int,
                        logger: Logger,
                        log: bool = True,
                        balancer=None,
                        dom=None):

    if balancer is None:
        balancer = create_loss_balancer(config, dict(config.weighting.init_weights))

    if first_order_steps > 0:
        print(f"Using first-order optimizer for {first_order_steps} steps...")

        for step in range(1, first_order_steps+1):
            start_time = time.time()
            # First-order optimization step
            model, batch, samplers = first_order_step(model, samplers, config, step, first_order_steps, balancer)

            # Logging and saving
            if log: 
                logger = logging_and_saving(model, config, step + step_offset, batch, evaluator, logger, start_time)

        step_offset += first_order_steps

    stats = jax.local_devices()[0].memory_stats()
    if stats:                                   # None on CPU backends
        peak_gib = stats['peak_bytes_in_use'] / 1024**3
        print(f"Peak GPU memory usage: {peak_gib:.2f} GiB") 
    if second_order_steps > 0:
        balancer = FixedBalancer()  # freeze weights during second-order optimization
        print(f"Using second-order optimizer for {second_order_steps} steps...")
        # if samplers are dict then resample only the residual sampler
        # test if the config has initial_sampler attribute and if it's true set the ics sampler to full
        if hasattr(config.dia, "initial_sampler") and config.dia.initial_sampler:
            samplers['ics'].mode = 'full'
        if isinstance(samplers, dict):
            model = resample_samplers(model, samplers)
            batch = {}
            for key, sampler in samplers.items():
                batch[key] = next(sampler)
        else:
            model = resample_samplers(model, samplers)
            batch = next(samplers)
        # keep point weights aligned with the freshly resampled batch
        res_sampler = samplers["res"] if isinstance(samplers, dict) else samplers
        if hasattr(res_sampler, "get_lambdas"):
            model.state = model.state.update_point_weights(
                point_weights=res_sampler.get_lambdas()
            )

        # Frozen QN batch: config.qn_batch.type selects a frozen-batch sampler
        # (domain_importance_adapt.get_qn_sampler); no qn_batch section keeps
        # the batch drawn above exactly as-is.
        if dom is None and not isinstance(samplers, dict):
            dom = _sampler_domain(res_sampler)
        qn_sampler = get_qn_sampler(config, dom, k_anneal=step_offset,
                                    anneal_max_iter=step_offset + second_order_steps,
                                    n=None if isinstance(batch, dict) else int(batch.shape[0]))
        refresh_sampler = None
        if qn_sampler is not None:
            if isinstance(samplers, dict):
                raise NotImplementedError(
                    "qn_batch.type is only supported for a single residual sampler"
                )
            qn_sampler.resample(model, model.state.params, model.state.weights)
            batch = next(qn_sampler)
            # served weights are sqrt(omega): the loss squares point weights
            model.state = model.state.update_point_weights(
                point_weights=qn_sampler.get_lambdas()
            )
            # let an instrumented evaluator record the construction stats
            if hasattr(evaluator, "on_qn_batch_info"):
                evaluator.on_qn_batch_info(qn_sampler.info)
            if qn_refresh_every(config) is not None:
                # a refresh re-draws with the same sampler and knobs (pool,
                # floor, weights, Nystrom); the loop re-seeds every draw
                refresh_sampler = qn_sampler
        model.state, logger, steps_done = \
                second_order_step(model=model,
                                config=config,
                                batch=batch,
                                step_offset=step_offset,
                                evaluator=evaluator,
                                logger=logger,
                                log=log,
                                max_iter=second_order_steps,
                                refresh_sampler=refresh_sampler)
        if hasattr(evaluator, "on_qn_end"):     # NaN / line-search stop is data
            evaluator.on_qn_end(steps_done)
        step_offset += steps_done
    
    if hasattr(config.logging, "log_preds") and config.logging.log_preds:
        params = model.state.params
        evaluator.log_preds(params)
        
    stats = jax.local_devices()[0].memory_stats()
    if stats:                                   # None on CPU backends
        peak_gib = stats['peak_bytes_in_use'] / 1024**3
        print(f"Peak GPU memory usage: {peak_gib:.2f} GiB") 
        
    return model, step_offset, logger
    
def init_and_train_and_evaluate(config: ml_collections.ConfigDict, 
                                dataset_fn,
                                model_cls,
                                evaluator_cls,
                                log=True,
                                restore=None):
    """Single-shot training: build the model (physics-informed init if
    configured), phase 1 (first-order) then phase 2 (quasi-Newton on the frozen
    batch). `restore` = {"params", "weights", "first_order_steps"} skips the
    model init and phase 1 and starts phase 2 from that handover checkpoint
    (the QN batch is then drawn at the restored parameters)."""
    # Initialize W&B
    wandb_config = config.wandb
    if log:
        wandb.init(project=wandb_config.project, name=wandb_config.name, reinit=True)

    # Initialize logger
    logger = Logger()

    # Get the dataset, the domain and the initial condition points

    sol_dict, axes_dict, axis_order, pde_params_dict = dataset_fn()
    
    
    dom = make_domain(axes_dict, axis_order)
    initial_dict = make_initial_dict(sol_dict, axes_dict=axes_dict)

    # Initialize the sampler
    res_sampler = get_sampler(config, dom)

    # Initialize the model
    first_order_steps, step_offset = config.optim.first_order_steps, 0
    if restore is not None:
        params = jax.tree_util.tree_map(jnp.asarray, restore["params"])
        pi = find_pi_init(params)
        if pi is not None:
            config.arch.pi_init = jnp.asarray(pi)
        model = model_cls(config=config, axes_dict=axes_dict,
                          initial_dict=initial_dict, pde_params_dict=pde_params_dict)
        weights = {k: jnp.asarray(v) for k, v in restore["weights"].items()}
        model.state = model.state.replace(params=params, weights=weights)
        first_order_steps, step_offset = 0, int(restore.get("first_order_steps", 0))
    elif hasattr(config, "use_pi_init") and config.use_pi_init:
        model = physics_informed_init(config, axes_dict, initial_dict, pde_params_dict, 
                                            model_cls)
    else:
        model = model_cls(config=config, axes_dict=axes_dict, 
                        initial_dict=initial_dict, pde_params_dict=pde_params_dict)

    #print number of trainable parameters
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(model.state.params))
    print(f"Number of trainable parameters: {num_params}")
    # Initialize evaluator
    evaluator = evaluator_cls(config=config, model=model, sol_dict=sol_dict)


    # train and evaluate
    model, _, logger = train_and_evaluate(config=config,
                                            first_order_steps=first_order_steps,
                                            second_order_steps=config.optim.second_order_steps,
                                            model=model,
                                            samplers=res_sampler,
                                            evaluator=evaluator,
                                            step_offset=step_offset,
                                            logger=logger,
                                            log=log,
                                            dom=dom)
    if config.saving.num_keep_ckpts > 0:
            print("Saving checkpoint")
            ckpt_path = os.path.join(os.getcwd(), config.wandb.name, "ckpt")
            save_checkpoint(model.state, ckpt_path, keep=config.saving.num_keep_ckpts)
    return model, evaluator

def train_curriculum(config: ml_collections.ConfigDict, 
                     changing_model_param,
                     model_cls,
                     dataset_fn,
                     evaluator_cls,
                     log=True,):
    
    assert len(config.optim.first_order_steps) == len(changing_model_param)
    # Initialize W&B
    wandb_config = config.wandb
    if log:
        wandb.init(project=wandb_config.project, name=wandb_config.name, reinit=True)

    # Initialize logger
    logger = Logger()
    old_model_params = None

    step_offset = 0
    all_first_order_steps = config.optim.first_order_steps
    all_second_order_steps = config.optim.second_order_steps

    # run 
    for first_order_steps, second_order_steps, parameter in \
        zip(all_first_order_steps, all_second_order_steps, changing_model_param):

        param_value = parameter
        print(f"Training with parameter value: {param_value}")

        # Get the dataset, the domain and the initial condition points
        sol_dict, axes_dict, axis_order, pde_params_dict = dataset_fn(param_value)
        dom = make_domain(axes_dict, axis_order)
        initial_dict = make_initial_dict(sol_dict, axes_dict=axes_dict)

        # Initialize the sampler
        res_sampler = get_sampler(config, dom)
        
        # Initialize the model
        model = model_cls(config=config, axes_dict=axes_dict, initial_dict=initial_dict, pde_params_dict=pde_params_dict)
        

        # if we have a pre-trained model, load the params
        if old_model_params is not None:
            new_model = model_cls(config=config, axes_dict=axes_dict, initial_dict=initial_dict, pde_params_dict=pde_params_dict)
            new_model.state = model.state.replace(params=old_model_params)
            model = new_model
        
        # Initialize evaluator
        evaluator = evaluator_cls(config=config, model=model, sol_dict=sol_dict)

        
        model, step_offset, logger = train_and_evaluate(config=config,
                                                first_order_steps=first_order_steps,
                                                second_order_steps=second_order_steps,
                                                model=model,
                                                samplers=res_sampler,
                                                evaluator=evaluator,
                                                step_offset=step_offset,
                                                logger=logger,
                                                log=log)

        old_model_params = model.state.params
    
    if config.saving.num_keep_ckpts > 0:
            print("Saving checkpoint")
            ckpt_path = os.path.join(os.getcwd(), config.wandb.name, "ckpt")
            save_checkpoint(model.state, ckpt_path, keep=config.saving.num_keep_ckpts)
    return model, evaluator

def track_time_windows_errors(window_error_table, window_error_history, log_dict, idx):

    # Keep only error metrics
    error_dict = {
        key: float(value)
        for key, value in log_dict.items()
        if "error" in key.lower()
    }

    # --- Lazy table creation (after first window) ---
    if window_error_table is None:
        # Table columns: window_idx + all error keys
        columns = ["row_type", "window_idx"] + list(error_dict.keys())
        window_error_table = wandb.Table(columns=columns, log_mode="MUTABLE")

        # Initialize history storage
        for key in error_dict:
            window_error_history[key] = []

    # --- Add row for this window ---
    row = ["window", idx] + [error_dict[k] for k in error_dict]
    window_error_table.add_data(*row)

    # --- Store for averaging ---
    for key, value in error_dict.items():
        window_error_history[key].append(value)

    # --- Log table (safe to call every window) ---
    wandb.log({"windows/l2_error_table": window_error_table})
    return window_error_table, window_error_history

def train_time_windows(config: ml_collections.ConfigDict, 
                     model_cls,
                     dataset_fn,
                     evaluator_cls,
                     log=True,): 
    if log:
        wandb_config = config.wandb
        wandb.init(project=wandb_config.project, name=wandb_config.name)

    sol_dict, axes_dict, axis_order, pde_params_dict = dataset_fn()
    initial_dict = make_initial_dict(sol_dict, axes_dict=axes_dict)

    # add t to the sol_dict for conflicts of slicing in the last window
    sol_dict["t"] = axes_dict["t"]

    time_window_bounds = get_time_window_bounds(
        len(axes_dict["t"]),
        config.training.num_time_windows,
    )
    interval_lengths = [stop - start - 1 for start, stop in time_window_bounds]
    if len(set(interval_lengths)) > 1:
        print(
            "Using uneven time windows to cover all time points. "
            f"Intervals per window: {interval_lengths}"
        )

    # Each window resets its local t-axis to start at zero.
    step_offset = 0
    logger = Logger()
    
    window_error_table = None            
    window_error_history = {}      
         
    # collect all final predictions for final evaluation, that is every key in sol_dict except t
    final_predictions = {key: [] for key in sol_dict if key != "t"}
    ref_solution = {key: [] for key in sol_dict if key != "t"}

    for idx, (window_start, window_stop) in enumerate(time_window_bounds):
        # Get the reference solution for the current time window
        sol_dict_window = get_sol_dict_window_by_indices(
            sol_dict,
            window_start,
            window_stop,
        )
        axes_dict['t'] = sol_dict_window['t']   

        # check first if config has the attr then if they are true
        if hasattr(config, "use_pi_init") and config.use_pi_init:
            model = physics_informed_init(config, axes_dict, initial_dict, pde_params_dict, 
                                              model_cls)
        else:
            model = model_cls(config=config, axes_dict=axes_dict, 
                          initial_dict=initial_dict, pde_params_dict=pde_params_dict)
            
        
        evaluator = evaluator_cls(config=config, model=model, sol_dict=sol_dict_window)

        dom = make_domain(axes_dict, axis_order)
        res_sampler = get_sampler(config, dom)
        # if inital or bc sampler do those somehow
        if hasattr(config.dia, "initial_sampler") and config.dia.initial_sampler:
            ICSampler = IC_Axis_Sampler(
                init_dict=initial_dict,
                axes_dict=axes_dict,
                batch_size=config.dia.initial_batch_size,
                mode=config.dia.initial_sampler_mode,
            )
            samplers = {
                "res": res_sampler,
                "ics": iter(ICSampler),
            }
        else:
            samplers = res_sampler

        model, step_offset, logger = train_and_evaluate(config=config,
                                                first_order_steps=config.optim.first_order_steps,
                                                second_order_steps=config.optim.second_order_steps,
                                                model=model,
                                                samplers=samplers,
                                                evaluator=evaluator,
                                                step_offset=step_offset,
                                                logger=logger,
                                                log=log)
        
        # compute l2 error for the current time window
        evaluator.log_errors(model.state.params)  
        log_dict = evaluator.log_dict
        window_error_table, window_error_history = track_time_windows_errors(window_error_table, window_error_history, log_dict, idx)
        
        stats = jax.local_devices()[0].memory_stats()
        if stats:                                   # None on CPU backends
            peak_gib = stats['peak_bytes_in_use'] / 1024**3
            print(f"Peak GPU memory usage: {peak_gib:.2f} GiB")

        # collect final predictions
        preds = model.pred_domain(model.state.params)
        # unpack the dict and append to final_predictions
        for key in final_predictions:
            if idx != config.training.num_time_windows - 1:
                # exclude last time step
                final_predictions[key].append(preds[key][:-1])
                ref_solution[key].append(sol_dict_window[key][:-1])
            else:
                final_predictions[key].append(preds[key])  
                ref_solution[key].append(sol_dict_window[key])  
        # Save checkpoint for the current time window
        if config.saving.num_keep_ckpts > 0:
            print("Saving checkpoint for time window {}".format(idx+1))
            ckpt_path = os.path.join(os.getcwd(), config.wandb.name, "ckpt", "time_window_{}".format(idx+1))
            save_checkpoint(model.state, ckpt_path, keep=config.saving.num_keep_ckpts)

        # Update the initial condition for the next time window
        new_initial_dict = model.pred_last_t(model.state.params)
        # check if new_init dict contains NaNs
        for key in new_initial_dict:
            if jnp.isnan(new_initial_dict[key]).any():
                raise ValueError(f"NaN values found in the new initial condition for key '{key}' at time window {idx+1}.")
        # check if keys match

        assert set(new_initial_dict.keys()) == set(initial_dict.keys()), "Keys of the new and old initial dict do not match!"
        initial_dict = new_initial_dict
        del model

    
    print("Training on all time windows completed. Evaluating final predictions...")
    # for all keys in final_predictions stack along time axis and save to wandb
    for key in final_predictions:
        full_pred = jnp.concatenate(final_predictions[key], axis=0)
        full_ref_solution = jnp.concatenate(ref_solution[key], axis=0)
        # print shape
        print(f"Final prediction shape for {key}: {full_pred.shape}")
        rel_l2_error = jnp.linalg.norm(full_ref_solution - full_pred) / jnp.linalg.norm(full_ref_solution)
        wandb.run.summary[f"final_rel_l2_error_{key}"] = float(rel_l2_error)
        print(f"Final relative L2 error for {key}: {rel_l2_error:.3e}")


# =============================================================================
# Two-phase runs: the phase-1 checkpoint, the off-batch loss, the runners
# =============================================================================
# Phase 1 (first-order) is trained ONCE per (benchmark, seed) and saved as a
# hand-over checkpoint; every second-order run of that seed restores it, so all
# sampling rules start from the same network. During the second-order phase the
# OffBatchTracker logs the loss on the frozen batch next to the loss on a fixed
# held-out set the optimizer never sees (the off-batch loss) and their ratio.

HELDOUT_SEED = 90210        # held-out set: UniformSampler(dom, N_HELDOUT, HELDOUT_SEED + seed)
N_HELDOUT = 65536
# refresh-event rows written per run to metrics.jsonl; every event still enters
# the summary aggregates, only the row stream is thinned
REFRESH_LOG_MAX = 4000


# ------------------------------------------------------------------ utilities

def _sha1(obj):
    return hashlib.sha1(str(obj).encode()).hexdigest()[:12]


def params_hash(params):
    """12-hex SHA-1 of the parameter leaves (bitwise identity of a network)."""
    h = hashlib.sha1()
    for leaf in jax.tree_util.tree_leaves(params):
        h.update(np.asarray(leaf).tobytes())
    return h.hexdigest()[:12]


def config_hash(config):
    try:
        return _sha1(json.dumps(config.to_dict(), sort_keys=True, default=str))
    except Exception:
        return _sha1(repr(config))


def _yamlable(obj):
    """Plain-YAML view of a ConfigDict tree: non-serializable leaves -> str."""
    if isinstance(obj, dict):
        return {str(k): _yamlable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_yamlable(v) for v in obj]
    if obj is None or isinstance(obj, (str, bool, int, float)):
        return obj
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return _yamlable(obj.tolist())
    return str(obj)


def dump_config(config, path):
    import yaml
    with open(path, "w") as fh:
        yaml.safe_dump(_yamlable(config.to_dict()), fh, sort_keys=False,
                       default_flow_style=False)


def save_params(params, path):
    with open(path, "wb") as fh:
        pickle.dump(jax.tree_util.tree_map(np.asarray, params), fh)


def read_json(path, default=None):
    if not os.path.exists(path):
        return default
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception as exc:
        print(f"could not read {path} ({exc})")
        return default


def _json_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (np.ndarray, jnp.ndarray)):
        return np.asarray(obj).tolist()
    return str(obj)


def _write_json(path, obj):
    tmp = f"{path}.{os.getpid()}.tmp"           # per process: no torn files
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)
    os.replace(tmp, path)


def gpu_info():
    """Accelerator of this process (device kind and backend; no host names)."""
    d = jax.devices()[0]
    info = {"device_kind": getattr(d, "device_kind", str(d)),
            "platform": jax.default_backend()}
    return info["device_kind"], info


def now_iso():
    return datetime.datetime.now().isoformat(timespec="seconds")


def l2re_of(evaluator, params):
    """Relative L2 error from an evaluator's log_errors (first '*error' key)."""
    evaluator.log_dict = {}
    try:
        evaluator.log_errors(params)
        errs = {k: float(v) for k, v in evaluator.log_dict.items()}
    except Exception:
        errs = {}
    return next((v for k, v in errs.items() if "error" in k), float("nan")), errs


# ------------------------------------------------------- phase-1 checkpoints

def save_phase1(path, params, weights, **meta):
    """Phase-1 hand-over checkpoint: numpy pytree + loss weights (+ metadata),
    CPU-loadable, restored by init_and_train_and_evaluate(restore=...)."""
    ck = {"params": jax.tree_util.tree_map(np.asarray, params),
          "weights": {k: float(v) for k, v in weights.items()},
          "phase1_ckpt_hash": params_hash(params), **meta}
    tmp = f"{path}.{os.getpid()}.tmp"           # per process: no torn files
    with open(tmp, "wb") as fh:
        pickle.dump(ck, fh)
    os.replace(tmp, path)
    return ck


def load_phase1(path):
    with open(path, "rb") as fh:
        ck = pickle.load(fh)
    ck["path"] = path
    return ck


def build_from_checkpoint(config, ckpt, dataset_fn, model_cls, evaluator_cls):
    """(model, evaluator, dom) at a phase-1 checkpoint, without training: the
    model is rebuilt with the checkpoint's physics-informed-init kernel (if
    any), its params and its frozen loss weights."""
    sol_dict, axes_dict, axis_order, pde_params_dict = dataset_fn()
    dom = make_domain(axes_dict, axis_order)
    initial_dict = make_initial_dict(sol_dict, axes_dict=axes_dict)
    params = jax.tree_util.tree_map(jnp.asarray, ckpt["params"])
    pi = find_pi_init(params)
    if pi is not None:
        config.arch.pi_init = jnp.asarray(pi)
    model = model_cls(config=config, axes_dict=axes_dict, initial_dict=initial_dict,
                      pde_params_dict=pde_params_dict)
    model.state = model.state.replace(
        params=params,
        weights={k: jnp.asarray(v) for k, v in ckpt["weights"].items()})
    evaluator = evaluator_cls(config=config, model=model, sol_dict=sol_dict)
    return model, evaluator, dom


# ----------------------------------------------------------- off-batch loss

class OffBatchTracker:
    """Evaluator wrapper that tracks the off-batch loss of the second-order phase.

    Behaves exactly like the wrapped evaluator during phase 1. Once
    second_order_step calls `on_qn_start`, every log step also records the
    UNWEIGHTED mean squared residual (mean r^2, first equation) on
        loss_frozen   the batch being optimized,
        loss_batch0   the first (hand-over) batch, fixed,
        loss_heldout  a fixed held-out set of n_heldout uniform points that the
                      optimizer never sees (the off-batch loss),
    and gap = loss_heldout / loss_frozen, the overfit of the frozen batch.
    Writes <out_dir>/metrics.jsonl (one row per log step and per logged
    refresh), and on write_summary params_end.pkl and summary.json.
    """

    REFRESH_KEYS = ("rule", "ess", "batch_distinct", "score_max_over_median",
                    "max_rba", "resample_s", "batch_wall_s", "probe_wall_s")

    def __init__(self, inner, model, dom, config, out_dir, meta=None,
                 n_heldout=N_HELDOUT):
        self.inner = inner
        self.model = model
        self.config = config
        self.meta = dict(meta or {})
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.jsonl = os.path.join(out_dir, "metrics.jsonl")
        open(self.jsonl, "w").close()           # a re-run starts a fresh series
        self.r_fn = raw_residual_fn(model)
        self.heldout = next(iter(UniformSampler(
            dom, n_heldout, HELDOUT_SEED + int(config.seed))))
        self.log_every = int(config.logging.log_every_steps)
        self.t0 = time.time()
        self.qn_started = False
        self.qn_calls = 0
        self.batch0 = None
        self.qn_info = None
        self.n_refresh = 0
        self.n_reset = 0
        self.refresh_rows = 0
        self._refresh_acc = {}
        self._stride = None
        self.best_l2re = None
        self.best_l2re_step = None
        self.handover_l2re = None
        self.phase1_ckpt_hash = None
        self.steps_done = None
        self.last = {}

    # ------------------------------------------------------------- hooks
    def on_qn_batch_info(self, info):
        self.qn_info = {k: v for k, v in info.items() if k != "omega"}

    def on_qn_start(self, params, batch, point_weights):
        self.qn_started = True
        self.batch0 = jnp.asarray(batch)
        self.handover_l2re = self._l2re(params)
        self.phase1_ckpt_hash = params_hash(params)
        print(f"[qn] handover_l2re={self.handover_l2re:.4g} "
              f"phase1_ckpt_hash={self.phase1_ckpt_hash}")

    def _refresh_stride(self):
        """Write at most REFRESH_LOG_MAX refresh rows per run."""
        if self._stride is None:
            period = qn_refresh_every(self.config) or 1
            expected = max(1, (int(self.config.optim.second_order_steps) - 1)
                           // max(period, 1))
            self._stride = max(1, -(-expected // REFRESH_LOG_MAX))
        return self._stride

    def on_qn_refresh(self, step, info):
        """One mid-phase batch refresh. `step` is global (phase 1 included);
        `qn_step` counts from the hand-over, as the metrics rows do."""
        ev = {"step": int(step), "qn_step": info.get("qn_step"),
              **{k: info.get(k) for k in self.REFRESH_KEYS}}
        if info.get("reset"):                       # *_reset rules only
            ev["reset"] = True
        for name in ("ess", "batch_distinct", "resample_s"):
            v = ev.get(name)
            if isinstance(v, (int, float)) and np.isfinite(v):
                a = self._refresh_acc.setdefault(
                    name, {"n": 0, "sum": 0.0, "min": v, "max": v, "last": v})
                a["n"] += 1
                a["sum"] += float(v)
                a["min"] = min(a["min"], v)
                a["max"] = max(a["max"], v)
                a["last"] = v
        self.n_refresh += 1
        self.n_reset += int(bool(info.get("reset")))
        if (self.n_refresh - 1) % self._refresh_stride():   # thinned row stream
            return
        self.refresh_rows += 1
        with open(self.jsonl, "a") as fh:
            fh.write(json.dumps({"event": "refresh", **ev}) + "\n")

    def on_qn_end(self, steps_done):
        """Completed QN iterations, as counted by second_order_step: less than
        the budget means a NaN loss or a line-search failure stopped it."""
        self.steps_done = int(steps_done)

    # ---------------------------------------------------------- internals
    def _mean_r2(self, params, X, chunk=32768):
        tot, n = 0.0, 0
        for i in range(0, X.shape[0], chunk):
            r = self.r_fn(params, X[i:i + chunk])
            tot += float(jnp.sum(r ** 2))
            n += r.shape[0]
        return tot / max(n, 1)

    def _l2re(self, params):
        return l2re_of(self.inner, params)[0]

    # ------------------------------------------------------------- __call__
    def __call__(self, state, batch):
        d = dict(self.inner(state, batch))
        if not self.qn_started:
            return d
        self.qn_calls += 1
        step = self.qn_calls * self.log_every
        params = state.params
        lf = self._mean_r2(params, batch)          # CURRENT batch
        l0 = self._mean_r2(params, self.batch0)    # FIRST batch, fixed
        lh = self._mean_r2(params, self.heldout)   # off-batch
        l2re = next((float(v) for k, v in d.items() if "error" in k),
                    float("nan"))
        if np.isfinite(l2re) and (self.best_l2re is None or l2re < self.best_l2re):
            self.best_l2re, self.best_l2re_step = float(l2re), int(step)
        rec = {"step": step, "wall_s": round(time.time() - self.t0, 3),
               "loss_frozen": lf, "loss_batch0": l0, "loss_heldout": lh,
               "gap": (lh / lf) if lf > 0 else float("inf"),
               "gap_batch0": (lh / l0) if l0 > 0 else float("inf"),
               "l2re": l2re, "best_l2re": self.best_l2re}
        with open(self.jsonl, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        self.last = rec
        d.update(frozen_loss=lf, batch0_loss=l0, heldout_loss=lh,
                 gap=rec["gap"])
        return d

    # ------------------------------------------------------------- summary
    def _refresh_aggregates(self):
        """Batch statistics over EVERY refresh (the thinning applies to the row
        stream only); the one-shot fields of the summary describe the FIRST
        batch."""
        if not self.n_refresh:
            return {}
        out = {"refresh_rows_logged": self.refresh_rows,
               "refresh_log_stride": self._refresh_stride()}
        for name, a in self._refresh_acc.items():
            out[f"refresh_{name}_mean"] = a["sum"] / a["n"]
            out[f"refresh_{name}_min"] = a["min"]
            out[f"refresh_{name}_max"] = a["max"]
            out[f"refresh_{name}_last"] = a["last"]
        if "resample_s" in self._refresh_acc:
            out["refresh_resample_s_total"] = self._refresh_acc["resample_s"]["sum"]
        return out

    def write_summary(self, extra=None):
        info = self.qn_info or {}
        final = self.last or {}
        save_params(self.model.state.params,
                    os.path.join(self.out_dir, "params_end.pkl"))
        budget = int(self.config.optim.second_order_steps)
        status = ("ok" if np.isfinite(final.get("loss_heldout", np.nan))
                  else "diverged")
        if status == "ok" and self.steps_done is not None and self.steps_done < budget:
            status = "truncated"        # NaN loss / line-search stop mid-run
        summary = {
            **self.meta,
            "status": status,
            "steps_done": self.steps_done,
            "n_batch": info.get("n_batch"),
            "batch_distinct": info.get("batch_distinct"),
            "batch_wall_s": info.get("batch_wall_s"),
            # the sampler's RESOLVED knobs and diagnostics
            **{k: info[k] for k in
               ("rad_k", "rad_c", "rad_floor", "max_rba", "k_anneal",
                # the Sobol-floor split of sled: n_floor Sobol points at weight 1
                # next to n_scored draws from the unfloored proposal
                "n_floor", "n_scored", "pool_size", "pool_mult",
                # floor_rho: the raw Spearman and whether it set the floor (the
                # RESOLVED floor is the "floor" field below)
                "floor_rho", "rho_floor",
                # sled_fixed (ablation a1): the Spearman measured next to the
                # fixed floor, which the floor did not use
                "rho_measured",
                # pinnacle_k: eigenpairs kept, the batch's convergence degree
                # against a uniform subset, embedding / selection wall time
                "pinnacle")
               if k in info},
            "qn_type": info.get("rule"),
            "refresh_every": qn_refresh_every(self.config),
            "dia_batch_size": int(self.config.dia.batch_size),
            "weight_convention": WEIGHT_CONVENTION,
            "score_stats": info.get("score_stats"),
            "ess": info.get("ess"),
            "floor": info.get("floor"),
            "floor_source": info.get("floor_source"),
            "pi_omega_sum": info.get("pi_omega_sum"),
            "probe_wall_s": info.get("probe_wall_s"),
            "handover_l2re": self.handover_l2re,
            "best_l2re": self.best_l2re,
            "best_l2re_step": self.best_l2re_step,
            "final_l2re": final.get("l2re"),
            "final_gap": final.get("gap"),
            "final_loss_frozen": final.get("loss_frozen"),
            "final_loss_batch0": final.get("loss_batch0"),
            "final_loss_heldout": final.get("loss_heldout"),
            "phase1_ckpt_hash": self.phase1_ckpt_hash,
            "params_end_hash": params_hash(self.model.state.params),
            "first_order_steps": int(self.config.optim.first_order_steps),
            "qn_steps": budget,
            "qn_refresh_events": self.n_refresh,
            # *_reset rules only: the optimizer re-initialised at every refresh
            **({"reset_memory": True, "qn_resets": self.n_reset}
               if qn_reset_memory(self.config) else {}),
            **self._refresh_aggregates(),
            "wall_clock_s": round(time.time() - self.t0, 1),
            **(extra or {}),
        }
        _write_json(os.path.join(self.out_dir, "summary.json"), summary)
        return summary


def off_batch_factory(evaluator_cls, dom, out_dir, meta=None, n_heldout=N_HELDOUT):
    """evaluator_cls drop-in for init_and_train_and_evaluate that wraps the
    benchmark evaluator in an OffBatchTracker; the built tracker is exposed as
    factory.instance for write_summary()."""
    def factory(config, model, sol_dict):
        inner = evaluator_cls(config=config, model=model, sol_dict=sol_dict)
        factory.instance = OffBatchTracker(inner, model, dom, config, out_dir,
                                           meta=meta, n_heldout=n_heldout)
        return factory.instance
    factory.instance = None
    return factory


# ------------------------------------------------------------------ runners

def train_phase1(config, dataset_fn, model_cls, evaluator_cls, ckpt_path,
                 log=True, force=False, meta=None):
    """Train phase 1 (first-order only) with `config` and save the hand-over
    checkpoint `ckpt_path` (+ phase1.json next to it). Skips when the
    checkpoint exists, after checking it was trained for the same number of
    first-order steps. Returns the phase-1 record."""
    ck_dir = os.path.dirname(os.path.abspath(ckpt_path))
    json_path = os.path.join(ck_dir, "phase1.json")
    want = int(config.optim.first_order_steps)
    if os.path.exists(ckpt_path) and not force:
        ck = load_phase1(ckpt_path)
        rec = read_json(json_path, None) or {
            k: v for k, v in ck.items() if k not in ("params", "weights")}
        have = int(ck.get("first_order_steps", -1))
        if have != want:
            raise SystemExit(
                f"[phase1] {ckpt_path} was trained for {have} first-order steps "
                f"but the config asks for {want}; give the other budget its own "
                "checkpoint directory instead of overwriting.")
        # the seed and the first-order sampler are part of a hand-over's
        # identity too (RT's phase 1 runs LAS); checkpoints written by other
        # code may lack the fields, then only the budget can be checked
        for key, val in (("seed", int(config.seed)),
                         ("dia_type", str(config.dia.type))):
            got = ck.get(key, rec.get(key))
            if got is not None and str(got) != str(val):
                raise SystemExit(f"[phase1] {ckpt_path} has {key}={got!r} but the "
                                 f"config asks for {val!r}")
        print(f"[phase1] {ckpt_path} exists (hash {ck['phase1_ckpt_hash']}, "
              f"{have} steps), skipping")
        return rec
    os.makedirs(ck_dir, exist_ok=True)
    cfg_hash = config_hash(config)
    config.optim.second_order_steps = 0
    dump_config(config, os.path.join(ck_dir, "config.yaml"))
    gpu, ginfo = gpu_info()
    t0 = time.time()
    model, evaluator = init_and_train_and_evaluate(
        config, dataset_fn=dataset_fn, model_cls=model_cls,
        evaluator_cls=evaluator_cls, log=log)
    wall = round(time.time() - t0, 1)
    run_id = wandb.run.id if (log and wandb.run) else None
    if log:
        wandb.finish()
    handover, errs = l2re_of(evaluator, model.state.params)
    ck = save_phase1(ckpt_path, model.state.params, model.state.weights,
                     **{**(meta or {}), "seed": int(config.seed),
                        "first_order_steps": want, "config_hash": cfg_hash,
                        "dia_type": str(config.dia.type),
                        "handover_l2re": handover})
    rec = {**(meta or {}), "seed": int(config.seed), "stage": "phase1",
           "status": "ok", "first_order_steps": want,
           "optimizer": str(config.optim.optimizer),
           "dia_type": str(config.dia.type),
           "batch_size": int(config.dia.batch_size),
           "handover_l2re": handover, "handover_errors": errs,
           "loss_weights": ck["weights"],
           "phase1_ckpt_hash": ck["phase1_ckpt_hash"],
           "config_hash": cfg_hash, "ckpt_path": ckpt_path,
           "wandb_project": config.wandb.project, "wandb_name": config.wandb.name,
           "wandb_run_id": run_id, "gpu": gpu, "gpu_info": ginfo,
           "started": datetime.datetime.fromtimestamp(t0).isoformat(timespec="seconds"),
           "finished": now_iso(), "wall_clock_s": wall}
    _write_json(json_path, rec)
    print(f"[phase1] handover_l2re={handover:.4g} hash={ck['phase1_ckpt_hash']} "
          f"wall={wall}s -> {ckpt_path}")
    return rec


class RunConflict(Exception):
    """A run directory holds a finished run of another configuration or from
    another phase-1 checkpoint; nothing was written."""


def run_key(config):
    """What identifies a second-order run beyond its seed's checkpoint: the
    frozen-batch settings, the optimizer and the budget. Two studies that ask
    for the same key share one run directory instead of training it twice."""
    qn = config.qn_batch.to_dict() if "qn_batch" in config else None
    return _sha1(json.dumps({"qn_batch": qn, "seed": int(config.seed),
                             "optimizer": str(config.optim.optimizer),
                             "qn_steps": int(config.optim.second_order_steps)},
                            sort_keys=True, default=str))


def train_phase2(config, ckpt_path, dataset_fn, model_cls, evaluator_cls, out_dir,
                 meta=None, log=True, force=False, n_heldout=N_HELDOUT):
    """The second-order phase of `config` from the phase-1 checkpoint
    `ckpt_path`, with the off-batch loss tracked. Writes config.yaml,
    metrics.jsonl, params_end.pkl and summary.json into `out_dir` and returns
    the summary. Skips when out_dir already holds a summary of the same
    run_key and phase-1 checkpoint whose status is not "failed" (a truncated or
    diverged run is a result, not an error); raises RunConflict when it holds a
    different configuration or a run from another checkpoint. Two processes
    must not run the same out_dir at the same time.

    `log=True` is required for the tracker: second_order_step only calls the
    evaluator when logging (set WANDB_MODE=offline/disabled to keep W&B local).
    """
    summary_path = os.path.join(out_dir, "summary.json")
    key = run_key(config)
    ck = load_phase1(ckpt_path)
    h = params_hash(ck["params"])
    if h != ck["phase1_ckpt_hash"]:
        raise SystemExit(f"checkpoint {ckpt_path} is corrupt: params hash {h} != "
                         f"stored {ck['phase1_ckpt_hash']}")
    prev = read_json(summary_path, None)
    if prev is not None and prev.get("status") != "failed" and not force:
        if prev.get("run_key") != key:
            raise RunConflict(
                f"[qn] {out_dir} holds a finished run of a DIFFERENT configuration "
                f"({prev.get('run_key')} != {key}); use another output directory "
                "or force=True to overwrite it")
        if prev.get("phase1_ckpt_hash") != h:
            raise RunConflict(
                f"[qn] {out_dir} holds a finished run from ANOTHER phase-1 "
                f"checkpoint ({prev.get('phase1_ckpt_hash')} != {h}, {ckpt_path}); "
                "use another output directory or force=True to overwrite it")
        print(f"[qn] {out_dir}: done ({prev.get('status')}), skipping")
        return prev
    config.optim.first_order_steps = int(ck["first_order_steps"])  # the checkpoint's
    os.makedirs(out_dir, exist_ok=True)
    cfg_hash = config_hash(config)
    sol_dict, axes_dict, axis_order, _ = dataset_fn()
    dom = make_domain(axes_dict, axis_order)
    factory = off_batch_factory(evaluator_cls, dom, out_dir, meta=meta,
                                n_heldout=n_heldout)
    dump_config(config, os.path.join(out_dir, "config.yaml"))
    gpu, ginfo = gpu_info()
    t0 = time.time()
    error = None
    try:
        init_and_train_and_evaluate(config, dataset_fn=dataset_fn,
                                    model_cls=model_cls, evaluator_cls=factory,
                                    log=log, restore=ck)
    except Exception as exc:                            # divergence is a result
        error = repr(exc)
        print(f"[qn] {out_dir}: second-order phase raised {error}")
    run_id = wandb.run.id if (log and wandb.run) else None
    if log:
        wandb.finish()
    extra = {"config_hash": cfg_hash, "run_key": key, "gpu": gpu, "gpu_info": ginfo,
             "optimizer": str(config.optim.optimizer),
             "phase1_ckpt_path": ckpt_path,
             "phase1_ckpt_hash_expected": ck["phase1_ckpt_hash"],
             "wandb_project": config.wandb.project, "wandb_name": config.wandb.name,
             "wandb_run_id": run_id,
             "started": datetime.datetime.fromtimestamp(t0).isoformat(timespec="seconds"),
             "finished": now_iso()}
    if factory.instance is None:                        # failed before training
        summary = {**(meta or {}), "status": "failed", "error": error,
                   "qn_steps": int(config.optim.second_order_steps), **extra}
        _write_json(summary_path, summary)
    else:
        summary = factory.instance.write_summary(extra=extra)
        if summary.get("phase1_ckpt_hash") != ck["phase1_ckpt_hash"]:
            summary["status"] = "failed"
            summary["error"] = (f"hand-over hash {summary.get('phase1_ckpt_hash')} "
                                f"!= checkpoint {ck['phase1_ckpt_hash']}")
        if error is not None:
            summary["status"] = "failed"
            summary["error"] = error
        _write_json(summary_path, summary)
    print(f"[qn] status={summary['status']} final_l2re={summary.get('final_l2re')} "
          f"best={summary.get('best_l2re')}@{summary.get('best_l2re_step')} "
          f"gap={summary.get('final_gap')} wall={summary.get('wall_clock_s')}s "
          f"-> {out_dir}")
    return summary


# tiny budgets for a smoke test of a study script (never for results); refresh
# periods are capped so a refreshing rule refreshes within the smoke budget
SMOKE = {"first_order_steps": 30, "qn_steps": 40, "log_every": 10,
         "n_heldout": 2048, "refresh_every": 10}


# The phase-1 hand-overs shipped with the repository: checkpoints/<benchmark>/
# seed<k>/phase1.pkl, the checkpoints the paper's comparison runs started from
# (checkpoints/README.md). Read-only: a study never trains into them.
SHARED_PHASE1_SEEDS = (2, 3, 5, 7, 11, 13, 17, 19, 23, 27)


def shared_phase1_dir(benchmark):
    """checkpoints/<benchmark>, relative to the working directory (a benchmark
    directory when a study script runs), so run records hold no absolute path."""
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "checkpoints")
    return os.path.relpath(os.path.join(root, benchmark))


def shared_seeds(seeds):
    """The seeds of `seeds` that have a shared hand-over."""
    return [int(s) for s in seeds if int(s) in SHARED_PHASE1_SEEDS]


def run_study(study, benchmark, runs, base_config, dataset_fn, model_cls,
              evaluator_cls, out_root="runs", phase1_dir=None, smoke=False,
              force=False, shared_phase1=False):
    """Two-phase study driver used by the benchmarks' study scripts.

    `runs` is a list of dicts with keys seed, name and config (a second-order
    config, e.g. configs.sampler_comparison.get_config(...)); any other keys
    (rule, n, ...) are recorded in the run's summary. Seeds are the outer loop:
    the phase-1 checkpoint of a seed (<phase1_dir>/seed<k>/phase1.pkl, default
    <out_root>/phase1) is trained with base_config.get_config(seed=seed) if it
    does not exist, and then every run of that seed restores it, writing
    <out_root>/<name>/seed<k>/. The run directory does not depend on the
    study: a run two studies share (same name and run_key) is trained once.
    Finished runs are skipped, so a study can be resumed by starting it again;
    a run whose directory holds a conflicting run (RunConflict) is reported and
    skipped, and the study goes on. `smoke` uses tiny budgets and writes to
    <out_root>_smoke/.

    `shared_phase1` starts every seed from the repository's shared hand-over
    (shared_phase1_dir: checkpoints/<benchmark>/seed<k>/phase1.pkl), which is
    never trained or written: a seed without one is refused before anything
    runs. The runs then go to <out_root>_shared/, so runs from shared and from
    locally trained hand-overs never meet in one directory; with `smoke` the
    hand-overs keep their budget and only phase 2 is shortened.
    """
    if shared_phase1:
        if phase1_dir is not None:
            raise ValueError("phase1_dir and shared_phase1 exclude each other")
        phase1_dir = shared_phase1_dir(benchmark)
        out_root = out_root.rstrip("/") + "_shared"
    if smoke:
        out_root = out_root.rstrip("/") + "_smoke"
    phase1_dir = phase1_dir or os.path.join(out_root, "phase1")
    seeds = list(dict.fromkeys(int(r["seed"]) for r in runs))
    if shared_phase1:
        missing = [s for s in seeds
                   if not os.path.exists(os.path.join(phase1_dir, f"seed{s}", "phase1.pkl"))]
        if missing:
            raise SystemExit(f"[{study}] no shared hand-over for seed(s) {missing} in "
                             f"{phase1_dir} (shared seeds: {list(SHARED_PHASE1_SEEDS)})")
    print(f"[{study}] {benchmark}: {len(runs)} runs over seeds {seeds} -> "
          f"{out_root} (phase 1: {phase1_dir})")
    summaries, conflicts = [], []
    for seed in seeds:
        ckpt_path = os.path.join(phase1_dir, f"seed{seed}", "phase1.pkl")
        config = base_config.get_config(seed=seed)
        if smoke and not shared_phase1:
            config.optim.first_order_steps = SMOKE["first_order_steps"]
            config.logging.log_every_steps = SMOKE["log_every"]
        train_phase1(config, dataset_fn, model_cls, evaluator_cls, ckpt_path,
                     meta={"benchmark": benchmark})
        for run in (r for r in runs if int(r["seed"]) == seed):
            config = run["config"]
            if smoke:
                config.optim.second_order_steps = SMOKE["qn_steps"]
                config.logging.log_every_steps = SMOKE["log_every"]
                if qn_refresh_every(config) is not None:
                    config.qn_batch.refresh_every = min(
                        qn_refresh_every(config), SMOKE["refresh_every"])
            out_dir = os.path.join(out_root, run["name"], f"seed{seed}")
            meta = {"benchmark": benchmark, "study": study,
                    **{k: v for k, v in run.items() if k != "config"}}
            print("\n" + "=" * 70 + f"\n=== {benchmark} {study} {run['name']} "
                  f"seed {seed} ===\n" + "=" * 70)
            try:
                summaries.append(train_phase2(
                    config, ckpt_path, dataset_fn, model_cls, evaluator_cls,
                    out_dir, meta=meta, force=force,
                    n_heldout=SMOKE["n_heldout"] if smoke else N_HELDOUT))
            except RunConflict as exc:
                print(f"REFUSED {exc}")
                conflicts.append(str(exc))
    if conflicts:
        print(f"\n[{study}] {len(conflicts)} run(s) refused (conflicting run "
              "directories, nothing written):")
        for msg in conflicts:
            print("   " + msg)
    return summaries
