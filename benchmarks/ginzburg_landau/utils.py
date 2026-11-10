import scipy.io
import jax.numpy as jnp


def get_dataset(time_range=(0.0, 0.8)):
    """
    Loads the Ginzburg-Landau dataset

    Parameters
    ----------
    time_range : tuple
        Fraction of the full time axis to keep, e.g. (0.0, 0.8).
        The kept time axis is shifted to start at t = 0.

    Returns
    -------
    sol : dict
        Solution dictionary containing 'u', 'v'
    axes : dict
        Axes dictionary containing 't', 'x', 'y'
    axis_order : tuple
        Order of the axes
    pde_extra_params : dict
        Containing 'eps' and 'k'
    """
    data = scipy.io.loadmat("data/ginzburg_landau_square.mat")

    u_ref = data["usol"]
    v_ref = data["vsol"]

    # PDE parameters
    eps = data["eps"].flatten()[0]
    k = data["k"].flatten()[0]

    t_star = data["t"].flatten()
    x_star = data["x"].flatten()
    y_star = data["y"].flatten()

    # Only use a fraction of the data
    start_time_step = int(time_range[0] * len(t_star))
    end_time_step = int(time_range[1] * len(t_star))

    u_ref = u_ref[start_time_step:end_time_step, :, :]
    v_ref = v_ref[start_time_step:end_time_step, :, :]
    t_star = t_star[start_time_step:end_time_step] - t_star[start_time_step]

    sol = {
        "u": jnp.asarray(u_ref),
        "v": jnp.asarray(v_ref),
    }
    axes = {
        "t": jnp.asarray(t_star),
        "x": jnp.asarray(x_star),
        "y": jnp.asarray(y_star),
    }
    axis_order = ("t", "x", "y")
    pde_extra_params = {
        "eps": eps,
        "k": k,
    }

    return sol, axes, axis_order, pde_extra_params
