import scipy.io
import jax.numpy as jnp


def get_dataset(time_range=[0.0, 0.35]):
    """
    Loads the Kuramoto-Sivashinsky dataset

    Parameters
    ----------
    time_range : list
        Fraction of the full time axis to keep, e.g. [0.0, 0.35].
        The kept time axis is shifted to start at t = 0.

    Returns
    -------
    sol : dict
        Solution dictionary containing:
            - 'u': solution array with shape (Nt, Nx)
    axes : dict
        Axes dictionary containing:
            - 't': time axis
            - 'x': spatial x-axis
    axis_order : tuple
        Order of the axes, given by ('t', 'x').
    pde_extra_params : None
    """
    data = scipy.io.loadmat("data/ks_chaotic.mat")
    u_ref = data["usol"]
    t_star = data["t"].flatten()
    x_star = data["x"].flatten()

    # Only use a fraction of the data
    start_time_step = int(time_range[0] * len(t_star))
    end_time_step = int(time_range[1] * len(t_star))

    u_ref = u_ref[start_time_step:end_time_step]
    t_star = t_star[start_time_step:end_time_step] - t_star[start_time_step]

    sol = {
        "u": jnp.asarray(u_ref),
    }
    axes = {
        "t": jnp.asarray(t_star),
        "x": jnp.asarray(x_star),
    }
    axis_order = ("t", "x")
    pde_extra_params = None

    return sol, axes, axis_order, pde_extra_params
