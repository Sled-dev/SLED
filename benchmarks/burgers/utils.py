import scipy.io
import jax.numpy as jnp

def get_dataset():
    """
    Loads the Burgers dataset
    Returns:
        sol: dict, solution dictionary containing 'u'
        axes: dict, axes dictionary containing 't_star' and 'x_star'
        axis_order: tuple, order of the axes
        None: for pde_extra_params_dict
    """
    data = scipy.io.loadmat("data/burgers_refined.mat")    
    sol = {"u": jnp.asarray(data["usol"])}

    axes = {
        "t": jnp.asarray(data["t"]).flatten(),
        "x": jnp.asarray(data["x"]).flatten(),
    }

    axis_order = ("t", "x")
    return sol, axes, axis_order, None

