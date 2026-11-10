import scipy.io
import jax.numpy as jnp

def get_dataset():
    """
    Loads the KdV dataset
    Returns:
        sol: dict, solution dictionary containing 'u'
        axes: dict, axes dictionary containing 't_star', 'x_star',
        axis_order: tuple, order of the axes 
        pde_extra_params: None
    """
    data = scipy.io.loadmat("data/kdv.mat")
    u_ref = data["usol"]
    t_star = data["t"].flatten()
    x_star = data["x"].flatten()

    sol = {
        "u": jnp.asarray(u_ref),
    }
    axes = {
        "t": jnp.asarray(t_star),
        "x": jnp.asarray(x_star),
    }
    axis_order = ("t", "x") 

    return sol, axes, axis_order, None
