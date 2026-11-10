import scipy.io
import jax.numpy as jnp


def get_dataset(kappa=5.0):
    """
    Loads the Allen-Cahn reference: u_t = 1e-4 u_xx + kappa (u - u^3) with
    kappa = 5 and the initial condition u0 = x^2 sin(2 pi x) (as in Kiyani et al.
    (2025)). Reference: data/allen_cahn_sin.mat
    (8176 Fourier modes, dt 1e-5, see data/gen_allen_cahn_sin.py), on the
    201 x 512 grid of the classic Allen-Cahn benchmark. `kappa` is passed to the
    model as a PDE parameter; the reference is the kappa = 5 solution.

    Returns:
        sol: dict, solution dictionary containing 'u'
        axes: dict, axes dictionary containing 't' and 'x'
        axis_order: tuple, order of the axes
        pde_extra_params: dict containing 'kappa'
    """
    data = scipy.io.loadmat("data/allen_cahn_sin.mat")

    sol = {
        "u": jnp.asarray(data["usol"])
    }

    axes = {
        "t": jnp.asarray(data["t"]).flatten(),
        "x": jnp.asarray(data["x"]).flatten(),
    }

    axis_order = ("t", "x")
    return sol, axes, axis_order, {"kappa": float(kappa)}
