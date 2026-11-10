import jax.numpy as jnp
import numpy as np

# the evaluation window of the Rayleigh-Taylor example of jaxpi2
# (github.com/sifanexisted/jaxpi2, examples/rayleigh_taylor: config.time_range)
TIME_RANGE = (0.1, 0.6)

# The certified spectral reference (see data/README_spectral.md): the exact solution of the
# Boussinesq IBVP from the analytic initial condition
#     T(x,y,0) = 0.5 (1 - tanh((y - 1 - 0.02 sin(pi x)) / 0.1)),  u = v = 0,
# computed with two independent pseudo-spectral solvers that agree to 6.3e-9 (T) over the
# benchmark window. It replaces the upstream jaxpi dataset, whose own error there is 4.5e-3 in T -
# larger than the PINN errors it was used to judge.
DATA_FILE = "data/rayleigh_taylor_spectral.npy"       # 40 outputs, t = 0.01 + 0.25 k

# The canonical output grid the benchmark was defined on: 40 slices, t = 0.01 + 0.25 k.
# `time_range` is resolved against THIS grid, so the evaluation window is the same physical
# interval no matter how finely the reference file samples time.
CANONICAL_NT, CANONICAL_T0, CANONICAL_DT = 40, 0.01, 0.25


def window_bounds(time_range=TIME_RANGE):
    """Physical end points [t_lo, t_hi] of the evaluation window (raw time).

    time_range is a pair of fractions, as in jaxpi2's loader: with the canonical
    40-slice grid, (0.1, 0.6) selects slices 4..23, i.e. t = 1.01 .. 5.76.
    """
    i0 = int(time_range[0] * CANONICAL_NT)
    i1 = int(time_range[1] * CANONICAL_NT)
    return (CANONICAL_T0 + CANONICAL_DT * i0,
            CANONICAL_T0 + CANONICAL_DT * (i1 - 1))


def get_dataset(time_range=TIME_RANGE, every=1, data_file=DATA_FILE):
    """
    Loads the Rayleigh-Taylor reference (Ra = 1e6, Pr = 0.71), restricted to the
    benchmark's evaluation window.

    Parameters
    ----------
    time_range : pair of fractions, as in jaxpi2's loader (default (0.1, 0.6)).
        Resolved against the canonical 40-slice grid to the physical interval
        t = 1.01 .. 5.76, then shifted so the first kept slice is t = 0
        (window t in [0, 4.75]). Because the bounds are physical, a reference
        file with a finer output cadence gives the SAME window, sampled more
        densely - the PINN's time domain and scaling are unchanged.
    every : int
        Keep only every `every`-th slice of the window.
    data_file : str
        Which reference to load.

    Returns
    -------
    sol : dict
        'u', 'v' (velocity components), 'p' (pressure), 'T' (temperature);
        each of shape (Nt, Npoints) on the flattened (x, y) mesh.
    axes : dict
        't' : time axis (Nt,), starting at 0
        'coords' : mesh (Npoints, 2), x in [0, 1] (periodic), y in [0, 2]
    axis_order : ('t', 'coords')
    pde_params : dict
        'alpha1'..'alpha4', 'Ra', 'Pr', 'Ge'
    """
    if jnp.zeros(1, dtype=jnp.float64).dtype != jnp.dtype("float64"):
        raise RuntimeError(
            "JAX x64 is disabled, so the reference would be rounded to float32 before the "
            "evaluator sees it - 1.9e-8 in the benchmark norm, i.e. coarser than the "
            "reference itself (6.3e-9 in T). Set JAX_ENABLE_X64=True before importing jax "
            "(the study scripts do).")

    data = np.load(data_file, allow_pickle=True).item()

    t_all = np.asarray(data["t"])
    t_lo, t_hi = window_bounds(time_range)
    tol = 1e-9
    keep = np.flatnonzero((t_all >= t_lo - tol) & (t_all <= t_hi + tol))[:: int(every)]
    if keep.size == 0:
        raise ValueError(f"no slices of {data_file} fall in t = [{t_lo}, {t_hi}]")

    t = jnp.asarray(t_all[keep] - t_all[keep[0]])
    velocity = jnp.asarray(np.asarray(data["velocity"])[keep])        # (Nt, Npoints, 2)
    pressure = jnp.asarray(np.asarray(data["pressure"])[keep])
    temperature = jnp.asarray(np.asarray(data["temperature"])[keep])
    coords = jnp.asarray(np.asarray(data["coords"]))                  # (Npoints, 2)

    sol = {
        "u": velocity[..., 0],
        "v": velocity[..., 1],
        "p": pressure,
        "T": temperature,
    }

    axes = {
        "t": t,
        "coords": coords,
    }

    axis_order = ("t", "coords")

    pde_param_dict = {
        "alpha1": data["alpha1"],
        "alpha2": data["alpha2"],
        "alpha3": data["alpha3"],
        "alpha4": data["alpha4"],
        "Ra": data["Ra"],
        "Pr": data["Pr"],
        "Ge": data["Ge"],
    }

    return sol, axes, axis_order, pde_param_dict
