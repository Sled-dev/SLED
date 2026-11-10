"""Independent verification of the reference solutions shipped in
benchmarks/*/data (see REFERENCE_SOLUTIONS.md).

Five of the six references are Fourier-spectral / ETDRK4 solutions of periodic
problems:

    benchmark        file                                   generated with
    burgers          burgers/data/burgers_refined.mat       2044 modes, dt 2.5e-6 (this solver)
    allen_cahn       allen_cahn/data/allen_cahn_sin.mat     8176 modes, dt 1e-5   (this solver)
    kdv              kdv/data/kdv.mat                       511 modes, dt 1e-5    (Chebfun spin)
    ks               ks/data/ks_chaotic.mat                 511 modes, dt 1e-5    (Chebfun spin)
    ginzburg_landau  ginzburg_landau/data/ginzburg_landau_square.mat
                                                            200^2 modes, dt 1e-3  (Chebfun spin2)

This script re-solves each of them with an independent NumPy implementation of
ETDRK4 (Kassam & Trefethen, SIAM J. Sci. Comput. 2005) and reports

  repro : rel. L2 distance between the shipped file and this solver at the
          generating resolution and time step -> the file is what the
          documented solve produces (no MATLAB/Chebfun needed)
  acc   : rel. L2 distance between the shipped file and a refined run
          (2x modes, dt/2)                    -> accuracy of the reference
  conv  : rel. L2 distance between the refined and a doubly refined run
          (4x modes, dt/4)                    -> the refined run is converged

The sixth, rayleigh_taylor/data/rayleigh_taylor_spectral.npy, is a
Fourier x Chebyshev pseudo-spectral solution of the Boussinesq system (see
benchmarks/rayleigh_taylor/data/README_spectral.md). It is not re-solved here;
the script checks its structure and the properties that need no solver: the
output times and the benchmark window, finiteness, the PDE coefficients, the
maximum principle 0 <= T <= 1 and the mirror symmetry about x = 1/2.

CPU only (NumPy/SciPy, no JAX). Run from anywhere:

    python verify_reference_solutions.py                # repro + acc
    python verify_reference_solutions.py --fast         # repro only
    python verify_reference_solutions.py --conv         # repro + acc + conv (hours)
    python verify_reference_solutions.py --only kdv ks rayleigh_taylor

The data/gen_*.py scripts import the solver functions of this module
(make_problem, solve_1d, rel_l2).
"""
import argparse
import hashlib
import os
import time

import numpy as np
import scipy.io as sio

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmarks")


def etdrk4_coeffs(L, dt, M=32):
    """ETDRK4 scalar coefficients via contour integral (Kassam-Trefethen)."""
    E = np.exp(dt * L)
    E2 = np.exp(dt * L / 2.0)
    r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
    LR = dt * L[:, None].astype(complex) + r[None, :]
    Q = dt * np.real(np.mean((np.exp(LR / 2.0) - 1.0) / LR, axis=1))
    f1 = dt * np.real(np.mean((-4.0 - LR + np.exp(LR) * (4.0 - 3.0 * LR + LR**2)) / LR**3, axis=1))
    f2 = dt * np.real(np.mean((2.0 + LR + np.exp(LR) * (-2.0 + LR)) / LR**3, axis=1))
    f3 = dt * np.real(np.mean((-4.0 - 3.0 * LR - LR**2 + np.exp(LR) * (4.0 - LR)) / LR**3, axis=1))
    return E, E2, Q, f1, f2, f3


def etdrk4_coeffs_complex(L, dt, M=64):
    """Same but for complex L (Ginzburg-Landau): contour around each dt*L."""
    E = np.exp(dt * L)
    E2 = np.exp(dt * L / 2.0)
    r = np.exp(2j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
    LR = dt * L[..., None] + r
    Q = dt * np.mean((np.exp(LR / 2.0) - 1.0) / LR, axis=-1)
    f1 = dt * np.mean((-4.0 - LR + np.exp(LR) * (4.0 - 3.0 * LR + LR**2)) / LR**3, axis=-1)
    f2 = dt * np.mean((2.0 + LR + np.exp(LR) * (-2.0 + LR)) / LR**3, axis=-1)
    f3 = dt * np.mean((-4.0 - 3.0 * LR - LR**2 + np.exp(LR) * (4.0 - LR)) / LR**3, axis=-1)
    return E, E2, Q, f1, f2, f3


def solve_1d(u0, L, nonlin_hat, dt, n_steps, save_every):
    """ETDRK4 for u_t = L u + N(u), periodic, real field. Returns (n_saves+1, N)."""
    E, E2, Q, f1, f2, f3 = etdrk4_coeffs(L, dt)
    v = np.fft.fft(u0)
    out = [u0.copy()]
    for n in range(1, n_steps + 1):
        Nv = nonlin_hat(v)
        a = E2 * v + Q * Nv
        Na = nonlin_hat(a)
        b = E2 * v + Q * Na
        Nb = nonlin_hat(b)
        c = E2 * a + Q * (2.0 * Nb - Nv)
        Nc = nonlin_hat(c)
        v = E * v + Nv * f1 + 2.0 * (Na + Nb) * f2 + Nc * f3
        if n % save_every == 0:
            out.append(np.real(np.fft.ifft(v)))
    return np.array(out)


def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


# ----------------------------------------------------------------------------
# 1D problem definitions (mirroring the gen_* scripts)
# ----------------------------------------------------------------------------
def make_problem(name, N):
    """Return (x, u0, L, nonlin_hat) on an N-point periodic grid.

    "allen_cahn" is the classic initial condition u0 = x^2 cos(pi x)
    (data/gen_allen_cahn_sin.py starts from it and replaces u0);
    "allen_cahn_sin" is the benchmark shipped here, u0 = x^2 sin(2 pi x).
    """
    if name in ("burgers", "allen_cahn", "allen_cahn_sin", "kdv"):
        dom_len, x0 = 2.0, -1.0
    else:  # ks_chaotic
        dom_len, x0 = 2.0 * np.pi, 0.0
    x = x0 + dom_len * np.arange(N) / N
    k = 2.0 * np.pi * np.fft.fftfreq(N, d=1.0 / N) / dom_len
    ik = 1j * k

    if name == "burgers":
        nu = 0.01 / np.pi
        L = -nu * k**2
        u0 = -np.sin(np.pi * x)

        def nonlin_hat(v):
            u = np.real(np.fft.ifft(v))
            return -0.5 * ik * np.fft.fft(u * u)

    elif name in ("allen_cahn", "allen_cahn_sin"):
        L = 5.0 - 1e-4 * k**2
        if name == "allen_cahn":
            u0 = x**2 * np.cos(np.pi * x)
        else:
            u0 = x**2 * np.sin(2.0 * np.pi * x)

        def nonlin_hat(v):
            u = np.real(np.fft.ifft(v))
            return np.fft.fft(-5.0 * u**3)

    elif name == "kdv":
        L = -(0.022**2) * ik**3  # purely imaginary (dispersive)
        u0 = np.cos(np.pi * x)

        def nonlin_hat(v):
            u = np.real(np.fft.ifft(v))
            return -0.5 * ik * np.fft.fft(u * u)

    elif name == "ks_chaotic":
        L = (100.0 / 16.0**2) * k**2 - (100.0 / 16.0**4) * k**4
        u0 = np.cos(x) * (1.0 + np.sin(x))

        def nonlin_hat(v):
            u = np.real(np.fft.ifft(v))
            return -(100.0 / 16.0) * 0.5 * ik * np.fft.fft(u * u)

    else:
        raise ValueError(name)
    return x, u0, L, nonlin_hat


def solve_1d_kdv_complexL(u0, L, nonlin_hat, dt, n_steps, save_every):
    """KdV has purely imaginary L -> use complex coefficient variant."""
    E, E2, Q, f1, f2, f3 = etdrk4_coeffs_complex(np.asarray(L, complex), dt)
    v = np.fft.fft(u0)
    out = [u0.copy()]
    for n in range(1, n_steps + 1):
        Nv = nonlin_hat(v)
        a = E2 * v + Q * Nv
        Na = nonlin_hat(a)
        b = E2 * v + Q * Na
        Nb = nonlin_hat(b)
        c = E2 * a + Q * (2.0 * Nb - Nv)
        Nc = nonlin_hat(c)
        v = E * v + Nv * f1 + 2.0 * (Na + Nb) * f2 + Nc * f3
        if n % save_every == 0:
            out.append(np.real(np.fft.ifft(v)))
    return np.array(out)


def solve_problem(problem, N, dt, T, n_out):
    """Solve `problem` on N modes with step dt to time T, n_out + 1 outputs."""
    _, u0, L, nl = make_problem(problem, N)
    n_steps = round(T / dt)
    if n_steps % n_out:
        raise ValueError(f"{n_steps} steps do not split into {n_out} output intervals")
    solver = solve_1d_kdv_complexL if problem == "kdv" else solve_1d
    return solver(u0, L, nl, dt, n_steps, n_steps // n_out)


# The shipped 1-D references: every file stores 512 points x = x0 + L j / 511,
# j = 0..511, the last one duplicating the first (periodic endpoint), so the
# 511 periodic values are a nested subgrid of any N = m * 511 mode solve.
SPECTRAL_1D = {
    # benchmark: (file under benchmarks/, problem, generating modes, generating dt)
    "burgers":    ("burgers/data/burgers_refined.mat", "burgers", 2044, 2.5e-6),
    "allen_cahn": ("allen_cahn/data/allen_cahn_sin.mat", "allen_cahn_sin", 8176, 1e-5),
    "kdv":        ("kdv/data/kdv.mat", "kdv", 511, 1e-5),
    "ks":         ("ks/data/ks_chaotic.mat", "ks_chaotic", 511, 1e-5),
}


def run_1d(benchmark, fast=False, conv=False):
    rel_path, problem, N0, dt0 = SPECTRAL_1D[benchmark]
    d = sio.loadmat(os.path.join(BASE, rel_path))
    usol = d["usol"]                      # (n_out + 1, 512) incl. duplicated endpoint
    n_grid = usol.shape[1] - 1            # 511 periodic grid values
    ref = usol[:, :n_grid]
    n_out = usol.shape[0] - 1             # output intervals
    T = float(d["t"].ravel()[-1])

    def on_grid(u, N):
        if N % n_grid:
            raise ValueError(f"{N} modes do not nest the {n_grid}-point grid")
        return u[:, :: N // n_grid]

    res = {}
    # 1) reproduction at the generating resolution / time step
    mine = solve_problem(problem, N0, dt0, T, n_out)
    res["repro"] = rel_l2(on_grid(mine, N0), ref)
    if fast:
        return res
    # 2) refined run: 2x modes, dt / 2 (its grid contains the file's grid)
    fine = solve_problem(problem, 2 * N0, dt0 / 2, T, n_out)
    res["acc"] = rel_l2(ref, on_grid(fine, 2 * N0))
    # 3) doubly refined: convergence of the refined run
    if conv:
        fine2 = solve_problem(problem, 4 * N0, dt0 / 4, T, n_out)
        res["conv"] = rel_l2(fine, fine2[:, ::2])
    return res


# ----------------------------------------------------------------------------
# 2D Ginzburg-Landau (complex field)
# ----------------------------------------------------------------------------
def solve_gl(N, dt, n_steps, save_every, x_target):
    """Solve GL on N-point periodic grid; save snapshots Fourier-evaluated at
    the tensor grid x_target (matching the .mat's linspace sampling)."""
    kk = 10.0
    eps = kk / (50.0 * 50.0)
    x = -1.0 + 2.0 * np.arange(N) / N
    X, Y = np.meshgrid(x, x, indexing="ij")
    u = (1j * X * 10.0 + Y * 10.0) * np.exp(-0.01 * (X**2 * 2500.0 + Y**2 * 2500.0))
    k1 = np.pi * np.fft.fftfreq(N, d=1.0 / N)
    KX, KY = np.meshgrid(k1, k1, indexing="ij")
    L = -eps * (KX**2 + KY**2)
    E, E2, Q, f1, f2, f3 = etdrk4_coeffs(L.ravel(), dt)
    E, E2, Q, f1, f2, f3 = (a.reshape(N, N) for a in (E, E2, Q, f1, f2, f3))
    # evaluation matrix  P[a, j] = exp(i k_j (x_a + 1)) / N   (grid starts at -1)
    P = np.exp(1j * np.outer(x_target + 1.0, k1)) / N

    def nonlin_hat(v):
        u = np.fft.ifft2(v)
        return np.fft.fft2(kk * u - kk * (1.0 + 1.5j) * u * np.abs(u) ** 2)

    def sample(v):
        return P @ v @ P.T

    v = np.fft.fft2(u)
    out = [sample(v)]
    for n in range(1, n_steps + 1):
        Nv = nonlin_hat(v)
        a = E2 * v + Q * Nv
        Na = nonlin_hat(a)
        b = E2 * v + Q * Na
        Nb = nonlin_hat(b)
        c = E2 * a + Q * (2.0 * Nb - Nv)
        Nc = nonlin_hat(c)
        v = E * v + Nv * f1 + 2.0 * (Na + Nb) * f2 + Nc * f3
        if n % save_every == 0:
            out.append(sample(v))
    return np.array(out)


def run_gl(fast=False, conv=False):
    d = sio.loadmat(os.path.join(BASE, "ginzburg_landau/data/ginzburg_landau_square.mat"))
    # MATLAB meshgrid in gen script stores snapshots as (t, y, x); transpose
    # to (t, x, y) to match this solver's orientation (verified vs t=0 init).
    ref = (d["usol"] + 1j * d["vsol"]).transpose(0, 2, 1)
    xt = np.linspace(-1.0, 1.0, 200)
    n_out = ref.shape[0] - 1
    dt0 = 1.0 / 200.0 / 5.0               # = 1/1000, as in gen script
    T = 1.0
    n_steps = round(T / dt0)

    res = {}
    mine = solve_gl(200, dt0, n_steps, n_steps // n_out, xt)
    res["repro"] = rel_l2(mine, ref)
    if fast:
        return res

    fine = solve_gl(400, dt0 / 2.0, 2 * n_steps, 2 * n_steps // n_out, xt)
    res["acc"] = rel_l2(ref, fine)

    if conv:
        fine2 = solve_gl(800, dt0 / 4.0, 4 * n_steps, 4 * n_steps // n_out, xt)
        res["conv"] = rel_l2(fine, fine2)
    return res


# ----------------------------------------------------------------------------
# Rayleigh-Taylor: structural and physical checks (no re-solve)
# ----------------------------------------------------------------------------
RT_FILE = "rayleigh_taylor/data/rayleigh_taylor_spectral.npy"
RT_T0, RT_DT, RT_NT = 0.01, 0.25, 40          # output times t = 0.01 + 0.25 k
RT_WINDOW = (1.01, 5.76)                      # benchmark window (slices 4..23)


def run_rt(fast=False, conv=False):
    """Checks that need no solver; each value is a max violation or a count."""
    d = np.load(os.path.join(BASE, RT_FILE), allow_pickle=True).item()
    t = np.asarray(d["t"])
    coords = np.asarray(d["coords"])
    vel = np.asarray(d["velocity"])
    p = np.asarray(d["pressure"])
    T = np.asarray(d["temperature"])
    Ra, Pr = float(d["Ra"]), float(d["Pr"])
    res = {}
    res["t_grid_err"] = float(np.abs(t - (RT_T0 + RT_DT * np.arange(RT_NT))).max()) \
        if t.shape == (RT_NT,) else float("inf")
    win = np.flatnonzero((t >= RT_WINDOW[0] - 1e-9) & (t <= RT_WINDOW[1] + 1e-9))
    res["window_slices"] = int(win.size)                       # expect 20
    res["window_span"] = float(t[win[-1]] - t[win[0]])         # expect 4.75
    res["nonfinite"] = int(sum(np.size(a) - np.isfinite(a).sum() for a in (vel, p, T)))
    res["alpha1_err"] = abs(float(d["alpha1"]) - np.sqrt(Pr / Ra))
    res["alpha4_err"] = abs(float(d["alpha4"]) - 1.0 / np.sqrt(Pr * Ra))
    # maximum principle of the temperature: 0 <= T <= 1
    res["T_below_0"] = float(max(0.0, -T.min()))
    res["T_above_1"] = float(max(0.0, T.max() - 1.0))
    # mirror symmetry about x = 1/2: T, v, p even, u odd
    xs, ix = np.unique(np.round(coords[:, 0], 12), return_inverse=True)
    ys, iy = np.unique(np.round(coords[:, 1], 12), return_inverse=True)
    res["mesh"] = f"{xs.size}x{ys.size}"
    if xs.size * ys.size == coords.shape[0] and np.allclose(xs + xs[::-1], 1.0, atol=1e-12):
        pos = np.empty((xs.size, ys.size), dtype=int)
        pos[ix, iy] = np.arange(coords.shape[0])
        mirror = pos[xs.size - 1 - ix, iy]                     # point -> its mirror

        def asym(f, sign=1.0):
            return float(np.linalg.norm(f - sign * f[:, mirror]) / np.linalg.norm(f))

        res["parity_T"] = asym(T)
        res["parity_v"] = asym(vel[..., 1])
        res["parity_u"] = asym(vel[..., 0], -1.0)
    else:
        res["parity"] = float("nan")     # not a mirror-symmetric tensor grid
    return res


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


JOBS = {
    "burgers": lambda fast, conv: run_1d("burgers", fast, conv),
    "allen_cahn": lambda fast, conv: run_1d("allen_cahn", fast, conv),
    "kdv": lambda fast, conv: run_1d("kdv", fast, conv),
    "ks": lambda fast, conv: run_1d("ks", fast, conv),
    "ginzburg_landau": run_gl,
    "rayleigh_taylor": run_rt,
}
FILES = {**{k: v[0] for k, v in SPECTRAL_1D.items()},
         "ginzburg_landau": "ginzburg_landau/data/ginzburg_landau_square.mat",
         "rayleigh_taylor": RT_FILE}


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.3e}"
    return str(v)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fast", action="store_true", help="repro only (no refined runs)")
    ap.add_argument("--conv", action="store_true",
                    help="also the doubly refined run (4x modes, dt/4); slow")
    ap.add_argument("--only", nargs="+", choices=list(JOBS), default=list(JOBS),
                    help="subset of benchmarks")
    ap.add_argument("--md5", action="store_true", help="print the md5 of every file")
    args = ap.parse_args()
    if args.fast and args.conv:
        ap.error("--fast and --conv exclude each other")

    for name in args.only:
        path = os.path.join(BASE, FILES[name])
        if not os.path.exists(path):
            print(f"{name:16s} SKIPPED: {FILES[name]} not found", flush=True)
            continue
        if args.md5:
            print(f"{name:16s} md5 {md5(path)}  {FILES[name]}", flush=True)
        t0 = time.time()
        try:
            r = JOBS[name](args.fast, args.conv)
            print(f"{name:16s} " + "  ".join(f"{k}={_fmt(v)}" for k, v in r.items())
                  + f"  ({time.time() - t0:.0f} s)", flush=True)
        except Exception as e:
            print(f"{name:16s} FAILED: {e!r}", flush=True)


if __name__ == "__main__":
    main()
