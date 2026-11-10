"""Reference solution of the Allen-Cahn *sin* variant
    u_t = 1e-4 u_xx + 5 u - 5 u^3,  x in [-1,1] periodic,  u0 = x^2 sin(2 pi x)
(the initial condition as written in Kiyani et al. (2025); the classic
Allen-Cahn benchmark of Raissi et al. (2019) uses u0 = x^2 cos(pi x)).

u0 is C^1 across the periodic seam (only u'' jumps), so Fourier convergence is
fast: 1024 modes are already at ~1.5e-8 (1024 vs 16384), 4096 at 3.6e-10.
Solved with the independent ETDRK4 solver of verify_reference_solutions.py at
N = 8176 = 16 x 511 modes, dt = 1e-5, so the 512-point evaluation grid
x = linspace(-1, 1, 512) of the benchmark is a nested subgrid (every 16th point).

    python gen_allen_cahn_sin.py [--cert path/to/N16384.npz]

Writes allen_cahn_sin.mat (201 x 512, loaded by utils.get_dataset) and
allen_cahn_sin_hires.mat (201 x 8177).
"""
import argparse
import hashlib
import os
import sys
import time

import numpy as np
import scipy.io as sio

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))          # repo root
import verify_reference_solutions as V                             # noqa: E402

N, DT, N_OUT = 8176, 1e-5, 200


def rel_l2_parseval(u, v):
    """Continuous relative L2 on [-1,1] between periodic fields on different grids."""
    cu, cv = np.fft.rfft(u, axis=1) / u.shape[1], np.fft.rfft(v, axis=1) / v.shape[1]
    d = np.zeros((u.shape[0], max(cu.shape[1], cv.shape[1])), complex)
    d[:, :cu.shape[1]] += cu
    d[:, :cv.shape[1]] -= cv
    w = lambda c: np.concatenate([[1.0], 2 * np.ones(c.shape[1] - 1)])
    return float(np.sqrt((w(d) * abs(d) ** 2).sum()) / np.sqrt((w(cv) * abs(cv) ** 2).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cert", default=None, help=".npz with key u16384 (or usol): finer solve to certify against")
    a = ap.parse_args()

    x, u0, L, nl = V.make_problem("allen_cahn", N)                 # grid, L = 5 - 1e-4 k^2, N(u) = -5 u^3
    u0 = x ** 2 * np.sin(2 * np.pi * x)
    n_steps = round(1.0 / DT)
    print(f"solving sin-IC Allen-Cahn: N={N} dt={DT} steps={n_steps}", flush=True)
    t0 = time.time()
    u = V.solve_1d(u0, L, nl, DT, n_steps, n_steps // N_OUT)      # (201, N) periodic values
    print(f"  done in {time.time() - t0:.0f} s, u range {u.min():.4f}..{u.max():.4f}, finite={np.isfinite(u).all()}")

    t = np.linspace(0, 1, N_OUT + 1)
    step = N // 511
    usol = np.concatenate([u[:, ::step], u[:, :1]], 1)             # 512 points incl. x = +1
    xg = np.linspace(-1, 1, 512)
    assert np.allclose(np.append(x[::step], 1.0), xg, atol=1e-12)
    meta = dict(gen_modes=N, gen_dt=DT, ic="x^2 sin(2 pi x)",
                method="Fourier spectral + ETDRK4 (verify_reference_solutions.py at the repo root)")

    out = os.path.join(HERE, "allen_cahn_sin.mat")
    sio.savemat(out, dict(t=t.reshape(1, -1), x=xg.reshape(1, -1), usol=usol, **meta))
    out_h = os.path.join(HERE, "allen_cahn_sin_hires.mat")
    sio.savemat(out_h, dict(t=t.reshape(1, -1), x=np.append(x, 1.0).reshape(1, -1),
                            usol=np.concatenate([u, u[:, :1]], 1), **meta))
    print(f"wrote {out} md5 {hashlib.md5(open(out, 'rb').read()).hexdigest()}")
    print(f"wrote {out_h}")

    if a.cert and os.path.exists(a.cert):
        z = np.load(a.cert)
        uc = z["u16384"] if "u16384" in z else z["usol"]
        print(f"certification vs {a.cert} ({uc.shape[1]} modes): rel L2 = {rel_l2_parseval(u, uc):.3e}")
    # the cos-IC reference, for the record (not part of this repository;
    # compared only if present)
    cos_path = os.path.join(HERE, "allen_cahn_refined.mat")
    if os.path.exists(cos_path):
        cos = sio.loadmat(cos_path)["usol"]
        print(f"difference to the cos-IC solution on the 512 grid: rel L2 = "
              f"{np.linalg.norm(usol - cos) / np.linalg.norm(cos):.3f}  (different problem)")


if __name__ == "__main__":
    main()
