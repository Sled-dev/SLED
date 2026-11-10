"""Generate the refined Burgers reference burgers_refined.mat.

Same PDE/IC/output protocol as gen_burgers.m (nu = 0.01/pi, u0 = -sin(pi x),
t in [0,1], 201 outputs) but at 2044 Fourier modes and dt = 2.5e-6 instead of
511 modes / dt = 1e-5. 2044 = 4 x 511, so the coarse evaluation grid is an
exact subset — burgers_refined.mat is a drop-in replacement for burgers.mat.

Afterwards runs a 4088-mode / dt = 1.25e-6 solve to certify the accuracy of
the saved file.
"""
import os
import sys
import numpy as np
import scipy.io as sio

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", ".."))   # repo root: verify_reference_solutions.py
import verify_reference_solutions as V

OUT = os.path.dirname(os.path.abspath(__file__))
NU = 0.01 / np.pi
N_OUT = 200  # intervals -> 201 outputs


def solve(N, dt):
    _, u0, L, nl = V.make_problem("burgers", N)
    n_steps = round(1.0 / dt)
    return V.solve_1d(u0, L, nl, dt, n_steps, n_steps // N_OUT)


print("solving N=2044, dt=2.5e-6 ...", flush=True)
u_fine = solve(2044, 2.5e-6)  # (201, 2044)
t = np.linspace(0.0, 1.0, N_OUT + 1)

# full-resolution file (2045 points incl. duplicated periodic endpoint)
x_hi = np.linspace(-1.0, 1.0, 2045)
usol_hi = np.concatenate([u_fine, u_fine[:, :1]], axis=1)
sio.savemat(f"{OUT}/burgers_refined_hires.mat",
            {"t": t, "x": x_hi, "nu": NU, "usol": usol_hi,
             "gen_modes": 2044, "gen_dt": 2.5e-6,
             "method": "Fourier spectral + ETDRK4 (verify_reference_solutions.py at the repo root)"})

# drop-in replacement on the original 512-point grid (2044 = 4 x 511)
x_lo = np.linspace(-1.0, 1.0, 512)
u_lo = u_fine[:, ::4]
usol_lo = np.concatenate([u_lo, u_lo[:, :1]], axis=1)
sio.savemat(f"{OUT}/burgers_refined.mat",
            {"t": t, "x": x_lo, "nu": NU, "usol": usol_lo,
             "gen_modes": 2044, "gen_dt": 2.5e-6,
             "method": "Fourier spectral + ETDRK4 (verify_reference_solutions.py at the repo root)"})
print("saved burgers_refined.mat and burgers_refined_hires.mat", flush=True)

# sanity: distance to the widely used 511-mode reference burgers.mat (expect ~4.0e-5);
# that file is not part of this repository, so the check runs only if it is present
if os.path.exists(f"{OUT}/burgers.mat"):
    old = sio.loadmat(f"{OUT}/burgers.mat")["usol"][:, :511]
    print(f"rel L2 (refined vs original burgers.mat): {V.rel_l2(old, u_lo):.3e}", flush=True)

# certification: 4088-mode / dt=1.25e-6 run
print("certifying against N=4088, dt=1.25e-6 ...", flush=True)
u_finer = solve(4088, 1.25e-6)
err = V.rel_l2(u_fine, u_finer[:, ::2])
print(f"rel L2 (2044-mode file vs 4088-mode run): {err:.3e}", flush=True)
print("=> accuracy of saved refined reference is at or below this level.", flush=True)
