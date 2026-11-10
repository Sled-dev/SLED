# Refined Burgers reference

The widely used Burgers reference `burgers.mat` (from `gen_burgers.m`: Chebfun
`spin`, 511 Fourier modes, dt = 1e-5; neither file is shipped here) has a
measured accuracy of only ~4.0e-5 relative L2 — the viscous front at x = 0
(width ~nu = 0.01/pi) is marginal at 511 modes, and the best PINN runs sit
exactly at this reference floor.

Refined reference, same PDE / initial condition / output protocol, generated
with the independent Fourier-spectral ETDRK4 solver
(`verify_reference_solutions.py` at the repository root, Kassam–Trefethen
scheme, float64) at **2044 modes, dt = 2.5e-6**:

- `burgers_refined.mat` — the file `utils.get_dataset` loads, a drop-in
  replacement for `burgers.mat`: identical keys (`t` 1×201, `x` 1×512, `nu`,
  `usol` 201×512) and identical grid (2044 = 4×511, so the 512-point grid is
  an exact subset — no interpolation).
- `burgers_refined_hires.mat` — full resolution (`usol` 201×2045) for
  higher-accuracy evaluation; not shipped (`*_hires.mat` is git-ignored),
  regenerate it with `gen_burgers_refined.py`.

**Certified accuracy: rel. L2 = 1.9e-11** against a further-refined 4088-mode,
dt = 1.25e-6 run (see `gen_burgers_refined.log`; generation script
`gen_burgers_refined.py` in this folder, run from here:
`python gen_burgers_refined.py`). Extra `.mat` keys `gen_modes`, `gen_dt`,
`method` record provenance. The distance to `burgers.mat` in the log
(4.005e-5) was measured with that file present; the script skips the check
when it is absent.

Note: relative L2 errors of runs evaluated against `burgers_refined.mat` differ
from values computed against `burgers.mat` by up to ~4e-5 (the original
reference's own error).
