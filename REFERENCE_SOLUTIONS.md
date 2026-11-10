# Reference solutions: provenance, generation, and accuracy

This document records how the six reference solutions shipped in
`benchmarks/*/data/` were obtained, how to regenerate or re-derive them (the
Rayleigh-Taylor solvers excepted, see below), and how accurate they are. Every
error is a relative L2 distance in float64.

| benchmark | file (loaded by `utils.get_dataset`) | how it was generated | grid | used by the benchmark | accuracy of the reference |
|---|---|---|---|---|---|
| Burgers | `burgers/data/burgers_refined.mat` | Fourier + ETDRK4 (`verify_reference_solutions.py`), `data/gen_burgers_refined.py` | 2,044 modes, dt 2.5e-6; stored 201 × 512 | t ∈ [0, 1] | **1.9e-11** (vs 4,088 modes, dt 1.25e-6) |
| Allen-Cahn, u0 = x² sin(2πx) | `allen_cahn/data/allen_cahn_sin.mat` | Fourier + ETDRK4 (`verify_reference_solutions.py`), `data/gen_allen_cahn_sin.py` | 8,176 modes, dt 1e-5; stored 201 × 512 | t ∈ [0, 1] | **5.9e-11** (vs 16,384 modes) |
| KdV | `kdv/data/kdv.mat` | Chebfun `spin` (Fourier + ETDRK4, MATLAB), `data/gen_kdv.m` | 511 modes, dt 1e-5; 201 × 512 | t ∈ [0, 1] | ≈ 5e-12 |
| Kuramoto-Sivashinsky | `ks/data/ks_chaotic.mat` | Chebfun `spin`, `data/gen_ks_chaotic.m` | 511 modes, dt 1e-5; 251 × 512 | first 35 % of t: 87 slices, t ∈ [0, 0.344] | ≈ 3e-9 |
| Ginzburg-Landau | `ginzburg_landau/data/ginzburg_landau_square.mat` | Chebfun `spin2`, `data/gen_ginzburg_landau_square.m` | 200 × 200 modes, dt 1e-3; 101 × 200 × 200 | first 80 % of t: 80 slices, t ∈ [0, 0.79] | ≈ 3e-8 |
| Rayleigh-Taylor | `rayleigh_taylor/data/rayleigh_taylor_spectral.npy` | two independent Fourier × Chebyshev pseudo-spectral solvers, `data/README_spectral.md` | 384 × 512 modes, dt 1.5625e-4; sampled on a 100 × 200 mesh, 40 outputs | slices 4 … 23 (raw t = 1.01 … 5.76), shifted to t ∈ [0, 4.75] | **T 6.3e-9**, u 1.9e-8, v 1.1e-8 over the window |

KdV, Kuramoto-Sivashinsky and Ginzburg-Landau use the jaxpi reference files
unchanged (identical bytes). Burgers, Allen-Cahn and Rayleigh-Taylor use new
references, for the reasons given below. The truncated time windows of KS
and GL are applied by the loaders (`utils.get_dataset(time_range=...)`, the
kept time axis is shifted to start at 0); the files hold the full horizon.

MD5 checksums of the shipped files (`python verify_reference_solutions.py --md5`):

| file | md5 |
|---|---|
| `burgers_refined.mat` | `4307eea1b692f384c76bef76318185a7` |
| `allen_cahn_sin.mat` | `fd97e0a17fcf8ce20edb493c12fd8e9e` |
| `kdv.mat` | `6c5cf86a42457cb89a215cc9e0482780` |
| `ks_chaotic.mat` | `72dc34d1a1049408c8aed2fec1bc0043` |
| `ginzburg_landau_square.mat` | `9fbb2fa7511c47e8aebd360b7062d706` |
| `rayleigh_taylor_spectral.npy` | `e58d9661dd03ad972e79998b7fdeb07d` |

## The problems

All 1-D problems are periodic in x; the `.mat` files store 512 points
x = x0 + L j / 511, j = 0 … 511, the last one duplicating the first.

- **Burgers** u_t + u u_x = ν u_xx, ν = 0.01/π, x ∈ [−1, 1],
  u0 = −sin(πx), t ∈ [0, 1], 201 output times.
- **Allen-Cahn** u_t = 10⁻⁴ u_xx + 5 (u − u³), x ∈ [−1, 1],
  u0 = x² sin(2πx), t ∈ [0, 1], 201 output times.
- **KdV** u_t + u u_x + 0.022² u_xxx = 0, x ∈ [−1, 1], u0 = cos(πx),
  t ∈ [0, 1], 201 output times.
- **Kuramoto-Sivashinsky** u_t + (100/16) u u_x + (100/16²) u_xx +
  (100/16⁴) u_xxxx = 0, x ∈ [0, 2π], u0 = cos x (1 + sin x), t ∈ [0, 1],
  251 output times.
- **Ginzburg-Landau** (complex A = u + i v) A_t = ε ΔA + k A − k (1 + 1.5 i) A |A|²,
  k = 10, ε = k/2500, (x, y) ∈ [−1, 1]² periodic,
  A0 = (10 i x + 10 y) exp(−25 (x² + y²)), t ∈ [0, 1], 101 output times sampled
  on a 200 × 200 tensor grid.
- **Rayleigh-Taylor** (Boussinesq, Ra = 10⁶, Pr = 0.71)
  u_t + u·∇u = −∇p + α1 Δu + α2 T e_y, ∇·u = 0, T_t + u·∇T = α4 ΔT with
  α1 = √(Pr/Ra), α2 = 1, α4 = 1/√(Pr Ra); x ∈ [0, 1] periodic, y ∈ [0, 2]
  with rigid isothermal walls (u = v = T = 0); T(x, y, 0) = ½ (1 − tanh((y − 1 −
  0.02 sin πx)/0.1)), u = v = 0; 40 output times t = 0.01 + 0.25 k. The file
  stores u, v, p, T on the 100 × 200 sin-stretched mesh of the upstream jaxpi
  dataset (x ∈ [0.0015, 0.9985], y ∈ [0.0015, 1.9985]) and the coefficients
  `alpha1` … `alpha4`, `Ra`, `Pr`, `Ge`.

## Burgers: the refined reference

The widely used Burgers reference `burgers.mat` (Chebfun `spin`, 511 modes,
dt 1e-5) is only
accurate to **4.0e-5**: the viscous front at x = 0 (width ~ν ≈ 3.2e-3) is
marginally resolved on a grid of spacing 2/511 ≈ 3.9e-3, and well-trained PINNs
sat exactly at that floor. `burgers_refined.mat` solves the same problem with
the same output protocol at 2,044 = 4 × 511 modes and dt = 2.5e-6, so the
512-point grid is an exact subset of the solver grid (no interpolation). A
4,088-mode, dt = 1.25e-6 solve certifies it to **1.907e-11**; its distance to
the 511-mode file is 4.005e-5, the error of that file (`data/gen_burgers_refined.log`,
`data/README_refined.md`). Extra keys `gen_modes`, `gen_dt` and `method` record
the provenance.

## Allen-Cahn: the sin initial condition

The classic Allen-Cahn benchmark (Raissi et al., 2019) starts from u0 = x² cos(πx), which has slopes
+2 and −2 on the two sides of the periodic seam: u0 is only C⁰ there, the exact
solution carries a layer of width √(4εt) along x = ±1 and Fourier convergence
is algebraic. The 511-mode reference of that problem is wrong by 1.6e-5, almost
all of it in the corner t < 0.1, |x| > 0.9, and a PINN with a smooth periodic
embedding resolves that layer no better than such a solver, so the benchmark
saturates at ~1e-5 whatever the reference.

This repository uses the initial condition of Kiyani et al. (2025),
**u0 = x² sin(2πx)** (same PDE, same grid). It is C¹ across the seam
(only u'' jumps), so the spectral solution converges fast: 1,024 modes are
already within 1.5e-8 of a 16,384-mode solve, 4,096 within 3.6e-10.
`allen_cahn_sin.mat` is solved at 8,176 = 16 × 511 modes with dt = 1e-5 (the
512-point grid is every 16th solver point) and certified against a 16,384-mode
solve to **5.896e-11** (`data/gen_allen_cahn_sin.log`; the 16,384-mode solution
is not shipped, `gen_allen_cahn_sin.py --cert` takes it as input). The `method`
key names the solver, `verify_reference_solutions.py` at the repository root;
the MD5 in `gen_allen_cahn_sin.log` predates an edit of that string (the arrays
are unchanged).

## KdV, Kuramoto-Sivashinsky, Ginzburg-Landau

These files come unchanged from the upstream jaxpi repository (PredictiveIntelligenceLab/jaxpi) and its PirateNets
extensions, together with the Chebfun scripts `gen_*.m` that produced them
(MATLAB with Chebfun, https://www.chebfun.org); for their licences see
Provenance below and `NOTICE`. Their accuracy is well below the errors the
benchmarks reach: KdV ≈ 5e-12, KS ≈ 3e-9, GL ≈ 3e-8 (measured below).

Caveat: `gen_kdv.m` sets `steps = 250` (251 output times) while `kdv.mat`
holds 201 (`steps = 200`). The solve is the same (dt = 1e-5 fixed), only the
output sampling differs; set `steps = 200` to reproduce the file.

## Rayleigh-Taylor: the spectral reference

The upstream Rayleigh-Taylor dataset of jaxpi (`rayleigh_taylor_high_Ra.npy`;
its generating script was never published) is coarser than
the PINN errors it is used to judge: over the benchmark window it is
1.1e-2 (u), 5.1e-3 (v) and 4.5e-3 (T) away from the solution below, and its
stored velocity is 2e-4 to 1.3e-3 (depending on slice and component) away from
the nearest divergence-free field.

`rayleigh_taylor_spectral.npy` replaces it with the same keys, dtypes, mesh and
output times. It is computed from the analytic initial condition above (the one
that reproduces the upstream t = 0.01 snapshot, T to 1.7e-6 outside the wall
layer) over the full horizon t = 0.01 … 9.76 by two solvers written
independently:

- a Fourier(384) × Chebyshev(512) vorticity-streamfunction solver with an
  influence-matrix wall treatment and IMEX SBDF3, dt = 1.5625e-4 (the
  delivered field), and
- a Fourier × Chebyshev primitive-variable collocation solver with SBDF4 and a
  different startup (the independent check).

The initial condition is projected onto the retained Fourier modes (Galerkin),
not sampled: sin(πx) is a |sin(πx)| corner as a period-1 function, and
sampling it leaves an O(Nx⁻²) error of ~2e-4.

The **certified accuracy is the difference between the two solvers**: over
the benchmark window (slices 4 … 23) 1.86e-8 (u), 1.12e-8 (v), 6.33e-9 (T),
1.06e-8 (p); over all 40 slices 2.77e-8, 1.65e-8, 1.70e-8, 1.31e-8. The
delivered field's own refinement budget is an order smaller (~2e-9). Its
velocity is divergence-free to 3e-14, 0 ≤ T ≤ 1 holds to round-off at every
slice, and the spectra fall by 14 – 16 orders with no pile-up.

`data/README_spectral.md` documents the solvers, their validation, the
refinement runs and the physics checks in detail. The two solvers, the
full-resolution spectral fields and the raw outputs behind those numbers are
not part of this artifact; `rayleigh_taylor_spectral.npy` and that README are
all that ships.

The benchmark evaluates the window `time_range = (0.1, 0.6)` (fractions of
the horizon, `utils.TIME_RANGE`), resolved against the 40-slice grid
(`utils.window_bounds`): slices 4 … 23,
raw t = 1.01 … 5.76, shifted to t ∈ [0, 4.75]. The PINN's initial condition is
the reference at the window's first slice. The loader refuses to run without
float64 (`JAX_ENABLE_X64`), which would round the reference by 1.9e-8.

## Verification: `verify_reference_solutions.py`

The script re-solves the five spectral references with an independent NumPy
implementation of ETDRK4 (Kassam & Trefethen, SIAM J. Sci. Comput. 2005; CPU
only) and reports

- **repro** — shipped file vs this solver at the generating resolution and
  time step: the file is what the documented solve produces;
- **acc** — shipped file vs a refined run (2× modes, dt/2): the accuracy of the
  reference;
- **conv** — refined vs doubly refined run (4× modes, dt/4, `--conv`): the
  refined run is converged.

For Rayleigh-Taylor, which it does not re-solve, it checks the output times and
the window (20 slices spanning 4.75), finiteness, the coefficients α1 and α4,
the maximum principle 0 ≤ T ≤ 1 and the mirror symmetry about x = ½ (T, v, p
even, u odd).

```bash
python verify_reference_solutions.py            # repro + acc (~35 min)
python verify_reference_solutions.py --fast     # repro only (~7 min)
python verify_reference_solutions.py --conv     # + conv (hours for the refined files)
python verify_reference_solutions.py --only kdv ks rayleigh_taylor --md5
```

Measured (float64, relative L2 over the whole stored space-time grid):

| benchmark | repro | acc (2× modes, dt/2) | conv (4× / dt/4) | CPU time |
|---|---|---|---|---|
| Burgers (2,044 modes, dt 2.5e-6) | 3.0e-16 | 1.91e-11 (4,088 modes, dt 1.25e-6) | — | 3 min repro, 14 min repro + acc |
| Allen-Cahn sin (8,176 modes, dt 1e-5) | 0 (bitwise) | 1.24e-11 (16,352 modes, dt 5e-6) | — | 3.5 min repro, 18 min repro + acc |
| KdV (511 modes, dt 1e-5) | 4.19e-13 | 5.48e-12 | 1.28e-11 | 1 min repro + acc, 4 min with conv |
| Kuramoto-Sivashinsky (511 modes, dt 1e-5) | 1.66e-11 | 2.98e-9 | 9.11e-10 | 1 min repro + acc, 4.5 min with conv |
| Ginzburg-Landau (200² modes, dt 1e-3) | 1.41e-9 | 3.17e-8 | 2.39e-9 | 40 s repro + acc, 7 min with conv |

- The two refined references were generated with this solver, so their repro
  is at round-off. The acc column certifies them: Burgers 1.91e-11 (the value of
  its generation log), Allen-Cahn 1.24e-11 against the 2× refined run and
  5.9e-11 against the 16,384-mode solve of its generation log. Their conv runs
  (8,176 and 32,704 modes) would take hours of CPU and were not run.
- KdV, KS and GL were generated with Chebfun; the repro values (4e-13 … 2e-11
  for the 1-D files) show the shipped files are exactly what the `gen_*.m`
  scripts produce, without MATLAB. GL's larger repro (1.4e-9) is the chebfun2
  interpolation used to sample the output grid, not a scheme difference.
- Rayleigh-Taylor: output times within 1.7e-12 of t = 0.01 + 0.25 k, a window
  of 20 slices spanning 4.75, no non-finite value, α1 and α4 exact,
  0 ≤ T ≤ 1 with no violation, mirror asymmetry 4.3e-16 (T), 7.2e-16 (v),
  8.3e-16 (u).

## Regenerating the references

- Burgers: `cd benchmarks/burgers/data && python gen_burgers_refined.py`
  (writes `burgers_refined.mat` and the full-resolution
  `burgers_refined_hires.mat`, then certifies against the 4,088-mode run).
- Allen-Cahn: `cd benchmarks/allen_cahn/data && python gen_allen_cahn_sin.py
  [--cert <16384-mode solution .npz>]` (~3 min CPU for the 8,176-mode solve).
- KdV, KS, GL: the `gen_*.m` scripts (MATLAB + Chebfun), or the independent
  Python solver of `verify_reference_solutions.py` at the same resolution
  (`repro` above).
- Rayleigh-Taylor: the two spectral solvers are not part of this artifact,
  so the file cannot be regenerated from it; `data/README_spectral.md`
  describes the method. An independent regeneration must Galerkin-project the
  initial condition in x, not sample it, to reproduce the file at better than
  ~1e-4.

`.mat` files written by `scipy.io.savemat` carry a creation time stamp, so a
regenerated file matches the shipped one in its arrays, not in its MD5
checksum.

## Provenance

- `kdv.mat`, `ks_chaotic.mat`, `ginzburg_landau_square.mat` and the `gen_*.m`
  scripts originate from the upstream jaxpi repository
  (PredictiveIntelligenceLab/jaxpi, branch `pirate`, the PirateNets code) and
  are mirrored unchanged. Byte-identical copies of the three
  `.mat` files and of `gen_kdv.m` and `gen_ks_chaotic.m` are distributed with
  jaxpi2 (sifanexisted/jaxpi2, Apache-2.0); `gen_ginzburg_landau_square.m` is
  found only in jaxpi. `NOTICE` gives the licence of each.
- `burgers_refined.mat` and `allen_cahn_sin.mat` were generated with
  `verify_reference_solutions.py` for this study; they replace
  `burgers.mat` (the 511-mode Chebfun solution of the same Burgers problem)
  and the Allen-Cahn problem with the cos initial condition.
- `rayleigh_taylor_spectral.npy` was generated for this study from the
  analytic initial condition; it replaces the upstream jaxpi dataset
  `rayleigh_taylor_high_Ra.npy`.
