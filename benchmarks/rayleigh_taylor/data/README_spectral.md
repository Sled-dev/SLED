# Rayleigh–Taylor benchmark — certified spectral reference solution

**Files in this directory**

| file | what it is |
|---|---|
| `rayleigh_taylor_spectral.npy` | the reference `utils.get_dataset` loads: a drop-in replacement for the upstream jaxpi dataset `rayleigh_taylor_high_Ra.npy`, with identical keys, dtypes, `coords` and `t`; 26 MB |
| `README_spectral.md` | this file |

The two solvers, the full-resolution spectral fields, the per-slice accuracy
tables and the raw outputs behind the numbers quoted here are not part of this
artifact (see §9).

---

## 1. Summary

The upstream dataset `rayleigh_taylor_high_Ra.npy` is internally inconsistent
at the ~5 × 10⁻³ level:

* its velocity is **4.4 × 10⁻⁴ … 1.3 × 10⁻³ (relative L2) away from the nearest
  divergence-free field** at *every* slice — a statement that needs no grid and
  no difference operator, so no incompressible solution can ever match it better
  than that;
* its temperature at the first output time is wrong by **2.6 × 10⁻²** in the
  bottom wall layer (only ~2 of its cells span the 0.0069-wide layer);
* its discrete divergence on its own grid is 1.5–3 × 10⁻³ at early times where
  the reference gives 2–7 × 10⁻⁴.

That is **larger than the PINN errors it is used to judge**.

`rayleigh_taylor_spectral.npy` replaces it with a solution computed from the
**analytic initial condition** by two independent spectral solvers over the
**entire** time domain `t = 0.01 … 9.76`. The two solutions agree to

| region | u | v | T | p |
|---|---|---|---|---|
| benchmark window (slices 4…23, t = 1.01…5.76) | 1.9 × 10⁻⁸ | 1.1 × 10⁻⁸ | 6.3 × 10⁻⁹ | 1.1 × 10⁻⁸ |
| full domain (all 40 slices) | 2.8 × 10⁻⁸ | 1.7 × 10⁻⁸ | 1.7 × 10⁻⁸ | 1.3 × 10⁻⁸ |

and they differ from the upstream data by 1.1 × 10⁻² (u), 5.1 × 10⁻³ (v),
4.5 × 10⁻³ (T) over the window. **The reference is 5–6 orders of magnitude
tighter than the dataset it replaces.**

---

## 2. The problem being solved

Boussinesq Rayleigh–Taylor, exactly as in `benchmarks/rayleigh_taylor/models.py`:

```
u_t + u u_x + v u_y = -p_x + a1 (u_xx + u_yy)
v_t + u v_x + v v_y = -p_y + a1 (v_xx + v_yy) + a2 T
u_x + v_y = 0
T_t + u T_x + v T_y = a4 (T_xx + T_yy)

a1 = sqrt(Pr/Ra) = 8.426149773176359e-04     (Ra = 1e6, Pr = 0.71)
a2 = 1.0
a4 = 1/sqrt(Pr*Ra) = 1.1867816581938534e-03
```

**Domain** `x ∈ [0,1]` with period 1, `y ∈ [0,2]`.

**Boundary conditions** rigid isothermal walls at `y = 0` and `y = 2`:
`u = v = T = 0`. In `x` the configuration is one symmetric half-cell of a
single-mode RT instability, so `T, v, p` are even and `u` is odd about both
`x = 0` and `x = 1/2`.

**Initial condition** (reconstructed from the stored `t = 0.01` snapshot and
independently verified, see §6):

```
T(x, y, 0) = 0.5 * (1 - tanh((y - 1 - 0.02 sin(pi x)) / 0.1)),    u = v = 0
```

with `T` forced to 0 on the two wall rows. **The data is discontinuous at
`y = 0`**: the wall value is 0 while the interior value just above is
`1 - 4.1e-9`. That jump is what creates the bottom thermal layer, whose width at
`t = 0.01` is `sqrt(4 a4 t) = 0.0069`.

### The one thing you must get right if you regenerate this

As a period-1 function, `sin(pi x)` is `|sin(pi x)|`, which has a **corner** at
`x = 0`; its Fourier coefficients decay only like `1/m²`. Sampling that function
on the `Nx`-point grid **aliases** the tail onto the retained modes and leaves an
`O(Nx⁻²)` error — about `2.4e-4` at `Nx = 384` — in the whole solution, which no
amount of refinement removes.

The initial condition here is therefore the **Galerkin projection** in `x`: the
Fourier coefficients are computed by quadrature on a 2¹⁸-point grid and truncated
to `|m| ≤ Nx/2 - 1`. Both solvers did this independently and their projected ICs
agree; each also ran the collocated variant and reproduced the `1e-4` failure.
Anyone who regenerates the reference by *evaluating* the formula on the grid will
not reproduce this file at better than ~10⁻⁴.

---

## 3. How it was produced

Two solvers, written independently, sharing no code path beyond `numpy`:

### A. Vorticity–streamfunction solver — the delivered run

Fourier(384) × Chebyshev–Gauss–Lobatto(512 on `[0,2]`) pseudo-spectral,
vorticity–streamfunction form with a Kleiser–Schumann influence matrix enforcing
`ψ = ψ_y = 0` at both walls; the `k = 0` mean flow is carried in primitive form.
IMEX **SBDF3** (implicit diffusion, explicit advection and buoyancy), exact 3/2
zero-padding in `x`, 2/3 Chebyshev truncation in `y`, and projection onto the
`x`-parity subspace every step.

*Validation (its own suite):* MMS order 2.00; spectral convergence; Rayleigh–
Bénard critical Rayleigh number 1707.7619 against the literature 1707.762; the
linear RT growth rate to 6.5 × 10⁻⁷ against an independent eigensolver; discrete
divergence 4 × 10⁻¹⁵; and a separately written 2nd-order MAC finite-volume
solver converges **to** it at exactly `h²` (measured ratios 3.97–4.00 per mesh
halving, all fields).

*Startup* `t = 0 → 0.01`: constant `dt₀ = 1.25e-6`, 8000 steps, the first four
with L-stable SBDF1 (Crank–Nicolson rings on the wall jump) and the rest SBDF3.
A **constant** step is the correct grading here: `h/t → 0` as `t` grows, so the
scheme is most accurate exactly where the surviving error is made. A geometric
ramp `h ~ c t` was measured to be 5 × 10⁵ times worse.

*Main run* `t = 0.01 → 9.76`: `dt = 1.5625e-4`, 62 400 steps.

### B. Primitive-variable solver — the independent check

Fourier(384) × Chebyshev(512) collocation in primitive variables, **SBDF4** IMEX
started with three ARS(4,4,3) stages, incompressibility by an exact per-Fourier-
mode (ω, ψ) influence-matrix projection, 3/2 dealiasing in `x` only, **no**
parity projection, and a completely different startup (`dt = 7.8125e-5`, 128
steps).

*Validation (its own suite):* ARS/SBDF order checks; a Stokes manufactured
solution to 3 × 10⁻¹⁴; the erf wall layer; an eigenvalue-problem growth rate;
divergence 3 × 10⁻¹⁵.

### Refinement partners

| run | grid | dt | purpose |
|---|---|---|---|
| **VS C (delivered)** | 384 × 512 | 1.5625e-4 | the reference |
| VS B | 384 × 512 | 3.125e-4 | time refinement |
| VS A | 256 × 384 | 3.125e-4 | space refinement |
| VS D | 384 × 768 | 3.125e-4 | space refinement in `y` |
| PV A (2nd reference) | 384 × 512 | 1.5625e-4 | independent formulation |
| PV B | 256 × 384 | 1.5625e-4 | its space partner |
| PV C | 384 × 512 | 2.0833e-4 | its time partner |

All CPU-only, `numpy`; ~1–2.5 h per production run.

---

## 4. Certified accuracy

All numbers are the evaluator's norm — relative Frobenius over a whole
100 × 200 slice. The table lists a selection of the 40 output times.

The certified number is **the difference between the two independent
references**. Two formulations that share no time integrator, no
incompressibility mechanism, no dealiasing convention and no startup, agreeing to
this level, is the strongest statement available without an exact solution. It is
an *upper bound* on the delivered field's error, not an estimate of it.

| t | u | v | T | p |
|---:|---:|---:|---:|---:|
| 0.01 | 1.4e-10 | 5.4e-10 | 1.5e-09 | 6.1e-10 |
| 0.51 | 2.2e-09 | 1.8e-09 | 6.8e-12 | 4.6e-10 |
| 1.01 | 4.3e-09 | 3.7e-09 | 2.9e-11 | 1.1e-10 |
| 2.01 | 8.0e-09 | 7.2e-09 | 3.3e-10 | 1.2e-10 |
| 3.01 | 1.1e-08 | 1.0e-08 | 1.8e-09 | 2.7e-09 |
| 4.01 | 1.3e-08 | 1.2e-08 | 6.0e-09 | 5.6e-09 |
| 5.01 | 1.8e-08 | 1.1e-08 | 1.2e-08 | 1.2e-08 |
| 5.76 | 2.1e-08 | 1.1e-08 | 1.6e-08 | 1.4e-08 |
| 7.01 | 2.3e-08 | 1.4e-08 | 2.5e-08 | 2.4e-08 |
| 8.51 | 2.5e-08 | 1.7e-08 | 2.9e-08 | 1.3e-08 |
| 9.76 | 6.3e-08 | 4.1e-08 | 4.5e-08 | 1.7e-08 |

| block | u | v | T | p |
|---|---|---|---|---|
| **window, slices 4…23** | 1.86e-08 | 1.12e-08 | 6.33e-09 | 1.06e-08 |
| **full domain, 40 slices** | 2.77e-08 | 1.65e-08 | 1.70e-08 | 1.31e-08 |

### Best estimate, and why it is smaller

The delivered run's own refinement budget — Richardson in `dt` (order measured
3.0–3.2) in quadrature with the spread over three spatial grids — is
**2.2 × 10⁻⁹ (u), 6.5 × 10⁻¹⁰ (v), 2.2 × 10⁻⁹ (T)** over the window and
4.1 / 2.4 / 2.1 × 10⁻⁹ over the full domain, i.e. an order below the certified
number.

### What separates the two references — a controlled experiment

Both solvers were restarted from **one bit-identical state** (the delivered run
at `t = 1.01`, 384 × 512) and integrated to `t = 3.01` with the same `dt`. That
removes the initial condition, the startup and all accumulated history; only the
discretisations remain. Differences at `t = 3.01` in `u`, same norm:

| comparison | difference in u |
|---|---:|
| VS restarted vs the VS production trajectory (restart artefact) | 2.3 × 10⁻¹³ |
| PV restarted vs the PV production trajectory (restart artefact) | 1.9 × 10⁻⁹ |
| VS **with** vs **without** 2/3 Chebyshev dealiasing in `y` | 2.4 × 10⁻¹⁴ |
| PV **with** vs **without** 2/3 Chebyshev dealiasing in `y` | 5.6 × 10⁻¹³ |
| VS **with** vs **without** the `x`-parity projection | 2.4 × 10⁻¹⁴ |
| **VS vs PV, both dealiasing in `y`** | **9.42 × 10⁻⁹** |
| **VS vs PV, neither dealiasing in `y`** | **9.42 × 10⁻⁹** |
| **VS vs PV, each in its production configuration** | **9.42 × 10⁻⁹** |
| the two production trajectories over the same interval | 1.06 × 10⁻⁸ |

Three readings:

1. **None of the configuration differences matters.** Turning the Chebyshev
   dealiasing on or off changes VS by 2.4 × 10⁻¹⁴ and PV by 5.6 × 10⁻¹³; the
   parity projection is worth 2.4 × 10⁻¹⁴. The two solvers separate by the *same*
   9.42 × 10⁻⁹ whether both dealias in `y`, neither does, or each keeps its own
   production setting. The separation is intrinsic to the two formulations.
2. **It is generated continuously**, at ~9.4 × 10⁻⁹ per two time units at this
   stage of the flow. Only about 10 % of the 1.06 × 10⁻⁸ production difference at
   `t = 3.01` is inherited from the different startups.
3. PV's production run is written in four checkpointed chunks, each
   re-bootstrapping SBDF4 with three ARS(4,4,3) stages; **each such restart costs
   1.9 × 10⁻⁹**. The VS restart costs 2.3 × 10⁻¹³.

**Where the difference comes from.** The VS family is internally consistent to
**1.3 × 10⁻¹¹** across three different spatial grids (256 × 384, 384 × 512,
384 × 768) at fixed `dt`, and to 2.2 × 10⁻⁹ across a factor-2 change of `dt`
(→ 3.1 × 10⁻¹⁰ for the delivered run by Richardson). The PV family is internally
scattered at **1.4 × 10⁻⁸** in space — and at `t = 3.01` its own space bound,
9.1 × 10⁻⁹, matches the 9.4 × 10⁻⁹ by which it separates from VS almost exactly.
PV's own analysis identifies that scatter as an **arithmetic floor**: refining
`y` to 384 × 640 and coarsening it to 384 × 384 move its answer by the same
*order* (5.2 × 10⁻⁹ and 4.0 × 10⁻⁹ at `t = 1.01`) rather than by the factor a
truncation error would give, and a 10⁻¹⁶ perturbation grows to 3.3 × 10⁻⁹ in two
time units in the late flow.

**Direct test of that claim.** The second solver was rerun from the same common
state to `t = 3.01` at three Chebyshev resolutions, everything else fixed.
Distance from the VS answer, in `u`:

| PV resolution | \|PV − VS\| at t = 3.01 |
|---|---:|
| 384 × **384** | 1.83 × 10⁻⁹ |
| 384 × **512** | 9.42 × 10⁻⁹ |
| 384 × **640** | 3.74 × 10⁻⁹ |

**It is not monotone in `Ny`** — refining 384 → 512 moves PV five times *further*
from VS, and refining again to 640 brings it back. No truncation error can do
that. And the three PV answers are further from **each other** than any of them
is from VS:

| | u | v | T |
|---|---:|---:|---:|
| mean \|PV − VS\| | 5.00 × 10⁻⁹ | 3.94 × 10⁻⁹ | 6.33 × 10⁻¹⁰ |
| mean \|PV − PV\| | 8.18 × 10⁻⁹ | 6.17 × 10⁻⁹ | 1.01 × 10⁻⁹ |
| \|VS − centroid(PV)\| / RMS radius of the PV cloud | 0.65 | 0.85 | 0.82 |

The VS answer sits **inside** the PV family's scatter, nearer its centroid than a
typical member. That is what you see when one solver has an arithmetic floor that
does not shrink under refinement and scatters about the true solution, with the
other solver at the centre of the scatter. A systematic error of VS would place
it *outside* the cloud.

On this evidence the cross-solver difference is dominated by the **second**
reference, and the delivered field is probably accurate to ~2 × 10⁻⁹. That
cannot be *certified* — proving it needs a third independent formulation — so
the certified number above remains the full cross-solver difference.

---

## 5. Is a PINN error of 10⁻⁴ / 10⁻⁶ / 10⁻⁸ measurable?

Measured directly: a synthetic model carrying a known smooth
error `eps` is scored against **both** independent references; the spread between
the two scores is the uncertainty of the measurement.

**Benchmark window (slices 4…23)**

| eps | field | score vs ref 1 | score vs ref 2 | uncertainty | verdict |
|---|---|---|---|---|---|
| 10⁻⁴ | u / v / T | 1.0000e-4 | 1.0000e-4 | < 0.01 % | **measurable** |
| 10⁻⁶ | u | 1.0000e-6 | 9.9645e-7 | 0.36 % | **measurable** |
| 10⁻⁶ | v | 1.0000e-6 | 9.9980e-7 | 0.02 % | **measurable** |
| 10⁻⁶ | T | 1.0000e-6 | 9.9993e-7 | 0.01 % | **measurable** |
| 10⁻⁸ | u | 1.0000e-8 | 1.9266e-8 | 93 % | **not resolved** |
| 10⁻⁸ | v | 1.0000e-8 | 1.4843e-8 | 48 % | **not resolved** |
| 10⁻⁸ | T | 1.0000e-8 | 1.1758e-8 | 18 % | marginal |

**Full domain (all 40 slices)**

| eps | uncertainty (u / v / T) | verdict |
|---|---|---|
| 10⁻⁴ | < 0.01 % / < 0.01 % / < 0.01 % | **measurable** |
| 10⁻⁶ | 0.13 % / 0.02 % / < 0.01 % | **measurable** |
| 10⁻⁸ | 189 % / 94 % / 96 % | **not resolved** |

**Plain answer.** 10⁻⁴ and 10⁻⁶ are measurable everywhere, with the reference
contributing well under 1 % of the reported number. 10⁻⁸ is **not** certified as
measurable — it is the same size as the difference between the two references —
except marginally for `T` inside the window. The *best estimate* of the
delivered field's error is ~2 × 10⁻⁹, so a 10⁻⁸ result is probably real, but
that cannot be certified with the evidence here.

For contrast, against the **upstream** dataset all three levels return the same
number — 1.1 × 10⁻² (u), 5.1 × 10⁻³ (v), 4.5 × 10⁻³ (T) over the window — and so
does a *perfect* model. The upstream data cannot distinguish a perfect model
from one with a 10⁻³ error, so **none** of the three levels is measurable
against it.

---

## 6. Physics checks on the delivered dataset

| check | result |
|---|---|
| incompressibility (spectral grid) | `‖u_x+v_y‖ / max(‖u_x‖,‖v_y‖)` = **2.65–3.54 × 10⁻¹⁴** at every one of the 40 outputs |
| wall values (absolute) | `max\|u\|` ≤ 2.6 × 10⁻¹³, `max\|v\|` ≤ 2.1 × 10⁻¹¹, `max\|T\|` ≤ 9.5 × 10⁻¹⁸ on `y = 0` and `y = 2`, all 40 slices |
| x-parity | `‖T(x)−T(−x)‖∞ / ‖T‖∞` ≤ 6.7 × 10⁻¹⁶; `v` even to 8.2 × 10⁻¹⁶; `u` odd to 9.3 × 10⁻¹⁶. (VS projects onto the parity subspace, so this is not an independent diagnostic — the independent control is the second reference, which does **not** symmetrise and whose parity violation reaches only 3.3 × 10⁻¹¹ over the window and 6.7 × 10⁻¹⁰ over the full domain.) |
| maximum principle | `0 ≤ T ≤ 1` to round-off: `T_min ≥ −8.3 × 10⁻¹⁸` and `T_max ≤ 1.0000000` at all 40 slices — no over- or undershoot anywhere, which a spectral method is not obliged to give and which fails immediately if the fields are under-resolved |
| spectral tails | for `t ≥ 0.26` the Fourier-x tail of `T` is 4.9 × 10⁻¹⁷ … 8.5 × 10⁻¹⁷ and the Chebyshev-y tail 2.5 × 10⁻¹⁵ … 6.0 × 10⁻¹⁵ — round-off. At slice 0 they are 2.8 × 10⁻¹¹ / 4.3 × 10⁻¹³, which is **correct**: the projected IC genuinely carries the `1/m²` tail of the `\|sin(πx)\|` corner, and diffusion removes it within `t ~ 0.1` |
| spectral decay | full spectra: tail/peak ratio ≤ **1.4 × 10⁻¹⁴** in both directions for all of `u, v, T` at `t = 1.01, 3.26, 5.76, 7.76, 9.76` — a fall of 14–16 orders with **no pile-up in the last modes**. Neither `Nx` nor `Ny` limits the solution at any output time |
| heat budget | `d/dt ∫T dy` vs `a4 [T_y]₀^Ly`: median relative residual **7.4 × 10⁻⁴**, and that is the `O(0.25⁴)` error of differencing the 0.25-spaced outputs, not an error of the reference |
| interface growth | single-valued to `t ≈ 3.8`, then rolls up; half-amplitude grows exponentially at rate 1.13–1.25 over `t = 1…2` before nonlinear saturation; bubble reaches the top wall at `t ≈ 8` |
| bubble/spike | spike (x = 0) falls 1.000 → 0.209, bubble (x = 1/2) rises 1.020 → 1.577 by `t = 5.01` — correct RT phenomenology |

### Agreement with the upstream data at early times

This is the check that the initial condition is the right one. At `t = 0.01`,
outside the upstream data's unresolved wall layer (`y > 0.05`):

| field | ‖ref‖/‖upstream‖ | correlation | relative difference |
|---|---|---|---|
| T | 1.0000000 | 1.0000000 | **1.74 × 10⁻⁶** |
| p | 1.0000150 | 1.0000000 | **6.0 × 10⁻⁵** |
| u | 1.0029690 | 0.9999970 | 3.85 × 10⁻³ |
| v | 1.0012200 | 0.9999979 | 2.38 × 10⁻³ |

The temperature agrees to **1.7 parts per million** and the pressure to 6 × 10⁻⁵
— the analytic IC is confirmed. The velocity disagreement is a pure **amplitude**
deficit (correlation 0.999997, ratio 1.0030) living 99 % at the interface: the
upstream solver's velocity at its first output is 0.3 % too small, the signature
of a low-order startup from rest. Agreement then degrades monotonically as the
instability rolls up, reaching 2–6 % at `t = 9.76`.

`T` agreement per slice: 1.7e-06 (t=0.01, y>0.05), 5.4e-05, 7.5e-05, 1.0e-04,
1.3e-04 (t=1.01) … 1.7e-03 (t=3.01) … 9.8e-03 (t=5.76) … 2.2e-02 (t=9.76).

### The upstream velocity is not incompressible — grid-independent test

The stored velocity is lifted to the spectral grid and L2-projected onto the
divergence-free / no-penetration subspace. The size of that projection is how far
the field is from *any* incompressible field, and it involves no difference
operator at all:

| t | upstream, u | upstream, v | reference, u | reference, v |
|---|---|---|---|---|
| 0.01 | 4.79e-04 | 5.16e-04 | 6.0e-16 | 3.8e-16 |
| 1.01 | 4.38e-04 | 4.38e-04 | 3.2e-16 | 3.3e-16 |
| 3.26 | 4.16e-04 | 2.31e-04 | 5.4e-16 | 3.3e-16 |
| 5.76 | 8.01e-04 | 2.20e-04 | 8.2e-16 | 3.3e-16 |
| 7.76 | 1.15e-03 | 4.53e-04 | 5.2e-16 | 3.4e-16 |
| 9.76 | 1.29e-03 | 4.43e-04 | 5.4e-16 | 4.0e-16 |

So the upstream velocity could not be matched below ~5 × 10⁻⁴ by *any*
incompressible reference, whatever the method. The new reference sits on the
divergence-free manifold to round-off.

### The upstream snapshots are not consistent with each other under the PDE

A restart test: a solver is restarted from a stored upstream snapshot and
integrated 0.25 or 0.5 time units, then compared with the next stored
snapshot. The residual is 5 × 10⁻⁵ … 7 × 10⁻³ and it **does not decrease**
from 192 × 384 to 384 × 768 to 512 × 1024 (e.g. `t = 1.26 → 1.76`, `T`:
1.109e-4 / 1.187e-4 / 1.204e-4; `t = 7.51 → 8.01`, `u`: 6.9e-3 / 5.6e-3 /
5.6e-3). A grid-converged nonzero residual means the upstream snapshots are not
consistent with each other under the equations they are supposed to solve.

The same test applied to this reference is the toggle experiment above: the
*other* solver, restarted from the delivered field and run for **two** time
units, lands within 9.4 × 10⁻⁹.

### Divergence of the upstream data vs the reference, on the stored grid

Measured with the same high-order operator on the same 100 × 200 points:

| t | upstream | new reference |
|---|---|---|
| 0.01 | 2.96e-03 | 7.35e-04 |
| 1.01 | 2.30e-03 | 2.41e-04 |
| 3.01 | 1.55e-03 | 2.87e-04 |
| 6.01 | 2.60e-02 | 2.66e-02 |
| 9.76 | 2.70e-02 | 2.69e-02 |

Read this carefully. The reference is solenoidal to 3 × 10⁻¹⁴ on its own grid, so
everything here is the *difference operator's* truncation error on the stretched
100 × 200 grid. Up to `t ≈ 4` that error is 3–10× smaller for the reference,
i.e. the upstream data carries a genuine divergence the reference does not. From
`t ≈ 6` both sit at ~3 × 10⁻², because there **the stored grid can no longer
differentiate the field at all** — see §7.

---

## 7. Known limitations

1. **10⁻⁸ is the floor, and it is not certified below ~2 × 10⁻⁸.** The certified
   number is the cross-solver difference. The delivered run's own budget says
   ~2 × 10⁻⁹, but that budget is blind to anything both of its refinement
   partners share.

2. **The stored 100 × 200 grid does not resolve the late-time field.**
   Interpolating the delivered `.npy` to the midpoints of its own grid and
   comparing against the exact spectral field gives (window maxima) `u`
   4.5 × 10⁻⁴, `v` 2.7 × 10⁻⁵, `T` 5.8 × 10⁻⁶, rising to ~10⁻³ (u) at
   `t ≈ 8`. Comparing *at* the stored points is unaffected — it is exact to the
   accuracy in §4 — but **any consumer who interpolates the `.npy` inherits those
   numbers**. The benchmark evaluates at the stored points only; the
   full-resolution spectral fields are not part of this artifact.

3. **The pressure is a reconstruction, not an evolved field.** Both formulations
   eliminate `p` exactly. It is recovered here as the Leray/Neumann pressure,
   `Δp = -div(u·∇u) + a2 T_y` with `p_y|wall = a1 v_yy|wall`, zero mean — the
   second solver's pressure operator applied to the delivered `(u, v, T)`.
   Applying that same operator to the two independent references gives fields
   agreeing to 1.06 × 10⁻⁸ (window), and the result agrees with the
   *upstream* `p` to 5.4 × 10⁻⁵ at `t ≤ 2`, which is far better than any other
   field — strong evidence that this is the pressure the original solver stored.
   It is nevertheless a *derived* quantity: it inherits the accuracy of
   `(u, v, T)` through a fixed operator but its own formulation was not
   independently certified.

4. **`t = 0` is not in the dataset and cannot be.** The initial–boundary-value
   problem is incompatible at `t = 0` by construction (`T = 0` imposed at the
   wall, interior value 1). `t = 0.01` is the earliest time at which the wall
   layer is representable on these grids to better than 10⁻¹².

5. **Run A of the VS family (256 × 384) must not be used at slice 0**: its
   `Nx = 256` truncation of the corner-bearing IC gives 3.4 × 10⁻⁷ in `v` there.
   `Nx ≥ 384` is clean (6.5 × 10⁻¹¹). The slice-0 space term of the refinement
   budget therefore uses the `y`-only refinement.

6. **Two full-domain resolutions per solver, not three.** The spectral tails are
   at round-off at every output time and the toggle/`Ny`-scaling experiments
   cover `t = 1.01 … 3.01` at four extra resolutions, but there is no third
   *full-domain* grid.

7. **The attribution argument is evidence, not proof.** The `Ny`-scaling and
   centroid tests show the second solver scattering about the first, which is
   what an arithmetic floor looks like; they do not *prove* the first solver is
   right. Settling that needs a third independent formulation, or extended
   precision. Until then the certified number stays at the cross-solver
   difference.

8. **Contended machine.** All runs shared a 128-core node with other jobs; wall
   times are not clean performance measurements.

---

## 8. How to use it

`benchmarks/rayleigh_taylor/utils.py` loads it (`DATA_FILE`) and restricts it
to the benchmark window:

```python
data = np.load("rayleigh_taylor_spectral.npy", allow_pickle=True).item()
# t (40,), coords (20000, 2), velocity (40, 20000, 2), pressure (40, 20000),
# temperature (40, 20000), and the scalars Ra, Pr, Ge, alpha1 ... alpha4;
# keys, dtypes, coords and t are identical to the upstream file's
```

Checked against the upstream file: `t` and `coords` differ by exactly 0.0,
the loader returns all four `sol` arrays with the same shape `(20, 20000)` and
dtype, and every `pde_param` is identical. The pressure carries the upstream
file's gauge (zero mean over the 100 × 200 grid, reproduced to 9 × 10⁻¹⁷).

## 9. Regeneration

The two spectral solvers of §3, the refinement, cross-check and attribution
scripts behind §§4–6, and their raw outputs are **not part of this artifact**;
the file cannot be regenerated from it. §§2–3 describe the method: the
problem, the Galerkin-projected initial condition, both discretisations, their
time steps and startups, and the refinement partners of the table in §3.
`verify_reference_solutions.py` at the repository root checks the shipped file
without re-solving it (output times, window, finiteness, coefficients, maximum
principle, mirror symmetry; see `REFERENCE_SOLUTIONS.md`).
