# GridapLFEM.jl — the algebraic LFE-M wave solver

A loop-free, serial-and-distributed Gridap implementation of the **LFE-M** (Layer-averaged Finite
Element Multilayer) depth-integrated, non-hydrostatic wave model of Yang & Liu (2024, *J. Fluid
Mech.* 999, A32). It uses a **stacked value-type layout** that turns every per-layer sum in the
governing equations into a native Gridap tensor contraction instead of a Julia `for`-loop.

Repository: `GridapLFEM.jl`. Module: `src/GridapLFEM.jl` → `module GridapLFEM`. Fully self-contained.

---

## The core idea

The model represents the horizontal velocity as a sum of vertical basis functions,
`u(x, σ, t) = Σⱼ uⱼ(x, t) φⱼ(σ)`, so every term in the depth-integrated momentum/continuity
equations carries a sum over the vertical (layer) index `j = 0..Nσ`. The direct approach — one
scalar FE field per layer per component (`1 + 2Nσ` fields), with the layer sums contracted by Julia
loops inside the residual — works but is verbose and slow to assemble. This solver instead **stacks
the layer index into the FE value type**:

```
MultiField = [η, 𝖴x, 𝖴y]        (exactly 3 fields, not 1+2Nσ)
𝖴x, 𝖴y  ∈  VectorValue{Nσ}       (all Nσ layers' x-/y-velocity in one field value)
```

Every vertical sum becomes a single algebraic operation on `VectorValue{Nσ}` /
`TensorValue{Nσ,Nσ}` / `ThirdOrderTensorValue{Nσ,Nσ,Nσ}` objects:

| model sum | stacked Gridap expression |
|---|---|
| `Σⱼ Mᵢⱼ u̇ⱼ` | `𝗠 ⋅ Ut`  (matvec) |
| `Σⱼ Φⱼ uⱼ` (depth average) | `𝚽 ⋅ Ux` |
| `Σₖⱼ 𝓜ᵢₖⱼ (uₖ·∇uⱼ)` | `double_contraction(𝗠3, Ux⊗∂x(Ux) + Uy⊗∂y(Ux))` |

No custom contraction helpers are needed — `⋅`, `⊗`, `double_contraction`, and `∇` are all
native Gridap operators on these value types. The authoritative derivation and rationale live in the
LaTeX project `building_files/LFEM_discretisation.zip` (§8, *Gridap solver implementation*), mirrored
in `building_files/LFEM_Gridap.md`, with the term-by-term operator simplifications in
`building_files/algebraic_residual_math.md`.

**Why it's worth doing:** collapsing the layer sums into single dense contractions keeps the
residual compact and well-typed, and makes assembly fast — on the fully nonlinear benchmark the
stacked residual runs **~3.6× faster with ~13× fewer allocations** than a per-layer assembly of the
same weak form (`building_files/DESIGN_RECORDS.md`), because there is no fused per-layer advection
integrand to expand. The same CellField algebra is forwarded to `DistributedCellField`, so the
identical residual runs sequentially and across MPI ranks.

---

## Physics implemented

The **complete** LaTeX §8 global residual, every term:

- Mass continuity (with internal Gaussian wavemaker source)
- H-weighted acceleration
- Nonlinear horizontal + vertical advection (`𝓜`, `𝓖` tensors)
- Gravity (integrated-by-parts energy form, rest-state safe)
- **Leading pressure `R_P`** — the model's entire frequency dispersion (on a flat bed it reduces to
  the classical `d²B` dispersion operator; the nonlinear core keeps the complete
  `P¹L¹+P²L²+P³L³` decomposition)
- Linear (bed-slope / surface-slope) non-hydrostatic pressure package
- **Full nonlinear pressure**, all eight `𝓝ₖⱼ` components in all three residual blocks — the native
  first-order set {3,6,7,8} plus the Class-III set {1,2,4,5} via exact integration-by-parts for the
  bed-slope half and frozen L²-projections for the surface-slope / leading-pressure halves
- Quadratic sponge absorption layers — damp **both the velocity and the free surface η**
  (`+∫ μ q η`, same μ profile/`mu_max`), so open-boundary modes are fully absorbed
- Solid-wall / open / **periodic** (in y) boundary conditions
- **Wave generation — three mechanisms** selected by `wave_gen`: `:inner_res` (interior Gaussian
  source: line ⇒ plane wave, point ⇒ ring wave), `:bc_gen_profile` (a **parametrised** wave generated
  from a boundary — a periodic plane wave from `A_wave`/`T_wave`/`wave_dir`, or a supplied `WaveInput`),
  `:bc_gen_airy` (a WaveSpec stochastic sea from a boundary). `:auto` (default) infers it from
  `wave_bc`, validated by `resolve_wave_gen`
- **Dirichlet boundary wave generation** (`:bc_gen_profile` / `:bc_gen_airy`) — waves enter through time-varying Dirichlet
  data (η, 𝖴x, and 𝖴y for directional seas) on a domain side, no interior source: regular waves,
  hand-built multichromatic seas, or **WaveSpec.jl stochastic sea states** (JONSWAP/TMA/…, angular
  spreading) via the `AiryState → WaveInput` converter. Discrete LFE-M eigenmode vertical
  polarization (`:model`, default) or Airy cosh sampling (`:airy`); Hann start-up ramp; optional
  generation/absorption relaxation zone (`relax_bc`). See `building_files/boundary_wave_generation.md`.

### Selecting the model — three orthogonal controls

The physics is chosen through three high-level switches, mapped to the internal residual flags by
`resolve_physics`:

| control | values | meaning |
|---|---|---|
| `regime` | `:linear` \| `:nonlinear` | small-amplitude linearised core vs. full finite-amplitude core (advection + nonlinear leading pressure) |
| `nl_pressure` | `:none` \| `:native` \| `:full` | nonlinear non-hydrostatic pressure: off / native set `{3,6,7,8}` / `+` Class-III `{1,2,4,5}` |
| `flat_bed` | `false` \| `true` | sea-bed geometry: variable bathymetry (∇h≠0) vs. flat bed (**∇h≡0** — every ∇h-term dropped at a single control point; ∇η/dispersion terms kept) |

`regime`/`nl_pressure` fix *which* physics the model carries; `flat_bed` fixes the *bed geometry* it is
solved over. `regime=:linear` with `nl_pressure≠:none` is rejected. The frequency dispersion (the
defining LFE-M feature) is present in every configuration through `R_P`. The low-level
`build_problem_raw` still exposes the individual internal booleans for the rare split the high-level
interface deliberately ties (used by the oracle-equivalence test).

Every configuration is validated against a hand-derived reference or an independent per-layer assembly
of the same weak form; see [Validation](#validation).

---

## Repository layout

```
GridapLFEM.jl/
├── src/                          # the solver package (module GridapLFEM)
│   ├── GridapLFEM.jl             module entry: deps, includes, exports, g/rho constants
│   ├── vertical.jl              Stage 1: σ-mesh + vertical tensor set (M, Φ, 𝓜, 𝓖, A, K, P, 𝓝, …)
│   ├── tensors.jl               constant-tensor constructors + pointwise Operation helpers
│   ├── horizontal.jl            Stage 2: 2D mesh + stacked FE spaces (serial + distributed; periodic-y)
│   ├── problem.jl               LFEMProblem, resolve_physics, residual, hand Jacobians
│   ├── nlpressure.jl            full nonlinear pressure (native / exact-IBP / frozen projections)
│   ├── reconstruct.jl           w(σ) / total-pressure σ-level VTK field reconstruction
│   ├── monitor.jl               runtime SolverMonitor + independent governing-equation residual checker
│   ├── timeloop.jl              sequential ODE solver factory (SDIRK / θ) + time loop
│   ├── utilities.jl             dispersion analysis, sponge/wavemaker, setup_and_run
│   ├── waveinput.jl             Dirichlet boundary wave generation + WaveSpec.jl coupling
│   ├── timeloop_dist.jl         distributed mesh + GMRES+Jacobi+Newton solver + time loop
│   └── utilities_dist.jl        setup_and_run_distributed
├── test/                         # 21 test files + test/cluster/ — see Validation
├── examples/
│   ├── plane_wave.jl · ring_wave.jl · periodic_plane_wave.jl     sequential interior-source
│   ├── bc_plane_wave.jl · bc_irregular_sea.jl · bc_directional_sea.jl   sequential BC generation
│   ├── validation/               physical benchmarks (Stokes, bar, soliton, ring, sideband) + README
│   ├── distributed/              8 env-configurable cluster MPI scripts + README
│   └── distributed_small/        5 parametric small-domain (50×20) scripts (line/point/bc-plane/dir/irreg)
├── run/                          # SLURM launchers (snellius/blue) + run/dist_small/ (16 small-domain jobs)
├── postprocessing/GridapLFEMPost # self-contained VTK/CSV analysis + plotting library (own env)
├── compile/                      # cluster sysimage build (compile.jl, warmup.jl, module loaders)
├── WaveSpec.jl/                  # vendored stochastic sea-state package (Pkg.develop'd)
├── building_files/               # LaTeX project (zip), design records, prototype, notebook
│   ├── LFEM_discretisation.zip   authoritative LaTeX derivation (compiles; §8 = this solver, §9 = validation)
│   ├── LFEM_Gridap.md            clean synthesis of the derivation → the scalar residual
│   ├── algebraic_residual_math.md   operator simplifications used to write the residual
│   ├── boundary_wave_generation.md  Dirichlet-generation math note (:model/:airy polarization)
│   ├── DESIGN_RECORDS.md         consolidated design/implementation records (residual, distributed,
│   │                               nonlinear pressure, BC generation, periodic BC, flat_bed switch)
│   ├── LFEM_runs.md              small-domain observation/comparison run-suite plan
│   ├── algebraic_lfem2D.jl / test_algebraic_lfem2D.jl   single-file residual prototype + its test
│   └── test_2HDmodel.ipynb       early exploratory notebook
├── Project.toml / Manifest.toml  standalone environment (pinned to the parent versions)
├── output/                       # VTK run output (created on demand)
└── CLAUDE.md                     # up-to-date status/notes for this folder
```

---

## Environment / activation

`GridapLFEM.jl/` is **not a registered Julia package** (its `Project.toml` has no
`name`/`uuid`/`version`) — loading it is always done by file-path `include(...)` +
`using .GridapLFEM` (note the leading dot), never `using GridapLFEM` after `Pkg.add`. Two
environments can supply the dependencies for that `include`:

- **The parent repository's environment** (`../Project.toml`, i.e. `TS_2HDmodel/`) — what every result
  in [Validation](#validation) was run against and what the [Quick start](#quick-start) / cluster
  examples assume (`--project=.` from the repo root).
- **This folder's own environment** (`GridapLFEM.jl/Project.toml` + `Manifest.toml`) — a
  self-contained, standalone environment pinned to the **exact same** package versions as the parent
  (Gridap 0.19.11, GridapDistributed 0.4.13, GridapSolvers 0.6.2, PartitionedArrays 0.3.5,
  MPI 0.20.26) so results are identical either way. Useful for extracting this solver independently.

Either way the loading code is the same:

```julia
include("src/GridapLFEM.jl")   # or "GridapLFEM.jl/src/GridapLFEM.jl" from one level up
using .GridapLFEM
```

---

## Quick start

```julia
include("GridapLFEM.jl/src/GridapLFEM.jl")
using .GridapLFEM

# Sequential run: nonlinear plane wave in a flat-bed flume
diags, vert, prob = setup_and_run(
    M            = 2,                          # vertical layers (LFE-2)
    domain       = ((0.0, 60.0), (0.0, 10.0)),
    partition    = (120, 20),                  # horizontal mesh
    p_horizontal = 2,                          # Q2 (≥2 required — Q1 zeroes dispersion)
    h_val        = 3.5, T_wave = 1.6, A_wave = 0.001,
    x_wm         = 12.0,                        # wavemaker position (line source; y_wm=nothing)
    sponge_wL    = 12.0, sponge_wR = 12.0, mu_max = 5.0,
    T_final      = 12.8, dt = 0.02,
    regime       = :nonlinear,                 # :linear for the linearised dispersive model
    nl_pressure  = :none,                       # :native / :full add the O(A³) non-hydrostatic pressure
    flat_bed     = true,                        # ∇h≡0; false + h_bathy=… for variable bathymetry
    gauges       = [(30.0, 5.0)],
    save_every   = 20, output_dir = "output/my_run",
)
```

`diags` is a `Vector` of `(t, eta_max, gauge_vals)` NamedTuples. VTK snapshots (`eta, u1x, u1y, …`,
plus reconstructed `w_s<σ>`/`p_s<σ>`) and a ParaView `.pvd` index are written to `output_dir` when
`save_every > 0`.

### Distributed (MPI)

```julia
# Same physics; cpu_grid=(px,py) ranks; call from an mpiexecjl -n (px*py) context
diags, vert, prob = setup_and_run_distributed(
    cpu_grid = (2, 2), M = 2, domain = (0.0,60.0,0.0,10.0), partition = (120,20),
    h_val = 3.5, T_wave = 1.6, A_wave = 0.001, x_wm = 12.0,
    sponge_wL = 12.0, sponge_wR = 12.0, mu_max = 5.0,
    T_final = 12.8, dt = 0.02,
    regime = :nonlinear, nl_pressure = :none, flat_bed = true,
)
```

```bash
~/.julia/bin/mpiexecjl --project=. -n 4 julia --project=. my_script.jl
```

Use Julia's own MPI launcher (`mpiexecjl`) — the system `mpiexec` fails on this development machine
with a PMIx version mismatch. `nx`/`ny` in `partition` must divide evenly by `px`/`py`. Distributed
`diags` carries `(t, eta_max)` only (point gauges need inter-rank communication). See
`examples/distributed/README.md` for cluster/SLURM usage, and `examples/distributed_small/` +
`run/dist_small/` for a ready-made small-domain (50×20) observation/comparison suite.

---

## `setup_and_run` / `setup_and_run_distributed` — key options

| Argument | Meaning |
|---|---|
| `M`, `p_vertical`, `c_bdy` | vertical layers, polynomial order, σ-node positions (defaults = Yang & Liu Table 1 optimised nodes for M≤4) |
| `domain`, `partition`, `p_horizontal` | horizontal mesh (`p_horizontal ≥ 2` required — Q1 zeroes the dispersion term) |
| `h_val` / `h_bathy` | constant depth, or `h_bathy(x)` for variable bathymetry (pair with `flat_bed=false`) |
| **`regime`** | `:linear` (linearised, no advection) \| `:nonlinear` (full finite-amplitude core) |
| **`nl_pressure`** | `:none` \| `:native` `{3,6,7,8}` \| `:full` `+{1,2,4,5}` — the O(A³) non-hydrostatic pressure |
| **`flat_bed`** | `true` = flat bed (∇h≡0) \| `false` = variable bathymetry; a driver warning flags a bed↔switch mismatch |
| **`wave_gen`** | wave-generation mechanism: `:inner_res` (interior Gaussian source) \| `:bc_gen_profile` (parametrised boundary wave) \| `:bc_gen_airy` (WaveSpec boundary sea); `:auto` (default) infers it from `wave_bc` |
| `T_wave`, `A_wave`, `x_wm`, `y_wm` | interior source (`:inner_res`): `y_wm=nothing` → line source (plane wave), else point source (ring wave) |
| `wave_dir` | propagation angle vs +x for a `:bc_gen_profile` boundary wave (0 = normal incidence) |
| `sponge_wL/wR/wB/wT`, `mu_max` | quadratic absorbing sponge widths and strength; damps **velocity AND η** (0 width = side off) |
| `eta0_func` | initial free-surface release `η₀(x)` — **requires `x_wall_bc=true`** (closed basin) |
| `wave_bc` | boundary-generation source: `:regular` (from `A_wave`/`T_wave`/`wave_dir`), a `WaveInput`, or a `WaveSpec.AiryWaves.AiryState` (auto-converted); disables the interior wavemaker |
| `bc_side`, `bc_profile` | generation boundary (`:left`/`:right`); vertical polarization (`:model` = discrete eigenmode, default / `:airy` = cosh sampling) |
| `T_ramp`, `ic_from_bc` | Hann start-up ramp (default 2 peak periods); hot start from the incident field (needs `T_ramp=0`) |
| `relax_bc`, `relax_width` | generation/absorption relaxation zone at the inflow (strength `mu_max`, default width one peak wavelength) |
| `y_wall_bc`, `x_wall_bc` | lateral BC **symbol** `{:wall, :open, :periodic}` (directional seas need `:open`/`:periodic`); `x_wall_bc::Bool` closed-x basin, mandatory for any IC-release problem |
| `solver_type`, `tableau` | time integrator: `:sdirk` (default, `SDIRK_2_2`, L-stable) \| `:theta` (Crank–Nicolson); `:rk3` (`SDIRK_3_4`) sequential-only |
| `write_w`, `write_pressure`, `rho` | reconstruct vertical velocity / total pressure at every σ-node into VTK |
| `save_every`, `output_dir`, `gauges` | VTK snapshot cadence, output directory, point-gauge stations (sequential only) |

Distributed adds `cpu_grid=(px,py)` and Krylov controls (`nl_iter`, `nl_tol`, `ls_rtol`,
`ls_maxiter`, `nlp_cg_rtol`, `nlp_cg_maxiter`), plus the same `solver_type`/`tableau`; the linear
solve is GMRES+Jacobi+Newton. It drops `gauges` (global reductions instead).

---

## Validation

All gates below are standalone Julia scripts in `test/`; run with
`julia --project=. GridapLFEM.jl/test/<name>.jl` (distributed ones via `mpiexecjl`, see file headers).
The full suite is **21 tests + `test/cluster/`**; representative highlights:

| Test | What it checks | Result |
|---|---|---|
| `test_vertical.jl` | vertical tensor identities, dispersion bridge vs Yang & Liu Table 1 | 15/15 PASS |
| `test_primitives.jl` | tensor index order, `∂x/∂y` orientation, contraction semantics | 9/9 PASS |
| `test_equivalence.jl` | virtual-work match vs an independent per-layer assembly, 3 configs | 10/10 PASS, rel ≤ 7e-15 |
| `test_dispersion.jl` | FEM phase speed vs linear theory at kd=3 | PASS, err 0.90% |
| `test_dispersion_curve.jl` | `Cm/Ce(kd)` sweep vs Airy, LFE-2/3/4 applicable-kd | 9/9 PASS (10.8/39.2/127.9) |
| `test_dispersion_nonlinear.jl` | full-NL solver ⇒ Airy at vanishing amplitude (kd 1/3/5) | 3/3 PASS |
| `test_shallow_water.jl` | `kd→0 ⇒ √(gd)` limit | 6/6 PASS |
| `test_sloshing.jl` | standing-wave period vs LFE-M theory | PASS, err 1.44% |
| `test_energy.jl` | non-dissipativity / amplitude preservation | 3/3 PASS |
| `test_conservation.jl` | mass conservation, closed basin, nonlinear advection | PASS, drift 7.8e-16 |
| `test_nlpressure.jl` | exact-IBP identity, structural scaling, dynamics (all pressure tiers) | 9/9 PASS |
| `test_mms.jl` | unsteady nonlinear manufactured solution over a curved bed | 3/3 PASS (~3e-9) |
| `test_convergence.jl` | Richardson temporal order (→2) | 2/2 PASS (q≈1.72) |
| `test_vertical_profile.jl` | reconstructed `w(σ)` vs Airy `sinh` shape | 7/7 PASS |
| `test_basic.jl` | smoke run, linear + fully nonlinear | 6/6 PASS |
| `test_waveinput.jl` | Dirichlet-generation data: identities, closures, AD, WaveSpec converter | 30/30 PASS |
| `test_bc_generation.jl` | Dirichlet-generated regular wave e2e (kd=3) | 11/11 PASS |
| `test_bc_spectrum.jl` | 3-component Dirichlet sea: Goda–Suzuki incident amplitudes | 8/8 PASS |
| `test_basic_distributed.jl` | 4-rank MPI, linear + fully nonlinear vs sequential | 6/6 PASS, rel ≤ 5e-9 |
| `test_nlpressure_distributed.jl` | 4-rank MPI, full nonlinear pressure vs sequential | 3/3 PASS |
| `test_bc_generation_distributed.jl` | 4-rank MPI Dirichlet generation vs sequential | 4/4 PASS, rel 3.1e-8 |
| `cluster/cluster_conservation.jl` · `cluster/cluster_mms.jl` | long-run mass conservation / nonlinear MMS at scale | dist |

The reference for `test_equivalence.jl` is an independent per-layer assembly of the same weak form in
`../LFE-M_2D_solver/`, differently structured, so agreement to machine precision cross-checks the
stacked residual. The full validation report is §9 of `building_files/LFEM_discretisation.zip`; design
records are in `building_files/DESIGN_RECORDS.md`.

---

## Design notes for anyone extending this solver

- **`flat_bed` is one control point.** The flat-bed reduction is realised by zeroing the bed gradient
  (`(∂ₓh,∂ᵧh) = flat_bed ? 0 : ∇h`) in `global_residual`/`jacobian_*` and skipping the bed-slope IBP
  half — that single switch drops every ∇h-term (bed-slope `𝓐` packages, `L¹`, `N{3,6}`, bed Hessian)
  while `∇H→∇η` keeps the surface-slope and dispersion terms. Physics content is fixed by
  `regime`/`nl_pressure`; `flat_bed` only chooses the bed geometry.
- **Default integrator is `RungeKutta(:SDIRK_2_2)`** (fully implicit, L-stable, 2nd-order) — markedly
  more robust in the stiff/nonlinear regime than Crank–Nicolson (`:theta`), which is still selectable.
- **Never apply `∇` to an `Operation`-composed expression that contains a test-function basis** —
  Gridap's block-array `copyto!` isn't implemented for that combination. Expand such derivatives by
  hand using linearity of the vertical contraction in the test, `∂ₐ(W⋅𝓣) = (∂ₐW)⋅𝓣`, and
  `Σₐ Ψ·∂ₐUₐ = Ψ·DU` (see `nlpressure.jl` for worked examples).
- **AD Jacobians are not viable** on this residual under Gridap 0.19.11 — the multifield autodiff
  split cannot dualize through `∂t(u)` (a missing `TransientMultiFieldCellField` constructor). Hand
  Jacobians (`jacobian_u`, `jacobian_u_t`) are the design; they are exact for everything except the
  `O(A³)` nonlinear-pressure terms, which are treated quasi-Newton.
- **Distributed linear solves must not use `lu()`** — no method for a partitioned `PSparseMatrix`. Use
  `GMRESSolver`/`CGSolver` with `JacobiLinearSolver()` from GridapSolvers, and allocate RHS/solution
  vectors **from the matrix** (`allocate_in_range`/`allocate_in_domain`), not from an independently
  `assemble_vector`-built vector.
- **Distributed initial conditions** must use `interpolate_everywhere` — `FEFunction(U, zeros(...))`
  builds a local array and fails on a partitioned space.
- Keep `ConsecutiveMultiFieldStyle` (the default) — `BlockMultiFieldStyle` breaks the Jacobi
  preconditioner's `diag`.
- **Solid-wall Dirichlet BCs must include the corner tags**, and IC-release problems need
  `x_wall_bc=true`, or corner DOFs / a spurious-forcing mode blow up.

---

## Known issues

- The vendored **`WaveSpec.jl`** `change_seed!` currently accesses a non-existent `state.spec` field
  (the struct field is `spectrum`), so `build_airy_state` errors — this blocks the Dirichlet **sea-state**
  runs (production `run_{irregular,directional}_sea_dist.jl` and the small sea cases). A one-line
  library fix; the run scripts themselves are correct. Regular-wave / multichromatic BC generation and
  all interior-wavemaker cases are unaffected.
