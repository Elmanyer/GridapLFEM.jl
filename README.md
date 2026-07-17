# GridapLFEM.jl — the algebraic LFE-M wave solver

A loop-free, serial-and-distributed Gridap implementation of the **LFE-M** (Layer-averaged Finite
Element Multilayer) depth-integrated, non-hydrostatic wave model of Yang & Liu (2024, *J. Fluid
Mech.* 999, A32). This is a from-scratch reimplementation of the model already validated in
`../LFE-M_2D_solver/`, using a **stacked value-type layout** that turns every per-layer sum in the
governing equations into a native Gridap tensor contraction instead of a Julia `for`-loop.

Repository: `GridapLFEM.jl`. Module: `src/GridapLFEM.jl` → `module GridapLFEM`. Fully
self-contained (no dependency on the older per-layer solver, which remains the validated oracle
it was checked against).

---

## The core idea

The model represents the horizontal velocity as a sum of vertical basis functions,
`u(x, σ, t) = Σⱼ uⱼ(x, t) φⱼ(σ)`, so every term in the depth-integrated momentum/continuity
equations carries a sum over the vertical (layer) index `j = 1..Nσ`. The old solver represents
each layer's velocity components as **separate scalar FE fields** (`1 + 2Nσ` fields total) and
contracts the layer sums with Julia loops inside the residual. This solver instead **stacks the
layer index into the FE value type**:

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
native Gridap operators on these value types. See `building_files/main.tex` §8 ("Field layout"
subsection) for the full derivation and rationale, and `building_files/algebraic_residual_math.md`
for the term-by-term operator simplifications.

**Why it's worth doing:** on the fully nonlinear benchmark, the stacked solver reproduces the
per-layer oracle to machine precision while running **~3.6× faster with ~13× fewer allocations**
(see `building_files/algebraic_solver_plan.md` §6) — the fused per-layer advection integrand is
what made the old solver need a matrix-free V⊗H workaround; the stacked residual never needs one.

---

## Physics implemented

The **complete** `building_files/main.tex` §8 global residual, every term:

- Mass continuity (with internal Gaussian wavemaker source)
- H-weighted acceleration
- Nonlinear horizontal + vertical advection (`𝓜`, `𝓖` tensors)
- Gravity (oracle-matching IBP energy form, rest-state safe)
- **Leading pressure `R_P`** — the model's entire linear frequency dispersion (flat-bed reduces to
  the classical `d²B` dispersion operator); optional full `P¹L¹+P²L²+P³L³` form (`P_full`)
- Linear (bed-slope/surface-slope) non-hydrostatic pressure package (`lin_pressure`)
- **Full nonlinear pressure**, all eight `𝓝ₖⱼ` components, in all three residual blocks
  (`nl_pressure68` for the native first-order set {3,6,7,8}; `nl_pressure_full` adds {1,2,4,5} via
  exact integration-by-parts for the bed-slope half and frozen L²-projections for the
  surface-slope/leading-pressure halves — see `building_files/algebraic_pressure_completion_plan.md`)
- Quadratic sponge absorption layers
- Solid-wall / open boundary conditions

Every flag is validated against a hand-derived reference or the per-layer oracle solver; see
[Validation](#validation) below.

---

## Repository layout

```
GridapLFEM.jl/
├── src/                         # the solver package
│   ├── GridapLFEM.jl             module entry: deps, includes, exports
│   ├── vertical.jl            Stage 1: σ-mesh + vertical tensor set (M, Φ, 𝓜, 𝓖, A, K, P, …)
│   ├── tensors.jl             constant-tensor constructors + pointwise Operation helpers
│   ├── horizontal.jl          Stage 2: 2D mesh + stacked FE spaces (serial + distributed)
│   ├── problem.jl             LFEMProblem struct, residual, hand Jacobians
│   ├── nlpressure.jl          full nonlinear pressure (native / exact-IBP / frozen projections)
│   ├── reconstruct.jl         w(σ) / total-pressure σ-level VTK field reconstruction
│   ├── timeloop.jl            sequential ODE solver factory + time loop
│   ├── utilities.jl           dispersion analysis, sponge/wavemaker, setup_and_run
│   ├── timeloop_dist.jl       distributed mesh + GMRES+Jacobi+Newton solver + time loop
│   └── utilities_dist.jl      setup_and_run_distributed
├── test/                        # 9 test files — see Validation
├── examples/                    # sequential examples (plane_wave.jl, ring_wave.jl)
│   └── distributed/               4 cluster-ready MPI examples + README (mpiexecjl/SLURM)
├── building_files/              # math derivation, design/implementation plans, superseded prototype
│   ├── main.tex / main.log        LaTeX derivation (authoritative math reference, §8 = this solver)
│   ├── LFEM_Gridap.md             clean synthesis of main.tex §1–§8
│   ├── wavemaker_sponge_BC.tex    wavemaker/sponge boundary-condition derivation
│   ├── algebraic_residual_math.md   operator simplifications used to write the residual
│   ├── algebraic_residual_plan.md   P1–P6 implementation plan for the residual
│   ├── algebraic_solver_plan.md     package port plan (module/test/example structure)
│   ├── algebraic_distributed_plan.md          distributed-memory port plan + results
│   ├── algebraic_pressure_completion_plan.md  full nonlinear-pressure implementation plan + results
│   ├── algebraic_lfem2D.jl        original single-file prototype (superseded by src/)
│   ├── test_algebraic_lfem2D.jl   prototype's validation script (superseded by test/)
│   └── test_2HDmodel.ipynb        exploratory notebook predating the package
├── output/                      # VTK run output (git-ignored in practice; created on demand)
└── CLAUDE.md                    # up-to-date status/notes for this folder
```

---

## Environment / activation

`GridapLFEM.jl/` is **not a registered Julia package** (its `Project.toml` has no
`name`/`uuid`/`version`) — loading it is always done by file-path `include(...)` +
`using .GridapLFEM` (note the leading dot), never `using GridapLFEM` after `Pkg.add`. Two
environments can supply the dependencies for that `include`:

- **The parent repository's environment** (`../Project.toml`, i.e. `TS_2HDmodel/`) — this is what
  every result in [Validation](#validation) was actually run against, and what the [Quick
  start](#quick-start) / cluster examples below assume (`--project=.` from the repo root).
- **This folder's own environment** (`GridapLFEM.jl/Project.toml` + `Manifest.toml`) — a
  self-contained, standalone environment for running `GridapLFEM.jl` on its own
  (`--project=GridapLFEM.jl` from one level up, or `--project=.` from inside this folder), pinned
  to the **exact same** package versions as the parent (Gridap 0.19.11, GridapDistributed 0.4.13,
  GridapSolvers 0.6.2, PartitionedArrays 0.3.5, MPI 0.20.26) so results are identical either way.
  Useful for extracting/publishing this solver independently of the rest of the repository.

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

# Sequential run: plane wave in a flume, linear regime
diags, vert, prob = setup_and_run(
    M           = 2,                       # vertical layers (LFE-2)
    domain      = ((0.0, 60.0), (0.0, 10.0)),
    partition   = (120, 20),               # horizontal mesh
    d_val       = 3.5, T_wave = 1.6, A_wave = 0.001,
    x_wm        = 12.0,                    # wavemaker position (line source; y_wm=nothing)
    sponge_wL   = 12.0, sponge_wR = 12.0, mu_max = 5.0,
    T_final     = 12.8, dt = 0.02,
    linearised  = false, advection = true, # fully nonlinear
    gauges      = [(30.0, 5.0)],
    save_every  = 20, output_dir = "output/my_run",
)
```

`diags` is a `Vector` of `(t, eta_max, gauge_vals)` NamedTuples. VTK snapshots (`eta, u1x, u1y,
…`) and a ParaView `.pvd` index are written to `output_dir` when `save_every > 0`.

### Distributed (MPI)

```julia
# Same physics, cpu_grid=(px,py) ranks; call from an mpiexecjl -n (px*py) context
diags, vert, prob = setup_and_run_distributed(
    cpu_grid = (2, 2), M = 2, domain = (0.0,60.0,0.0,10.0), partition = (120,20),
    d_val = 3.5, T_wave = 1.6, A_wave = 0.001, x_wm = 12.0,
    sponge_wL = 12.0, sponge_wR = 12.0, mu_max = 5.0,
    T_final = 12.8, dt = 0.02, linearised = false, advection = true,
)
```

```bash
~/.julia/bin/mpiexecjl --project=. -n 4 julia --project=. my_script.jl
```

Use Julia's own MPI launcher (`mpiexecjl`) — the system `mpiexec` fails on this development
machine with a PMIx version mismatch. `nx`/`ny` in `partition` should divide evenly by `px`/`py`.
`diags` distributed carries `(t, eta_max)` only (no point gauges — those need inter-rank
communication, not yet implemented). See `examples/distributed/README.md` for cluster/SLURM usage.

---

## `setup_and_run` / `setup_and_run_distributed` — key options

| Argument | Meaning |
|---|---|
| `M`, `p_vert`, `c_bdy` | vertical layers, polynomial order, σ-node positions (defaults = Yang & Liu Table 1 optimised nodes for M≤4) |
| `domain`, `partition`, `fe_order` | horizontal mesh (`fe_order ≥ 2` required — Q1 zeros the dispersion term) |
| `d_val` / `d_func` | constant depth, or `d_func(x,y)` for variable bathymetry |
| `T_wave`, `A_wave`, `x_wm`, `y_wm` | wavemaker: `y_wm=nothing` → line source (plane wave), else point source (ring wave) |
| `sponge_wL/wR/wB/wT`, `mu_max` | quadratic absorbing sponge layer widths and strength |
| `linearised`, `advection` | drop H-weighting / include nonlinear advection |
| `lin_pressure`, `P_full` | linear slope-pressure package; full leading-pressure form |
| `nl_pressure68`, `nl_pressure_full` | nonlinear pressure: native set / full set (see [Physics](#physics-implemented)) |
| `eta0_func` | initial free-surface release `η₀(x,y)` — **requires `x_wall_bc=true`** (closed basin) |
| `y_wall_bc`, `x_wall_bc` | solid-wall BCs; `x_wall_bc=true` mandatory for any IC-release problem |
| `write_w`, `write_pressure`, `rho` | reconstruct vertical velocity / total pressure at every σ-node into VTK |
| `save_every`, `output_dir`, `gauges` | VTK snapshot cadence, output directory, point-gauge stations (sequential only) |

Distributed adds `cpu_grid=(px,py)` and Krylov tolerances (`nl_iter`, `nl_tol`, `ls_rtol`,
`ls_maxiter`, `nlp_cg_rtol`, `nlp_cg_maxiter`); drops `gauges` and `solver_type` (always `:theta`
with GMRES+Jacobi+Newton).

---

## Validation

All gates below are implemented as standalone Julia scripts in `test/`; run with
`julia --project=. GridapLFEM.jl/test/<name>.jl` (distributed ones via `mpiexecjl`, see file headers).

| Test | What it checks | Result |
|---|---|---|
| `test_vertical.jl` | vertical tensor identities, dispersion bridge vs Yang & Liu Table 1 | 15/15 PASS |
| `test_primitives.jl` | tensor-constructor index order, `∂x/∂y` orientation, contraction semantics | 9/9 PASS |
| `test_equivalence.jl` | virtual-work match vs the per-layer oracle solver, 3 flag configs | 10/10 PASS, rel ≤ 7e-15 |
| `test_basic.jl` | smoke run, linearised + fully nonlinear | 6/6 PASS |
| `test_dispersion.jl` | FEM phase speed vs linear theory at kd=3 | PASS, err 0.90% |
| `test_conservation.jl` | mass conservation, closed basin, nonlinear advection | PASS, drift 7.8e-16 |
| `test_sloshing.jl` | standing-wave period vs LFE-M dispersion theory | PASS, err 1.44% |
| `test_nlpressure.jl` | exact-IBP identity (machine precision), structural scaling, dynamics with all pressure flags on | 9/9 PASS, rel 4e-15 |
| `test_basic_distributed.jl` | 4-rank MPI, linear + fully nonlinear vs sequential | 6/6 PASS, rel ≤ 5e-9 |
| `test_nlpressure_distributed.jl` | 4-rank MPI, full nonlinear pressure vs sequential | 3/3 PASS, rel 4.6e-9 |

The oracle for `test_equivalence.jl` is the independently validated per-layer solver in
`../LFE-M_2D_solver/`. See the `building_files/algebraic_*_plan.md` files for full derivations, implementation
notes, and detailed results of each validation pass.

---

## Design notes for anyone extending this solver

- **Never apply `∇` to an `Operation`-composed expression that contains a test-function basis** —
  Gridap's block-array `copyto!` isn't implemented for that combination. Expand such derivatives
  by hand using linearity of the vertical contraction in the test,
  `∂ₐ(W⋅𝓣) = (∂ₐW)⋅𝓣`, and `Σₐ Ψ·∂ₐUₐ = Ψ·DU` (see `nlpressure.jl` for worked examples).
- **AD Jacobians are not viable** on this residual under Gridap 0.19.11 — the multifield
  autodiff split cannot dualize through `∂t(u)` (a missing `TransientMultiFieldCellField`
  constructor, not a scale/compile-cost issue). Hand-coded Jacobians (`jacobian_u`,
  `jacobian_u_t`) are the permanent design, not a workaround; they are exact for everything
  except the flag-gated `O(A³)` nonlinear-pressure terms, which are quasi-Newton (as in the oracle).
- **Distributed linear solves must not use `lu()`** — it has no method for a partitioned
  `PSparseMatrix`. Use `GMRESSolver`/`CGSolver` with `JacobiLinearSolver()` from GridapSolvers,
  and always allocate RHS/solution vectors **from the matrix**
  (`allocate_in_range`/`allocate_in_domain`) rather than from an independently
  `assemble_vector`-built vector — the two are only isomorphic, not identical, PRanges, and
  `solve!`'s internal `mul!` asserts exact equality.
- **Distributed initial conditions** must use `interpolate_everywhere` — `FEFunction(U, zeros(...))`
  builds a local array and fails on a partitioned space.
- Keep `ConsecutiveMultiFieldStyle` (the default) — `BlockMultiFieldStyle` breaks the Jacobi
  preconditioner's `diag`.
