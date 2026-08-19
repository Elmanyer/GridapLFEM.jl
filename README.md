# GridapBALFEM.jl — the algebraic BALFE-M wave solver

A loop-free, serial-and-distributed Gridap implementation of the **BALFE-M**
(*Basis-Agnostic Layer-integrated Finite Element*, `M` vertical elements) family of depth-integrated,
non-hydrostatic wave models — this project's generalisation of the **LFE-M** model of Yang & Liu
(2024, *J. Fluid Mech.* 999, A32) from a fixed piecewise-linear vertical basis to an arbitrary one.
A concrete member is named **P*p*LFE-*M*** after the basis it uses, so P1LFE-2 is the two-layer
piecewise-linear model (the direct counterpart of Yang & Liu's LFE-2); *LFE-M* is reserved for their
published models. It uses a **stacked value-type layout** that turns every per-layer sum in the
governing equations into a native Gridap tensor contraction instead of a Julia `for`-loop.

Repository: `GridapBALFEM.jl`. Module: `src/GridapBALFEM.jl` → `module GridapBALFEM`. Fully self-contained.

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
in `building_files/BALFEM_Gridap.md`, with the term-by-term operator simplifications in
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
- **Wave generation — two mechanisms** selected by `wave_gen`: `:inner_res` (interior Gaussian
  source: line ⇒ plane wave, point ⇒ ring wave) and `:bc_gen` (boundary Dirichlet generation). For
  `:bc_gen` the boundary source is dispatched on the **type** of `wave_bc` — a *parametrised* regular
  plane wave from `A_wave`/`T_wave`/`wave_dir`, a supplied `WaveInput`, or a **WaveSpec `AiryState`**
  stochastic sea — all feeding the same Dirichlet machinery (they differ only in how the `WaveInput`
  component table is populated). `:auto` (default) infers it (`wave_bc===nothing` ⇒ `:inner_res`, else
  `:bc_gen`)
- **Dirichlet boundary wave generation** (`:bc_gen`) — waves enter through time-varying Dirichlet
  data (η, 𝖴x, and 𝖴y for directional seas) on a domain side, no interior source: regular waves,
  hand-built multichromatic seas, or **WaveSpec.jl stochastic sea states** (JONSWAP/TMA/…, angular
  spreading) via the `AiryState → WaveInput` converter. Discrete BALFE-M eigenmode vertical
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
defining BALFE-M feature) is present in every configuration through `R_P`. The low-level
`build_problem_raw` still exposes the individual internal booleans for the rare split the high-level
interface deliberately ties (used by the oracle-equivalence test).

Every configuration is validated against a hand-derived reference or an independent per-layer assembly
of the same weak form; see [Validation](#validation).

---

## Repository layout

```
GridapBALFEM.jl/
├── src/                          # the solver package (module GridapBALFEM)
│   ├── GridapBALFEM.jl             module entry: deps, includes, exports, g/rho constants
│   ├── vertical.jl              Stage 1: σ-mesh + vertical tensor set (M, Φ, 𝓜, 𝓖, A, K, P, 𝓝, …)
│   ├── tensors.jl               constant-tensor constructors + pointwise Operation helpers
│   ├── horizontal.jl            Stage 2: 2D mesh + stacked FE spaces (serial + distributed; periodic-y)
│   ├── problem.jl               BALFEMProblem, resolve_physics, residual, hand Jacobians
│   ├── nlpressure.jl            full nonlinear pressure (native / exact-IBP / frozen projections)
│   ├── reconstruct.jl           w(σ) / total-pressure σ-level VTK field reconstruction
│   ├── monitor.jl               runtime SolverMonitor + independent governing-equation residual checker
│   ├── timeloop.jl              sequential ODE solver factory (SDIRK / θ) + time loop
│   ├── utilities.jl             dispersion analysis, sponge/wavemaker, setup_and_run
│   ├── waveinput.jl             Dirichlet boundary wave generation + WaveSpec.jl coupling
│   ├── timeloop_dist.jl         distributed mesh + GMRES+Jacobi+Newton solver + time loop
│   └── utilities_dist.jl        setup_and_run_distributed
├── test/                         # 27 test files + runtests.jl + test/cluster/ — see Validation
│   ├── runtests.jl              batch runner; verdict from GATE OUTPUT, never the exit code
│   └── local/                   5 machinery gates on quasi-1D flumes + runner (minutes, 6 cores)
├── examples/
│   ├── plane_wave.jl · ring_wave.jl · periodic_plane_wave.jl     sequential interior-source
│   ├── bc_plane_wave.jl · bc_irregular_sea.jl · bc_directional_sea.jl   sequential BC generation
│   ├── inspect_run.jl            stdlib-only: diagnostics.csv → a health verdict for any run
│   ├── validation/               physical benchmarks (Stokes, bar, soliton, ring, sideband) + README
│   ├── distributed/              8 env-configurable cluster MPI scripts + README
│   ├── distributed_small/        5 parametric small-domain (50×20) scripts (line/point/bc-plane/dir/irreg)
│   ├── local_1d/                 parametric quasi-1D flume (local sequential + cluster MPI)
│   └── local_2d/                 parametric small 2-D case (25×10 m, 6 ranks)
├── run/                          # SLURM launchers — all sysimage-based (see Running on a cluster)
│   ├── balfem_env.sh              shared helper: modules + sysimage + balfem_run <nranks> <script.jl>
│   ├── run_*.sh                 9 production cases (snellius; run_blue.sh = DelftBlue)
│   ├── dist_small/              20 small-domain (50×20) jobs + 7 quasi-1D flume jobs
│   └── local/                   balfem_local.sh + 15 workstation launchers (≤6 cores) + README
├── postprocessing/GridapBALFEMPost # self-contained VTK/CSV analysis + plotting library (own env)
├── compile/                      # cluster sysimage build (compile.jl, warmup.jl, module loaders) + README
├── Project.toml / Manifest.toml  package manifest (name/uuid/version) + its environment
└── CLAUDE.md                     # up-to-date status/notes for this folder
```

Not shown (present locally, excluded by `.gitignore`): `WaveSpec.jl/` — the vendored stochastic
sea-state dependency, `Pkg.develop`ed from its GitHub repository version; `building_files/` — the
LaTeX derivation project and the design/plan records; and `output/` — VTK run output, created on
demand.

---

## Environment / activation

`GridapBALFEM.jl/` **is a Julia package** — `name = "GridapBALFEM"`,
`uuid = 43e94d05-4d7d-4679-96a4-d46e2615da34`, `version = 0.1.0`. Activate this directory as the
project and load it by name:

```julia
# julia --project=/path/to/GridapBALFEM.jl
using GridapBALFEM
```

Never `include("src/GridapBALFEM.jl")`. Being a package is not cosmetic: it is what allows
PackageCompiler to bake the solver *and its Gridap specialisations* into the cluster system image
(`compile/`), and what gives sequential runs a cached precompile (~3.5 s to load). An `include`d
module is rebuilt in every process — which is what made the 32–64 rank cluster runs recompile the
whole FEM stack per rank and get OOM-killed.

This directory is **both the package and the working environment**: the tests, examples and compile
tooling are run directly against it (`julia --project=. test/test_basic.jl`), which is why `Test`,
`BlockArrays`, `MPIPreferences` and `Preferences` sit in `[deps]` rather than `[extras]`.

> **⚠ Gridap is a FORK (since 2026-08-15).** `Manifest.toml` pins
> `https://github.com/Elmanyer/Gridap.jl.git`, branch `fix-transient-multifield-ad`, based on the
> **`v0.20.8` tag** (not master) so the only delta from the validated stack is one commit,
> `fa860899c`. It adds a single outer constructor to `TransientMultiFieldCellField` normalising a
> `Tuple` to the `Vector` the struct declares. Without it, `time_derivative` on a multifield builds
> that argument with `map`, which yields a Tuple on the AD path and a Vector on the hand path, so
> **automatic differentiation of transient multifield residuals raised a `MethodError`** — long
> recorded here as an insurmountable Gridap limitation. It is not, and it is not a `ForwardDiff.Dual`
> problem: no `Dual` appears in the error. **Consequence: `use_ad=true` works**, and AD is an oracle
> for the **Jacobians** — it is what `test/test_jacobians_ad.jl` compares against.
> **⚠ AD is NOT an oracle for the RESIDUAL.** It differentiates the *same assembled residual*, so
> its agreeing (or converging where the hand path fails) proves residual↔Jacobian **consistency**
> and says nothing about correctness. That exact inference was recorded here as fact and retracted
> on 2026-08-15: the residual was at the time double-counting the `𝓐/𝓚` package, and AD agreed with
> the hand Jacobian throughout. Only the analytic MMS verifies the residual. Analysis:
> `building_files/AD_ISSUE.md` (untracked — `building_files/` is gitignored). The
> version-controlled guard that the fork keeps working is **gate A0 of
> `test/test_jacobians_ad.jl`** — an instant `hasmethod` check for the fork's Tuple-argument
> `TransientMultiFieldCellField` constructor, which stock Gridap does not provide.
> Rebuild the cluster sysimage after changing this — `run/balfem_env.sh` hashes `src/*.jl` only and
> will NOT notice a changed dependency.
>
> **Dependency versions.** This environment **and the cluster** run **Gridap 0.20.x /
> GridapDistributed 0.4.17 / GridapSolvers 0.7.1 / PartitionedArrays 0.3.5 / MPI 0.20.26**.
> `[compat]` also admits `Gridap 0.19` / `GridapSolvers 0.6`, because the two stacks were *measured*
> equivalent (2026-08-05): `test_basic.jl` gives identical `max η = 0.00410 m` and `408` Newton
> iterations under both (measured at the then-default `nl_tol=1e-6`; today's default gives 240 on
> both stacks alike, so the equivalence conclusion is unaffected), and the package precompiles on
> both. The parent repository environment
> (`../Project.toml`) is a **separate** environment for the legacy 1D/2D solvers and remains on
> Gridap 0.19.11 / GridapSolvers 0.6.2; it does not need to match. A system image is only valid for
> the versions it was built against — build it in the environment you run in.

---

## Quick start

```julia
using GridapBALFEM

# Sequential run: nonlinear plane wave in a flat-bed flume
diags, vert, prob = setup_and_run(
    M            = 2,                          # vertical layers (P1LFE-2)
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
`diags` carries `(t, eta_max)` only (point gauges need inter-rank communication).

### Running on a cluster (SLURM + system image)

On the cluster, runs are launched from `run/` and **always against a prebuilt system image**
(`GridapBALFEM_sysimage.so`). Without it every rank JIT-compiles the full Gridap FEM stack: ~30–45 min
of wall time per run and a ~4–8 GB/rank memory spike that OOM'd whole nodes at 32–128 ranks. With
it the ranks mmap one shared, already-compiled image — no compile, no spike — and the jobs fit the
cheap `rome`/L1 budget.

> **Memory: the launchers request `--mem-per-cpu=4G`, and that is not optional.** Running at the
> `rome` node default (2 GB/core) was tried in 2026-08 and the job was OOM-killed with the same
> error as the earlier crashes. The compile spike is therefore not the only consumer, and the
> attribution experiment (per-rank JIT vs GMRES cache vs a per-step leak vs a baseline footprint
> above 2 GB/core) is open — see `building_files/EXECUTED_PLANS.md` §2.2. Each launcher
> sizes its request to fit a `rome` node (256 GB / 128 cores).

```bash
# 1. build the image once (~45-60 min; rebuild after changing src/*.jl or upgrading packages)
cd ~/GridapBALFEM.jl/compile && sbatch compile_snellius.sh

# 2. submit any case — all launchers already use the image
cd ~/GridapBALFEM.jl
sbatch run/dist_small/run_nl_periodic_plane_flat_small.sh   # small-domain (50x20) suite, 20 cases
sbatch run/run_irregularsea.sh                              # production case, 9 total
```

Every launcher is a thin wrapper: its `#SBATCH` header, the case's `BALFEM_*` env overrides, and one
call to the shared helper `run/balfem_env.sh`, which loads the cluster modules the image was built
with, verifies the image, and launches:

```bash
source $HOME/GridapBALFEM.jl/run/balfem_env.sh
export BALFEM_PX=8; export BALFEM_PY=4          # 8*4 = 32 ranks
export BALFEM_REGIME=linear
balfem_run 32 examples/distributed_small/run_periodic_plane_small.jl
```

Helper knobs: `BALFEM_PROJ`, `BALFEM_CLUSTER` (`snellius`/`blue`), `BALFEM_SYSIMAGE`, and
`BALFEM_NO_SYSIMAGE=1` to drop `-J` and fall back to the JIT path (useful while the image is stale or
rebuilding). A missing image aborts the job immediately rather than silently taking the slow path.

Because the image bakes a *compiled copy* of `src/*.jl`, editing the solver without rebuilding would
otherwise make the job run the old code silently. The helper therefore checks freshness before
launching and **warns** if the image no longer matches `src/`: exactly, by comparing a hash of
`src/*.jl` against the stamp written at build time (so a `touch` or re-clone does not cry wolf), or
by mtime when an image predates that stamp. Set `BALFEM_STRICT_SYSIMAGE=1` to abort instead of warning
— worth it for long production jobs. Full build walkthrough and troubleshooting:
`compile/README.md`. Per-case physics/geometry env vars: `examples/distributed/README.md`,
`examples/distributed_small/`.

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
| **`wave_gen`** | wave-generation mechanism: `:inner_res` (interior Gaussian source) \| `:bc_gen` (boundary Dirichlet generation — the source, a regular wave / `WaveInput` / `AiryState`, is dispatched on the type of `wave_bc`); `:auto` (default) infers it from `wave_bc` |
| `T_wave`, `A_wave`, `x_wm`, `y_wm` | interior source (`:inner_res`): `y_wm=nothing` → line source (plane wave), else point source (ring wave) |
| `wave_dir` | propagation angle vs +x for a `:bc_gen` boundary wave (0 = normal incidence) |
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
| **`diag_every`, `diag_csv`** | field-diagnostics sampling (`0` → follow `print_every`, `−1` → off) and the `diagnostics.csv` step log: **where** max\|η\| sits, its interior/damped split, \|u\|/\|η\|, mass & energy, GMRES saturation, per-rank RSS. Overhead measured within noise |
| **`eta_ref`, `div_factor`** | the *relative* divergence guard — abort at `div_factor·eta_ref` (default 20×) instead of the blind absolute `1e4`. `eta_ref` is inferred from the forcing when omitted (`A_wave` / the sea state's `Hs` / peak `η₀`) |

Distributed adds `cpu_grid=(px,py)`, the Krylov controls (`nl_iter`, `nl_tol`, `ls_rtol`,
`ls_maxiter`, `krylov_m`, `nlp_cg_rtol`, `nlp_cg_maxiter`) and **`precond`**
(`:jacobi` | `:schwarz` | `:gs`), plus the same `solver_type`/`tableau`; the linear solve is
GMRES+Jacobi+Newton. It drops `gauges` (global reductions instead).

> **Solver tolerances (retuned 2026-08-11/12, on measurement).** `ls_rtol = 1e-5`, `nl_tol = 1e-5`.
> The two are **not** the same kind of knob:
> * `ls_rtol` does not affect the answer anywhere in 1e-9…1e-5, while each order costs GMRES
>   iterations — 1e-9 → 1e-6 cut them **40 %**, 1e-6 → 1e-5 a further **19 %**, `max|η|` unmoved and
>   Newton unchanged in both.
> * `nl_tol` **does** affect the answer, as a step function: Newton runs an integer number of
>   iterations, so at 1e-6 it takes 3/step and at 1e-5 (and 1e-4 — bit-identical) it takes 2, moving
>   `max|η|` by 1.9e-5–3.8e-5 relative. It is adopted on an **error budget** — the dropped iteration
>   refines a `~3e-6` residual against a measured `O(Δt²)` time-discretisation error of
>   `‖R‖∞ ≈ 1.8e-3` — not because it changes nothing.
>
> Consequence for the reference below: `test_basic.jl` reports **240** Newton iterations at the
> default tolerance (it was 408 at `nl_tol=1e-6`); `max η` and gauge amplitude are unchanged.
> Full analysis: `building_files/LOCAL_TESTS_RESULTS.md` §5.4. Tests that need a sharper answer pin
> `nl_tol = 1e-8` explicitly.

---

## Validation

All gates below are standalone Julia scripts in `test/`; run with
`julia --project=GridapBALFEM.jl GridapBALFEM.jl/test/<name>.jl` from the parent repo, or
`julia --project=. test/<name>.jl` from inside `GridapBALFEM.jl/` (distributed ones via `mpiexecjl`,
see file headers). The project must be the **package** environment — since the migration the tests
do `using GridapBALFEM`, which the parent repository's environment cannot resolve unless you also
`Pkg.develop(path="GridapBALFEM.jl")` there.
The full suite is **27 test files + `runtests.jl` + `test/cluster/` + `test/local/`**. As of
2026-08-17, all re-measured after two solver fixes: **sequential 21/21 PASS**
(`julia --project=. test/runtests.jl`), **distributed 13/13 gates PASS** on 4 ranks,
**Jacobian-vs-AD 8/8 models PASS**. Representative highlights.

**Read the first block separately from the rest.** Those five files are *verification*: the forcing
is derived from the governing equations and never touches `problem.jl` (a grep gate enforces it), so
a self-consistently wrong residual cannot hide in them. Everything below that block is
*self-consistency or physics agreement* — strong evidence, but it compares the solver against
expectations computed with the solver's own residual, where an error can cancel.

| Verification (analytic MMS + Jacobian gate) | What it checks | Result |
|---|---|---|
| `test_mms_forcing.jl` | forcing-level gates for the linear models, no FE solve; includes the grep gate that keeps `src/mms.jl` independent of the residual | 5/5 PASS |
| `test_mms_forcing_nonlinear.jl` | forcing-level gates for the nonlinear models: Model 4 at constant `h` ≡ Model 3 exactly, `ε²` amplitude scaling, `H>0` guard | 10/10 PASS |
| `test_mms_convergence.jl` | **order of accuracy**, Model 1 (linear, flat bed), on the `Q3/Q2` pairing | see §FE pairing |
| `test_mms_convergence_nonlinear.jl` | order of accuracy, Models 3 and 4 (nonlinear) | **4/4 PASS** (run end-to-end 2026-08-17) — M3 `p_η=2.996`/`p_u=3.995`, M4 `2.996`/`3.997`, both at theoretical order, at the DEFAULT `nl_iter` |
| `test_jacobians_ad.jl` | hand `∂R/∂u`, `∂R/∂u̇` vs **AD of the same residual**, matrix by matrix, all 8 models; gates the linear branch on equality and the nonlinear one on how the gap **scales with amplitude** (an O(1) gap is a defect, a vanishing one is the deliberate quasi-Newton choice) | `∂R/∂u̇` exact for all 8 |
| `test_linear_newton_gate.jl` | a LINEAR problem must converge in **one Newton iteration per implicit stage**, on a **sloping** bed — the configuration no flat-bed test can reach | 10/10 PASS |

| Test | What it checks | Result |
|---|---|---|
| `test_vertical.jl` | vertical tensor identities, dispersion bridge vs Yang & Liu Table 1 | 15/15 PASS |
| `test_primitives.jl` | tensor index order, `∂x/∂y` orientation, contraction semantics | 9/9 PASS |
| `test_equivalence.jl` | virtual-work match vs the older per-layer solver | **RETIRED — not a correctness gate.** The external solver was not maintained in step with the weak form (it lacks the leading-pressure term `R_P`), so a disagreement measures its age, not a defect here |
| `test_dispersion.jl` | FEM phase speed vs linear theory at kd=3 | PASS, err 0.90% |
| `test_dispersion_curve.jl` | `Cm/Ce(kd)` sweep vs Airy, P1LFE-2/3/4 applicable-kd | 9/9 PASS (10.8/39.2/127.9) |
| `test_dispersion_nonlinear.jl` | full-NL solver ⇒ Airy at vanishing amplitude (kd 1/3/5) | 3/3 PASS |
| `test_shallow_water.jl` | `kd→0 ⇒ √(gd)` limit | 6/6 PASS |
| `test_sloshing.jl` | standing-wave period vs BALFE-M theory | PASS, err 1.44% |
| `test_energy.jl` | non-dissipativity / amplitude preservation | 3/3 PASS |
| `test_conservation.jl` | mass conservation, closed basin, nonlinear advection | PASS, drift 7.8e-16 |
| `test_nlpressure.jl` | exact-IBP identity, structural scaling, dynamics (all pressure tiers) | 9/9 PASS |
| `test_selfconsistency.jl` | **support, not model validation**: hand Jacobians are exact derivatives of the residual; multi-step integration self-consistent. Its forcing *is* the solver's own residual, so a wrong residual would still pass — see the file header | 3/3 PASS (~3e-9) |
| `test_convergence.jl` | Richardson temporal order (→2) | 2/2 PASS (q≈1.72) |
| `test_vertical_profile.jl` | reconstructed `w(σ)` vs Airy `sinh` shape | 7/7 PASS |
| `test_basic.jl` | smoke run, linear + fully nonlinear | 6/6 PASS — references **bit-identical** through both 2026-08-17 fixes (max η 0.00410, gauge 0.00212, Newton 240) |
| `test_waveinput.jl` | Dirichlet-generation data: identities, closures, AD, WaveSpec converter | 30/30 PASS |
| `test_bc_generation.jl` | Dirichlet-generated regular wave e2e (kd=3) | 11/11 PASS |
| `test_bc_spectrum.jl` | 3-component Dirichlet sea: Goda–Suzuki incident amplitudes | 8/8 PASS |
| `test_basic_distributed.jl` | 4-rank MPI, linear + fully nonlinear vs sequential | 6/6 PASS, rel 2.5e-7 (**both reference constants re-measured 2026-08-17 — the linear one was 9.7 % stale**) |
| `test_nlpressure_distributed.jl` | 4-rank MPI, full nonlinear pressure vs sequential over a tanh bar | 3/3 PASS, rel 1.6e-5. **The only test in the repo that pins a VALUE on `:nonlinear` + `∇h≠0`** — it caught the gravity defect at rel 5.9e-1 the first time it was run; its reference moved 58 % as a result |
| `test_bc_generation_distributed.jl` | 4-rank MPI Dirichlet generation vs sequential | 4/4 PASS, rel 6.6e-7 (reference re-measured 2026-08-17; was 11.7 % stale) |
| `cluster/cluster_conservation.jl` · `cluster/cluster_selfconsistency.jl` | long-run mass conservation / nonlinear solver self-consistency at scale (**not** an MMS validation — renamed for the same reason `test_mms.jl` became `test_selfconsistency.jl`) | dist |

### The local suite — `test/local/`

A second, faster tier that gates the solver's **internal machinery** rather than its physics, on
quasi-1D flumes small enough to run in minutes on a workstation. It exists because the components
that failed on the cluster — the sponge, the relaxation zone, the open-boundary behaviour — failed
*silently*, and a multi-hour job is the wrong instrument for finding out why.

| Test | What it gates | Result |
|---|---|---|
| `test_reststate_1d.jl` | an undisturbed surface stays at rest to machine precision, flat bed **and** sloping bed (the IBP energy-form gravity term must cancel exactly) | **8/8**, `max\|η\| = 0` exactly |
| `test_sponge_1d.jl` | the envelope inside the sponge follows the quadratic-μ damping law `ln a ∝ −μ_max(x−x_R)³/(3w²c_g)` across `μ_max = 5/20/40`; decay monotone and saturating; reflection < 5 %; **η damped, not just u** | `R² = 0.9995/0.9866/0.9835`, reflection 1.0–1.4 % |
| `test_relaxation_1d.jl` | inside the inflow zone the state *is* the prescribed incident wave; and the zone absorbs an outgoing wave — measured as the energy left in the basin after a hump release, against a `relax_bc=false` control | **9/9**; amplitude err ≤ 0.1 %, phase 0.9°, absorption **145×** the control |
| `test_boundary_modes_1d.jl` | the open-boundary mode's five-clause signature (growth rate, boundary localisation, location of max\|η\|, kinematic ratio, invariants), **with a negative control that must fail** so a future regression cannot silence the detector | **16/16**; control detected in **6 s** of simulated time |

```bash
test/local/run_local_tests.sh              # all four, 4 concurrent processes
JOBS=6 test/local/run_local_tests.sh sponge relaxation
```

They run **sequentially** (one process per test, several cases inside each): every measurement is
gauge-based and only the sequential driver evaluates point gauges. The cores go to running tests
concurrently, not to MPI. They are not part of the 21-file suite gate yet.

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
- **Jacobian coverage, precisely.** `∂R/∂u̇` (`jacobian_u_t`) is **EXACT in both regimes** — mass,
  `H`-weighted acceleration, leading pressure `R_P`, and the `𝓐/𝓚` slope package; the `𝓝` blocks
  carry no `u̇`-dependence at all. Verified `0.000e+00` against AD in **all 8 models**.
  `∂R/∂u` (`jacobian_u`) is **quasi-Newton in the nonlinear branch by choice**: advection is
  differentiated in full, but the pressure packages contribute no η-derivative and `𝓝` is absent.
  Its gap is `2.9e-2` (`:none`) → `8.3e-1` (`:full`) and **vanishes at order 1.11–1.16** in the
  state amplitude — which is what makes it benign, because Newton drives the *residual* to zero.
  **An omission is benign only if it is higher-order in amplitude.** The `𝓐/𝓚` block previously
  missing from `∂R/∂u̇` had prefactor `H·∇h`, which does not scale with the solution — an `O(1)`
  error that made Newton converge to the wrong fixed point and stalled the nonlinear variable-bed
  MMS outright. Measure the scaling with `test/test_jacobians_ad.jl`; do not assume it.
- **AD Jacobians are viable** since the Gridap fork (see the Environment note): `use_ad=true` builds
  them, and they are the reference `test_jacobians_ad.jl` checks the hand ones against.
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

- **On the word "validated" in this repository.** A test validates a term only if it can *resolve*
  that term's contribution — if switching the term off moves the measured quantity by less than the
  measurement's resolution, a passing test shows the code executes, not that the term is correct.
  The clearest case is `nl_pressure=:full`, whose `𝓝{1,2,4,5}` block changes `max|η|` by only
  0.013–0.094 % at `A = 1e-3`: the local observation runs cannot validate it, although
  `test_nlpressure.jl` G1 *does* validate its `∇h` half to machine precision against analytic
  derivatives. Prefer precise phrasing ("unvalidated by the local set", "validated structurally but
  not by value") over a bare "unvalidated". Full discussion, including the cheap experiment that
  would close the gap (the same case at `A = 1e-2`, where the effect grows as `A²` to ~1.3 %):
  `building_files/LOCAL_TESTS_RESULTS.md` §7b.

- **`test_equivalence.jl` is retired and must not be read as a correctness gate.** It compared the
  package residual against the older per-layer solver in `../LFE-M_2D_solver/`. That external code
  was not maintained in step with the model as the weak form was completed — most consequentially
  it lacks the leading-pressure term `R_P` — so the two no longer discretise the same equations and
  its 1/10 result measures the reference's age, not a defect here. The construction (layout-
  independent virtual work) remains the right instrument should a *current* second implementation
  ever exist.
- **Residual verification — ALL FOUR `:none` MODELS DONE (2026-08-17); the `𝓝` tiers open.** The
  analytic MMS (`src/mms.jl`, `src/errors.jl`, `src/mms_driver.jl`, `test/test_mms_*.jl`) derives its
  forcing from the governing equations *without touching* `problem.jl`, so — unlike every other test
  here — the error does not cancel and the measured **order of accuracy** certifies the operator.

  | Model | `regime` / bed / `nl_pressure` | `p_η` (opt 3) | `p_u` (opt 4) |
  |---|---|---|---|
  | 1 | `:linear` / flat / `:none` | optimal | optimal |
  | 2 | `:linear` / **variable** / `:none` | 3.000 | 4.000 |
  | 3 | `:nonlinear` / flat / `:none` | 2.996 | 3.995 |
  | 4 | `:nonlinear` / **variable** / `:none` | 2.996 | 3.997 |

  So the verified scope is the **complete `:none` operator over arbitrary bathymetry** —
  `H`-weighting, advection, the full three-component leading pressure, the `O(ε²)` surface-slope
  package and the bed-slope terms. Only `nl_pressure=:native`/`:full` remain unverified, because
  their MMS forcing is not available (three nested ForwardDiff levels; diagnosis **unconfirmed**).
  **Say "the `:none` models are verified", never "the residual is verified".**

  ⚠ Model 4 needed **two solver fixes** to get there, and *neither was findable by any other test in
  the suite*: the `𝓐/𝓚` package was missing from `∂R/∂u̇` (an `O(1)` omission — Newton stalled), and
  the nonlinear gravity branch was missing the `−η∇h` half of its own integration by parts. The
  latter was absent from the residual **and** from its own Jacobian, so every self-consistency check
  cancelled the error identically. Full record: `building_files/MMS_CAMPAIGN.md`,
  `building_files/RESIDUAL_AUDIT.md`.
- **⚠ Equal-order FE spaces cost one order of convergence** (found by the above). `Q_p/Q_p` for
  `(η, 𝖴)` converges at `p`, not `p+1`. Mixed order `Q_p/Q_{p-1}` fixes the **surface** universally
  and the **velocity** at `Q3/Q2` — measured over a 12-study campaign
  (`building_files/MMS_CAMPAIGN.md`); the two fields must be stated separately:
  **`η` hits its optimal `p_e+1` in 12/12 studies** (`2.000/3.000/4.000` in 1-D, `1.980/2.997/3.994`
  in 2-D, static ≡ transient to 4 figures), while **`u` hits `p_u+1` only at `Q3/Q2`** (3.991 / 3.930);
  at `Q2/Q1` it stalls near 2.4 (optimal 3) and at `Q4/Q3` near 4.65 (optimal 5), rates still falling.
  The 1-D `Q4` figure of 3.345 is a **round-off floor, not a rate** (`e_u` 1.79e-11 → 1.14e-11).
  **`Q3/Q2` is the pairing to prefer on current evidence.**
  `η` enters momentum undifferentiated via `∇·v` after integration
  by parts, so it plays the pressure role and equal order is inf-sup deficient — the Stokes analogue,
  fixed by the Taylor–Hood pairing. This is also *what verified the residual*: the operator code is
  identical across those runs, and a wrong coefficient cannot be repaired by changing FE spaces.
  `build_fe_spaces(...; p_eta=…)` exposes the choice and **defaults to equal order (unchanged)**.
  Do not switch production on the rate alone — at `nx=24` `Q3/Q3` was 40× more accurate than `Q3/Q2`
  despite the worse rate; an error-vs-DOF study at production resolution is not yet run.


- The vendored **`WaveSpec.jl`** must track the **GitHub repository version, not a tagged release**:
  the release's `change_seed!` accesses a non-existent `state.spec` field (the struct field is
  `spectrum`), which makes `build_airy_state` error and blocks every Dirichlet **sea-state** run.
  The repo version carries the fix and is what is vendored here — if sea-state runs start failing
  in `change_seed!`, check that `WaveSpec.jl/` has not been reset to a release tag.
- The cluster **system image must be rebuilt after editing `src/*.jl`** — it bakes a compiled copy of
  the solver, so a stale image runs old code. This is now *detected* (the launchers warn, or abort
  under `BALFEM_STRICT_SYSIMAGE=1`) but not prevented: rebuild with `compile/compile_snellius.sh`, or
  launch with `BALFEM_NO_SYSIMAGE=1` to run the current sources via JIT.
