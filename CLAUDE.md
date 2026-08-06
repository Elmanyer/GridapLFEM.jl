# CLAUDE.md — `GridapLFEM.jl/` (the 2D LFE-M algebraic wave solver)

**What this folder is.** The production home of the 2D LFE-M solver: the self-contained
serial+distributed package `src/GridapLFEM.jl` (stacked `[η,𝖴x,𝖴y]` layout, loop-free residual),
its test suite (`test/`), sequential and cluster examples (`examples/`), the standalone
postprocessing library (`postprocessing/GridapLFEMPost`), the vendored sea-state package
(`WaveSpec.jl/`), and the design/paper material in `building_files/` (the LaTeX derivation project
`LFEM_discretisation.zip`, the synthesis notes, the validation report §9 within it). An independent per-layer
implementation lives in `../LFE-M_2D_solver/` and is used by `test_equivalence.jl` as a second,
differently-structured assembly of the same weak form — a cross-check that the two agree to
machine precision.

**How to read this document.** The feature summary is the at-a-glance capability list; §1 maps
every file; §2–§9 are the authoritative explanation of *how* the residual is built and *why* the
design is shaped the way it is (stacked value types, index conventions, pressure-term treatment —
the rationale future modifications must respect). To extend the solver, read the feature summary,
the §1 table, and the section relevant to the change (§3 layout rules, §5 residual terms,
§6 nonlinear pressure, §8 solver conventions).

---

## 0. Orientation — the LFE-M model in one paragraph

LFE-M (Yang & Liu 2024, *JFM* 999 A32) is a depth-integrated, **non-hydrostatic** free-surface wave
model. The water column `σ∈[0,1]` (terrain-following) is discretised with `M` vertical finite
elements of order `p`, so the horizontal velocity is `u_h(x,σ,t)=Σ_j u_j(x,t) φ_j(σ)`. The vertical
velocity `w` and the non-hydrostatic pressure `p_nh` are eliminated **analytically** (no pressure
Poisson solve) into a set of small, precomputable **vertical σ-tensors**. What remains is a system of
`Nσ+2` coupled **2-D PDEs** in the unknowns `(H, u_0,…,u_{Nσ})`, solved on a horizontal FE mesh by
Gridap. `H=d+η` (still-water depth `d`, free surface `η`). The structure is therefore a
**(small dense vertical algebra) ⊗ (large sparse horizontal FE)** — cheap dense vertical work built
once, wrapped by the large sparse horizontal solve.

`Nσ` **convention:** `Nσ = num_free_dofs(U_phi) = M·p+1` = the number of vertical nodes; velocity
modes are 1-based `j=1..Nσ`, one per node.

---

## 1. Files in this folder

| File | What it is |
|------|-----------|
| `Project.toml` / `Manifest.toml` | **The Julia package manifest** — `name = "GridapLFEM"`, `uuid = 43e94d05-4d7d-4679-96a4-d46e2615da34`, `version = 0.1.0` (migrated 2026-08-04; see `building_files/PACKAGE_MIGRATION_PLAN.md`). The solver is loaded with **`using GridapLFEM`**, never `include()`. Being a real package is what lets PackageCompiler bake the solver *and its Gridap specialisations* into the cluster sysimage — under the previous `include()` loading they lived in a throw-away `Main.GridapLFEM`, were discarded from the image, and were recompiled in every rank (the cause of the 2026-08-04 OOM kills). It also gives every sequential run a cached precompile (~3.5 s to load instead of a cold compile). This directory is **both the package and the working environment**: tests, examples and the compile tooling are run directly against it (`julia --project=. test/test_basic.jl`), so `Test`, `BlockArrays`, `MPIPreferences` and `Preferences` are kept in `[deps]` rather than `[extras]`/`[targets]` — under `[extras]` those direct invocations would not resolve. `BlockArrays` is needed by the differently-structured **oracle** `LFEModel2D` that `test_equivalence.jl` loads. `[compat]` admits **both** minors (`Gridap = "0.19, 0.20"`, `GridapSolvers = "0.6, 0.7"`) because they were measured to give identical results (§Solver conventions); this environment and the cluster both run Gridap 0.20.x / GridapSolvers 0.7.1, while the parent `../` — a separate environment for the legacy solvers — stays on 0.19.11 / 0.6.2. A sysimage is only valid for the versions it was built against, so build it in the environment you run in. If you add a new `using X` to any `src/*.jl`, add it to `[deps]` here too. |
| `LFEM_discretisation.zip` (LaTeX project) | The authoritative LaTeX derivation, a multi-file project (compiles with pdflatex). **Structure** (Overleaf layout, adopted 2026-08-04): `main.tex` (preamble/macros) inputs the chapter files — `SigmaEulerModel.tex` (chapter `σ-Euler Model`, two sections: governing equations, σ-transformation), then the wrapper `\chapter{Vertical Multilayer Discretisation}` declared in `main.tex` whose five **sections** live in `VerticalFESemiDiscretisation/` (`VerticalFEapprox`, `wDerivation`, `pDerivation`, `VerticalProjection`, `VerticalSemiDiscreteSystem` — the last incl. the **flat-bed reduction** of the full nonlinear model as subsections), then the standalone root-level chapters `LinearModel.tex` and **`StokesWaveFourierAnalysis.tex`** (the analytical theory chapter, mirroring Yang & Liu §3 for an **arbitrary-order** vertical basis: the Stokes–Fourier hierarchy; the dispersion functional `R(μ)=Φᵀ(M+μ|B|)⁻¹Φ` and four basis-independent properties — basis/space invariance, the variational bound `Cm ≤ Ce`, the `[Nσ/(Nσ+1)]` Padé-type structure, exact shallow-water + `O((kd)²)` consistency; group velocity and shoaling gradient; the **second-order bound-harmonic transfer function** with its hand-assembled forcing, reproducing Stokes to 1% and the published LFE-2 nonlinear range `kd=6.0`; third-order solvability + wave–current outlook; and the **vertical grid optimisation** — minimax formulation, the design rule `Δσ_top ≈ 2.94/kd_max`, and "grading beats order" at fixed DOF), and finally `NumericalImplementation/` — `GlobalResidual`, **`GridapImplementation.tex` (§8** — stacked layout, residual implementation, 𝓛/𝓝 pressure treatment, the `regime`/`nl_pressure`/**`flat_bed`** model-setups subsection, wave generation/sponge/BC incl. Dirichlet boundary generation with the `:model`/`:airy` polarization) and **`ValidationTests.tex` (§9** — the validation report incl. the BC-generation gates, Goda–Suzuki, spectral fidelity). Sign convention: `R_P` is a positive integral **subtracted** in the global sum, uniformly with `R_lin`/`R_nonlin`. All prior standalone `*Improved.tex`/`.md` section drafts are integrated into this zip. The document is a **`report`** (chapters + table of contents); the `§8`/`§9` labels above are the corresponding chapters. **Conventions** (enforced 2026-08-04, verified mechanically): structural labels carry a prefix matching their level — `\label{chap: …}` / `\label{sec: …}` / `\label{subsec: …}`, always `prefix: name` with a space; every chapter/section/subsection cross-reference is written `\S\ref{…}` (no `Section~\ref{}`/`Chapter~\ref{}` prose forms; ranges are `\S\ref{a}--\S\ref{b}`). Header numbering is the **report-class default**; the Roman-chapter/custom scheme is retained commented-out in `main.tex`. Both the `.zip` and an unzipped working copy `LFEM_discretisation/` live in `building_files/` — **the folder is the working copy; edit it and re-zip to sync** (the `.zip` is only an export/import channel for Overleaf; if the two disagree, the folder is authoritative). The `.cls`/toolchain needs `newtx`, so the document does not compile on this dev machine; `StokesWaveFourierAnalysis.tex` requires `multirow`. |
| `LFEM_Gridap.md` | Synthesis of the derivation §1–§8 leading to the single scalar residual, in the LaTeX notation; §9 bridges that notation to the solver code. |
| `algebraic_residual_math.md` | Operator reference: how each §8 residual term is written with native Gridap tensor ops (no MultiField decomposition, no vertical-index loops) — the `L`/`N` pressure stacks, the leading-pressure `R_P`, IBP of second-derivative terms, and the verified Gridap operator table. |
| `DESIGN_RECORDS.md` | Consolidated **historical design records** for the completed features — algebraic residual + Jacobians, package layout, distributed solver, nonlinear-pressure completion, boundary wave generation, periodic-y BC, the `flat_bed` switch, and the production sea-state scripts. Provenance only (*why* the code is shaped as it is); the code, this `CLAUDE.md`, and the LaTeX derivation are authoritative. |
| `src/` (`GridapLFEM.jl` + submodules) | **The self-contained serial + distributed solver package** (`module GridapLFEM`): vertical tensors (incl. `Pcal`), constant-tensor + `Operation` helpers, stacked FE spaces (distributed-safe MultiField dispatch + transient-Dirichlet inflow variants), the loop-free residual + hand Jacobians (the same code runs distributed — `Operation` is forwarded for `DistributedCellField`), the time loops (default fully-implicit `RungeKutta(:SDIRK_2_2)`, `:theta` Crank–Nicolson selectable via `solver_type`; sequential LU+Newton / distributed GMRES+Jacobi+Newton), per-component VTK (`eta,u1x,u1y,…`) plus reconstructed `w_s<σ>`/`p_s<σ>` fields (`reconstruct.jl`), and `waveinput.jl` (Dirichlet boundary wave generation + WaveSpec.jl coupling: component tables, `:model`/`:airy` polarizations, ramp, relaxation zone). Drivers: `setup_and_run` and `setup_and_run_distributed`. |
| `test/` (21 tests + `test/cluster/`) | **Base suite:** `test_vertical` 15/15, `test_primitives` 9/9, `test_equivalence` **⚠ 1/10 — FAILING at HEAD 2026-08-06** (was documented 10/10 ≤7e-15; see the open item), `test_basic` 6/6, `test_dispersion` (kd=3 err 0.90%), `test_basic_distributed` 6/6 (4 ranks; rel ~1e-5 vs the sequential references), `test_nlpressure` 9/9 (+ `_distributed`), `test_sloshing` (1.44%), `test_conservation` (drift 7.8e-16). **Validation batch:** `test_dispersion_curve` 9/9 (closed-form Cm/Ce(kd), kd_app 10.8/39.2/127.9), `test_mms` 3/3 (unsteady nonlinear MMS), `test_convergence` 2/2, `test_vertical_profile` 7/7 (sinh shape), `test_energy` 3/3, `test_dispersion_nonlinear` 3/3 (full-NL⇒Airy, kd=1/3/5 err 0.93/0.36/3.05%), `test_shallow_water` 6/6 (kd→0 ⇒ √(gd), ΦᵀM⁻¹Φ=1). **BC-generation batch:** `test_waveinput` 30/30, `test_bc_generation` 11/11, `test_bc_spectrum` 8/8 (Goda–Suzuki), `test_bc_generation_distributed` 4/4 (rel 3.05e-8). **`test/cluster/`:** `cluster_conservation` (2 ranks, drift 5.8e-10), `cluster_mms` (all 𝓝 at scale) + SLURM template. **`test/local/`** (new 2026-08-06): the LOCAL validation suite — quasi-1D flumes (`Ly=3, ny=3`, y-periodic) that gate the solver's *internal machinery* in minutes on 6 cores instead of hours on the cluster: `test_reststate_1d` (rest state, flat + sloping bed), `test_sponge_1d` (the sponge's damping law vs its μ-profile, + reflection), `test_relaxation_1d` (inflow zone: generation fidelity + absorption, differential against `relax_bc=false`), `test_boundary_modes_1d` (the open-boundary-mode signature, **with a negative control that must fail**), shared helpers in `_local_common.jl`, runner `run_local_tests.sh` (process-level concurrency, not MPI — these need point gauges, which only the sequential driver has). |
| `examples/` (+ `validation/`, `distributed/`) | Sequential: `plane_wave.jl`, `ring_wave.jl`, `periodic_plane_wave.jl`, and BC-generation `bc_plane_wave.jl`, `bc_irregular_sea.jl` (JONSWAP, gauge CSV), `bc_directional_sea.jl`; `examples/distributed/` — 6 env-configurable cluster scripts (plane/ring wave, IC hump, bathymetry, `run_irregular_sea_dist.jl`, `run_directional_sea_dist.jl` — sea-state env vars + `build_airy_state()` in `_dist_common.jl`) + README; `examples/validation/` — physical benchmarks (`stokes_harmonics`, `submerged_bar`, `solitary_wave`, `ring_spreading`, `bichromatic_sideband`) + `dispersion_sweep.jl` + `spectral_fidelity.jl` (JONSWAP component-wise amplitude+dispersion transfer) + README; `examples/distributed_small/` — **5 parametric** small-domain (50×20 m) scripts (`run_periodic_plane_small` [interior line source], `run_ring_small` [point source], `run_bc_plane_small` [`:bc_gen` boundary plane wave], `run_directional_sea_small`, `run_irregular_sea_small` [`:bc_gen` WaveSpec sea]) grouped by wave-generation type; the 20 observation/comparison cases are their **launchers** in `run/dist_small/`, each overriding only the env vars that change (regime / nl_pressure / flat_bed→bar / amplitude / period). See the "Current Implementation Stage" small-domain-suite entry. **`examples/local_1d/run_flume_1d.jl`** — the parametric **quasi-1D flume** (narrow domain, `ny≥3`, y-periodic; the solver is structurally 2-D, so this is how a 1-D horizontal problem is posed — it is *not* a 1-D model, see the script header), sequential-with-gauges locally and MPI `(px,1)` on the cluster, `LFEM_WAVE_GEN ∈ {inner,bc,sea}`. **`examples/local_2d/run_small_2d.jl`** — the parametric small 2-D case (25×10 m, 6 ranks as `3×2`), `LFEM_WAVE_GEN ∈ {line,point,bc,sea}`, each case a scaled-down sibling of a named `run/dist_small/` job. **`examples/inspect_run.jl`** — stdlib-only reader that turns a run's `diagnostics.csv` into a health verdict (works on cluster output too). |
| `postprocessing/` (`GridapLFEMPost`) | Self-contained postprocessing library with its own environment (ReadVTK, Plots+GR, FFTW, Interpolations — pinned separately from the solver). Reads VTK (`solution.pvd`/`sol_t_*.vtu`) + CSV → `WaveSimulation` (auto-`regularize!`s the duplicated Q2 node cloud to a Cartesian grid). Modules: `io, probes, spectral, diagnostics, reconstruct, plotting, seastate`. Gauges/DFT/celerity/harmonics/radial/conservation; heatmap/animation(GIF)/Hovmöller/dispersion/profile plots; `seastate.jl` — Welch PSD, JONSWAP target overlay, spectral moments/Hs, zero-upcrossing heights, Rayleigh exceedance (+ `spectral_validation.jl` example). `reconstruct.jl` rebuilds `w(σ)`/`p_nh(σ)` from the stored velocity modes at any σ (analytic σ-basis, Gauss quad, no Gridap; matches solver `w_s` to 4–8%). No dependency on the solver. |
| `WaveSpec.jl/` (repo-vendored package) | Stochastic sea-state synthesis (CMOE-TUDelft; JONSWAP/TMA/… spectra, sampling strategies, angular spreading, `AiryState`). Tracks the **GitHub repository version, not a tagged release** — the release's `change_seed!` reads a non-existent `state.spec` field and breaks every sea-state run; the repo version has the fix. `Pkg.develop`ed in both environments; `using WaveSpec` is re-exported by `GridapLFEM`. The `WaveInput` converter (`src/waveinput.jl`) snapshots seeded amplitudes/phases into plain arrays and re-solves the wavenumbers with the solver's `g` (WaveSpec uses 9.80665). |
| `compile/` (cluster sysimage build) | Builds a precompiled system image so distributed cluster ranks **load** the solver instead of JIT-compiling it (removes the ~30–45 min/rank compile and the associated OOM). `set_preferences.jl` pins MPI.jl to the **system** OpenMPI via `use_system_binary()`; `compile.jl` bakes the deps + traces `warmup.jl`, with an MPI preflight and **`include_transitive_dependencies=false`** — load-bearing: it stops PackageCompiler force-loading `OpenMPI_jll`, whose baked initializer would otherwise `dlopen` the JLL artifact and crash the image at launch (`undefined symbol: opal_single_threaded`). `compile_snellius.sh` runs the chain; full walkthrough + troubleshooting in `compile/README.md`. |
| `run/local/` | **Local launchers (this workstation, ≤6 cores)** — `lfem_local.sh` (the helper: project resolution, a hard rank cap, `lfem_local_run` sequential / `lfem_local_mpi` MPI, exit-143 handling for the benign `MPI_Finalize` OFI error) plus 7 `run_1d_*.sh` and 8 `run_2d_*.sh` case launchers and a `README.md`. Deliberately shares no code with `run/lfem_env.sh`, which is cluster-only (modules + sysimage). The point is a minutes-long feedback loop instead of the multi-hour cluster one. |
| `run/` (+ `run/dist_small/`) | The SLURM launchers: 9 production cases in `run/` (plane/ring/periodic-plane wave, IC hump, bathymetry, irregular + directional sea, and the DelftBlue `run_blue.sh`) and 20 small-domain observation/comparison cases in `run/dist_small/`. **All of them run against the prebuilt sysimage**, through the shared helper **`run/lfem_env.sh`**: it loads the cluster modules the image was built with, resolves + verifies `GridapLFEM_sysimage.so`, and exposes `lfem_run <nranks> <script.jl>` (the `mpiexecjl --project=… -n … julia --project=… -J<sysimage> …` invocation). A launcher is therefore only its `#SBATCH` header, its `LFEM_*` env overrides, and one `lfem_run` call — nothing sysimage-specific is repeated. Helper knobs: `LFEM_PROJ`, `LFEM_CLUSTER` (`snellius`/`blue`), `LFEM_SYSIMAGE`, `LFEM_NO_SYSIMAGE=1` (escape hatch: drop `-J` and JIT-compile, for when the image is stale or mid-rebuild) and `LFEM_STRICT_SYSIMAGE=1` (below). A missing image **fails the job immediately** rather than falling back to a silent 45-min/rank compile. It also runs `lfem_check_sysimage_freshness` before launching and **warns if the image is stale w.r.t. `src/`** — the image bakes a *compiled copy* of the solver, so editing `src/*.jl` without rebuilding would otherwise run old code silently. Exact check: `compile_*.sh` calls `lfem_write_sysimage_stamp` after a successful build, writing `GridapLFEM_sysimage.so.src.sha256` (hash of all `src/*.jl`); the launcher recomputes and compares, so edits/additions/deletions trip it but a `touch`, re-clone or content-restoring `git checkout` does not. Images predating the stamp fall back to an mtime comparison (coarser, can cry wolf). Warns and continues by default; `LFEM_STRICT_SYSIMAGE=1` makes it a hard abort (use for long production jobs). |
| `boundary_wave_generation.md` | Dirichlet boundary wave generation math note (nodal trace, `:model` discrete-eigenmode polarization derivation, ramp, well-posedness/reflection, relaxation zone, WaveSpec contract, validation map). The design record is in `DESIGN_RECORDS.md`. |
| `LFEM_runs.md` | Plan/record for the small-domain observation + comparison run suite (all three batches; `examples/distributed_small/` + `run/dist_small/`). |
| `SPONGE_WAVEGEN_PLAN.md` | Plan + post-implementation addendum for the surface-damping sponge (`+∫ μ q η`) and the explicit `wave_gen` interface (the two boundary symbols were later collapsed to a single `:bc_gen`). |
| `CFC2027_LFEMultilayer_abstract/` (+ `.zip`) | Conference abstract (CFC 2027, *Advanced Computational Methods for Free-surface Water Waves* symposium) presenting the **arbitrary-order** LFE-M model and its Gridap solver: `CFC2027_abstracts.tex` + `.cls` (the class needs the `newtx` fonts to build). |
| `algebraic_lfem2D.jl` / `test_algebraic_lfem2D.jl` | The single-file prototype of the residual and its validation script; the `src/` package is the maintained form. |
| `test_2HDmodel.ipynb` | An early exploratory notebook; its vertical pre-compute is sound and it is kept only for reference. |

Notation note: `LFEM_Gridap.md` and `main.tex` use `M^V, 𝓜^V, 𝓖^V, A^V, K^V, 𝓐^V, 𝓚^V, Φ, φ_j`;
the **solver code** uses `Mmat, Mcal, Gcal, A, K, Acal, Kcal, Phi/D/C`, and stores the unit basis as
`w_j = −φ_j_int`. The bridge table is in `LFEM_Gridap.md` §9 and §4 below.

---

## Feature summary (what the library provides)

**Solver core** (`src/`): the stacked `[η,𝖴x,𝖴y]` loop-free residual + hand Jacobians; time loops
that default to the fully-implicit `RungeKutta(:SDIRK_2_2)` (L-stable 2nd-order, robust in the stiff
deep-water regime; `:theta` Crank–Nicolson also selectable) — sequential LU+Newton / distributed
GMRES+Jacobi+Newton; the full nonlinear physics — advection, the full leading pressure `R_P`, all
eight 𝓝 nonlinear-pressure components (native {3,6,7,8} + frozen-projection {1,2,4,5}) — in **both**
serial and distributed. The model is selected through three orthogonal high-level controls
`regime`/`nl_pressure`/`flat_bed` (see §6): `flat_bed` chooses a flat sea bed (∇h≡0, every ∇h-term
dropped, ∇η/dispersion kept) vs variable bathymetry, with a driver-side consistency warning.
Wavemaker/sponge/wall-BC; runtime solver monitoring plus an independent governing-equation residual
checker (`monitor.jl`, transient-aware); and `w_s`/`p_s` VTK reconstruction.

**Dirichlet boundary wave generation + WaveSpec coupling** (`waveinput.jl`): waves generated purely
from time-varying Dirichlet data on a domain side — regular / multichromatic / WaveSpec `AiryState`
stochastic sea states (seeded phases ⇒ rank-deterministic). The `:model` discrete-eigenmode
polarization (default) prescribes an exact discrete transport `Φ·Uamp = Aω/(kd)`, so a generated
wave is a solution of the discrete equations at the boundary and radiates cleanly; `:airy` samples
the continuous cosh profile instead. Includes a model-dispersion wavenumber solver, a Hann ramp or
hot start, and an optional generation/absorption relaxation zone. Driver kwargs
`wave_bc/bc_side/bc_profile/T_ramp/ic_from_bc/relax_bc/relax_width` on both drivers; transient trial
spaces work sequentially and distributed.

**Wave-generation selector** (`wave_gen`, both drivers): an explicit control with **two** mechanisms —
`:inner_res` (interior Gaussian source: line ⇒ plane, point ⇒ ring) and `:bc_gen` (Dirichlet boundary
generation). For `:bc_gen` the boundary source is dispatched on the **type** of `wave_bc` — a
parametrised regular plane wave from `A_wave`/`T_wave`/`wave_dir`, a caller-supplied `WaveInput`, or a
**WaveSpec `AiryState`** — all feeding the same Dirichlet machinery (they differ only in how the
`WaveInput` component table is populated, not in the boundary mechanism; a boundary wave must be a
consistent solution — surface *and* velocity — so it radiates cleanly, which is why it is always a
`WaveInput`). `resolve_wave_gen` validates the choice; the default `:auto` infers it
(`wave_bc===nothing`→`:inner_res`, else `:bc_gen`) so existing calls keep working. `wave_dir` sets the
boundary-wave propagation angle vs +x.

**Tests** (`test/`, 21 + `cluster/`): all pass — see the §1 table for the per-file scores. Highlights:
cross-check virtual-work equivalence **⚠ now failing at HEAD, see the open item**; the asymptotic-consistency pair (full-NL⇒Airy across the
band, kd→0⇒√(gd)); unsteady nonlinear MMS ~3e-9; BC generation 30/30 + 11/11 + 8/8 (Goda–Suzuki) +
distributed 4/4 (rel 3.05e-8); distributed agreement ≤5e-9 throughout.

**Production cluster scripts** (`examples/distributed/`, 6 total): including `run_irregular_sea_dist.jl`
and `run_directional_sea_dist.jl` (env-configured JONSWAP ± spreading via `build_airy_state()`;
smoke-validated on 2 ranks, governing-equation check at 4e-12 under transient Dirichlet). Validation
runs on record: `spectral_fidelity.jl` at 60 Tp — Hs transfer 1.023, dispersion 14/14 within 5% of the
model wavenumber, incident amplitudes 9/14 within 10%; `bc_irregular_sea.jl` — near-inflow Welch Hs
ratio 0.979.

**Postprocessing** (`postprocessing/GridapLFEMPost`): VTK/CSV → analysis/plots, from-modes
`w(σ)`/`p_nh(σ)` reconstruction, and the sea-state module (Welch PSD, JONSWAP target overlay, Hs,
Rayleigh exceedance), validated against solver output.

**Docs**: `building_files/LFEM_discretisation.zip` (authoritative LaTeX derivation; §8 the Gridap
implementation incl. the `flat_bed` model-setups subsection and Dirichlet generation; §9 the validation
report incl. BC gates — compile-checked), `LFEM_Gridap.md`, `algebraic_residual_math.md`,
`boundary_wave_generation.md`, and the consolidated `DESIGN_RECORDS.md`.

## Current Implementation Stage

**Working & validated.** The 2-D LFE-M solver is complete and validated in **both** serial and
distributed forms: the stacked loop-free residual + hand Jacobians, the full nonlinear physics
(advection, the complete leading pressure `R_P`, all eight 𝓝 nonlinear-pressure components), the
SDIRK/θ time integrators, wavemaker/sponge/wall/periodic BCs, and Dirichlet boundary wave generation
with WaveSpec coupling. The `test/` suite (21 files + `cluster/`) passes — cross-check virtual-work
equivalence **⚠ now failing at HEAD (see the open item)**, asymptotic consistency (full-NL⇒Airy, kd→0⇒√(gd)), unsteady nonlinear MMS ~3e-9,
BC generation 30/30 + 11/11 + Goda–Suzuki 8/8, distributed agreement ≤5e-9. Postprocessing
(`GridapLFEMPost`) and the authoritative LaTeX derivation are in place and compile-checked.

**Recently completed (this branch).**
- **Local validation loop + runtime diagnostics** (2026-08-06) — the work package planned in
  `building_files/LOCAL_VALIDATION_PLAN.md`, motivated by the fact that the four archived cluster
  runs burned 8 h / 44 h / 58 h / 72 h before failing, *silently*.
  * **Instrumentation** (§Runtime monitoring): per-rank RSS, GMRES saturation flagging, a
    **relative** divergence guard (`div_factor·eta_ref`, replacing the blind absolute `1e4`), the
    location of `max|η|` and its interior-vs-damped split, `|u|/|η|`, mass/energy invariants
    seeded from t=0, RK stage disambiguation, a `diagnostics.csv` step log, and a banner read back
    from the constructed solver objects. **Overhead measured within noise** (−0.8 % sampling every
    step, +2.2 % every 10th).
  * **`test/local/`** — four machinery gates on quasi-1D flumes, **51 gates, 0 failures**:
    rest state **8/8** (`max|η| = 0` exactly, flat and sloping bed), sponge **18/18**
    (`R² = 0.9995/0.9866/0.9835`, reflection 1.0–1.4 %), relaxation zone **9/9** (in-zone
    amplitude error ≤ 0.1 %, phase 0.9°, absorption **145×** its control), boundary modes
    **16/16** including a negative control that must fail. Budget **≈45 min at `JOBS=2`**
    (concurrent Julia processes contend heavily — three at once each run at ~⅓ solo speed).
  * **Non-regression**: `test_basic` reproduces its documented references **exactly**
    (max η = 0.00410 m, 408 Newton iterations, gauge amp = 0.00212 m); `test_vertical`,
    `test_primitives`, `test_conservation` re-pass.
  * **`examples/local_1d/` + `examples/local_2d/` + `run/local/`** — parametric scripts, a local
    launcher helper and 15 case launchers; 7 cluster 1-D launchers in `run/dist_small/`.
  * **Measured on this machine**: `using GridapLFEM` 2.8 s; first `setup_and_run` ~207 s of JIT;
    warm 0.349 s/step at 4669 free DOFs. **A Julia+Gridap process sits at ~1.4 GB RSS before
    solving anything** — see the memory entry below, this is the leading explanation of the OOM.
  * **`ny ≥ 3` is mandatory for a y-periodic mesh** (Gridap `CartesianGrids.jl:39`); `ny=1`/`ny=2`
    are rejected at mesh construction.
- **Distributed GMRES configuration fixed** (2026-08-05, `f7fe62d`) — `ls_maxiter` was being passed
  as `GMRESSolver`'s positional `m` (basis size) instead of the `maxiter` keyword (see §7). Now
  `GMRESSolver(krylov_m=100; restart=true, maxiter=ls_maxiter, …)`, with `ls_maxiter` **2000 → 1000**
  (measured convergence is ~150 iterations, so ~7× headroom). Measured on the 4-rank distributed
  test: `gmres` no longer pinned (149–152, converging), **Newton 8–10 → 4 per step**, 120/120 steps,
  caches **139 MB → 5.6 MB per rank**. `krylov_m` is a kwarg on both drivers, forwarded through all
  12 distributed run scripts, and settable as `LFEM_KRYLOV_M`. The banner previously advertised a
  2000-iteration cap that never existed; it now reports `GMRES(m) restarted`.
- **`test_basic_distributed.jl` reference amplitudes re-measured** (`4bf90fd`) —
  `REF_EMAX_LIN = 0.0037461259`, `REF_EMAX_NL = 0.0041032781`, obtained by running the *sequential*
  solver (LU + Newton) with the distributed test's exact configuration. Verified 6/6 PASS at
  rel ≈ 1e-5, three orders inside `REF_RTOL = 2e-3` (so that tolerance did **not** need loosening).
  The previous constants were traced by bisection (see the open-boundary entry below).
- **`GridapLFEM` is now a Julia package** (2026-08-04) — `name`/`uuid`/`version` in `Project.toml`,
  loaded everywhere with `using GridapLFEM`; the 36 consumers (21 tests + `test/cluster/`,
  13 examples, `compile/warmup.jl`, `examples/distributed/_dist_common.jl`) were migrated off
  `include(src/GridapLFEM.jl)` + `using .GridapLFEM`, and `:GridapLFEM` was added to
  `create_sysimage` with a preflight that fails the build if the package is not resolvable.
  **Why it matters:** an `include()`d module lives in a throw-away `Main.GridapLFEM`, so neither the
  solver nor — far more expensive — the Gridap FEM specialisations keyed on its types were retained
  in the sysimage, and every rank recompiled them (ranks were caught in `typeinf`/`optimize` in one
  kill-time traceback). The package also precompiles natively (**3.5 s** to load).
  **Correction of an earlier claim:** this was first recorded here as *the* fix for the 2026-08-04
  cluster OOM kills. That attribution was wrong. The demonstrated OOM mechanism is the GMRES cache
  over-allocation above (139 MB/rank per numerical setup, churned across stages and steps until RSS
  crossed the 2 GB/core limit) — the failing job advanced **140 steady steps** before dying, whereas
  a compile spike occurs before step 1. Both defects were real and both are fixed; packaging alone
  would not have prevented the OOM. `test_equivalence.jl` keeps its `include()` of the *oracle*
  `../LFE-M_2D_solver/` — that is a genuinely external second implementation. Plan and design
  record: `building_files/PACKAGE_MIGRATION_PLAN.md`.
  Two pre-existing problems surfaced by the migration, both fixed/recorded: `test_vertical.jl` used
  `VectorValue` without importing it (it failed identically under the old loading — the documented
  "15/15 PASS" was stale; now genuinely 15/15), and an apparent dependency drift to
  **Gridap 0.20.8 / GridapSolvers 0.7.1** against prose claiming 0.19.11 / 0.6.2.
  **Reconciled 2026-08-05** (see §Solver conventions): the cluster also runs GridapSolvers 0.7.1, so
  the prose — not the environment — was stale; and the two stacks were measured to give *identical*
  results, so `[compat]` keeps both minors on evidence rather than as a hedge.
- **Launcher memory re-sized, and the 2 GB/core hypothesis REFUTED by experiment** (2026-08-06) —
  `--mem-per-cpu=4G` is restored and sized to *fit* a rome node (256 GB / 128 cores): ranks are
  spread `nodes × ntasks-per-node` so no launcher requests more than 256 GB/node (128 ranks ⇒ 2×64).
  A small-domain job was then launched on `rome` at the **node-default 2 GB/core** (no
  `--mem-per-cpu`) and was **killed for lack of memory, with the same error as the previous
  crashes**. So the 4 GB/core request is **not** provisional compile headroom that a working
  sysimage would make redundant — it is required, and *why* is now the open question, not *whether*.
  All 28 launcher headers carrying the old "drop this once the sysimage is proven" comment were
  rewritten accordingly. The memory-attribution experiment (four hypotheses — per-rank JIT, GMRES
  cache, a per-step leak, or a baseline footprint that simply exceeds 2 GB/core — each with the
  measurement that decides it) is **`building_files/LOCAL_VALIDATION_PLAN.md` §2.2**; it depends on
  the new per-rank RSS reporting (§Runtime monitoring).
- **Physics-selection interface** (§6): the six residual booleans are driven by three orthogonal
  high-level controls `regime` ∈ {`:linear`,`:nonlinear`}, `nl_pressure` ∈ {`:none`,`:native`,`:full`},
  and **`flat_bed`** ∈ {`false`,`true`} via `resolve_physics`. `flat_bed` replaced the former
  `slope_pressure` switch: it selects a flat sea bed (∇h≡0 — drops the bed-slope `𝓐` packages, `L¹`,
  `N{3,6}`, the bed-slope IBP half) vs variable bathymetry, at a single control point
  (`dhx,dhy = flat_bed ? 0 : ∂h`) in `global_residual`/`jacobian_*`; ∇η/dispersion terms are kept.
  Runtime-verified with a differential residual test (9/9 across all `nl_pressure` tiers: `flat_bed`
  changes a sloped-bed residual, is bit-exact on a flat bed); the `flat_bed=false` path is identical to
  the prior validated baseline, confirmed at the time by `test_equivalence` re-passing 10/10 (oracle
  virtual-work match ≤7.4e-15) — **that test no longer passes at HEAD; see the open item**. Drivers emit a bathymetry↔switch consistency warning
  (`check_flat_bed_consistency`). Derivation: the flat-bed reduction of the full nonlinear model in
  `LFEM_discretisation.zip` (`VerticalSemiDiscreteSystem.tex`); design record: `DESIGN_RECORDS.md` §7.
- **Default integrator** SDIRK_2_2 (fully implicit, L-stable); **y-periodic** lateral BC option.
- **Surface-damping sponge** — the sponge now damps the free surface η as well as the velocity
  (`+∫ μ q η` in continuity, same μ profile/μ_max; residual + hand Jacobian, `problem.jl`). This
  cures the open-boundary spurious mode that a velocity-only sponge left under-damped (a linear
  plane wave that previously grew to 10⁴× and NaN'd). Closed-basin tests are bit-identical (μ≡0
  there). Design record: `DESIGN_RECORDS.md`; plan: `building_files/SPONGE_WAVEGEN_PLAN.md`.
- **Explicit `wave_gen` selector** — two mechanisms `:inner_res` / `:bc_gen` (interior source vs
  boundary Dirichlet generation); for `:bc_gen` the source (parametrised regular wave / `WaveInput` /
  WaveSpec `AiryState`) is dispatched on the type of `wave_bc`, all feeding the same Dirichlet machinery
  (`resolve_wave_gen`, both drivers; `:auto` keeps old calls working). New driver kwarg `wave_dir`
  (boundary-wave angle). New example `run_bc_plane_small.jl` (BC plane wave, left Dirichlet +
  relaxation + strong far sponge) with flat/varbed × lin/nonlinear launchers. Small-domain
  `mu_max` defaults raised 10→40 to kill waves fast in the sponge.
- **Small-domain run suite** — **5 parametric scripts** (`examples/distributed_small/`, grouped by
  wave-generation type: `run_periodic_plane_small` [line source], `run_ring_small` [point source],
  `run_bc_plane_small` [`:bc_gen` boundary plane wave], `run_directional_sea_small` /
  `run_irregular_sea_small` [`:bc_gen` WaveSpec Dirichlet BC]) driving **20 launchers**
  (`run/dist_small/`), one per case. Each script bakes in the common geometry/mesh/partition/solver
  defaults and reads the physics from env (`get!` base defaults ⇒ banner ⇔ solver consistent; a
  `flat_bed=0` toggle builds the submerged bar; the output dir is auto-tagged by config so cases don't
  clobber). Each launcher overrides only what changes (regime / nl_pressure / flat_bed→bar / amplitude
  / period). Common 50×20 m domain, short dynamic timescales (`d=3.5, T=2.0, kd≈3.5`), `save_every=10`;
  partition **8×4 = 32 ranks** (plane/irregular, 200×40) or **8×8 = 64 ranks** (ring/directional,
  200×80 square cells) — 250 cells/rank throughout. The 20 cases span the configuration space (batch 1),
  a controlled comparison matrix around the flat periodic plane wave (batch 2), and sea-state comparisons
  (batch 3). Verified: all 5 parse; every launcher resolves to its intended config with `n = px·py =
  ntasks`. Smoke-validated on 2 ranks (big-amplitude full-nonlinear flat plane wave: Newton-converged,
  finite). Plan: `building_files/LFEM_runs.md`.
- **Cluster sysimage build chain** (`compile/`) — working and validated end to end: the system-MPI
  pin (`use_system_binary`), the MPI preflight, and `include_transitive_dependencies=false` (which
  removed the last `OpenMPI_jll`-baked-into-the-image crash). Verify a built image with
  `strings GridapLFEM_sysimage.so | grep -i OpenMPI_jll` → must be empty; `… | grep libmpi` → only the
  system `/sw/.../OpenMPI/5.0.3/lib/libmpi.so`. Removes the ~30–45 min/rank JIT compile and the OOM.
  A production run launched against the image was confirmed to run correctly.
- **All launchers moved onto the sysimage** — the `-J` path is no longer one demo script: all 29
  launchers (9 in `run/`, 20 in `run/dist_small/`) go through the new shared helper
  **`run/lfem_env.sh`** (modules + image resolution + `lfem_run <nranks> <script.jl>`), so each
  launcher is just its `#SBATCH` header, its `LFEM_*` overrides, and one call. Consequences of no
  longer JIT-compiling per rank: every Snellius launcher moved **`fat_rome` → `rome`** (L2→L1
  budget) with **walltimes unchanged**. `--mem-per-cpu` was at that point dropped in favour of the
  `rome` node default (2 GB/core), on the argument that the 4 GB/core request only ever covered the
  compile spike — **that argument was subsequently tested and refuted** (the 2 GB/core job was
  OOM-killed; see the "Launcher memory" entry above), and 4 GB/core is back in every header. The
  `fat_rome` → `rome` move stands. The redundant
  `run_lin_periodic_plane_small_sysimage.sh` demo was removed (its content is now the norm; its
  explanatory header lives in `run/lfem_env.sh` and `compile/README.md`). A missing image aborts
  the job with the build command instead of silently falling back to the slow path;
  `LFEM_NO_SYSIMAGE=1` is the deliberate escape hatch. Verified: `bash -n` on all 29, helper
  behaviour smoke-tested against a stub cluster (image present / missing / disabled / bad script
  path), and every launcher re-checked for `px·py = lfem_run ranks = SBATCH ntasks` with an
  existing target script.
- **Sysimage staleness detection** — closes the "nothing detects a stale image" gap above.
  `lfem_check_sysimage_freshness` (in `run/lfem_env.sh`, called by `lfem_run`) warns when the image
  no longer matches `src/`; `lfem_write_sysimage_stamp` (called by `compile_snellius.sh` /
  `compile_blue.sh` after a successful build) writes the `…so.src.sha256` content hash it compares
  against, with an mtime fallback for pre-stamp images. `LFEM_STRICT_SYSIMAGE=1` upgrades the
  warning to an abort. Verified against a stub cluster: content edit / file added / file deleted all
  warn; bare `touch` and content-restore do **not**; strict mode exits 1 without launching;
  `LFEM_NO_SYSIMAGE=1` skips the check; empty and missing `src/` neither hang nor crash
  (`xargs -r`). `.gitignore` gained `GridapLFEM_sysimage.so` + `*.src.sha256` (the ~1 GB image and
  its stamp were previously untracked-but-not-ignored). Rebuilding remains manual — staleness is
  now *detected*, not prevented.
- **WaveSpec `change_seed!` fixed (upstream)** — the vendored `WaveSpec.jl` was moved from the
  tagged release to the **GitHub repository version**, which carries the fix (`change_seed!` now
  reads `state.spectrum`, not the non-existent `state.spec`). `build_airy_state()` works again, so
  every Dirichlet sea-state run is unblocked: production `run_{irregular,directional}_sea_dist.jl`
  and the four small sea launchers. The run scripts themselves never needed changing.
- **Docs** — LaTeX derivation restructured into a **`report`** (chapters + table of contents); the
  drivers `setup_and_run`/`setup_and_run_distributed` were re-commented for clarity (unrolled multi-arg
  calls, expanded large ternaries to `if/else`; no behavioural change). A **CFC 2027 conference
  abstract** on the arbitrary-order model was drafted (`building_files/CFC2027_LFEMultilayer_abstract/`).

**Under development / open:**
- **⚠ ORACLE EQUIVALENCE IS FAILING AT HEAD — `test_equivalence.jl` gives 1 PASS / 9 FAIL**
  (discovered 2026-08-06 while re-running the suite for non-regression). This is the repository's
  designated **acceptance** test — it compares the package residual against the independent
  per-layer implementation in `../LFE-M_2D_solver/` on identical analytic states.
  * **It is pre-existing, not caused by the 2026-08-06 instrumentation work.** Verified by
    `git stash`-ing all `src/` changes and re-running against pristine HEAD `051befa`: the failure
    is **bit-for-bit identical**, same 9 failures with the same relative errors to every digit.
    Independently, the test calls only `assemble_vertical_tensors`, `build_fe_spaces`,
    `build_problem_raw` and `global_residual` — none of which that work touched.
  * **What still passes**: the vertical tensors match the oracle exactly
    (`Mmat, Phi, B, Mcal, Gcal, A, K, P, Acal, Kcal`). So Stage 1 is fine and the discrepancy is in
    the **residual assembly**.
  * **What fails**: virtual work, all 3 configs × 3 test sets, `rel = 2.0e-2 … 1.5` against a
    `1e-10` gate. Crucially **config A fails too** (`rel = 1.2e-1`) — that is the *simplest*
    setting: `linearised=true, advection=false, lin_pressure=false, P_full=false,
    nl_pressure68=false`, flat bed. So the disagreement is already present in the linear core
    (mass + acceleration + gravity + the `P³L³` dispersion carrier), not in the nonlinear or
    slope-pressure packages. Configs B and C differ from each other only marginally
    (2.0e-2 / 1.3 / 9.2e-2 vs 2.0e-2 / 1.5 / 9.6e-2).
  * **The documented "10/10 PASS, ≤7e-15" was stale** — the same class of stale claim as the
    `test_vertical.jl` one found during the package migration. All such claims in `CLAUDE.md` and
    `README.md` are now flagged.
  * **Not investigated further, deliberately**: resolving this means changing validated physics
    code (`problem.jl`) or the oracle, which is a design-level decision. Candidate leads worth
    starting from: commit `3f442be` (the `regime`/`nl_pressure`/`flat_bed` flag refactor, which
    changed what `build_problem_raw`'s booleans mean and is the most recent change to the
    flag→term mapping the test relies on), the parent repo's "corrected G tensors" change to the
    oracle, and the Gridap 0.19→0.20 move (the oracle was written against an older minor).
    **A first triage step that costs nothing: bisect `test_equivalence.jl` over the commits since
    it last passed.**
- **Re-run the small-domain suite with the 2026-08 fixes in place.** Every cluster run on record
  predates at least one of: the surface-damping sponge (`95f5ec6`), the GMRES configuration fix
  (`f7fe62d`), and the `mu_max` 8→40 raise. The archived outputs are therefore *not* a baseline —
  `small_irregular_nonlinear_full_flat_Hs0.2_M2` (NaN at t=13.8) is a diagnosed pre-fix failure, not
  a solver defect. What to check on the re-run: `gmres=` no longer pinned (now also flagged
  automatically — a truncated solve prints a WARN), Newton ~4/step, and — read straight off the new
  `diagnostics.csv` — `x_at_max` staying in the interior with `eta_max_damped/eta_max_int < 1`.
  **Rebuild the sysimage first**: `src/*.jl` changed on 2026-08-06.
  *Local proxies now exist and should be run first, since they cost minutes rather than hours:*
  `run/local/run_2d_*.sh` are scaled-down siblings of eight of these cases, and
  `test/local/test_boundary_modes_1d.jl` reproduces the boundary-mode failure deliberately
  (its negative control reaches `damped/interior = 65` within **6 s of simulated time**, against
  the 60× measured post-hoc on the 44 h cluster job).
- **Where is the cluster memory actually going?** (supersedes "is the sysimage removing the
  per-rank compile?") A `rome` job at the node-default **2 GB/core was OOM-killed** (2026-08-06),
  so the compile-spike explanation is not sufficient on its own and `--mem-per-cpu=4G` stays in
  every launcher. Four hypotheses remain — per-rank JIT (image absent/stale/incomplete), GMRES
  cache allocation, a per-step leak (Gridap caches, `nl_pressure=:full` frozen projections,
  VTK buffers), or a baseline footprint that simply exceeds 2 GB/core — each with a decisive
  measurement, in `building_files/LOCAL_VALIDATION_PLAN.md` §2.2. Needs `sacct MaxRSS` + the new
  per-rank RSS log line; the cheapest decisive job is
  `run/dist_small/run_lin_periodic_plane_small.sh`.
- **`Hs=0.2 m` at `d=3.5 m` with `nl_pressure=:full`** is far outside the conservative `A ≤ 0.001`
  guidance and is the most aggressive case in the suite — if it still destabilises after the fixes,
  the sponge/relaxation widths for the longest components are the next thing to size (see §8).
- At-scale physical benchmarks on the cluster (Stokes harmonics, Dingemans bar) — scripts exist
  (`examples/validation/`, `examples/distributed/`); quantitative overlays on the paper's data pending.
- Production-length (200+ Tp) irregular/directional sea runs — scripts ready and now **unblocked**
  (the `change_seed!` fix above); not yet run at length.
- A run-and-reconstruct **pressure** profile test; a distributed-gauge utility.
- Sheared-current focusing (paper §4 case 5) — needs an ambient-current term (a modelling extension).
- Boundary-generation follow-ups: `:bottom`/`:top` generation sides; second-order (bound-wave)
  irregular BC corrections.

---

## 2. `main.tex` §8 — the single scalar residual

Gridap's `MultiFieldFESpace` solver consumes a **single scalar** = total virtual work `∫_Ω R·v`.
`main.tex` §8 gives it (the framed `eq: multifield general residual`). With `U=[u_0…u_{Nσ}]` (stacked
velocities), `V=[v_0…v_{Nσ}]`, `q`/`H` the continuity test/trial:

```
Global Residual = ∫_Ω [ q ∂H/∂t − ∇q·(H U)·Φ                                    (mass)
        + ( H·M^V·U̇                                                             (acceleration)
          + H·F_M(U) + F_G(H,U)                                                  (advection)
          + gH∇η·Φ                                                              (gravity, η=H−h)
          − H[ ∇h·(L:A^V + N⫶𝓐^V) + ∇H·(L:K^V + N⫶𝓚^V) ] ) · V                 (lin+nonlin slope pressure)
        − H²[ (L:P^V) + (N⫶𝓟^V) ] · (∇·V) ] dΩ                                  (leading pressure = DISPERSION)
```
with `F_M(U)_i = Σ_kj 𝓜_ikj (u_k·∇u_j)`, `F_G_i = Σ_kj 𝓖_ikj (∇·[Hu_k]) u_j`;
`L_j = [−u̇_j·∇h, u̇_j·∇H, −∇·(Hu̇_j)]` (3 comp); `N_kj` = 8 comp (see §6 / math doc §7);
`(∇·V)_i = ∇·v_i` (test-divergence vector); `P^V_ij = ∫θ_j φᵢ_int dσ` (3 comp, `P[3]=−B`),
`𝓟^V_ikj = ∫Θ_kj φᵢ_int dσ` (8 comp). **Sign convention:** all three pressure blocks are positive
integrals **subtracted** uniformly, `Global = R_mass + R_Acc + R_Adv + R_Grav − R_P − R_lin − R_nonlin`
with `R_P = +∫H²[(L:P^V)+(N⫶𝓟^V)]·(∇·V)`.

The leading-pressure term `R_P` is the load-bearing part of this equation: it is the only O(η)
non-hydrostatic term on a flat bed, and it *is* the entire frequency dispersion of the model. Only the
boundary part of the integration by parts that produces it vanishes; the volume part remains and must
be assembled — without it the model reduces to non-dispersive shallow-water. The wavemaker adds
`−∫ q S(x,t)` to mass; the sponge adds `+∫ μ (M^V U)·V` to momentum (`main.tex` §8, wave-generation
subsection).

---

## 3. The core design decision: stack the layer index into the value type

The whole solver turns on **one decision**: *stack the vertical (layer) index into the FE value type,
not into a Julia array.*

* **Velocity = two vector-valued fields** `𝖴x, 𝖴y ∈ VectorValue{Nσ}` (all layers' x- resp. y-velocity).
  The MultiField is `[η, 𝖴x, 𝖴y]` — **3 fields**, not `1+2Nσ` scalar fields.
* The static vertical arrays become **constant tensors**: `M^V→TensorValue{Nσ,Nσ}`,
  `𝓜^V,𝓖^V→ThirdOrderTensorValue{Nσ,Nσ,Nσ}`, and the component-indexed `A^V/K^V/𝓐^V/𝓚^V` are **split
  per component** into 3 (resp. 8) such constants.
* Then **every layer sum `Σ_j`, `Σ_{kj}` is a matvec / tensor double-contraction** — no loops.
* The **spatial index is only 2-D** and is written explicitly with `e_x=(1,0)`, `e_y=(0,1)`;
  `∂_x f ≡ e_x·∇f`, `∂_y f ≡ e_y·∇f`.
* We touch the MultiField only for `η=U[1]`, `𝖴x=U[2]`, `𝖴y=U[3]` — **no `Nσ`-decomposition**.

**Why this design.** Expressing the layer sums as native tensor contractions makes the residual
well-typed by construction: there are no `VectorValue{3}`-of-vectors rank mismatches, no illegal
`U[2:end]` slices of a `TransientMultiFieldCellField`, and no gradient-of-a-divergence or `∂²η` terms
appearing on the trial space (those are handled by IBP or auxiliary fields in §6). It is also fast: the
fused per-layer advection cost collapses into single dense contractions, and the same CellField algebra
is transparently forwarded to `DistributedCellField`, so the identical residual and Jacobians run
sequentially and across MPI ranks with no separate code path.

### Verified Gridap facts (rely on them)
1. `∇` of a `VectorValue{Nσ}` field = `TensorValue{2,Nσ}` (**spatial index first**,
   `(∇f)[d,j]=∂f_j/∂x_d`). So `∂_x f = e_x·∇f`, `∂_y f = e_y·∇f`. (`f=(x,2y,x+y) ⇒
   e_x·∇f=(1,0,1)`, `e_y·∇f=(0,2,1)`.) **Use `e·∇f`, not `∇f·e`.**
2. `double_contraction(𝓣::ThirdOrderTensorValue, S::TensorValue{Nσ,Nσ})` contracts the **trailing
   two** indices → `VectorValue{Nσ}` = `Σ_{k,j} 𝓣_{ikj} S_{kj}`. **Native — no helper needed.**
3. `W::VectorValue{Nσ} ⋅ 𝓣::ThirdOrderTensorValue` contracts the **first** index → `TensorValue{Nσ,Nσ}`.
   Native.
4. `⊗` (outer, `VectorValue{Nσ}⊗VectorValue{Nσ}→TensorValue{Nσ,Nσ}`), `⊙` (Frobenius `A:B→scalar`),
   `Operation(VectorValue)(a,b)` (build a `VectorValue{2}` from two scalar CellFields) — all native.

**Consequence: the residual needs no custom contraction primitives.** The only hand-written helpers
are the build-time constructors that turn the assembled Float arrays into the constant tensors.

### Variable / operator dictionary

| symbol | type | meaning | Gridap |
|--------|------|---------|--------|
| `η, H` | scalar CellField | free surface, total depth `H=d+η` | `U[1]`; `H=d_cf+η` |
| `𝖴x, 𝖴y` | `VectorValue{Nσ}` | stacked layer velocities (x, y comp) | `U[2]`, `U[3]` |
| `𝖶x, 𝖶y` | `VectorValue{Nσ}` | test functions | `V[2]`, `V[3]` |
| `d_cf` | scalar | still-water depth `d(x)`; `∇h=∇d`, `∇²h` analytic | `CellField(d_func,Ωₕ)` |
| `𝚽` | `VectorValue{Nσ}` const | depth-average weights `Φ_j` | `alg_to_vec(Φvec)` |
| `𝗠` | `TensorValue{Nσ,Nσ}` const | vertical mass `M^V` | `alg_to_tensor2(M2)` |
| `𝗠3,𝗚3` | `ThirdOrderTensorValue` const | advection `𝓜^V,𝓖^V`, index `[i,k,j]` | `alg_to_tensor3(...)` |
| `A[c],K[c]` | 3×`TensorValue{Nσ,Nσ}` | linear-pressure `A^V,K^V` per component | component split |
| `P[c]` | 3×`TensorValue{Nσ,Nσ}` | leading-pressure `P^V` (`P[3]=−𝗕`, dispersion) | `alg_to_tensor2(vert.P[:,:,c])` |
| `𝗔3[c],𝗞3[c]` | 8×`ThirdOrderTensorValue` | nonlinear-pressure `𝓐^V,𝓚^V` per component | component split |
| `DU`, `DW` | `VectorValue{Nσ}` | per-layer divergence of trial / test | `∂x(𝖴x)+∂y(𝖴y)`, `∂x(𝖶x)+∂y(𝖶y)` |
| `S` | `VectorValue{Nσ}` | `∇·(H u_j) = H·DU + u_j·∇H` | product rule |
| `ū, W̄` | `VectorValue{2}` | depth-avg velocity/test `(Φ·𝖴x, Φ·𝖴y)` | `Operation(VectorValue)(…)` |

**Index order rule:** store `𝓜^V,𝓖^V,𝓐^V,𝓚^V` as `[i,k,j]` = `[output/test layer, u_k / ∇·(Hu_k)
layer, ∇u_j / u_j layer]`, so contraction over the trailing two indices directly gives the mode-i
momentum contribution.

---

## 4. The vertical σ-tensors (built once)

On the σ-mesh (`M` elements, order `p`, optimised nodes `c_bdy`), build `φ_j` (basis), `φ_j'`,
`varphi_j=∫_0^σ φ_j` (solved as a BVP `dφ/dσ=φ_j, φ(0)=0`, degree `p+1`), `Φ_j=∫_0^1 φ_j`. Then:

```
M2[i,j]   = ∫ φ_i φ_j                         → M^V     (vertical mass)
M3[i,j,k] = ∫ φ_i φ_j φ_k                      → 𝓜^V     (horizontal advection)
G3[i,j,k] = ∫ (σΦ_k − varphi_k) φ_j' φ_i       → 𝓖^V
A2[i,j]   = ∫ φ_i θ_j            (θ_j 3-vec)    → A^V     (linear pressure, ∇h coupling)
K2[i,j]   = ∫ θ_j (varphi_i − σφ_i)            → K^V     (linear pressure, ∇H coupling)
A3[i,j,k] = ∫ Θ_kj φ_i          (Θ_kj 8-vec)   → 𝓐^V     (nonlinear pressure, ∇h)
K3[i,j,k] = ∫ Θ_kj (varphi_i − σφ_i)           → 𝓚^V     (nonlinear pressure, ∇H)
θ_j  = [φ_j, σφ_j, varphi_j]                                   (linear shape vector)
Θ_kj = [σΦ_k φ_j, Φ_k varphi_j, φ_j φ_k, σφ_j φ_k, varphi_j φ_k,
        σΦ_j φ_k'−varphi_j φ_k',
        σΦ_j φ_k+σ²Φ_j φ_k'−varphi_j φ_k−σ varphi_j φ_k',
        σΦ_j φ_k−varphi_j φ_k]                                 (nonlinear shape tensor, 8 comp)
```

Solver-code names: `Mmat, Mcal, Gcal, A, K, Acal, Kcal`, plus the leading-pressure tensor
`P[i,j,c] = ∫θ_j[c]·φᵢ_int dσ` (`P[:,:,3] = −B`, the dispersion carrier the `R_P` term requires),
the nonlinear leading tensor `Pcal[i,k,j,c] = ∫Θ_kj[c]·φᵢ_int dσ` (the 𝓟-part of `R_P`), and the
diagnostic `B = −∫ varphi_i varphi_j`. `assemble_vertical_tensors` builds all of these once; the
package reshapes them into `TensorValue`/`ThirdOrderTensorValue` constants (peel the component index,
store as `[i,k,j]`) in `build_problem`.

---

## 5. The residual, term by term (`src/problem.jl`)

From `algebraic_residual_math.md` (§2–§7). `∂x(f)=e_x·∇(f)`, `∂y(f)=e_y·∇(f)`:

```julia
DU  = ∂x(Ux)+∂y(Uy);  DW = ∂x(Wx)+∂y(Wy)          # per-layer div of trial / TEST (VectorValue{Nσ})
UgH = ∂x(H)*Ux+∂y(H)*Uy;  Ugh = ∂x(d)*Ux+∂y(d)*Uy;  S = H*DU+UgH
ū   = Operation(VectorValue)(𝚽⋅Ux, 𝚽⋅Uy)

mass  : ∫ q*ηt − H*(∇(q)⋅ū) − q*S(x,t)               # wavemaker source enters with a MINUS
acc   : ∫ H*( (Wx⋅(𝗠⋅Uxt)) + (Wy⋅(𝗠⋅Uyt)) )
grav  : − ∫ (g/2)*(H*H−d*d)*(𝚽⋅DW)                   # IBP energy form; lin: −∫g*η*(𝚽⋅DW)
disp  : − ∫ H²*(sP⋅DW)   with  sP = P[1]⋅L1+P[2]⋅L2+P[3]⋅L3   (R_P = the dispersion)
        (P_full=false → sP = P[3]⋅L3 only;  lin: −∫ d²*((P[3]⋅L3lin)⋅DW))
adv   : ∫ H*( double_contraction(𝗠3,TMx)⋅Wx + double_contraction(𝗠3,TMy)⋅Wy )
          + ( double_contraction(𝗚3,TGx)⋅Wx + double_contraction(𝗚3,TGy)⋅Wy )
        TMx=Ux⊗∂x(Ux)+Uy⊗∂y(Ux);  TMy=Ux⊗∂x(Uy)+Uy⊗∂y(Uy);  TGx=S⊗Ux; TGy=S⊗Uy
linP  : − ∫ H*( ∂x(d)*πAx + ∂y(d)*πAy + ∂x(H)*πKx + ∂y(H)*πKy )
        L1=−(∂x(d)*Uxt+∂y(d)*Uyt); L2=∂x(H)*Uxt+∂y(H)*Uyt; L3=−(H*(∂x(Uxt)+∂y(Uyt))+L2)
        πAx = (Wx⋅A[1])⋅L1+(Wx⋅A[2])⋅L2+(Wx⋅A[3])⋅L3   (πAy,πKx,πKy analogous)
nlP   : comps {3,6,7,8} native; comps {1,2,4,5} split (§6); + 𝓟-part of R_P (uses Pcal)
sponge: + ∫ μ*( q*η + (Wx⋅(𝗠⋅Ux)) + (Wy⋅(𝗠⋅Uy)) )   # damps the free surface η AND velocity (same μ profile/μ_max)
```

**Gravity** uses the integrated-by-parts energy form `−(g/2)(H²−d²)(𝚽·DW)`. Subtracting the
still-water baseline `(H²−d²)` makes the discrete rest state exactly force-free, so an undisturbed
surface at an open wall stays at rest instead of drifting.

**Linear pressure `L`** is built as **three separate `VectorValue{Nσ}` fields** `L1,L2,L3`, and
`L:A^V = A[1]·L1 + A[2]·L2 + A[3]·L3` (a sum of matvecs). Keeping the three components separate (rather
than as one `VectorValue{3}`-of-vectors) is what keeps the term well-typed and first-order.

---

## 6. The nonlinear pressure `N` (`src/nlpressure.jl`)

`N_kj` (8 comp) as `(k,j)` `TensorValue{Nσ,Nσ}` fields, split by how many derivatives each component
carries:

* **Components {3,6,7,8} are first-order** (outer products), so they are assembled directly in all
  three blocks (𝓐/𝓚 slope halves + the 𝓟 leading part), serial and distributed — `nl_pressure=:native`.
* **Components {1,2,4,5} carry second derivatives** of the unknowns (`∂𝖲=∇(∇·(H𝖴))`, "gradient of a
  divergence", and `∂²η`), which cannot act directly on `Q2`. They are handled by class —
  `nl_pressure=:full`:
  * **∇h half** → **integrated by parts onto the test function** (exact; uses the analytic bed Hessian
    `∇²h`), machine-verified.
  * **∇H half + 𝓟 part** → the `∂²η` factor is irreducible, so it is evaluated from **per-step frozen
    L²-projections** `π𝖲, π𝖻` (project `𝖲` and `𝖻` onto the velocity FE space, SPD mass solves, and use
    `∂_a(π𝖲), ∂_a(π𝖻)` lagged one step). This is `O(dt)` on an already `O(A³)` term, so the lag is
    negligible; the mass solve is a direct factorisation sequentially and CG + Jacobi distributed.

All 𝓝 blocks are `O(A²–A³)` and treated quasi-Newton (they add to the residual but not to the
Jacobian — their contribution to convergence is negligible at these amplitudes). The full linear-regime
physics (dispersion, sloshing, small-amplitude shoaling) is available with `nl_pressure=:none`;
`:native`/`:full` add the finite-amplitude harmonics.

### Physics selection interface

The drivers (and `build_problem`) select the physics through **three coupled controls** rather than
six independent booleans (which allowed inconsistent/redundant combinations):

| control | values | meaning |
|---------|--------|---------|
| `regime` | `:linear` \| `:nonlinear` | `:linear` = linearised core, no advection; `:nonlinear` = full nonlinear core + advection |
| `nl_pressure` | `:none` \| `:native` \| `:full` | nonlinear non-hydrostatic pressure: off / `{3,6,7,8}` / `+{1,2,4,5}` |
| `flat_bed` | `Bool` | sea-bed geometry: `false` = variable bathymetry (∇h≠0, full model); `true` = flat bed (∇h≡0 — every term carrying an explicit or implicit factor ∇h is dropped, ∇η/dispersion terms kept). Orthogonal to `regime`/`nl_pressure`; see the "Flat-bed reduction" subsection of `VerticalSemiDiscreteSystem.tex` in `LFEM_discretisation.zip`. |

`resolve_physics` (in `src/problem.jl`) maps these to the seven internal booleans stored on
`LFEMProblem` (`linearised, advection, lin_pressure, P_full, nl_pressure68, nl_pressure_full, flat_bed`).
The model's pressure content is intrinsic to `regime`/`nl_pressure` (`P_full = advection`,
`lin_pressure = advection || !flat_bed`); `flat_bed` then zeroes ∇h at a single point in
`global_residual`/`jacobian_*` (`dhx,dhy = flat_bed ? 0 : ∂h`), which uniformly drops the bed-slope
`𝓐` packages, `L¹`, `N{3,6}` and the bed-slope IBP half while `∇H → ∇η` keeps the surface-slope terms.
`resolve_physics` rejects the footgun `regime=:linear` with `nl_pressure≠:none`; the drivers emit a
**warning** on a `flat_bed` ↔ bathymetry mismatch (`check_flat_bed_consistency`). **`build_problem`**
takes the three high-level controls; **`build_problem_raw`** is the low-level escape hatch taking the
seven booleans directly, for the rare combination the high-level interface deliberately does not expose
— notably `lin_pressure` without `P_full`, used by the oracle-equivalence test.

---

## 7. Jacobians and time integration

**Jacobians are hand-written** (`jacobian_u`, `jacobian_u_t` in `src/problem.jl`). Splitting
`∂R/∂u̇` (the effective mass — acceleration + `R_P` dispersion) from `∂R/∂u` (the spatial block) lets
the integrator form its per-stage system `J = ∂R/∂u + (1/aΔt)∂R/∂u̇` directly. The advection block is
differentiated in full (quadratic Newton); the slope-pressure packages are linearised with `H` frozen
in the u̇-carrying terms (a quasi-Newton choice that keeps the Jacobian sparse). Automatic
differentiation is not used because Gridap's multifield AD cannot dualize through `∂t(u)` (there is
no `TransientMultiFieldCellField` constructor for the dual), so the hand Jacobians are the design;
`build_ode_operator_ad` exists only as a cross-check path. That limitation was established on
Gridap 0.19.11 and has **not been re-tested on 0.20.x** — but the hand Jacobians are the intended
design either way, so it is not on the critical path.

**Coding rule (block arrays):** never apply `∇` to an `Operation`-composed expression containing a
test basis — expand by hand via `∂_a(W⋅𝓣) = (∂_aW)⋅𝓣` (see `nlpressure.jl`).

**Time integrators** (`build_ode_solver` / `build_ode_solver_distributed`): the default is the
fully-implicit `RungeKutta(nls, ls, dt, :SDIRK_2_2)` (L-stable 2nd-order, diagonally implicit), which
is robust in the stiff deep-water regime; `:theta` selects Crank–Nicolson. Driver kwargs
(both drivers): `solver_type=:sdirk` (default), `tableau=:SDIRK_2_2`, `nl_iter=50`, `nl_tol=1e-6`
(production; convergence/physical-reproducibility tests pin `1e-8`), distributed `ls_maxiter=1000` /
`krylov_m=100` / `ls_rtol=1e-9` (the GMRES solve is kept accurate so Newton gets good steps),
`print_every`, `check_every=50`, `check_tol=1e-8`. The distributed run scripts expose these as
`LFEM_SOLVER/TABLEAU/NL_ITER/NL_TOL/LS_MAXITER/LS_RTOL/LFEM_KRYLOV_M` env vars.

**`krylov_m` vs `ls_maxiter` — two different bounds (do not conflate).** `GMRESSolver`'s first
POSITIONAL argument is `m`, the stored **Krylov basis size** (a *memory* bound: it allocates `m+1`
distributed vectors **and a dense local `(m+1)×m` Hessenberg per rank**, up front in
`get_solver_caches`); `maxiter` is a **keyword** giving the iteration budget (a *time* bound), whose
library default is **100**. The distributed factory therefore builds
`GMRESSolver(krylov_m; restart=true, maxiter=ls_maxiter, …)`. `restart=true` is load-bearing: the
library default `restart=false` lets the basis *grow* past `m` via `expand_krylov_caches!`, i.e.
unbounded memory. Passing the iteration cap positionally (the pre-2026-08-05 bug) reserved a
2000-vector basis — 139 MB/rank where 5.6 MB was needed — while silently capping every solve at 100
iterations, so `ls_rtol` was never reached. Symptom to watch for: **`gmres=` pinned at exactly the
same number every step**, with Newton needing 8–24 iterations instead of 3–5.

**Runtime monitoring** (`src/monitor.jl`, serial + distributed):
* `SolverMonitor` — a transparent `NonlinearSolver` wrapper (pass via `monitor=`) that harvests per
  step: Newton iterations, initial→final residual, convergence flag, GMRES iteration counts
  (distributed), and nonlinear-solve wall time. For Runge–Kutta it accumulates over the stages —
  `nl_iters` is the SUM over stages and `ncalls` is the stage count, both reported.
  It also tracks **linear-solver saturation**: `lin_min`/`lin_max` over the step's stages and a
  `lin_sat` flag raised when a solve reaches the iteration budget, which prints a WARN. That is the
  `gmres=100`-pinned signature that hid the `ls_maxiter`/`krylov_m` bug for months — it is now
  impossible to miss. The banner is likewise built from the values **read back out of the
  constructed `GMRESSolver`/`NewtonSolver` objects** (`ls.m`, `ls.log.tols.maxiter`, …), not from
  the caller's kwargs, so a mis-passed argument shows on line 1 of the log.
* **Field diagnostics** (`RunDiagnostics`, `build_run_diagnostics`, `field_diagnostics`) — sampled
  every `diag_every` steps (default `= print_every`; `−1` disables) and written to
  `<output_dir>/diagnostics.csv` as well as the step line. Measured overhead: **within noise**
  (−0.8 % at every step, +2.2 % at every 10th, on a 4669-DOF case). Contents:
  * **where** `max|η|` sits (`x_at_max`) and its split into **interior vs damped zone**
    (sponge ∪ relaxation) — the discriminator that identifies the open-boundary mode (§8);
  * `max|u|` and hence `|u|/|η|` — a genuine wave gives `ω/tanh(kd)` at the surface nodes
    (`ω/(kd)` for a depth-averaged probe), the η-dominated mode far less;
  * mass `∫η` and energy, with drift against the **t=0** state (baselines are seeded from `u0`, not
    from the first sample, so a forced run's gain is visible);
  * per-rank RSS, current and peak (`/proc/self/statm`, `Sys.maxrss()` fallback), reduced over
    ranks — the missing datum in every OOM post-mortem.
  * a **relative divergence guard** replacing the blind absolute `emax > 1e4`: the run aborts at
    `div_factor · eta_ref` (default 20×), with `eta_ref` inferred from the forcing (`A_wave`, the
    sea state's `Hs`, or the peak of `η₀`) via `resolve_eta_ref`. The 2026-08 archived run that
    grew from 1e-3 m to 44 m over 8 h would have been stopped in minutes.
  Driver kwargs (both drivers): `diag_every`, `diag_csv`, `eta_ref`, `div_factor`;
  env vars `LFEM_DIAG_EVERY`, `LFEM_DIV_FACTOR`. Read a run with
  `julia --project=. examples/inspect_run.jl <output_dir>` (stdlib-only, works on cluster output).
* `ResidualChecker` + `check_residuals` — every `check_every` steps the governing equations are
  reassembled independently through a separate code path: (a) the θ-scheme discrete residual (which
  should sit at the Newton tolerance, ~1e-13, and prints a WARN otherwise) — this self-consistency
  check is meaningful only for a single-stage scheme, so it runs **only under `solver_type=:theta`**
  (`res_theta=NaN` and no θ-WARN under SDIRK); (b) the instantaneous PDE residual (= the O(Δt)
  time-discretisation error), which is reassembled for any integrator.
* Both time loops print a solver-configuration banner, a per-step line
  (`step, t, eta_max, NL its, r0→r, [conv], gmres, solve s, ETA`), timed VTK writes,
  non-convergence warnings, and an end-of-run summary.

---

## 8. Solver conventions and key rules

* **`fe_order ≥ 2`** — Q1 elements zero the `R_P` dispersion term (they cannot represent the
  second-order pressure coupling), so the horizontal FE order must be at least 2.
* **Solid-wall Dirichlet BCs must include the corner tags** — omitting them leaves the corner DOFs
  unconstrained and the run diverges; `build_fe_spaces` includes them.
* **IC-release problems need `x_wall_bc=true`** — a free x-wall together with the dispersion term forms
  a spurious-forcing mode that an initial perturbation excites directly, so closed-basin IC cases run
  with solid x-walls.
* **`A_wave ≤ 0.001`** keeps the fully nonlinear runs stable over long integrations.
* **Sponge strength saturates: past `μ_max ≈ 5ω`, width is the lever, not strength**
  (measured 2026-08-06, `test/local/test_sponge_1d.jl`). The envelope inside the sponge follows
  the quadratic-μ damping law `ln a ∝ −μ_max(x−x_R)³/(3w²c_g)` with excellent fidelity —
  `R² = 0.9995 / 0.9866 / 0.9835` at `μ_max = 5 / 20 / 40` — but the *rate* does not scale with
  `μ_max`: `rate/μ_max = 0.639 / 0.487 / 0.451`. The law's constant is derived from
  `ω → ω + iμ ⇒ k + iμ/c_g`, valid only for `μ ≪ ω`, and `μ/ω = 1.27 / 5.09 / 10.19` at these
  settings. Beyond that the sponge stops being a slowly-varying absorber and becomes an evanescent
  barrier, so extra stiffness buys progressively less absorption. Reflection is small and does
  **not** grow with strength (2.1 / 2.9 / 3.1 %, and 1.0–1.4 % on the settled re-run).
  **Corollary — the constant cannot be validated here**: probing the weak-damping limit
  (`μ_max = 1`, `μ/ω = 0.25`) *diverges*, because a sponge that weak no longer holds the
  η-dominated boundary mode. The WKB regime and open-boundary stability are mutually exclusive at
  a free outflow.
* **The open-boundary mode is η-dominated — the sponge MUST damp η, not just velocity.** At a free
  (`x_wall_bc=false`) outflow the model supports a boundary-localised mode that carries large surface
  displacement with little velocity, so a velocity-only sponge (`∫ μ (W⋅𝗠U)`) is structurally blind
  to it and **no value of `mu_max` absorbs it**. The `+∫ μ q η` continuity term (giving
  `ηt = … − μη`) is what absorbs it; it is not optional. **How to recognise it in a run** (diagnosed
  on `output/small_irregular_nonlinear_full_flat_Hs0.2_M2`, a job that predated the fix and NaN'd at
  t=13.8): bin `max|η|` by `x` from the VTK output — the mode peaks **at the last boundary node and
  decays exponentially inward**, is spatially **disconnected** from the incident field (60× its
  immediate upstream neighbour while the wave front is still mid-domain), grows with an e-folding
  time comparable to `Tp`, and has `|u|/|η| ≈ 0.4` against ≈ `ω/kd` (≈0.9) for a genuine wave and
  ≈1.9 in the incident train. Once it exceeds the sea state, `eta_max` in the log stops reporting the
  physics and reports the mode. Sponge width matters too: it must cover the **longest** component
  (`kd_min` ⇒ λ_max), not the peak — 12 m is under one wavelength for `kd=0.9` at `d=3.5`.
* **`B_stored = −B̃` (the stored dispersion tensor ≤ 0)** and the explicit `(−1)` factors in the `R_P`
  and slope-pressure terms give those terms their correct sign — they are load-bearing.
* **Distributed linear solve = `NewtonSolver(GMRESSolver(Pr=Jacobi))`** (GridapSolvers) — a direct LU
  factorisation does not scale to the partitioned matrices at cluster size.
* **Global max|η|** is reduced from `own_values` + `reduce(max, …; init=0.0)` (the ∞-norm of a PVector
  is not usable directly here).
* **MPI:** launch with `~/.julia/bin/mpiexecjl` so the runtime matches the MPI build; the first
  full-FEM compile takes tens of minutes; Julia buffers stdout to files (use `flush`); `MPI_Finalize`
  prints a benign OFI error and exits 143.
* **Stack (reconciled 2026-08-05):** this package's environment **and the cluster** run
  **Gridap 0.20.x, GridapDistributed 0.4.17, GridapSolvers 0.7.1**, PartitionedArrays 0.3.5,
  MPI 0.20.26. `[compat]` admits `Gridap = "0.19, 0.20"` and `GridapSolvers = "0.6, 0.7"` because the
  two stacks were **measured equivalent** (see below), not as a hedge. The parent environment `../`
  is a *different* environment for the legacy 1D/2D solvers and stays on Gridap 0.19.11 /
  GridapSolvers 0.6.2 — it does not need to match, and the previous claim that the two envs were
  "pinned to the same versions" was stale. Run Julia via the `julia-mcp` tool. Transient API:
  `TransientFEOperator(res,jac,jac_t, U,V)`, `res(t,u,v)` with `∂t(u)`, `TransientCellField` in
  `Gridap.ODEs`, `solve(solver,op,t0,tF,u0)` iterator yields `(t,uh)`.
* **Gridap minor equivalence (measured 2026-08-05).** `test_basic.jl` run in two pinned
  environments gives **identical** results — `max η = 0.00410 m`, `408` Newton iterations
  (`3.40`/step), `gauge amp = 0.00212 m` — under *both* `Gridap 0.19.11 + GridapSolvers 0.6.2` and
  `Gridap 0.20.8 + GridapSolvers 0.7.1`, and the package precompiles cleanly on both. The Gridap
  minor therefore has **no effect** on this solver's results, and is *not* the explanation for the
  reference constants in `test_basic_distributed.jl`, which were traced instead to `95f5ec6` and have since been re-measured.
* **Identifying the cluster's versions without shell access:** Julia package directory slugs are
  content-addressed and stable across machines, so a cluster traceback names its versions. The OOM
  logs reference `GridapSolvers/WuCdi` = **0.7.1**, `PartitionedArrays/MVmxR` = 0.3.5, `MPI/pvbg6` =
  0.20.26 — which is how the cluster stack above was established.

**Field reconstruction** (`src/reconstruct.jl`): the eliminated vertical kinematics are rebuilt for
output. `w` is the exact modal vertical-velocity FE; the total pressure is `p = ρgH(1−σ)`
(hydrostatic) `− ρ Σⱼ div(u̇ⱼ) d² Π³ⱼ` (non-hydrostatic, linear flat-bed, `u̇` by backward FD), where
`Π³_j = ∫_σ^1 φ_j_int` vanishes at `σ=1` so the reconstructed non-hydrostatic pressure satisfies the
free-surface condition `p_nh(1)=0` exactly.

---

## 9. Conventions for editing here

* Keep `main.tex` the single source of mathematical truth; `LFEM_Gridap.md` mirrors its notation. If
  you change the math, update both, and note any solver-code discrepancy in `LFEM_Gridap.md` §9.
* When quoting solver tensor names versus paper names, always cross-reference the §4 table — the
  notation mismatch is the most common source of confusion.
* `test_equivalence.jl` cross-checks the residual against the independent per-layer implementation in
  `../LFE-M_2D_solver/` by comparing the assembled virtual-work scalars on the same analytic state
  (interpolated into both layouts). Do not map DOF vectors between the two layouts — compare the
  layout-independent scalars `dot(get_free_dof_values(v_interp), assemble_vector(r,V))`.
