> ## ⇨ START HERE: `building_files/HANDOVER.md` (read its §0 first — it retracts §1–§2)
> Solver state, method lessons, and the document map. **Its §1–§2 describe a bug that is now FIXED
> and whose diagnosis was WRONG**; §0 records the retraction. Then read
> `building_files/RESIDUAL_TERM_AUDIT_PLAN.md` (what the defect actually was and how the residual is
> now audited) and `building_files/MMS_NONLINEAR_PLAN.md` (where the nonlinear verification stands).
>
> **One-line status (2026-08-16):** the **linear** core is VERIFIED on both flat and variable
> bathymetry (analytic MMS at theoretical order, `p_η=3.000`/`p_u=4.000`), and the **nonlinear
> flat-bed core** is verified too (`2.996`/`3.995`). The `regime=:linear, flat_bed=false` bug is
> fixed: it was a **residual** defect — the `𝓐/𝓚` slope package was assembled **twice** — not a
> Jacobian defect as previously recorded. Still open: the nonlinear **variable-bed** MMS (Newton
> stalls on the quasi-Newton Jacobian) and the `𝓝` forcing tiers (nested-AD failure, diagnosis
> **unconfirmed**). Gridap is a **fork** (`Elmanyer/Gridap.jl` @ `fix-transient-multifield-ad`, one
> commit on `v0.20.8`) making transient-multifield **AD work** — see `building_files/AD_ISSUE.md`
> and the standing guard **gate A0 of `test/test_jacobians_ad.jl`**.
>
> **The specification the residual is audited against** is now the term classification in
> `LFEM_discretisation/NumericalImplementation/GridapImplementation.tex`
> §`subsec: term classification` (`tab: term classification`): every term of the full model tagged
> by amplitude order × bed-slope class × activation condition on each of the three switches.

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
| `LFEM_discretisation.zip` (LaTeX project) | The authoritative LaTeX derivation, a multi-file project (compiles with pdflatex). **Structure** (Overleaf layout, adopted 2026-08-04): `main.tex` (preamble/macros) inputs the chapter files — `SigmaEulerModel.tex` (chapter `σ-Euler Model`, two sections: governing equations, σ-transformation), then the wrapper `\chapter{Vertical Multilayer Discretisation}` declared in `main.tex` whose five **sections** live in `VerticalFESemiDiscretisation/` (`VerticalFEapprox`, `wDerivation`, `pDerivation`, `VerticalProjection`, `VerticalSemiDiscreteSystem` — the last incl. the **flat-bed reduction** of the full nonlinear model as subsections), then the standalone root-level chapters `LinearModel.tex` and **`StokesWaveFourierAnalysis.tex`** (the analytical theory chapter, mirroring Yang & Liu §3 for an **arbitrary-order** vertical basis: the Stokes–Fourier hierarchy; the dispersion functional `R(μ)=Φᵀ(M+μ|B|)⁻¹Φ` and four basis-independent properties — basis/space invariance, the variational bound `Cm ≤ Ce`, the `[Nσ/(Nσ+1)]` Padé-type structure, exact shallow-water + `O((kd)²)` consistency; group velocity and shoaling gradient; the **second-order bound-harmonic transfer function** with its hand-assembled forcing, reproducing Stokes to 1% and the published LFE-2 nonlinear range `kd=6.0`; third-order solvability + wave–current outlook; and the **vertical grid optimisation** — minimax formulation, the design rule `Δσ_top ≈ 2.94/kd_max`, and "grading beats order" at fixed DOF), and finally `NumericalImplementation/` — `GlobalResidual`, **`GridapImplementation.tex` (§8** — stacked layout, residual implementation, 𝓛/𝓝 pressure treatment, the `regime`/`nl_pressure`/**`flat_bed`** model-setups subsection, wave generation/sponge/BC incl. Dirichlet boundary generation with the `:model`/`:airy` polarization) and **`ValidationTests.tex` (§9** — the validation report incl. the BC-generation gates, Goda–Suzuki, spectral fidelity). **Two chapters were substantially extended 2026-08-16:** `GridapImplementation.tex` §`subsec: term classification` adds the **term-by-term anatomy** — the full model expanded with `H=h+η` distributed, 25 tagged terms (C1–C4, M1–M21), the classification table, the three activation predicates and the **assembly invariant**; and `ValidationTests.tex` §`sec: mms` is restructured into **four model subsections** (linear/nonlinear × flat/variable bed) with `nl_pressure` subsubsections, sharing one parent strong form `eq: mms strong general` from which all four are restrictions. Sign convention: `R_P` is a positive integral **subtracted** in the global sum, uniformly with `R_lin`/`R_nonlin`. All prior standalone `*Improved.tex`/`.md` section drafts are integrated into this zip. The document is a **`report`** (chapters + table of contents); the `§8`/`§9` labels above are the corresponding chapters. **Conventions** (enforced 2026-08-04, verified mechanically): structural labels carry a prefix matching their level — `\label{chap: …}` / `\label{sec: …}` / `\label{subsec: …}`, always `prefix: name` with a space; every chapter/section/subsection cross-reference is written `\S\ref{…}` (no `Section~\ref{}`/`Chapter~\ref{}` prose forms; ranges are `\S\ref{a}--\S\ref{b}`). Header numbering is the **report-class default**; the Roman-chapter/custom scheme is retained commented-out in `main.tex`. Both the `.zip` and an unzipped working copy `LFEM_discretisation/` live in `building_files/` — **the folder is the working copy; edit it and re-zip to sync** (the `.zip` is only an export/import channel for Overleaf; if the two disagree, the folder is authoritative). The `.cls`/toolchain needs `newtx`, so the document does not compile on this dev machine; `StokesWaveFourierAnalysis.tex` requires `multirow`. |
| `LFEM_Gridap.md` | Synthesis of the derivation §1–§8 leading to the single scalar residual, in the LaTeX notation; §9 bridges that notation to the solver code. |
| `algebraic_residual_math.md` | Operator reference: how each §8 residual term is written with native Gridap tensor ops (no MultiField decomposition, no vertical-index loops) — the `L`/`N` pressure stacks, the leading-pressure `R_P`, IBP of second-derivative terms, and the verified Gridap operator table. |
| `building_files/HANDOVER.md` | **START HERE, but read §0 first.** Solver state, method lessons, document map. §0 (2026-08-15) **retracts §1–§2**: the bug they describe is fixed and their diagnosis was wrong (it was a residual double-count, not a Jacobian defect). Kept for provenance of how it was hunted — including two localisations later overturned. |
| `building_files/RESIDUAL_TERM_AUDIT_PLAN.md` | **The residual audit.** Uses `tab: term classification` as a *specification*: maps every `∫` in `global_residual`/`jacobian_u`/`jacobian_u_t` to a tagged term, states the defects found, fixes them, verifies. Contains the **assembly invariant** (each classification row must have exactly one consumer, guarded by the *conjunction* of its three activation conditions) and the execution record with all measured numbers. |
| `building_files/MMS_NONLINEAR_PLAN.md` | **The nonlinear-MMS plan and its execution record.** Design of the single parent forcing evaluator, the forcing-level gate set (N1–N7), the study order, and §5b: what was delivered, Model 3 verified, and the two blockers then open. **⚠ Its §5b B2 diagnosis (Model 4 needs more Newton iterations) was disproved on 2026-08-17** — the cause was an `O(1)` block missing from `∂R/∂u̇`, now fixed, and Model 4 is verified. The `𝓝`-tier blocker B1 stands, with its own diagnosis flagged unconfirmed. |
| `building_files/archive/` | Executed/superseded plans (`LOCAL_VALIDATION_PLAN`, `PACKAGE_MIGRATION_PLAN`, `SPONGE_WAVEGEN_PLAN`, `LFEM_runs`, `SOLVER_ASSESSMENT_2026-08`, `MMS_ANALYTIC_PLAN`) + a README saying where each outcome now lives. Provenance only — not current instructions. |
| `building_files/AD_ISSUE.md` (**MOVED** from the repo root; note `building_files/` is **gitignored**, so this analysis is NOT under version control) | The Gridap transient-multifield AD bug: root cause (`Tuple` vs `Vector` in `time_derivative`, NOT `ForwardDiff.Dual`), the one-line fix, and the verification plan. The tracked guard that the fork stays effective is **gate A0 of `test/test_jacobians_ad.jl`** (added 2026-08-17) — an instant `hasmethod` check for the fork's Tuple-argument constructor. Before it, nothing in the repo exercised the fork at all. |
| `DESIGN_RECORDS.md` | Consolidated **historical design records** for the completed features — algebraic residual + Jacobians, package layout, distributed solver, nonlinear-pressure completion, boundary wave generation, periodic-y BC, the `flat_bed` switch, and the production sea-state scripts. Provenance only (*why* the code is shaped as it is); the code, this `CLAUDE.md`, and the LaTeX derivation are authoritative. |
| `src/` (`GridapLFEM.jl` + submodules) | **The self-contained serial + distributed solver package** (`module GridapLFEM`): vertical tensors (incl. `Pcal`), constant-tensor + `Operation` helpers, stacked FE spaces (distributed-safe MultiField dispatch + transient-Dirichlet inflow variants), the loop-free residual + hand Jacobians (the same code runs distributed — `Operation` is forwarded for `DistributedCellField`; **`∂R/∂u̇` is EXACT in both regimes since 2026-08-17**, `∂R/∂u` deliberately quasi-Newton in the nonlinear branch — both verified matrix-by-matrix against AD by `test/test_jacobians_ad.jl`), the time loops (default fully-implicit `RungeKutta(:SDIRK_2_2)`, `:theta` Crank–Nicolson selectable via `solver_type`; sequential LU+Newton / distributed GMRES+Jacobi+Newton), per-component VTK (`eta,u1x,u1y,…`) plus reconstructed `w_s<σ>`/`p_s<σ>` fields (`reconstruct.jl`), and `waveinput.jl` (Dirichlet boundary wave generation + WaveSpec.jl coupling: component tables, `:model`/`:airy` polarizations, ramp, relaxation zone). **`mms.jl` — the analytic (verification) MMS**, whose independence from `problem.jl` is the whole point and is enforced by a grep gate: `mms_forcing_stage1` (Model 1 closed form) and **`strong_residual_model`** — ONE parent evaluator for all four models, with `regime`/`flat_bed`/`nl_pressure` applied at one control point each, mirroring `resolve_physics` so a forcing/model mismatch is unrepresentable. `mms_forcing` is the single entry point (memoised one point deep; `H>0` guarded). Drivers: `setup_and_run`, `setup_and_run_distributed`, and `run_mms_case`/`run_conv_study` (`mms_driver.jl`, which take the same three physics symbols plus `nl_iter`). |
| `test/` (27 test files + **`runtests.jl`** + `test/cluster/` + `test/local/`) | **THE WHOLE SEQUENTIAL SUITE, THE DISTRIBUTED TRIO AND THE MMS RATE STUDIES WERE ALL RE-RUN 2026-08-17 AFTER TWO SOLVER FIXES — these are measured numbers, not carried-over ones.** **Sequential: 21/21 files PASS** via `runtests.jl` (`test_mms_forcing` 5/5, `test_mms_forcing_nonlinear` 10/10, `test_linear_newton_gate` 10/10, `test_vertical` 15/15, `test_primitives` 9/9, `test_basic` 6/6 with references **bit-identical** (max η 0.00410, gauge 0.00212, Newton 240), `test_dispersion` 1/1, `test_nlpressure` 9/9, `test_sloshing` 2/2, `test_conservation` 2/2, `test_dispersion_curve` 9/9, `test_selfconsistency` 3/3, `test_convergence` 2/2, `test_vertical_profile` 7/7, `test_energy` 3/3, `test_dispersion_nonlinear` 3/3, `test_shallow_water` 6/6, `test_waveinput` 30/30, `test_bc_generation` 11/11, `test_bc_spectrum` 8/8; `test_equivalence` RETIRED and correctly not counted). **Distributed: 13/13 gates PASS** on 4 ranks — `test_basic_distributed` 6/6 (seq↔dist 2.5e-7), `test_nlpressure_distributed` 3/3 (1.6e-5), `test_bc_generation_distributed` 4/4 (6.6e-7). **⚠ All three distributed REFERENCE CONSTANTS were wrong and are corrected** — see the stale-reference entry under "Recently completed". **VERIFICATION TIER (the only tests that can detect a self-consistently wrong residual):** **`test_jacobians_ad.jl` (NEW 2026-08-17) — 8/8 models PASS.** Assembles the hand `∂R/∂u` and `∂R/∂u̇` and compares them ENTRY BY ENTRY against AD of the *same* residual, on a **sloping** bed. Gates the linear branch on **equality** (`0.000e+00`, bit-exact) and the nonlinear branch on how the gap **SCALES WITH STATE AMPLITUDE** — the distinction that matters, because a vanishing gap is the deliberate quasi-Newton choice while an `O(1)` one is a defect. `∂R/∂u̇` is now exact (`0.000e+00`) in all 8; `∂R/∂u` vanishes at order 1.11–1.16. Gate A0 additionally pins the vendored Gridap fork by `hasmethod`. **This test found the `∂R/∂u̇` defect that had stalled Model 4.** `test_mms_convergence_nonlinear.jl` **4/4** end-to-end — Model 3 `p_η=2.996`/`p_u=3.995`, **Model 4 `2.996`/`3.997`**, both at theoretical order at the DEFAULT `nl_iter`. `test_mms_forcing` 5/5 + `test_mms_forcing_nonlinear` 10/10 (forcing gates, no FE solve; the latter's documented "9/9" had **never been real** — the file carried a syntax error and could not parse). `test_mms_convergence` — **FIXED 2026-08-17**, see the two-defect entry under "Recently completed"; now `Q3/Q2` with per-field optima and a mirror isolation guard (G10). **`test/runtests.jl` (NEW 2026-08-17):** the package had no runner at all, so `Pkg.test()` failed and nothing batch-ran the suite — which is how four tests stayed broken for three weeks. Runs each file in its OWN subprocess and takes its verdict from **GATE OUTPUT, never the exit code**: a file emitting no PASS/FAIL lines is reported **BLANK and counted as a failure**. Tiers via `LFEM_TESTS=fast|default|all|<files>`; MPI and `test/local/` are printed as explicit not-covered commands. **`test/cluster/`:** `cluster_conservation` (2 ranks, drift 5.8e-10), **`cluster_selfconsistency`** (renamed from `cluster_mms` — its forcing is the solver's own residual, exactly why `test_mms.jl` became `test_selfconsistency.jl`) + SLURM template. **`test/local/`:** the quasi-1D machinery suite (rest state, sponge, relaxation, boundary modes, `test_2d_reduces_to_1d`), runner `run_local_tests.sh` — **its verdict now also comes from gate output, not exit codes**. ⚠ `test/local/` has **not** been re-run since the 2026-08-17 fixes. |
| `examples/` (+ `validation/`, `distributed/`) | Sequential: `plane_wave.jl`, `ring_wave.jl`, `periodic_plane_wave.jl`, and BC-generation `bc_plane_wave.jl`, `bc_irregular_sea.jl` (JONSWAP, gauge CSV), `bc_directional_sea.jl`; `examples/distributed/` — 6 env-configurable cluster scripts (plane/ring wave, IC hump, bathymetry, `run_irregular_sea_dist.jl`, `run_directional_sea_dist.jl` — sea-state env vars + `build_airy_state()` in `_dist_common.jl`) + README; `examples/validation/` — physical benchmarks (`stokes_harmonics`, `submerged_bar`, `solitary_wave`, `ring_spreading`, `bichromatic_sideband`) + `dispersion_sweep.jl` + `spectral_fidelity.jl` (JONSWAP component-wise amplitude+dispersion transfer) + README; `examples/distributed_small/` — **5 parametric** small-domain (50×20 m) scripts (`run_periodic_plane_small` [interior line source], `run_ring_small` [point source], `run_bc_plane_small` [`:bc_gen` boundary plane wave], `run_directional_sea_small`, `run_irregular_sea_small` [`:bc_gen` WaveSpec sea]) grouped by wave-generation type; the 20 observation/comparison cases are their **launchers** in `run/dist_small/`, each overriding only the env vars that change (regime / nl_pressure / flat_bed→bar / amplitude / period). See the "Current Implementation Stage" small-domain-suite entry. **`examples/local_1d/run_flume_1d.jl`** — the parametric **quasi-1D flume** (narrow domain, `ny≥3`, y-periodic; the solver is structurally 2-D, so this is how a 1-D horizontal problem is posed — it is *not* a 1-D model, see the script header), sequential-with-gauges locally and MPI `(px,1)` on the cluster, `LFEM_WAVE_GEN ∈ {inner,bc,sea}`. **`examples/local_2d/run_small_2d.jl`** — the parametric small 2-D case (25×10 m, 6 ranks as `3×2`), `LFEM_WAVE_GEN ∈ {line,point,bc,sea}`, each case a scaled-down sibling of a named `run/dist_small/` job. **`examples/inspect_run.jl`** — stdlib-only reader that turns a run's `diagnostics.csv` into a health verdict (works on cluster output too). |
| `postprocessing/` (`GridapLFEMPost`) | Self-contained postprocessing library with its own environment (ReadVTK, Plots+GR, FFTW, Interpolations — pinned separately from the solver). Reads VTK (`solution.pvd`/`sol_t_*.vtu`) + CSV → `WaveSimulation` (auto-`regularize!`s the duplicated Q2 node cloud to a Cartesian grid). Modules: `io, probes, spectral, diagnostics, reconstruct, plotting, seastate`. Gauges/DFT/celerity/harmonics/radial/conservation; heatmap/animation(GIF)/Hovmöller/dispersion/profile plots; `seastate.jl` — Welch PSD, JONSWAP target overlay, spectral moments/Hs, zero-upcrossing heights, Rayleigh exceedance (+ `spectral_validation.jl` example). `reconstruct.jl` rebuilds `w(σ)`/`p_nh(σ)` from the stored velocity modes at any σ (analytic σ-basis, Gauss quad, no Gridap; matches solver `w_s` to 4–8%). No dependency on the solver. |
| `WaveSpec.jl/` (repo-vendored package) | Stochastic sea-state synthesis (CMOE-TUDelft; JONSWAP/TMA/… spectra, sampling strategies, angular spreading, `AiryState`). Tracks the **GitHub repository version, not a tagged release** — the release's `change_seed!` reads a non-existent `state.spec` field and breaks every sea-state run; the repo version has the fix. `Pkg.develop`ed in both environments; `using WaveSpec` is re-exported by `GridapLFEM`. The `WaveInput` converter (`src/waveinput.jl`) snapshots seeded amplitudes/phases into plain arrays and re-solves the wavenumbers with the solver's `g` (WaveSpec uses 9.80665). |
| `compile/` (cluster sysimage build) | Builds a precompiled system image so distributed cluster ranks **load** the solver instead of JIT-compiling it (removes the ~30–45 min/rank compile and the associated OOM). `set_preferences.jl` pins MPI.jl to the **system** OpenMPI via `use_system_binary()`; `compile.jl` bakes the deps + traces `warmup.jl`, with an MPI preflight and **`include_transitive_dependencies=false`** — load-bearing: it stops PackageCompiler force-loading `OpenMPI_jll`, whose baked initializer would otherwise `dlopen` the JLL artifact and crash the image at launch (`undefined symbol: opal_single_threaded`). `compile_snellius.sh` runs the chain; full walkthrough + troubleshooting in `compile/README.md`. |
| `run/local/` | **Local launchers (this workstation, ≤6 cores)** — `lfem_local.sh` (the helper: project resolution, a hard rank cap, `lfem_local_run` sequential / `lfem_local_mpi` MPI, exit-143 handling for the benign `MPI_Finalize` OFI error) plus 7 `run_1d_*.sh` (**sequential** — measured faster than any MPI split, see the scaling entry) and 8 `run_2d_*.sh` (12-rank MPI) case launchers, `run_all_1d.sh` (runs the 1-D set side by side), **`bench_solver_config.sh`** (compares preconditioner × `ls_rtol` configurations on one fixed case and reports iterations, wall time *and* `max|η|`, so a cheaper setting is rejected if it changes the answer), and a `README.md` documenting how to read `diagnostics.csv`. Deliberately shares no code with `run/lfem_env.sh`, which is cluster-only (modules + sysimage). The point is a minutes-long feedback loop instead of the multi-hour cluster one. |
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

**Tests** (`test/`, 27 files + `runtests.jl` + `cluster/` + `local/`): see the §1 table for per-file
scores; all re-measured 2026-08-17. **Sequential 21/21 PASS, distributed 13/13 gates PASS,
Jacobian-vs-AD 8/8 models PASS.** Highlights: the **analytic-MMS verification of all four `:none`
models** at theoretical order (the strongest correctness evidence in the repo — the forcing never
touches `problem.jl`, so the error cannot cancel); **`test_jacobians_ad.jl`**, which compares the
hand Jacobians against AD of the same residual matrix-by-matrix and gates the nonlinear branch on
*amplitude scaling* rather than equality (`∂R/∂u̇` exact in all 8; `∂R/∂u` vanishing at order
1.11–1.16) and which found the `O(1)` `∂R/∂u̇` defect that had stalled Model 4; the **linear
one-Newton-iteration gate** on a sloping bed, 10/10; the asymptotic-consistency pair (full-NL⇒Airy
across the band, kd→0⇒√(gd)); BC generation 30/30 + 11/11 + 8/8 (Goda–Suzuki); distributed↔sequential
agreement 2.5e-7 … 1.6e-5. **`runtests.jl`** batch-runs the suite in per-file subprocesses and takes
its verdict from **gate output, never exit codes** (a file printing no PASS/FAIL is BLANK = failure).
`test_equivalence` is **RETIRED** and no longer `error()`s (its external per-layer reference predates
the completion of the weak form — it lacks `R_P` — so its result measured the reference's age).

**Production cluster scripts** (`examples/distributed/`, 6 total): including `run_irregular_sea_dist.jl`
and `run_directional_sea_dist.jl` (env-configured JONSWAP ± spreading via `build_airy_state()`;
smoke-validated on 2 ranks, governing-equation check at 4e-12 under transient Dirichlet). Validation
runs on record: `spectral_fidelity.jl` at 60 Tp — Hs transfer 1.023, dispersion 14/14 within 5% of the
model wavenumber, incident amplitudes 9/14 within 10%; `bc_irregular_sea.jl` — near-inflow Welch Hs
ratio 0.979.

**Postprocessing** (`postprocessing/GridapLFEMPost`): VTK/CSV → analysis/plots, from-modes
`w(σ)`/`p_nh(σ)` reconstruction, and the sea-state module (Welch PSD, JONSWAP target overlay, Hs,
Rayleigh exceedance), validated against solver output.

**Docs**: `building_files/LFEM_discretisation.zip` (authoritative LaTeX derivation — **fully revised
2026-08-16**, all eleven compiled chapters read end to end; §8 the Gridap implementation incl. the
`flat_bed` model-setups subsection, the **term classification** that now serves as the residual's
specification, and Dirichlet generation; §9 the validation report incl. BC gates and the **four-model
MMS section**; compile-checked, 139 pp, 0 undefined refs), `LFEM_Gridap.md`,
`algebraic_residual_math.md`, `boundary_wave_generation.md`, the consolidated `DESIGN_RECORDS.md`, and
the two current plans `RESIDUAL_TERM_AUDIT_PLAN.md` / `MMS_NONLINEAR_PLAN.md`.
**Reminder: `building_files/` is gitignored**, so none of the above is version-controlled.

## Current Implementation Stage

**Working.** The 2-D LFE-M solver is feature-complete in **both** serial and distributed forms: the
stacked loop-free residual + hand Jacobians, the full nonlinear physics (advection, the complete
leading pressure `R_P`, all eight 𝓝 nonlinear-pressure components), the SDIRK/θ time integrators,
wavemaker/sponge/wall/periodic BCs, and Dirichlet boundary wave generation with WaveSpec coupling.

**Suite status, all re-measured 2026-08-17 after two solver fixes** (not carried over):
**sequential 21/21 files PASS** (via the new `runtests.jl`), **distributed 13/13 gates PASS** on
4 ranks, **Jacobian-vs-AD 8/8 models PASS**, **nonlinear MMS 4/4 PASS** end-to-end. Includes
asymptotic consistency (full-NL⇒Airy, kd→0⇒√(gd)), BC generation 30/30 + 11/11 + Goda–Suzuki 8/8,
distributed↔sequential agreement 2.5e-7 … 1.6e-5, and the linear one-Newton-iteration gate 10/10.
Not re-run since the fixes: **`test/local/`** (51 gates) and the 13 local observation cases.

Note the distinction this section then draws: **"the suite passes" is not "the model is verified"**
— and 2026-08-17 is the sharpest illustration on record. The suite reported 21 green files while the
nonlinear gravity term was missing its `−η∇h` half, because the one sequential test able to see that
configuration asserts only BOUNDEDNESS. For what is actually verified see the scope table below,
now **four of four `:none` models**.
Postprocessing (`GridapLFEMPost`) and the authoritative LaTeX derivation are in place and
compile-checked.

**What "validated" and "verified" mean here — read this before claiming correctness.** Most of the
suite is **self-consistency**: each test compares the solver against an analytical or physical
expectation *using the solver's own residual*. That is strong evidence — a wrong residual would have
to be wrong in a way that preserves the dispersion relation, the shallow-water limit, the Airy
vertical profile, the rest state, mass conservation, and the `O(A²)` ordering of every physics tier
simultaneously — but it is not verification: such a test cannot detect a residual that is
self-consistently wrong, because the error cancels.

**Verified scope — as of 2026-08-17, ALL FOUR `:none` models.** The analytic MMS (`src/mms.jl`)
derives its forcing from the governing equations without touching `problem.jl` (enforced by a grep
gate), so the error does **not** cancel, and reaching the theoretical order of accuracy certifies
that the discretised operator is the intended one. Measured, `Q3/Q2`, 1-D static unless noted:

| Model | `regime` / `flat_bed` / `nl_pressure` | `p_η` (opt) | `p_u` (opt) | |
|---|---|---|---|---|
| 1 | `:linear` / flat / `:none` | `p_e+1` | `p_u+1` | ✅ verified 2026-08-12 |
| 2 | `:linear` / **variable** / `:none` | **3.000** (3) | **4.000** (4) | ✅ verified 2026-08-15 |
| 3 | **`:nonlinear`** / flat / `:none` | **2.996** (3) | **3.995** (4) | ✅ verified 2026-08-16 |
| 4 | `:nonlinear` / **variable** / `:none` | **2.996** (3) | **3.997** (4) | ✅ verified 2026-08-17 |
| 3–4 | `:nonlinear` / any / `:native`,`:full` | — | — | ❌ forcing not available (B1) |

Model 2 was additionally confirmed transient (`2.999`/`3.998`) and in 2-D (`3.000`/`3.963`).
**So the verified scope is the complete `:none` model over arbitrary bathymetry** — `H`-weighting,
advection, the full three-component leading pressure, the `O(ε²)` surface-slope package and the
bed-slope terms. Only the `𝓝` tiers (`:native`, `:full`) remain unverified, because their MMS
forcing is not available (blocker B1); for those the self-consistency caveat above applies in full.
Say *"the `:none` models are verified"*, never bare *"the residual is verified"*.

**⚠ Model 4 took TWO fixes to get here, and neither was findable by the rest of the suite.**
It had never completed a convergence study before 2026-08-17 — it always stalled in Newton first —
so the sequence matters: (i) `∂R/∂u̇` was missing the `𝓐/𝓚` package, an **O(1)** omission that made
Newton converge to the wrong fixed point (found by `test_jacobians_ad.jl`'s amplitude-scaling gate);
fixing it made the study *runnable* and immediately exposed (ii) the nonlinear gravity branch was
missing the `−η∇h` half of its own integration by parts, which pinned `p_u` at **0.00** with `e_u`
converging to a constant `4.400e-4` while `p_η` held its optimal 2.99. Fixing (ii) gives the row
above. **Neither defect was reachable by any self-consistency test**: the gravity term was absent
from the residual *and* from its own Jacobian, so AD agreed with the hand Jacobian throughout and
every check comparing the solver against itself cancelled the error identically. This is the
clearest demonstration in the repo of why the analytic MMS exists.

**Recently completed (this branch).**
- **🔑 SECOND RESIDUAL DEFECT: the nonlinear gravity branch was missing the `−η∇h` half of its own
  integration by parts** (2026-08-17, `src/problem.jl`). Found by the analytic MMS the moment
  Model 4 became runnable, and fixed in `global_residual` + `jacobian_u`.
  * **The maths.** Gravity is assembled in IBP energy form. The identity is EXACT:
    `∇((H²−h²)/2) = H∇H − h∇h = H∇η + η∇h`, hence `H∇η = ∇((H²−h²)/2) − η∇h`, so
    `∫ g Φᵢ(H∇η)·vᵢ = −∫ (g/2)(H²−h²) Φ·(∇·v) − ∫ g η Φ·(v·∇h)` — **two** pieces. The linear branch
    (same identity with `H→h`) had both; the nonlinear branch had only the first. Because the second
    piece is *identical* in the two regimes, it is now assembled ONCE outside the branch.
  * **The signature, and why it is worth memorising:** `e_u` converged to a CONSTANT `4.400e-4`
    (rate `−0.009`, `−0.001` over three levels) while `e_η` held its optimal `2.997`/`2.990` on the
    same mesh sequence. One field optimal and the other pinned is the analytic-MMS fingerprint of a
    genuinely wrong operator — and NOT the saturation trap of §8, because the co-refining `e_η`
    proves isolation from within the same run.
  * **Localised by 3-case bisection**, not by reading code: (A) sloping bed `p_u=−0.001`;
    (B) *same* `flat_bed=false` code path with `a_b=0` so `∇h≡0` numerically → `p_u=3.993, 3.998`;
    (C) `flat_bed=true` control → **bit-identical to B**. One variable differed between A and B, so
    the defect was in the ∇h VALUES, not the code path. (C≡B also independently confirms the
    `flat_bed` switch is sound.)
  * **Result:** Model 4 `p_u` **−0.009 → 3.995**, `e_u` at `nx=8` **4.37e-4 → 1.29e-5** (34×), and
    the sloping-bed `e_u` now matches the zero-slope value (1.29e-5 vs 1.32e-5) — as it must once the
    operators agree. `p_η` unchanged at 2.997. **Verified scope now covers all four `:none` models.**
  * **⚠ NO SELF-CONSISTENCY TEST COULD HAVE FOUND THIS.** The term was absent from the residual AND
    from its own Jacobian, so AD agreed with the hand Jacobian throughout, `test_basic` passed, and
    dispersion/conservation/sloshing all passed. Writing `R = R_true + E`, every check that compares
    the solver against itself cancels `E` identically. Only a forcing derived independently of
    `problem.jl` exposes it.
  * **Flat beds are bit-unchanged** (`dhx/dhy` are already zeroed under `flat_bed`), verified
    digit-for-digit: Model 3's three levels are IDENTICAL pre- and post-fix
    (`1.321070e-05`, `8.299700e-07`, `5.194051e-08`).
  * **🔴 THE COVERAGE GAP THAT LET IT SURVIVE — worth more than the fix itself, and it is NOT
    simply "no test ran that configuration".** Most of the suite cannot reach the term at all:
    `test_sloshing` is `:linear`; `test_shallow_water` is `flat_bed=true`; `test_conservation`,
    `test_energy` and `test_basic` use a CONSTANT `h_bathy` (so `∇h=0` numerically);
    `test_linear_newton_gate` uses a sloping bed but is `:linear` by construction.
    **But `test_nlpressure.jl` G3 DOES run `:nonlinear` + `:full` over a tanh bar with
    `flat_bed=false` — precisely the configuration — and still passed 9/9 while the quantity it
    computes moved by 58 %.** Its gate is `emax < 20A`, i.e. BOUNDEDNESS, and a wrong coefficient
    that leaves the run bounded sails straight through.
    **The lesson is therefore sharper than "add a case": A BOUNDS CHECK ON THE RIGHT
    CONFIGURATION IS NOT A VALUE CHECK.** Running the right physics proves nothing if the assertion
    is only that nothing exploded. The one test in the repo that *did* pin a value on this
    configuration — `test_nlpressure_distributed` (`REF_EMAX`) — caught it immediately at
    rel 5.85e-01 the first time it was run in this session; it had simply not been run, because the
    MPI tests sit outside every runner.
    Now covered quantitatively by three things: the analytic-MMS Model 4 study,
    `test_jacobians_ad.jl` rows M4/M6/M8, and that distributed reference. **Do not delete any
    without replacing the coverage.** General rule for guards that are CONJUNCTIONS
    (`nonlinear ∧ ∇h≠0`): the suite needs a case satisfying the conjunction *and asserting a value*
    — conjuncts satisfied separately, or satisfied together but only bounded, both look green.
- **TEST AND RUN FILES BROUGHT BACK IN LINE WITH THE SOLVER — and the audit was MECHANICAL, not
  by eye** (2026-08-17). The method is the reusable part: the accepted keyword set of every
  `function` in `src/` was parsed out, then every call site in `test/`, `examples/` and `compile/`
  was checked against it. That is what found the breakages; reading the files had not.
  * **Seven files could not run at all.** Six examples (`plane_wave`, `periodic_plane_wave`,
    `ring_wave`, `bc_plane_wave`, `bc_irregular_sea`, `bc_directional_sea`) still passed the
    **retired** `linearised=`/`advection=` pair and would raise `MethodError` on the first call —
    a **half-finished migration**: `flat_bed=` and `p_horizontal=` had been added to the same
    argument lists while the old pair was left behind. And `test_mms_forcing_nonlinear.jl`
    contained a **syntax error** (an escaped `\"` inside a `$(...)` interpolation), so the file
    was unparseable and its documented "9/9" had never been measured. Now 10/10.
  * **`test_mms_convergence.jl` asserted an unreachable target.** Its G6 gate demanded the
    theoretical 3 while running the equal-order pairing, which converges at `p`. Moved to
    `Q3/Q2` with each field gated on its own optimum: **`p_η=2.995` (opt 3), `p_u=3.770` (opt 4)**.
    Not one tolerance was loosened — the *pairing* was the defect, exactly as the campaign
    predicted.
  * **The two batch runners judged tests by EXIT CODE.** `test/runtests.jl` did not exist at all
    (so `Pkg.test()` failed), and `test/local/run_local_tests.sh` used `wait $pid`. Both now take
    their verdict from **gate output**, and a file emitting no PASS/FAIL lines is reported
    **BLANK and counted as a failure**. This is the direct, mechanical fix for the failure mode
    that let four tests stay broken for three weeks, and for the `PROGRAM_FILE` guard that let
    `test_dispersion_nonlinear.jl` be recorded as passing twice while executing nothing — that
    guard is also now removed, so the file runs like every other.
  * **`use_ad` had ZERO coverage anywhere in the repository.** The repo pins a *fork* of Gridap
    for the sole purpose of making transient-multifield AD work, and nothing exercised it; the
    pin could have rotted silently and the loss would only have surfaced mid-debugging, when the
    cross-check was most needed. Now guarded by gate A0 of `test/test_jacobians_ad.jl`.
  * **`p_eta` — the FE pairing — was unreachable from either production driver.** It existed in
    `build_fe_spaces` but neither `setup_and_run` nor `setup_and_run_distributed` forwarded it,
    so the error-vs-DOF comparison this file lists as outstanding was *impossible to perform with
    the run scripts*. Now plumbed through both drivers and 14 parametric run scripts as
    `LFEM_P_ETA`, **defaulting to 0 = equal order, so nothing changes until opted into**.
  * **`bench_solver_config.sh` still benchmarked two configurations already measured to fail** —
    Schwarz (`SingularException`) and Gauss-Seidel (**zero steps in 12 h**). Running it as written
    would have burned half a day re-deriving a known negative. They are now opt-in behind
    `BENCH_FAILED_PRECOND=1`, and the ladder brackets the *adopted* `ls_rtol=1e-5` instead of
    searching for it.
  * Also: `cluster_mms.jl` → **`cluster_selfconsistency.jl`** (its forcing is the solver's own
    residual — the same reason `test_mms.jl` was renamed, and leaving it named "mms" reintroduced
    exactly the confusion that rename existed to kill); `test_equivalence.jl` retirement now
    enforced *in the file* rather than only in prose; `flat_bed=true` set on the constant-depth
    validation examples; `run_mms_validate.sh` extended to the nonlinear forcing gates;
    `run_varbed_study.jl` and `examples/local_mms/run_mms_convergence.jl` generalised to select
    any of the four MMS models.
  * **⚠ `src/*.jl` changed (`utilities.jl`, `utilities_dist.jl`, `mms_driver.jl`, `errors.jl`), so
    any existing cluster sysimage is STALE.** `lfem_check_sysimage_freshness` will warn; rebuild
    before the next cluster job.
- **THE WHOLE SEQUENTIAL SUITE WAS ACTUALLY RUN, and four tests were silently broken** (2026-08-16).
  The numbers in the `test/` table below had been *documented* rather than *re-measured*; running them
  found `test_shallow_water` erroring before it executed a single gate, and `test_energy`,
  `test_bc_spectrum`, `test_bc_generation` and `test_convergence` all failing on the L-stable default
  integrator (§8, first rule). All are now repaired **at the cause, never at the threshold** — not one
  tolerance was changed — and all pass. Two method points earned the hard way:
  * *Falsify the obvious hypothesis before acting on it.* These tests mesh at 6 cells/λ, so
    under-resolution was the natural suspect; it was the true cause for `test_shallow_water` and the
    **wrong** one for the other four, where doubling the mesh made things worse. The discriminating
    experiment (refine space and time *separately*) costs one extra run and settles it.
  * *A clean exit code is not evidence a test ran.* `test_dispersion_nonlinear.jl` is wrapped in
    `if abspath(PROGRAM_FILE) == @__FILE__`, so under `include()` it printed three header lines and
    returned "OK" — twice — having executed nothing. Any batch runner must assert on **gate output**
    (`PASS`/`FAIL` lines), not on the absence of an exception. It is the only file with that guard.
- **Full revision of the LaTeX derivation** (2026-08-16), all eleven compiled chapters read end to end.
  Errors of substance found and fixed: **trial/test terminology inverted** in `GlobalResidual`; the
  vertical tensors written as **functions of `σ`** when they are precomputed numbers
  (`VerticalProjection`) — contradicting the chapter's own premise; a **missing normal vector** in a
  divergence-theorem boundary term; the linear regime's depth weight given as **`H→1` instead of
  `H→h`** (`GlobalResidual`) and the linear leading pressure described as the **flat-bed form evaluated
  at a varying `d(x)`** (`GridapImplementation`) — both stale pre-`2579621` descriptions of exactly the
  defect that commit fixed; a `\mathbf{Q}` **symbol collision** inside one chapter; and the blanket
  `O(A³)` claim for the whole `𝓝` package (the `𝓟` block is `O(A²)`). Plus `\Gmesh`/`\Kweight`/
  `\divHu`/`\intcol` macros, section labels for `GlobalResidual`, and explanatory additions — notably
  that `𝓛ⱼ` *is* the time derivative of the `w` expression, which motivates its three-component shape.
  Compiles clean: 0 errors, 0 undefined refs, 139 pp.
  **⚠ None of that is under version control** — `building_files/` is in `.gitignore`, so the LaTeX
  sources, `RESIDUAL_TERM_AUDIT_PLAN.md` and `MMS_NONLINEAR_PLAN.md` are untracked even though the
  commit messages reference them. Worth ignoring the artefacts (`*.zip`, built `*.pdf`) instead of the
  directory.
- **🔑 THE `regime=:linear, flat_bed=false` BUG IS FIXED — and it was a RESIDUAL defect, not a
  Jacobian one** (2026-08-15). `global_residual` had **two** `lin_pressure` consumers: the
  *linearised* `𝓐/𝓚` slope package inside the `if lin` branch (row M14), and the *nonlinear* form of
  the same package at top level, guarded only by `lin_pressure = advection ∨ ¬flat_bed`. Under
  `:linear, flat_bed=false` **both fired** — the package assembled twice, plus four `O(ε²)` terms
  added to a linear model. `jacobian_u_t` implemented the intended single-count model, which is why
  it appeared to disagree. **Fix:** guard the top-level block with the conjunction
  `!lin && prob.lin_pressure` (`src/problem.jl`). Verification: exact `∂R/∂u̇` and `∂R/∂u` (extracted
  column-by-column from the residual — no AD, no FD) now agree with the hand Jacobians to
  **1.7e-16**, was `2.8e-02`; `test_basic` reproduces its references **exactly** (max η 0.00410,
  gauge 0.00212, Newton **240**); `test_nlpressure` 9/9. Full audit:
  `building_files/RESIDUAL_TERM_AUDIT_PLAN.md`.
  **⚠ This retracts a claim that is still written in `HANDOVER.md` §1.3.** "The residual is correct
  because AD converges where hand fails" is an invalid inference — AD converging proves
  residual↔Jacobian *consistency*, never residual *correctness*. **Every `use_ad=true` variable-bed
  result predating the fix was computed against the wrong operator and must be discarded**, incl.
  `e_eta = 7.974294e-05` (its exact equality with the flat-bed value was already a red flag).
- **The term classification is now the residual's specification** (LaTeX
  §`subsec: term classification`). The full model expanded with `H=h+η` distributed, 25 tagged terms,
  each with amplitude order × bed-slope class × activation condition per switch. The table
  **factorises**: `regime` is a truncation in amplitude order, `flat_bed` a projection onto
  `∇h≡0`, `nl_pressure` a component filter on `𝓝` — which is the formal statement that the three
  switches are orthogonal. The `O(ε)` rows are exactly eight, and they *are* the linearised model.
  Two things it surfaces that the un-expanded form hides: the **IBP of the gravity term generates an
  explicit `∇h` contribution** with no counterpart in the strong form, and the advection block is
  **not homogeneous** (M5/M7/M8 are `O(ε²)`, M6/M9 `O(ε³)`, and M8 is a bed-slope term *inside* it).
- **The assembly invariant** — the rule the above defect violated, now stated in the LaTeX and in
  `problem.jl`: *each classification row must have exactly one consumer, and its guard must be the
  **conjunction** of its three activation conditions.* A guard testing only the bed condition, or
  only the amplitude condition, is a defect whenever the physics has a second representation in the
  other regime — which, for the `𝓐/𝓚` and `𝓐/𝓚`-quadratic packages, it always does.
- **Standing gate: a linear problem must converge in ONE Newton iteration per stage**
  (`test/test_linear_newton_gate.jl`, 10/10). Free, needs no reference value, and — crucially — it is
  run **on a sloping bed**. That closes the structural blind spot that let the defect survive the
  whole suite: every bed-slope term vanishes when `∇h≡0`, so no flat-bed test can reach rows C3,
  M3-IBP, M10b or M14. Measured: exactly 1 iteration/stage (θ: 1/step, SDIRK: 2/step), residual
  `1.9e-15`–`5.4e-15`.
- **🔑 `∂R/∂u̇` IS NOW EXACT IN THE NONLINEAR BRANCH — the missing `𝓐/𝓚` block was NOT a benign
  quasi-Newton omission, and it is what stalled Model 4** (2026-08-17, `src/problem.jl`).
  Found by the new `test/test_jacobians_ad.jl`, which compares the assembled hand Jacobians against
  AD of the same residual, matrix entry by matrix entry, and — the decisive part — measures how the
  gap SCALES WITH STATE AMPLITUDE.
  * **Measured before the fix:** `‖Δ(∂R/∂u̇)‖/‖·‖ = 1.11e-2` over a sloping bed, and halving the
    amplitude left it at `1.09e-2` — **order 0.03, i.e. it did not vanish at all**. On a flat bed the
    same gap scaled at order 0.95 and was genuinely benign. The discriminator is the prefactor:
    the `𝓐` half carries `H·∇h`, which does **not** scale with the solution, so on a sloping bed the
    omission is `O(1)` in the EFFECTIVE MASS MATRIX.
  * **Why the old justification was wrong.** "This costs Newton iterations, never accuracy" is valid
    only for omissions of HIGHER ORDER IN AMPLITUDE. An `O(1)` error makes Newton converge to a fixed
    point of the *wrong map*, so it can prevent convergence outright — and no iteration budget helps.
    This is exactly blocker **B2**: Model 4 stalled at `‖r‖=4.8e-8`, and the plan's proposed next step
    (raise `nl_iter` to 200–400) could never have worked.
  * **The fix** assembles the `𝓐/𝓚` package in `jacobian_u_t`'s nonlinear arm, mirroring the residual
    block term for term with `u̇ → du̇` — exact, because the block is strictly linear in `u̇` (its
    prefactors `H, ∇h, ∇H` depend only on `η`). `dL1/dL2/dL3` were hoisted out of `if P_full`, since
    `build_problem_raw` can set `lin_pressure` without `P_full`.
  * **Verified:** `∂R/∂u̇` gap `3.38e-3 → 0.000e+00` (M3) and `1.11e-2 → 0` (M4); M1/M2 stay
    bit-exact (no regression); `∂R/∂u` unchanged at `2.94e-2`, order 1.10 — still deliberately
    quasi-Newton, untouched. **B2 CLEARED: Model 4 now converges at the DEFAULT `nl_iter=50`**, the
    very configuration that errored before, and reaches `p_η = 2.997` (optimal 3).
  * **`test_basic` is structurally blind to this** — it already ran at 2.00 Newton iterations/step,
    which for 2-stage SDIRK is 1/stage, the floor. Its references are unchanged (`max η 0.00410`,
    gauge `0.00212`, Newton 240): the fix changed the iteration path, not the operator.
  * ⚠ **Model 4 is still NOT verified.** With the stall gone the study runs to completion and
    reveals a SEPARATE, pre-existing defect: `p_u ≈ 0` (`e_u` flat at ~4.4e-4 while `e_η` falls at
    exactly 2³ per refinement). Controls bound it — Model 3 (nonlinear, flat) gives `p_u=3.995` and
    Model 2 (linear, variable bed) gives `p_u=4.000` — so the suspect is the **nonlinear ∇h velocity
    terms**, in the residual or the forcing. It was invisible until now only because Model 4 had
    never completed a study. The Jacobian fix made Model 4 MEASURABLE; this gravity fix is what
    made it CORRECT. Both were needed, in that order.
  * `∂R/∂u` remains quasi-Newton by choice (no η-derivative in the pressure packages; `𝓝` absent).
    **Do not extend those omissions without re-measuring every nonlinear reference value**, and use
    `test_jacobians_ad.jl` to check the amplitude scaling rather than assuming an omission is benign.
- **Nonlinear analytic MMS: parent evaluator + Model 3 verified** (2026-08-16). `src/mms.jl` gained
  **`strong_residual_model`** — a *single* evaluator for all four models, with the three restrictions
  applied at one control point each, mirroring `resolve_physics`; `strong_residual_linear` is now a
  thin wrapper over it (agrees with the retained legacy implementation to **2.1e-17**).
  `mms_forcing` gained `nl_pressure`, a one-entry memo (Gridap evaluates `Seta`/`Sx`/`Sy` at the same
  point consecutively — 3 evaluations collapse to 1) and an `H>0` guard. Drivers gained
  `regime`/`nl_pressure`/`nl_iter` passthroughs. Forcing gates 9/9, incl. **Model 4 at constant `h`
  ≡ Model 3 to exactly 0.0** and the `ε²` amplitude scaling at 3.913. Plan and execution record:
  `building_files/MMS_NONLINEAR_PLAN.md`.
- **`test_shallow_water.jl` repaired — and the fix was RESOLUTION, not tolerances** (2026-08-16). It
  had been broken since the physics-interface change (it called `setup_and_run` with the retired
  `linearised=`/`advection=` kwargs and raised `MethodError` before running), so its documented
  "6/6" was stale. After the kwarg repair the dynamic gates failed at 6.47 %. Isolating the two
  discretisations settled it: refining **time** alone changed nothing (−6.47 % → −6.53 %); refining
  **space** alone fixed it (−0.05 %). The test built its mesh at **6 cells per wavelength**, and a
  celerity gate is a *phase* measurement. Fix: 6 → 12 cells/λ, thresholds untouched; now **6/6** at
  0.05 % against a 5 % gate, and *faster* than before (4:23 vs 7:14) because `flat_bed=true` skips
  assembling `∇h` terms that vanish. **Lesson: a physics gate is only as sharp as the discretisation
  feeding it — loosening the threshold to ~7 % would have destroyed its diagnostic value.**
- **LOCAL OBSERVATION CAMPAIGN — 13 cases run and analysed** (2026-08-06 → 08-11). Full results and
  the reasoning behind every number: **`building_files/LOCAL_TESTS_RESULTS.md`**. Read that file
  first if you are picking this up. Headlines:
  * **7 quasi-1-D cases** (60×3 m, dx=0.25 = 16 cells/λ, sequential LU, 16 periods) and **6 small
    2-D cases** (40×15 m, dx=dy=0.417, 98.6k DOFs, 12 ranks, 10 periods). **All 13 completed** —
    bounded, no NaN, no GMRES saturation, no boundary mode, including `nl_pressure=:full` over a bar
    at 10× amplitude.
  * **The physics-tier ladder collapses to the linear result**, as `O(A²)` requires: 0.78 % spread
    1-D, 1.94 % 2-D. Crucially it *separates* advection from nonlinear pressure by two orders —
    advection contributes 0.77 % (1-D) / 1.84 % (2-D), the whole `{1,2,4,5}` hierarchy a further
    **0.013 % / 0.094 %**. **Practical consequence: run production at `nl_pressure=:native` unless
    the amplitude justifies `:full`** — and these cases *cannot validate* `:full`, its effect being
    below everything else's noise.
  * **1-D and 2-D agree to 2–3 %** on the same interior source, through *different linear solvers*
    (LU vs GMRES) at *different resolutions* — a meaningful cross-check of the 2-D machinery.
  * **Kinematics track Airy**: `|u|/|η|` = 2.91–3.59 vs `ω/tanh(kd) = 3.93`; the clean
    single-direction BC trains give the closest values (3.28–3.50).
  * **The two generation mechanisms differ by design and by a factor ≈3.7**: interior Gaussian line
    source gives **3.9 × A** (it radiates both ways and superposes), Dirichlet BC gives **1.05 × A**
    (it prescribes). An interior-source run's `A_wave` is *not* the wave amplitude in the domain.
  * **Three diagnostic-interpretation rules**, each learned from a measurement that misled first:
    (i) never read `growth` without `x_at_max` — a rising `eta_max` with a *moving* `x_at_max` is a
    filling domain, with a *pinned* one at a boundary it is the spurious mode; (ii) `dmp/int` means
    "≫1 is trouble", never "<1 is fine" (healthy band measured **0.29–0.96**; the boundary mode
    gives ~65); (iii) mass drift is an invariant **only** in a closed unforced basin (1.2e-17 there)
    — in a forced run it measures *injected* mass and should scale with `A` (confirmed: 10× the
    amplitude gave exactly 10× the drift).
  * **The `c_g` transit trap is the standing hazard of this regime.** At `kd = 5.5`,
    `c_g = 1.25 m/s`, so filling a flume takes tens of seconds. It produced misleading measurements
    **three separate times**. Always budget `t_settle ≈ (x_sponge − x_source)/c_g + 3T` before
    reading a steady state. The 1-D driver's default duration is now transit-aware (16 periods for
    interior generation, **26** for boundary/sea).
- **Solver tolerances retuned on measurement, and the `nl_tol` question CLOSED** (2026-08-11/12).
  Defaults are now **`ls_rtol=1e-5`, `nl_tol=1e-5`**. Two distinct results, and they are not the
  same kind of result:
  * **`ls_rtol` is free.** On one fixed 2-D case, 1e-9 → 1e-6 cut GMRES **491–515 → 294–314 (−40 %)**
    and 1e-6 → 1e-5 cut it a further **295–313 → 238–253 (−19 %)**, with `max|η|` unmoved in both
    steps and Newton unchanged — the control proving the extra linear accuracy was discarded, not
    used.
  * **`nl_tol` is NOT free, and the original justification for it was wrong.** It had been argued
    that 1e-6 → 1e-5 was harmless because Newton would stop at the same iteration. Measured: Newton
    drops **3 → 2 iterations/step** and `max|η|` shifts **1.9e-5 (2-D) / 3.8e-5 (`test_basic`)**
    relative — 19–38× the ~1e-6 gate that had been set, which the change therefore *fails*.
    `nl_tol` is a **step function** (integer iteration counts): 1e-5 and 1e-4 are **bit-identical**,
    and the whole discrepancy is the 3rd Newton iteration. It is adopted on an **error budget** —
    that iteration polishes a `~3e-6` residual against a **measured** `O(Δt²)` time-discretisation
    error of `‖R‖∞ ≈ 1.8e-3`, i.e. ~600× larger — **not** on a null result. Full analysis and the
    replaced acceptance criterion: `LOCAL_TESTS_RESULTS.md` §5.4 "RESOLVED 2026-08-12".
    Consequence: `test_basic.jl`'s reference Newton count is **240, was 408** (max η and gauge amp
    unchanged); `test_basic_distributed.jl`'s `REF_EMAX_*` stand (shift is ~50× inside `REF_RTOL`).
- **Both alternative preconditioners failed** (same benchmark). Additive Schwarz dies with
  `SingularException` — `SchwarzLinearSolver(LUSolver())` factorises each rank's local block, but
  `partition(::PSparseMatrix)` returns own+**ghost** rows with the ghost rows unassembled, i.e.
  structurally zero. Symmetric Gauss–Seidel is worse: it completed **zero steps in 12 h** (against
  1.9 h for a *complete* Jacobi run) and was stopped. **Both cheap drop-ins are out**; what remains
  is field-split/Schur (medium effort), geometric multigrid (high), or restricted Schwarz (needs
  library support GridapSolvers 0.7.1 does not expose).
- **Strong scaling measured — partition size must follow the problem.** On the *identical* 20.2k-DOF
  quasi-1-D mesh: **1 rank (direct LU) 6.45 s/step vs 14.4 / 13.1 / 19.5 / 18.3 at 2 / 4 / 6 / 12
  ranks**, with the GMRES count **rank-independent at ~758**. So a direct LU beats every MPI split
  2–3× at that size, and the 1-D launchers were switched back to sequential (which also restores
  point gauges). For the 99k-DOF 2-D problem the ordering reverses and 12 ranks win. **Spend spare
  cores on more *cases*, not on decomposing one small case further.**
- **`test_mms.jl` renamed `test_selfconsistency.jl`** — it is *not* a manufactured-solution
  validation. Its forcing is `f = R(u*)` built with the solver's own residual, so writing
  `R = R_true + E` the error cancels identically and it passes for **any** `R`; it also measures no
  convergence rate. It is kept because it certifies what nothing else does — the hand Jacobians are
  the exact derivatives of the residual, and the multi-step bookkeeping is consistent — but as
  **code support, not model validation**. The file header now states this at length.
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
    (max η = 0.00410 m, 408 Newton iterations, gauge amp = 0.00212 m — at the then-default
    `nl_tol=1e-6`; the count is **240** under the current `nl_tol=1e-5`, see the tolerance entry);
    `test_vertical`,
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
  `GMRESSolver(krylov_m=100; restart=true, maxiter=ls_maxiter, …)`, with `ls_maxiter` **2000 → 1000**. Measured on the 4-rank distributed
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
- **~~🔴 sequential and distributed disagree by 2.5 % on the `:full` bar case~~ — RETRACTED
  2026-08-17, same day. THE DISCREPANCY WAS NEVER REAL.** It came from comparing
  `test_nlpressure.jl` G3's configuration (52x4 domain, sponge 6/8, `mu_max=20`) against
  `test_nlpressure_distributed.jl`'s (60x2, sponge 8/8, `mu_max=30`) — the same tanh bar and the
  same physics switches, but **two different problems**. Measured on the distributed file's OWN
  configuration, sequential LU and distributed CG+Jacobi give `0.0028640` and `0.0028640`: identical
  to the printed precision. The `:full` frozen-projection path is consistent between the solvers,
  and tightening `nlp_cg_rtol` 1e-10 → 1e-14 moves nothing (rel `2.9e-6` either way), confirming it
  was never a tolerance question either.
  **Kept visible rather than deleted, because the failure mode is easy to repeat:** two tests that
  share a bathymetry function and a physics tier can still be different problems, and a
  reference-vs-reference comparison is only meaningful when *every* discretisation parameter
  matches. The reference in `test_nlpressure_distributed.jl` still needed correcting
  (`0.0068979802 → 0.0028640`, 58 %), but that is the gravity fix landing, not a solver
  inconsistency.
- **🟠 THE MPI TESTS ARE NOT REACHED BY ANY RUNNER, and it cost three stale references.**
  `test/runtests.jl` (new 2026-08-17) covers the sequential suite; the distributed trio needs
  `mpiexecjl` and is only ever run by hand. Consequence, all found on 2026-08-17 the first time they
  were run in this session: `REF_EMAX_LIN` 9.67 % stale, `test_bc_generation_distributed`'s
  `REF_EMAX` 11.7 % stale (both fallout from the linear h-weighting, whose plan explicitly said
  references must be re-checked), and the `:full` discrepancy above sitting undetected.
  **Fix worth doing: give `runtests.jl` an MPI tier that runs them when `mpiexecjl` resolves and
  SKIPS LOUDLY when it does not** — a silent skip would recreate the same blind spot.
- **🟠 `test_nlpressure.jl` G3 asserts only BOUNDEDNESS on the one sequential configuration that can
  see the bed-slope physics.** It runs `:nonlinear` + `:full` over the tanh bar with
  `flat_bed=false` and gates `emax < 20A`; it passed 9/9 while the quantity it computes moved 58 %
  under the gravity fix. Pin a reference value there so the check lives in the suite that actually
  runs, instead of only in the distributed twin that does not.
- **🔴 NONLINEAR MMS — two blockers, both diagnosed, neither fixed** (2026-08-16). Full record:
  `building_files/MMS_NONLINEAR_PLAN.md` §5b.
  * **B1 — the `𝓝` forcing tiers (`nl_pressure=:native`/`:full`) are not available.** Components
    `{1,2,4,5}` are second derivatives of `u*`, and the leading pressure differentiates the result
    once more ⇒ a **third** derivative ⇒ three nested ForwardDiff levels. The outer spatial
    derivative returns a `Dual` instead of a scalar, i.e. the perturbations nest inverted. Tried and
    failed: two `derivative` calls, and a single `gradient` (returns a 2-partial `Dual`).
    `strong_residual_model` now **refuses loudly** for these tiers — a mis-nested forcing would be a
    plausible-looking wrong number that shows up as a collapsed rate and reads as a solver defect.
    **⚠ THE DIAGNOSIS IS UNCONFIRMED.** The tag-precedence story does not actually explain the case
    that failed: the first failure was `:native` on a **flat** bed, where `{3,6}` are zeroed by
    `∇h≡0` and only the *first-order* components `{7,8}` are live — two levels, not three — and an
    isolated rebuild of exactly that case evaluated cleanly. Remaining suspects: the `Z` placeholder
    (typed from `promote_type` of possibly-dual coordinates) mixing with computed components in one
    tuple, or the runtime `comps` ternaries producing a union-typed tuple. **Next step is cheap:
    bisect `Nvec` by returning one component at a time and differentiating `Ψ`** — that either
    confirms precedence or points at the typing issue, which would be a far smaller fix than the
    currently-planned "supply all spatial derivatives analytically".
  * **~~B2 — Model 4 Newton stall~~ ✅ RESOLVED 2026-08-17, and the recorded diagnosis was WRONG.**
    Kept for the method lesson. It read: *"convergence is linear and slow … raise `nl_iter` to
    200–400"*. Newton was **not** converging slowly; it was converging to the fixed point of the
    WRONG MAP. `∂R/∂u̇` was missing the `𝓐/𝓚` slope package, whose prefactor `H·∇h` does not scale
    with the solution — an `O(1)` error in the effective mass matrix. **No iteration budget could
    ever have fixed it**, and the proposed next step would have burned hours confirming that.
    Assembling the block (`src/problem.jl`) made Model 4 converge at the DEFAULT `nl_iter=50`, the
    very setting that had errored. **The lesson: "converging slowly" and "converging to the wrong
    thing" look identical in a solver log** — `test_jacobians_ad.jl` tells them apart by measuring
    how the hand↔AD gap scales with state amplitude (vanishing ⇒ slow; flat ⇒ wrong). The standing
    advice not to loosen `nl_tol` was, and remains, correct.

- **Residual verification: ALL FOUR `:none` MODELS DONE (see "Verified scope"); only the `𝓝`
  tiers open.** *(Entry below written 2026-08-12, when only Model 1 was done; kept because its
  statement of WHY the analytic MMS verifies where `test_selfconsistency.jl` cannot is still the
  clearest in this file. Its "Stages 2–4 remain unverified" is superseded: Model 2 (linear variable
  bed) and Model 3 (nonlinear flat bed) are now verified at theoretical order.)*
  The analytic MMS is
  implemented (`src/mms.jl`, `src/errors.jl`, `src/mms_driver.jl`) and **verifies the linear
  flat-bed operator** — mass, `M`-acceleration, `gΦ∇η`, `d²B` dispersion. Forcing derived from the
  governing equations in closed form, never touching `problem.jl` (enforced by a grep gate), so
  unlike `test_selfconsistency.jl` the error does **not** cancel. Gates: eigenmode `𝓛(u*)=3.6e-15`,
  closed-form ≡ ForwardDiff `1.8e-15`. **Still open: Stage 2 (+advection), 3 (+curved bed),
  4 (+𝓝 blocks)** — the architecture stages into them; nothing beyond the linear core is verified.
  Full record: `building_files/MMS_ANALYTIC_PLAN.md`.
- **⚠ The verification also found that the EQUAL-ORDER FE PAIRING COSTS ONE ORDER OF CONVERGENCE.**
  Measured: `Q2/Q2` converges at `p`(=2), `Q3/Q3` at `p`(=3) — uniformly `p`, not the theoretical
  `p+1`. Mixed order `Q_p/Q_{p-1}` fixes the **surface** universally and the **velocity** at `Q3/Q2`.
  **Confirmed by a 12-study campaign** (3 pairings × 1-D/2-D × static/transient,
  `building_files/MMS_CONVERGENCE_CAMPAIGN.md`) — state the two fields SEPARATELY:
  * **`η` reaches its optimal `p_e+1` in 12/12 studies**, exactly: `2.000/3.000/4.000` (1-D),
    `1.980/2.997/3.994` (2-D), and static ≡ transient to 4 significant figures.
  * **`u` reaches its optimal `p_u+1` only at `Q3/Q2`** (3.991 1-D, 3.930 2-D — four independent
    studies). At `Q2/Q1` it stalls near 2.4 (optimal 3) and at `Q4/Q3` near 4.65 (optimal 5), with
    the pairwise rate still *falling* at the finest level in both. Genuinely suboptimal vs merely
    pre-asymptotic is **undetermined**.
  ⚠ The 1-D `Q4/Q3` fitted `p_u=3.345` is a **round-off floor, not a rate** — `e_u` goes
  `1.79e-11 → 1.14e-11` over the last refinement and the pairwise rate collapses to 0.657. Discard
  it; `Q4` in 1-D cannot be refined further in double precision.
  **On this evidence `Q3/Q2` is the pairing to prefer** — the only one measured optimal in both fields.
  Cause: `η` enters momentum undifferentiated via `∇·v` after IBP, so it plays the pressure role and
  equal-order continuous spaces are inf-sup deficient (the Stokes analogue); velocity one order above
  the surface is the Taylor–Hood pairing. **This is what verified the residual**: the operator code is
  byte-identical across those runs, and a wrong coefficient cannot be repaired by changing the FE
  spaces, so reaching `p+1` under a stable pairing proves the operator is the intended one.
  Time integration and `R_P` were each eliminated first by experiment (steady `ω=0` field reproduces
  the same rates to 3 decimals; `B=0` makes `η` *worse*, so `R_P` is stabilising, not harmful).
  `build_fe_spaces(...; p_eta=…)` exposes the pairing, **defaulting to the existing equal-order
  behaviour** — nothing changes until deliberately opted in.
  **Do not switch production on the rate alone:** at `nx=24`, `Q3/Q3` gives `e_eta=5.97e-7` vs
  `Q3/Q2`'s `2.40e-5` — 40× *more accurate at that mesh* despite the worse rate, because `η` sits in
  a richer space. An error-vs-DOF comparison at production resolution is **not yet run**.
- **Run `test_2d_reduces_to_1d.jl`** — written, registered in the runner, not yet executed. A
  y-invariant 2-D run must reproduce the quasi-1-D flume to 1e-6. It is the sharpest check available
  short of the analytic MMS, because it tests the solver against a *symmetry the model must respect*
  rather than against theory via its own residual, and it exercises the `Ey`/`𝖴y` machinery no 1-D
  case touches.
- **Preconditioner replacement** — the single biggest performance item, now with the cheap options
  eliminated (see the benchmark entry above). GMRES needs ~300 iterations *after* the tolerance fix
  where a well-preconditioned solve of this size should need tens; the count is rank-independent and
  grows with **mesh anisotropy** (the 4:1-celled 1-D flume needs ~760 against ~480 for the isotropic
  2-D mesh) and with **solid-wall BCs** (ring 654–695 vs plane 451–517 on an identical mesh — an
  isolated, cheap, reproducible test bed: a preconditioner that closes *that* gap likely helps
  everywhere). Order of effort: field-split/Schur → geometric multigrid → restricted Schwarz.
  *Derived design rule meanwhile: keep horizontal cells near-isotropic, or pay for it in the linear
  solve.*
- **Four output gaps** (`SOLVER_ASSESSMENT_2026-08.md` §6): the discrete-equation check does not run
  under the default SDIRK integrator (it is `is_theta`-gated, so `res_theta = NaN`); linear counts
  are per RK stage, not per Newton iteration; no assembly-vs-solve time split; the linear solve's
  achieved tolerance is not recorded.
- **`nl_pressure=:full` is unvalidated *by the local observation set*** — and the qualifier is
  load-bearing. **The resolution principle: a test validates a term only if the test can resolve
  that term's contribution.** Switching `:full` off changes `max|η|` by **0.0129 % (1-D) / 0.094 %
  (2-D)**, so the 13 local cases establish that it *runs, converges and stays bounded* — useful,
  since it is the path that NaN'd on the cluster — but they carry no information about whether its
  coefficients are right. Always ask the counterfactual: *if this term were wrong, would this test
  have noticed?*
  **What IS validated, and must not be discarded when quoting this:** the `∇h` half of
  `𝓝{1,2,4,5}` is validated **to machine precision** (~4e-15) by `test_nlpressure.jl` **G1**, an
  exact-IBP identity against *analytic* second derivatives; **G2** pins every block's nonlinear
  order by amplitude scaling (∇h-IBP → 4, ∇H-frozen → 8, 𝓟-frozen → 4). What is missing is the
  *value* of the ∇H-frozen and 𝓟-frozen halves, and any dynamic confirmation at realistic amplitude.
  Say *"unvalidated by the local set"*, never bare *"unvalidated"*.
  **The sensitivity is tunable and the missing experiment is cheap:** the block is `O(A³)` against
  `O(A)` leading terms, so its relative effect scales as `A²` — 0.013 % at `A = 1e-3`, **~1.3 % at
  `A = 1e-2`**, i.e. comfortably measurable. The 2-D bar case already ran at `A = 0.01` with
  `:full`; its `:none` counterpart was never run, and that single pair would both resolve the
  difference and confirm the `A³` scaling dynamically. The deeper fix is the analytic MMS staged to
  that tier — MMS puts the term in the *forcing*, isolating it, instead of hoping a small
  contribution to a global quantity is visible. Full statement: `LOCAL_TESTS_RESULTS.md` §7b.
- **Only one vertical resolution and one `kd` tested locally** (LFE-2, `Nσ = 3`, `kd = 5.5`).
  LFE-3/LFE-4 and the rest of the band are untouched by the local campaign. The sponge in particular
  is verified at `kd = 5.5` only, while `CLAUDE.md` §8 requires it to cover the *longest* component
  — a `T` sweep measuring reflection at `kd = 1, 3, 5.5` would test that directly, and it is the
  failure mode behind the archived irregular-sea crash.
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
the integrator form its per-stage system `J = ∂R/∂u + (1/aΔt)∂R/∂u̇` directly. **Coverage differs by
regime, deliberately** (corrected 2026-08-15 — the previous description of this was wrong):
* the **LINEAR** branch is **EXACT**. The residual is affine in `(u,u̇)` there and every assembled row
  has its exact derivative. Consequence and standing gate: Newton **must** converge in one iteration
  per implicit stage, at any amplitude, on any bathymetry (`test/test_linear_newton_gate.jl`).
* **`∂R/∂u̇` is EXACT IN BOTH REGIMES** (since 2026-08-17). Every `u̇`-dependent term is
  differentiated exactly — mass, the `H`-weighted acceleration, the leading pressure `R_P`, and the
  `𝓐/𝓚` slope package; the `𝓝` blocks carry no `u̇`-dependence at all, so nothing is missing.
  Verified matrix-by-matrix against AD: `‖Δ(∂R/∂u̇)‖ = 0.000e+00` in **all eight** models
  (`test/test_jacobians_ad.jl`).
* the **NONLINEAR** `∂R/∂u` remains **QUASI-NEWTON by choice**. Advection is differentiated in full,
  but the leading- and slope-pressure packages contribute no η-derivative and the `𝓝` blocks add to
  the residual but not the Jacobian. Measured gap: `2.9e-2` (`:none`) to `8.3e-1` (`:full`),
  vanishing at order **1.11–1.16** in the state amplitude. That is what makes it benign — it costs
  Newton iterations, never accuracy, since Newton drives the *residual* to zero. Do not "complete"
  it without re-measuring every nonlinear reference value.
* **THE DISTINCTION THAT MATTERS, and that was blurred until 2026-08-17:** an omission is benign
  only if it is **higher order in amplitude**. The `𝓐/𝓚` block previously missing from `∂R/∂u̇`
  carried a prefactor `H·∇h`, which does **not** scale with the solution — an `O(1)` error in the
  effective mass matrix, which made Newton converge to a fixed point of the WRONG map and stalled
  Model 4 outright (blocker B2). No iteration budget can rescue that. **Never assume an omission is
  higher-order — measure it**, with `test_jacobians_ad.jl`'s amplitude-scaling gate.

Automatic
differentiation is not used because Gridap's multifield AD cannot dualize through `∂t(u)` (there is
no `TransientMultiFieldCellField` constructor for the dual), so the hand Jacobians are the design;
`build_ode_operator_ad` exists only as a cross-check path. That limitation was established on
Gridap 0.19.11 and held through 0.20.8 — **but it is now FIXED (2026-08-15)**. The cause was never `ForwardDiff.Dual` at all: `time_derivative(::TransientMultiFieldCellField)` builds its third constructor argument with `map(cellfield, derivatives...)`, and `map` returns a **Tuple** when fed a Tuple but a **Vector** when fed an array-like MultiField container — while the struct field is declared `transient_single_fields::Vector{<:TransientCellField}`. The hand path passes a `MultiFieldFEFunction` (array-like ⇒ Vector ⇒ works); the AD path passes a plain Tuple ⇒ `MethodError`. **A one-line additive constructor fixes it** (`Tuple`→`Vector`), carried on the fork `Elmanyer/Gridap.jl`, branch `fix-transient-multifield-ad`, based on the `v0.20.8` tag, commit `fa860899c`; `Manifest.toml` pins it. Analysis: `building_files/AD_ISSUE.md` (untracked); the standing guard that the pin has not rotted is **gate A0 of `test/test_jacobians_ad.jl`** — an instant `hasmethod` check for the fork's Tuple-argument constructor, which stock Gridap does not have.
  **Consequences — this changes the design, not just a dependency.** (a) `use_ad=true` now works, so **AD is an ORACLE for the hand Jacobians**: it differentiates the same assembled residual, so where AD converges and hand does not, the residual is right and the hand Jacobian is wrong. (b) Measured 2026-08-15: on a flat bed AD and hand agree **bit-for-bit** (`e_eta=7.974294e-05` both) — cross-validating both. (c) **⚠ RETRACTED:** it was recorded here that "on `flat_bed=false` AD converges where hand fails, which PROVES the variable-bed residual is correct and the defect is confined to the hand Jacobian's `sAK` block". **That inference is invalid.** AD differentiates the same assembled residual, so its converging proves residual↔Jacobian *consistency*, never residual *correctness* — and the residual was in fact double-counting the `𝓐/𝓚` package (see the fix entry under "Recently completed"). Every `use_ad=true` variable-bed result predating 2026-08-15 must be discarded. (d) The hand Jacobians remain the fast production path; AD is a cross-check, useful but **not** an oracle for correctness — only the analytic MMS is that, because only it is derived independently of the residual.

**Coding rule (block arrays):** never apply `∇` to an `Operation`-composed expression containing a
test basis — expand by hand via `∂_a(W⋅𝓣) = (∂_aW)⋅𝓣` (see `nlpressure.jl`).

**Time integrators** (`build_ode_solver` / `build_ode_solver_distributed`): the default is the
fully-implicit `RungeKutta(nls, ls, dt, :SDIRK_2_2)` (L-stable 2nd-order, diagonally implicit), which
is robust in the stiff deep-water regime; `:theta` selects Crank–Nicolson. Driver kwargs
(both drivers): `solver_type=:sdirk` (default), `tableau=:SDIRK_2_2`, `nl_iter=50`,
**`nl_tol=1e-5`** (production; convergence/physical-reproducibility tests pin `1e-8`), distributed
`ls_maxiter=1000` / `krylov_m=100` / **`ls_rtol=1e-5`**, `print_every`, `check_every=50`,
`check_tol=1e-8`.

**Tolerance ladder (retuned 2026-08-11, `nl_tol` closed on measurement 2026-08-12).** Defaults are
**`ls_rtol=1e-5`, `nl_tol=1e-5`**; no run script overrides them. The two tolerances behave
completely differently here, and conflating them is the mistake this entry exists to prevent:

* **`ls_rtol` does not affect the answer** anywhere in 1e-9…1e-5, while each order costs GMRES
  iterations: 1e-9 → 1e-6 cut **491–515 → 294–314 (−40 %)**, 1e-6 → 1e-5 a further
  **295–313 → 238–253 (−19 %)**, `max|η|` unmoved and Newton unchanged throughout
  (`run/local/bench_solver_config.sh`, one fixed 2-D case).
* **`nl_tol` DOES affect the answer, as a step function.** Newton runs an integer number of
  iterations, so only threshold crossings matter: at 1e-6 it needs 3 iterations/step, at 1e-5 *and*
  1e-4 it needs 2 — those two are **bit-identical to 12 digits** — and dropping that third iteration
  moves `max|η|` by **1.9e-5 / 3.8e-5** relative. Do **not** describe this as "no effect".

Justification for `nl_tol=1e-5` is an **error budget, not a null result**: the dropped iteration
refines a `~3e-6` residual, while the `O(Δt²)` time-discretisation error already in the answer is
`‖R‖∞ ≈ 1.8e-3` (measured by the run's own residual checker) — ~600× larger. Note the adopted pair
leaves **no separation** between `ls_rtol` and `nl_tol`, deliberately: the older "keep the linear
solve one order tighter" rule remains sound general guidance, but it is not what governs accuracy
here, and `ls_rtol` was measured not to limit Newton at any point in the range. Tests that need a
sharper answer pin `nl_tol=1e-8`. Full analysis and the superseded acceptance criterion:
`building_files/LOCAL_TESTS_RESULTS.md` §5.4. The distributed run scripts expose these as
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

* **A REFINEMENT STUDY MEASURES THE RATE OF WHICHEVER ERROR DOMINATES — so isolation must be
  VERIFIED, in BOTH directions, before any slope is interpreted** (learned twice, 2026-08-16/17).
  A saturated slope and a genuinely wrong coefficient produce the SAME observable — the error stops
  decreasing — so the legend "slope 0 ⇒ wrong coefficient" is only valid once the other
  discretisation has been ruled out. Two independent instances:
  * `test_shallow_water` measured a 6.47 % celerity error that was **spatial** under-resolution
    (6 cells/λ); refining time alone changed nothing, refining space alone fixed it.
  * `test_mms_convergence` G7 measured temporal orders of **0.024 and 0.195** on a perfectly correct
    operator, because it integrated over **1.7 % of the manufactured solution's period** and was
    therefore sitting on its spatial error floor — the tell being that `e_eta` flatlined at exactly
    G6's finest spatial error.
  **Guards come in pairs.** `test_mms_convergence` had `G8` (halving `dt` must not move the spatial
  study) but no mirror, which is why the temporal study stayed mis-specified; `G10` (refining the
  mesh must not move the temporal study) now closes it. When you fix one study of a refinement pair,
  check the mirror guard exists — a one-sided guard is a blind spot by construction. And when a
  temporal study is designed, check `T_final` against the solution's OWN timescale: refining `Δt` is
  pointless if nothing is evolving.
* **FE PAIRING: equal order costs one convergence order (measured 2026-08-12).** `Q_p/Q_p` for
  `(η, 𝖴)` converges at `p`, not `p+1`; `Q3` velocity with `Q2` surface recovers `p+1` for both.
  `η` enters momentum undifferentiated via `∇·v` after IBP, so it is the pressure of a Stokes-like
  pairing and equal-order continuous spaces are inf-sup deficient. `build_fe_spaces(...; p_eta=…)`
  selects the surface order and **defaults to equal order (unchanged behaviour)**. Before switching
  production, compare error-vs-DOF at realistic mesh sizes: the higher rate does not automatically
  win at a given mesh (`Q3/Q3` was 40× more accurate than `Q3/Q2` at `nx=24`). Evidence and the full
  diagnosis trail: `building_files/MMS_ANALYTIC_PLAN.md`.
* **ASSEMBLY INVARIANT — every residual term must have exactly ONE consumer, guarded by the
  CONJUNCTION of its three activation conditions** (`tab: term classification`). A guard testing only
  the bed-slope condition (`flat_bed`) or only the amplitude condition (`regime`) is a defect
  whenever the same physics has a second representation in the other regime — which, for the
  `𝓐/𝓚` slope-pressure packages, it always does: the linear regime carries the linearised form
  `h∇h·𝓛ˡⁱⁿ·(A+K)` and the nonlinear regime the un-expanded `H[∇h(𝓛·A)+∇H(𝓛·K)]`, and the second
  *contains* the first. They are alternatives, never addends. This exact mistake cost weeks: the
  package was assembled twice under `regime=:linear, flat_bed=false`, an `O(ε)` error invisible to
  every flat-bed test because both forms vanish when `∇h≡0`. Corollary: **a flat-bed regression can
  never test `∇h` code** — new bed-slope terms need a sloping-bed gate from the start.
* **THE DEFAULT INTEGRATOR IS DISSIPATIVE — any test measuring a non-dissipative property MUST pin
  `solver_type=:theta`.** `RungeKutta(:SDIRK_2_2)` is **L-stable**, i.e. it damps by construction;
  that is exactly why it is the right production default in the stiff deep-water regime, and exactly
  why it destroys the quantity certain tests exist to measure. When the default changed on
  2026-07-23, **four tests silently broke** and stayed broken for three weeks because the full suite
  was not re-run (measured 2026-08-16):

  | test | measures | SDIRK_2_2 | `:theta` (CN) |
  |---|---|---|---|
  | `test_energy` | non-dissipativity | `ΔE/E₀ = −1.34e-2` | **`+3.65e-14`** |
  | `test_bc_spectrum` | amplitude transfer, 3 components | 12.5 / 21.9 / 32.2 % | **2.5 / 4.6 / 9.4 %** |
  | `test_bc_generation` | generated wave amplitude | 23.1 % | **8.4 %** |
  | `test_convergence` | temporal order (Richardson) | **0.01** | ≈2 |

  All four now pin `:theta` with the measurement recorded inline; **do not remove those pins**, and do
  not "fix" such a failure by moving a threshold — the test's subject is a property CN has and SDIRK
  deliberately lacks, so on the default it measures the integrator rather than the model.
  **How to recognise this failure mode:** amplitude is damped while **phase is correct**
  (`test_bc_generation`: amplitude 23 % low, phase speed 0.6–2.4 %), the error **grows with
  frequency**, and — the decisive discriminator — **refining the mesh does not help**
  (`test_bc_spectrum` at 12 cells/λ got *worse* at high frequency, 32.2 → 41.6 %). That last check is
  what separates this from genuine under-resolution, which is a real and separate failure mode in this
  suite (see `test_shallow_water`, where refining space alone fixed a 6.47 % celerity error and
  refining time changed nothing).
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
  (measured at the then-default `nl_tol=1e-6`; the count is 240 at today's `nl_tol=1e-5`, which
  changes both stacks equally and so does not affect this equivalence conclusion)
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
