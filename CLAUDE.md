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
| `Project.toml` / `Manifest.toml` | A standalone Julia environment (deliberately not a registered package — `GridapLFEM` is loaded via `include()`+`using .GridapLFEM`, never `Pkg.add`). It is pinned to the same package versions as the parent `../Project.toml` (Gridap 0.19.11, GridapDistributed 0.4.13, GridapSolvers 0.6.2, PartitionedArrays 0.3.5, MPI 0.20.26) so both environments behave identically. If you add a new `using X` to any `src/*.jl` file, run `Pkg.add("X")` here too (`--project=GridapLFEM.jl`) to keep the two environments in sync. `BlockArrays` is a direct `[deps]` entry (already resolved in the Manifest) only so the differently-structured **oracle** `LFEModel2D` loaded by `test_equivalence.jl` can `using BlockArrays` from this env. |
| `LFEM_discretisation.zip` (LaTeX project) | The authoritative LaTeX derivation, a multi-file project (compiles with pdflatex): `main.tex` (preamble/macros) inputs `GoverningEquations`, `SigmaTransformation`, `VerticalFEapprox`, `wDerivation`, `pDerivation`, `VerticalProjection`, `VerticalSemiDiscreteSystem` (incl. the **flat-bed reduction** of the full nonlinear model), `LinearModel`, `GlobalResidual`, **`GridapImplementation.tex` (§8** — stacked layout, residual implementation, 𝓛/𝓝 pressure treatment, the `regime`/`nl_pressure`/**`flat_bed`** model-setups subsection, wave generation/sponge/BC incl. Dirichlet boundary generation with the `:model`/`:airy` polarization) and **`ValidationTests.tex` (§9** — the validation report incl. the BC-generation gates, Goda–Suzuki, spectral fidelity). Sign convention: `R_P` is a positive integral **subtracted** in the global sum, uniformly with `R_lin`/`R_nonlin`. All prior standalone `*Improved.tex`/`.md` section drafts are now integrated into this zip. |
| `LFEM_Gridap.md` | Synthesis of the derivation §1–§8 leading to the single scalar residual, in the LaTeX notation; §9 bridges that notation to the solver code. |
| `algebraic_residual_math.md` | Operator reference: how each §8 residual term is written with native Gridap tensor ops (no MultiField decomposition, no vertical-index loops) — the `L`/`N` pressure stacks, the leading-pressure `R_P`, IBP of second-derivative terms, and the verified Gridap operator table. |
| `DESIGN_RECORDS.md` | Consolidated **historical design records** for the completed features — algebraic residual + Jacobians, package layout, distributed solver, nonlinear-pressure completion, boundary wave generation, periodic-y BC, the `flat_bed` switch, and the production sea-state scripts. Provenance only (*why* the code is shaped as it is); the code, this `CLAUDE.md`, and the LaTeX derivation are authoritative. |
| `src/` (`GridapLFEM.jl` + submodules) | **The self-contained serial + distributed solver package** (`module GridapLFEM`): vertical tensors (incl. `Pcal`), constant-tensor + `Operation` helpers, stacked FE spaces (distributed-safe MultiField dispatch + transient-Dirichlet inflow variants), the loop-free residual + hand Jacobians (the same code runs distributed — `Operation` is forwarded for `DistributedCellField`), the time loops (default fully-implicit `RungeKutta(:SDIRK_2_2)`, `:theta` Crank–Nicolson selectable via `solver_type`; sequential LU+Newton / distributed GMRES+Jacobi+Newton), per-component VTK (`eta,u1x,u1y,…`) plus reconstructed `w_s<σ>`/`p_s<σ>` fields (`reconstruct.jl`), and `waveinput.jl` (Dirichlet boundary wave generation + WaveSpec.jl coupling: component tables, `:model`/`:airy` polarizations, ramp, relaxation zone). Drivers: `setup_and_run` and `setup_and_run_distributed`. |
| `test/` (21 tests + `test/cluster/`) | **Base suite:** `test_vertical` 15/15, `test_primitives` 9/9, `test_equivalence` 10/10 (cross-check virtual-work, ≤7e-15), `test_basic` 6/6, `test_dispersion` (kd=3 err 0.90%), `test_basic_distributed` (4 ranks), `test_nlpressure` 9/9 (+ `_distributed`), `test_sloshing` (1.44%), `test_conservation` (drift 7.8e-16). **Validation batch:** `test_dispersion_curve` 9/9 (closed-form Cm/Ce(kd), kd_app 10.8/39.2/127.9), `test_mms` 3/3 (unsteady nonlinear MMS), `test_convergence` 2/2, `test_vertical_profile` 7/7 (sinh shape), `test_energy` 3/3, `test_dispersion_nonlinear` 3/3 (full-NL⇒Airy, kd=1/3/5 err 0.93/0.36/3.05%), `test_shallow_water` 6/6 (kd→0 ⇒ √(gd), ΦᵀM⁻¹Φ=1). **BC-generation batch:** `test_waveinput` 30/30, `test_bc_generation` 11/11, `test_bc_spectrum` 8/8 (Goda–Suzuki), `test_bc_generation_distributed` 4/4 (rel 3.05e-8). **`test/cluster/`:** `cluster_conservation` (2 ranks, drift 5.8e-10), `cluster_mms` (all 𝓝 at scale) + SLURM template. |
| `examples/` (+ `validation/`, `distributed/`) | Sequential: `plane_wave.jl`, `ring_wave.jl`, and BC-generation `bc_plane_wave.jl`, `bc_irregular_sea.jl` (JONSWAP, gauge CSV), `bc_directional_sea.jl`; `examples/distributed/` — 6 env-configurable cluster scripts (plane/ring wave, IC hump, bathymetry, `run_irregular_sea_dist.jl`, `run_directional_sea_dist.jl` — sea-state env vars + `build_airy_state()` in `_dist_common.jl`) + README; `examples/validation/` — physical benchmarks (`stokes_harmonics`, `submerged_bar`, `solitary_wave`, `ring_spreading`, `bichromatic_sideband`) + `dispersion_sweep.jl` + `spectral_fidelity.jl` (JONSWAP component-wise amplitude+dispersion transfer) + README; `examples/distributed_small/` — **4 parametric** small-domain (50×20 m) scripts (`run_periodic_plane_small`, `run_ring_small`, `run_directional_sea_small`, `run_irregular_sea_small`) grouped by wave-generation type; the 16 observation/comparison cases are their **launchers** in `run/dist_small/`, each overriding only the env vars that change (regime / nl_pressure / flat_bed→bar / amplitude / period). See the "Current Implementation Stage" small-domain-suite entry. |
| `postprocessing/` (`GridapLFEMPost`) | Self-contained postprocessing library with its own environment (ReadVTK, Plots+GR, FFTW, Interpolations — pinned separately from the solver). Reads VTK (`solution.pvd`/`sol_t_*.vtu`) + CSV → `WaveSimulation` (auto-`regularize!`s the duplicated Q2 node cloud to a Cartesian grid). Modules: `io, probes, spectral, diagnostics, reconstruct, plotting, seastate`. Gauges/DFT/celerity/harmonics/radial/conservation; heatmap/animation(GIF)/Hovmöller/dispersion/profile plots; `seastate.jl` — Welch PSD, JONSWAP target overlay, spectral moments/Hs, zero-upcrossing heights, Rayleigh exceedance (+ `spectral_validation.jl` example). `reconstruct.jl` rebuilds `w(σ)`/`p_nh(σ)` from the stored velocity modes at any σ (analytic σ-basis, Gauss quad, no Gridap; matches solver `w_s` to 4–8%). No dependency on the solver. |
| `WaveSpec.jl/` (repo-vendored package) | Stochastic sea-state synthesis (CMOE-TUDelft; JONSWAP/TMA/… spectra, sampling strategies, angular spreading, `AiryState`). `Pkg.develop`ed in both environments; `using WaveSpec` is re-exported by `GridapLFEM`. The `WaveInput` converter (`src/waveinput.jl`) snapshots seeded amplitudes/phases into plain arrays and re-solves the wavenumbers with the solver's `g` (WaveSpec uses 9.80665). |
| `boundary_wave_generation.md` | Dirichlet boundary wave generation math note (nodal trace, `:model` discrete-eigenmode polarization derivation, ramp, well-posedness/reflection, relaxation zone, WaveSpec contract, validation map). The design record is in `DESIGN_RECORDS.md`. |
| `LFEM_runs.md` | Plan/record for the small-domain observation + comparison run suite (all three batches; `examples/distributed_small/` + `run/dist_small/`). |
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

**Tests** (`test/`, 21 + `cluster/`): all pass — see the §1 table for the per-file scores. Highlights:
cross-check virtual-work equivalence ≤7e-15; the asymptotic-consistency pair (full-NL⇒Airy across the
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
equivalence ≤7e-15, asymptotic consistency (full-NL⇒Airy, kd→0⇒√(gd)), unsteady nonlinear MMS ~3e-9,
BC generation 30/30 + 11/11 + Goda–Suzuki 8/8, distributed agreement ≤5e-9. Postprocessing
(`GridapLFEMPost`) and the authoritative LaTeX derivation are in place and compile-checked.

**Recently completed (this branch).**
- **Physics-selection interface** (§6): the six residual booleans are driven by three orthogonal
  high-level controls `regime` ∈ {`:linear`,`:nonlinear`}, `nl_pressure` ∈ {`:none`,`:native`,`:full`},
  and **`flat_bed`** ∈ {`false`,`true`} via `resolve_physics`. `flat_bed` replaced the former
  `slope_pressure` switch: it selects a flat sea bed (∇h≡0 — drops the bed-slope `𝓐` packages, `L¹`,
  `N{3,6}`, the bed-slope IBP half) vs variable bathymetry, at a single control point
  (`dhx,dhy = flat_bed ? 0 : ∂h`) in `global_residual`/`jacobian_*`; ∇η/dispersion terms are kept.
  Runtime-verified with a differential residual test (9/9 across all `nl_pressure` tiers: `flat_bed`
  changes a sloped-bed residual, is bit-exact on a flat bed); the `flat_bed=false` path is identical to
  the prior validated baseline, confirmed by `test_equivalence` re-passing 10/10 (oracle virtual-work
  match ≤7.4e-15). Drivers emit a bathymetry↔switch consistency warning
  (`check_flat_bed_consistency`). Derivation: the flat-bed reduction of the full nonlinear model in
  `LFEM_discretisation.zip` (`VerticalSemiDiscreteSystem.tex`); design record: `DESIGN_RECORDS.md` §7.
- **Default integrator** SDIRK_2_2 (fully implicit, L-stable); **y-periodic** lateral BC option.
- **Small-domain run suite** — **4 parametric scripts** (`examples/distributed_small/`, grouped by
  wave-generation type: `run_periodic_plane_small` [line source], `run_ring_small` [point source],
  `run_directional_sea_small` / `run_irregular_sea_small` [Dirichlet BC]) driving **16 launchers**
  (`run/dist_small/`), one per case. Each script bakes in the common geometry/mesh/partition/solver
  defaults and reads the physics from env (`get!` base defaults ⇒ banner ⇔ solver consistent; a
  `flat_bed=0` toggle builds the submerged bar; the output dir is auto-tagged by config so cases don't
  clobber). Each launcher overrides only what changes (regime / nl_pressure / flat_bed→bar / amplitude
  / period). Common 50×20 m domain, short dynamic timescales (`d=3.5, T=2.0, kd≈3.5`), `save_every=10`;
  partition **8×4 = 32 ranks** (plane/irregular, 200×40) or **8×8 = 64 ranks** (ring/directional,
  200×80 square cells) — 250 cells/rank throughout. The 16 cases span the configuration space (batch 1),
  a controlled comparison matrix around the flat periodic plane wave (batch 2), and sea-state comparisons
  (batch 3). Verified: all 4 parse; every launcher resolves to its intended config with `n = px·py =
  ntasks`. Smoke-validated on 2 ranks (big-amplitude full-nonlinear flat plane wave: Newton-converged,
  finite). Plan: `building_files/LFEM_runs.md`.

**Under development / open:**
- **WaveSpec `change_seed!` bug** — `WaveSpec.AiryWaves.change_seed!` accesses a non-existent
  `state.spec` field (the struct field is `spectrum`), so `build_airy_state` (in
  `examples/distributed/_dist_common.jl`) currently errors. Blocks every Dirichlet-sea run (production
  `run_{irregular,directional}_sea_dist.jl` and the small sea cases). A one-line library fix; the run
  scripts themselves are correct.
- At-scale physical benchmarks on the cluster (Stokes harmonics, Dingemans bar) — scripts exist
  (`examples/validation/`, `examples/distributed/`); quantitative overlays on the paper's data pending.
- Production-length (200+ Tp) irregular/directional sea runs (scripts ready; blocked by the WaveSpec bug).
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
sponge: + ∫ μ*((Wx⋅(𝗠⋅Ux))+(Wy⋅(𝗠⋅Uy)))
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
differentiation is not used because Gridap 0.19.11's multifield AD cannot dualize through `∂t(u)`
(there is no `TransientMultiFieldCellField` constructor for the dual), so the hand Jacobians are the
design; `build_ode_operator_ad` exists only as a cross-check path.

**Coding rule (block arrays):** never apply `∇` to an `Operation`-composed expression containing a
test basis — expand by hand via `∂_a(W⋅𝓣) = (∂_aW)⋅𝓣` (see `nlpressure.jl`).

**Time integrators** (`build_ode_solver` / `build_ode_solver_distributed`): the default is the
fully-implicit `RungeKutta(nls, ls, dt, :SDIRK_2_2)` (L-stable 2nd-order, diagonally implicit), which
is robust in the stiff deep-water regime; `:theta` selects Crank–Nicolson. Driver kwargs
(both drivers): `solver_type=:sdirk` (default), `tableau=:SDIRK_2_2`, `nl_iter=50`, `nl_tol=1e-6`
(production; convergence/physical-reproducibility tests pin `1e-8`), distributed `ls_maxiter=2000` /
`ls_rtol=1e-9` (the GMRES solve is kept accurate so Newton gets good steps), `print_every`,
`check_every=50`, `check_tol=1e-8`. The distributed run scripts expose these as
`LFEM_SOLVER/TABLEAU/NL_ITER/NL_TOL/LS_MAXITER/LS_RTOL` env vars.

**Runtime monitoring** (`src/monitor.jl`, serial + distributed):
* `SolverMonitor` — a transparent `NonlinearSolver` wrapper (pass via `monitor=`) that harvests per
  step: Newton iterations, initial→final residual, convergence flag, last GMRES iteration count
  (distributed), and nonlinear-solve wall time. For Runge–Kutta it accumulates over the stages.
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
* **`B_stored = −B̃` (the stored dispersion tensor ≤ 0)** and the explicit `(−1)` factors in the `R_P`
  and slope-pressure terms give those terms their correct sign — they are load-bearing.
* **Distributed linear solve = `NewtonSolver(GMRESSolver(Pr=Jacobi))`** (GridapSolvers) — a direct LU
  factorisation does not scale to the partitioned matrices at cluster size.
* **Global max|η|** is reduced from `own_values` + `reduce(max, …; init=0.0)` (the ∞-norm of a PVector
  is not usable directly here).
* **MPI:** launch with `~/.julia/bin/mpiexecjl` so the runtime matches the MPI build; the first
  full-FEM compile takes tens of minutes; Julia buffers stdout to files (use `flush`); `MPI_Finalize`
  prints a benign OFI error and exits 143.
* **Stack:** Gridap 0.19.11, GridapDistributed 0.4.13, GridapSolvers 0.6.2, PartitionedArrays 0.3.5,
  MPI 0.20.26. Run Julia via the `julia-mcp` tool. Transient API: `TransientFEOperator(res,jac,jac_t,
  U,V)`, `res(t,u,v)` with `∂t(u)`, `TransientCellField` in `Gridap.ODEs`, `solve(solver,op,t0,tF,u0)`
  iterator yields `(t,uh)`.

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
