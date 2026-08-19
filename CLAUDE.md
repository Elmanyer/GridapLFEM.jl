# CLAUDE.md — `GridapBALFEM.jl/` · the 2D BALFE-M algebraic wave solver

> ## ⇨ START HERE
>
> This file is the map and the standing rules. The detail lives in seven focused documents in
> `building_files/` (**gitignored — not under version control**):
>
> | document | what it answers |
> |---|---|
> | [`MODEL.md`](building_files/MODEL.md) | the maths: σ-tensors, the global residual term by term, nonlinear pressure, Jacobians |
> | [`ARCHITECTURE.md`](building_files/ARCHITECTURE.md) | code structure and workflow: the stacked layout, `src/` map, FE spaces, time loops, distributed path, diagnostics |
> | [`VERIFICATION.md`](building_files/VERIFICATION.md) | what is *proven*: the MMS campaign, verified scope, the Jacobian oracle, scope boundaries |
> | [`TEST_SUITE.md`](building_files/TEST_SUITE.md) | every test, its score, and **what it cannot detect** |
> | [`CONFIGURATION.md`](building_files/CONFIGURATION.md) | the settings, the evidence behind each, measured performance |
> | [`WAVE_GENERATION.md`](building_files/WAVE_GENERATION.md) | sources, Dirichlet generation, sponge, WaveSpec coupling |
> | [`RUNNING.md`](building_files/RUNNING.md) | how to launch: local, cluster, sysimage, env vars |
> | [`OPEN_ITEMS.md`](building_files/OPEN_ITEMS.md) | open work, each with its decisive next step |
>
> **One-line status (2026-08-19).** The solver is feature-complete in serial and distributed. The
> analytic MMS verifies **six of the eight models** — all four `:none` and both `:native` — at
> theoretical order. The remaining two, the **`:full` pair, are permanently outside MMS's reach by
> construction** (frozen L² projections vs an exact forcing); their error floor is quantified
> instead. Suite: sequential 20/20 files, distributed 13/13, Jacobian-vs-AD 17/17 over 8 models,
> nonlinear MMS 8/8, `test/local/` 50/50 — one failure in the whole campaign, and it is a
> gate-window specification defect, not a solver defect.

---

## 0. What this folder is

The production home of the 2D BALFE-M solver: the self-contained serial + distributed package
`src/GridapBALFEM.jl` (stacked `[η,𝖴x,𝖴y]` layout, loop-free residual), its test suite (`test/`),
sequential and cluster examples (`examples/`), the standalone postprocessing library
(`postprocessing/GridapBALFEMPost`), the vendored sea-state package (`WaveSpec.jl/`), and the design
and paper material in `building_files/`.

**BALFE-M** — *Basis-Agnostic Layer-integrated Finite Element*, `M` vertical elements — generalises
Yang & Liu (2024, *JFM* 999 A32) LFE-M to an **arbitrary vertical FE basis**. It is a
depth-integrated, **non-hydrostatic** free-surface wave model. The water column `σ ∈ [0,1]` is
discretised with `M` vertical elements of order `p`, so `u_h(x,σ,t) = Σ_j u_j(x,t) φ_j(σ)`. The
vertical velocity `w` and the non-hydrostatic pressure `p_nh` are eliminated **analytically** — no
pressure Poisson solve — into small, precomputable vertical σ-tensors. What remains is `Nσ+2`
coupled 2-D PDEs in `(H, u_1…u_Nσ)`, `H = d + η`, solved on a horizontal FE mesh by Gridap. The
structure is **(small dense vertical algebra) ⊗ (large sparse horizontal FE)**.

`Nσ = num_free_dofs(U_phi) = M·p + 1` = the number of vertical nodes; velocity modes are 1-based
`j = 1..Nσ`, one per node.

**Three names, kept strictly distinct in all prose** (defined in the LaTeX at
`\label{par: nomenclature}`):

* **BALFE-`M`** — the family this project derives, arbitrary vertical basis.
* **P`p`LFE-`M`** — a concrete member implemented/run/tabulated. Every instantiation here is `p=1`.
* **LFE-`M`** — Yang & Liu's published piecewise-linear models, used only when citing or comparing.

Tables comparing our numbers against theirs must not label both sides the same way.

---

## 1. Repository map

| path | what it is |
|---|---|
| `Project.toml` / `Manifest.toml` | the Julia package manifest — `name = "GridapBALFEM"`, `uuid = 43e94d05-4d7d-4679-96a4-d46e2615da34`. Loaded with **`using GridapBALFEM`, never `include()`** (§5). This directory is **both the package and the working environment**, so `Test`, `BlockArrays`, `MPIPreferences`, `Preferences` are in `[deps]`, not `[extras]`. `[compat]` admits two Gridap minors **on measured evidence** — see `CONFIGURATION.md` §1 |
| `src/` | the solver package — 16 files, mapped in `ARCHITECTURE.md` §2 |
| `test/` | 27 test files + `runtests.jl` + `test/cluster/` + `test/local/` — inventory and scores in `TEST_SUITE.md` |
| `examples/` | sequential, `distributed/`, `distributed_small/`, `validation/`, `local_1d/`, `local_2d/`, `local_mms/`, `inspect_run.jl` — `RUNNING.md` §2 |
| `run/` | 9 production SLURM launchers + `run/dist_small/` (20 small-domain cases) + `run/local/` (17 local launchers + benchmark), all through `run/balfem_env.sh` — `RUNNING.md` §3–4 |
| `compile/` | the cluster sysimage build chain — `RUNNING.md` §5 |
| `postprocessing/` | `GridapBALFEMPost` — self-contained, own environment, **no dependency on the solver** |
| `WaveSpec.jl/` | vendored stochastic sea-state synthesis (CMOE-TUDelft). Tracks the **GitHub repository version, not a tagged release** — the release's `change_seed!` is broken |
| `Gridap.jl/` | the vendored **fork** (`Elmanyer/Gridap.jl` @ `fix-transient-multifield-ad`, one commit on `v0.20.8`) making transient-multifield AD work — `CONFIGURATION.md` §2 |
| `BALFEM_models/` + `.zip` | **THE CURRENT LaTeX project** — the authoritative derivation (§2) |
| `LFEM_discretisation/` + `.zip` | **SUPERSEDED** pre-rename LaTeX, kept for provenance. Do not edit |
| `CFC2027_LFEMultilayer_abstract/` | conference abstract (CFC 2027); its class needs the `newtx` fonts |
| `building_files/` | the seven documents above. **Gitignored** |
| `algebraic_balfem2D.jl`, `test_algebraic_balfem2D.jl`, `test_2HDmodel.ipynb` | the single-file prototype and an early notebook, kept for reference; `src/` is the maintained form |
| `../LFE-M_2D_solver/` | a genuinely external per-layer implementation of the same weak form. Legacy, deliberately not renamed |

---

## 2. The LaTeX project (`BALFEM_models/`)

*Derivation, Study and Implementation of Basis-Agnostic Layer-integrated Finite Element (BALFE-$M$)
Models.* `main.tex` inputs `SigmaEulerModel.tex`; the chapter *Vertical Multilayer Discretisation*
over five sections in `VerticalFESemiDiscretisation/` (`VerticalFEapprox` — carrying the nomenclature
paragraph — `wDerivation`, `pDerivation`, `VerticalProjection`, `VerticalSemiDiscreteSystem` incl.
the flat-bed reduction); then `LinearModel.tex`, `StokesWaveFourierAnalysis.tex` (the analytical
core: the Stokes–Fourier hierarchy, the dispersion functional `R(μ) = Φᵀ(M+μ|B|)⁻¹Φ` and its four
basis-independent properties, group velocity and shoaling gradient, the second-order bound harmonic
reproducing the published `kd=6.0` for P1LFE-2, third-order solvability, and the vertical grid
optimisation `Δσ_top ≈ 2.94/kd_max`); then `NumericalImplementation/` — `GlobalResidual`,
`GridapImplementation.tex` and `ValidationTests.tex`.

> **`GridapImplementation.tex` §`subsec: term classification` IS THE RESIDUAL'S SPECIFICATION.**
> Every term of the full model tagged by amplitude order × bed-slope class × activation condition on
> each of the three switches. The table **factorises** — `regime` is a truncation in amplitude
> order, `flat_bed` a projection onto `∇h ≡ 0`, `nl_pressure` a component filter on `𝓝` — which is
> the formal statement that the three switches are orthogonal. The `O(ε)` rows are exactly eight,
> and they *are* the linearised model.

**Working rules:**

* ⚠ **COMPARE MTIMES BEFORE EDITING.** The author edits in Overleaf and exports the `.zip`, so the
  ZIP is newer after any round-trip. Correct procedure: unzip to scratch, `diff`, sync the folder
  **from the newer side** (back it up first), edit the folder, re-zip.
* ⚠ **After re-zipping, extract and compile FROM THE EXTRACT.** A `zip -x '*.pdf'` intended to skip
  a compiled `main.pdf` also matched `Figures/*.pdf` and silently shipped a zip missing four figures
  the document uses. Three clean compiles missed it because each ran against a *copy of the folder*,
  never against the zip — **the artefact that was verified was not the artefact being shipped.**
  The folder holds no build artefacts, so the zip needs no exclusion list at all.
* Conventions: structural labels `\label{chap:|sec:|subsec: name}`; cross-references written
  `\S\ref{…}`.
* Bibliography is **biblatex + biber**, 16 entries. ⚠ `biblatex`/`biber` are **not installed on this
  workstation** (only `bibtex`), so the production bibliography cannot be compile-tested here —
  validate via a bibtex shim in a scratch copy.
* Compilation: `pdflatex` ×3 → 0 errors, 0 undefined refs, 147 pp, with `pstricks` (not installed,
  unused) commented out **in the scratch copy only**; `multirow` required.

Table 4.1's Yang & Liu column matches ours: all three `C` entries agree to three significant figures
(our error at their `kd` is 2.04 / 1.99 / 2.00 %), and the `C_g`/`γ` offsets are **tolerance
conventions** — theirs implies ≈2.5 % on `C_g` and ≈0.09 absolute on `γ` against our 2 % and 0.02.

---

## 3. Feature summary

**Solver core.** The stacked `[η,𝖴x,𝖴y]` loop-free residual + hand Jacobians; time loops defaulting
to the fully-implicit `RungeKutta(:SDIRK_2_2)` (`:theta` Crank–Nicolson selectable) — sequential
LU+Newton, distributed GMRES+Jacobi+Newton; the full nonlinear physics (advection, the full leading
pressure `R_P`, all eight `𝓝` components) in **both** regimes. Wavemaker / sponge / wall / periodic
BCs; runtime monitoring plus an independent governing-equation residual checker; `w_s`/`p_s` VTK
reconstruction.

**Dirichlet boundary wave generation + WaveSpec coupling.** Regular, multichromatic, or WaveSpec
`AiryState` stochastic sea states (seeded phases ⇒ rank-deterministic). The `:model`
discrete-eigenmode polarization prescribes an exact discrete transport, so a generated wave is a
solution of the discrete equations at the boundary and radiates cleanly.

**Verification.** The analytic MMS (`src/mms.jl`), whose independence from `problem.jl` is enforced
by a grep gate; `test_jacobians_ad.jl`, comparing the hand Jacobians against AD of the same residual
matrix-by-matrix and gating the nonlinear branch on *amplitude scaling*; the linear
one-Newton-iteration gate on a sloping bed.

**Postprocessing.** VTK/CSV → analysis and plots, from-modes `w(σ)`/`p_nh(σ)` reconstruction, and a
sea-state module (Welch PSD, JONSWAP overlay, Hs, Rayleigh exceedance), validated against solver
output.

---

## 4. Physics selection — three orthogonal controls

| control | values | meaning |
|---|---|---|
| `regime` | `:linear` \| `:nonlinear` | linearised core, no advection / full nonlinear core + advection |
| `nl_pressure` | `:none` \| `:native` \| `:full` | 𝓝 off / components `{3,6,7,8}` / `+{1,2,4,5}` |
| `flat_bed` | `Bool` | `true` ⇔ **`∇h ≡ 0`** (every ∇h-term dropped, ∇η/dispersion kept); `false` = variable bathymetry |

`resolve_physics` maps these onto the seven internal booleans; `build_problem_raw` is the low-level
escape hatch. Pressure content is intrinsic to the model: `P_full = advection`,
`lin_pressure = advection ∨ ¬flat_bed`. `flat_bed` acts at a **single control point** —
`dhx,dhy = flat_bed ? 0 : ∂h` in `global_residual`/`jacobian_*`. Full term table: `MODEL.md` §6.

**`nl_pressure=:native` is the production tier.** The whole `{1,2,4,5}` hierarchy contributes
**0.013 % (1-D) / 0.094 % (2-D)** on top of advection's 0.77 % / 1.84 % at `A=1e-3`, and `:full`
carries a mesh-independent velocity-error floor (`VERIFICATION.md` §4).

---

## 5. Current state

**Working.** Feature-complete in both serial and distributed forms: the stacked loop-free residual +
hand Jacobians, the full nonlinear physics, the SDIRK/θ integrators, all boundary treatments, and
Dirichlet boundary wave generation with WaveSpec coupling.

**Verified scope — six of eight models** (`Q3/Q2`, 1-D static unless noted):

| model | `regime` / `flat_bed` / `nl_pressure` | `p_η` (opt 3) | `p_u` (opt 4) | |
|---|---|---|---|---|
| 1 | `:linear` / flat / `:none` | `p_e+1` exactly | `p_u+1` exactly | ✅ |
| 2 | `:linear` / **variable** / `:none` | 3.000 | 4.000 | ✅ |
| 3 | `:nonlinear` / flat / `:none` | 2.996 | 3.995 | ✅ |
| 4 | `:nonlinear` / **variable** / `:none` | 2.996 | 3.997 | ✅ |
| 5 | `:nonlinear` / flat / **`:native`** | 2.996 | 3.997 | ✅ |
| 6 | `:nonlinear` / **variable** / **`:native`** | 2.996 | 3.998 | ✅ |
| 7–8 | `:nonlinear` / any / **`:full`** | 2.99 → 2.59 | **−0.00** | ⛔ not MMS-verifiable **by construction** |

Model 2 additionally confirmed transient (2.999/3.998) and 2-D (3.000/3.963).

> **Say "the `:none` and `:native` models are verified."** Never bare *"the residual is verified"*
> (which would wrongly include `:full`), and never *"the `𝓝` tiers are verified"* (same error).

**Suite** (all measured 2026-08-18/19, not carried over): sequential **20/20 files**, distributed
**13/13 gates** on 4 ranks, `test_jacobians_ad` **17/17** over 8 models, `test_mms_convergence_nonlinear`
**8/8**, `test/local/` **50/50**. Per-file scores: `TEST_SUITE.md` §2.

**Open work** — nothing is half-built in the solver; the open items are verification gaps,
performance, and follow-through. Full list with decisive next steps: `OPEN_ITEMS.md`.

* 🔴 cluster memory attribution (4 GB/core is required; *why* is open — H4 leads)
* 🔴 `test_mms_convergence` G7 — a gate-window specification decision, not a fix
* 🟠 no MPI tier in `runtests.jl` (cost three stale reference constants)
* 🟠 preconditioner replacement — the single biggest performance item
* 🟠 four run-output gaps
* naming follow-through outside this checkout: GitHub repo, cluster checkout, sysimage rebuild

---

## 6. The design decision everything rests on

**The vertical (layer) index lives in the FE value type, not in a Julia array.**

* Velocity is two vector-valued fields `𝖴x, 𝖴y ∈ VectorValue{Nσ}`. The MultiField is
  `[η, 𝖴x, 𝖴y]` — **3 fields, not `1+2Nσ`**.
* The static vertical arrays become constant `TensorValue` / `ThirdOrderTensorValue` objects.
* Every layer sum `Σ_j`, `Σ_{kj}` is therefore a matvec or tensor double-contraction — **the
  residual contains no vertical-index loops**.
* The MultiField is touched only for `η=U[1]`, `𝖴x=U[2]`, `𝖴y=U[3]`.

This makes the residual well-typed by construction, ~3.6× faster with ~13× fewer allocations than
the fused per-layer form, and — because `Operation` is forwarded for `DistributedCellField` —
**one residual and one pair of Jacobians serve both sequential and MPI execution**. Details,
including the verified Gridap contraction facts and the variable dictionary: `ARCHITECTURE.md` §1.

---

## 7. Standing rules

These are the rules that cost something to learn. Each is stated where it is enforced; the
supporting measurement is in the linked document.

### Model and residual

1. **`R_P` is the entire frequency dispersion of the model.** Only the *boundary* part of its
   integration by parts vanishes; the volume part must be assembled. Without it the model degenerates
   to non-dispersive shallow water.
2. **`fe_order ≥ 2`.** `Q1` elements zero `R_P` and disable all non-hydrostatic physics.
3. **`B_stored = −B̃ ≤ 0`**, and the explicit `(−1)` factors in the `R_P` and slope-pressure terms
   are load-bearing.
4. **THE ASSEMBLY INVARIANT: every classification row must have exactly ONE consumer, guarded by the
   CONJUNCTION of its three activation conditions.** A guard testing only the bed condition or only
   the amplitude condition is a defect whenever the physics has a second representation in the other
   regime — which, for the `𝓐/𝓚` packages, it always does. They are **alternatives, never addends**.
   **Corollary: a flat-bed regression can never test `∇h` code.** (`MODEL.md` §7)
5. **`∂R/∂u̇` is EXACT in all eight models; nonlinear `∂R/∂u` is QUASI-NEWTON by choice**, its gap
   vanishing at order 1.11–1.16 in amplitude. **An omission is benign only if it is HIGHER ORDER IN
   AMPLITUDE** — a block whose prefactor does not scale with the solution is an `O(1)` error in the
   effective mass matrix, and Newton then converges to the fixed point of the *wrong map*. **Never
   assume; measure** with `test_jacobians_ad.jl`'s amplitude-scaling gate. Do not "complete" the
   omissions without re-measuring every nonlinear reference value.
6. **Never apply `∇` to an `Operation`-composed expression containing a test basis** — expand by
   hand via `∂_a(W⋅𝓣) = (∂_aW)⋅𝓣`.
7. **NESTED CLOSURES MUST DECLARE `local`.** In Julia, a nested function assigning a name already
   local to an enclosing function **assigns the enclosing variable**. This codebase is full of long
   functions with nested helpers, so the hazard is **structural**. One instance cost two days.
   **Re-run the mechanical audit after adding any nested helper.** (`VERIFICATION.md` §7)

### Boundaries and stability

8. **Solid-wall Dirichlet BCs must include the corner tags** — otherwise 4 corner DOFs are
   unconstrained and the run diverges exponentially.
9. **IC-release problems need `x_wall_bc=true`** — a free x-wall plus the dispersion term is a
   spurious-forcing mode that an initial perturbation excites directly.
10. **The open-boundary spurious mode is η-dominated, so the sponge MUST damp η** (`+∫ μ q η`), not
    just velocity. No value of `mu_max` absorbs it otherwise. (`WAVE_GENERATION.md` §3)
11. **Sponge strength saturates past `μ_max ≈ 5ω` — WIDTH is the lever.** And the width must cover
    the **longest** component (`kd_min` ⇒ `λ_max`), not the peak.
12. **`ny ≥ 3` is mandatory for a y-periodic mesh** (Gridap `CartesianGrids.jl:39`).
13. **`A_wave ≤ 0.001 m`** for stable long fully-nonlinear integrations.
14. **The `c_g` transit trap.** At `kd = 5.5`, `c_g = 1.25 m/s` — filling a 45 m flume takes 22.5
    periods. Budget `t_settle ≈ (x_sponge − x_source)/c_g + 3T` before reading any steady state. It
    caught three separate measurements.

### Solver and execution

15. **The default integrator (`SDIRK_2_2`) is L-stable, i.e. DISSIPATIVE BY CONSTRUCTION.** Any test
    measuring a non-dissipative property must pin `solver_type=:theta`. **Do not remove those pins,
    and never fix such a failure by moving a threshold.** Recognise it by: amplitude damped while
    **phase is correct**, error **growing with frequency**, and **refining the mesh does not help** —
    that last check separates it from genuine under-resolution. (`TEST_SUITE.md` §4)
16. **Distributed linear solve = `NewtonSolver(GMRESSolver(Pr=Jacobi))`.** A direct LU does not scale
    to partitioned matrices at cluster size.
17. **`krylov_m` (basis size, memory) and `ls_maxiter` (iteration budget, time) are different
    bounds**, and `restart=true` is load-bearing. Symptom of getting this wrong: **`gmres=` pinned at
    exactly the same number every step.** (`ARCHITECTURE.md` §5)
18. **`norm(PVector, Inf)` is broken** (PartitionedArrays 0.3.5) — reduce over `own_values`.
19. **Distributed ICs use `interpolate_everywhere`**, never `FEFunction(U, zeros(…))`.
20. **Keep `ConsecutiveMultiFieldStyle`** — `BlockMultiFieldStyle` breaks Jacobi's `diag`.
21. **Launch MPI with `~/.julia/bin/mpiexecjl`**; the system `mpiexec` fails on a PMIx mismatch.
    `MPI_Finalize` prints a benign OFI error and exits 143 on this machine.
22. **Keep horizontal cells near-isotropic, or pay for it in the linear solve** — a 4:1-celled mesh
    needs ~760 GMRES iterations against ~480 for an isotropic one.
23. **Use MPI when the problem is big enough to give each rank a meaningful share, and a direct LU
    otherwise.** Spend spare cores on more *cases*, not on decomposing one small case further.
24. **`nl_tol` is a step function and DOES change the answer** (integer Newton counts); `ls_rtol`
    does not, anywhere in 1e-9…1e-5. Do not conflate them. (`CONFIGURATION.md` §4)
25. **A sysimage is only valid for the versions it was built against** — build it in the environment
    you run in, and rebuild after **any** `src/*.jl` edit. Staleness is *detected*, not prevented.
26. **Julia buffers stdout when redirected to a file**; use `flush` or poll.
27. **Revise does not hot-swap signature changes** — restart before trusting a number after any
    signature or struct change.

### Testing and measurement

28. **"The suite passes" is not "the model is verified."** Most of the suite is
    **self-consistency**: writing `R = R_true + E`, the error appears on both sides and cancels
    identically, so such a test passes for **any** residual. Only the analytic MMS — whose forcing
    never touches `problem.jl` — can detect a self-consistently wrong residual. (`VERIFICATION.md` §0)
29. **AD is an oracle for the JACOBIAN, never for the RESIDUAL.** "AD converges where hand fails,
    therefore the residual is correct" is an invalid inference — AD differentiates the *same*
    assembled residual.
30. **A BOUNDS CHECK ON THE RIGHT CONFIGURATION IS NOT A VALUE CHECK.** A gate that only asserts
    nothing exploded will pass while the quantity it computes moves 58 %.
31. **For a guard that is a CONJUNCTION (`nonlinear ∧ ∇h ≠ 0`), the suite needs a case satisfying the
    conjunction AND asserting a value.** Conjuncts satisfied separately, or together but only
    bounded, both look green.
32. **A refinement study measures the rate of whichever error DOMINATES — verify isolation in BOTH
    directions before interpreting any slope.** A saturated slope and a wrong coefficient produce the
    *same* observable. **Guards come in pairs.**
33. **Read the pairwise rate SEQUENCE, not the fitted slope**, and check error *magnitude* before
    trusting a fine-level high-order rate.
34. **THE RESOLUTION PRINCIPLE: a test validates a term only if it can RESOLVE that term's
    contribution.** Always ask: *if this term were wrong, would this test have noticed?*
35. **A batch runner must take its verdict from GATE OUTPUT, never from exit codes.** A clean exit
    code is not evidence a test ran.
36. **A mirrored guard that collapses two per-field measurements into one pass/fail can hide the
    very asymmetry it exists to detect.**
37. **Two tests sharing a bathymetry function and a physics tier can still be different problems.**
    A reference-vs-reference comparison is meaningful only when *every* discretisation parameter
    matches.
38. **When an error message names a library type, that is where the bug SURFACED, not where it
    lives.** Print the type of *every* input to the failing expression before searching that library.
39. **Test a diagnosis against a case it cannot explain, rather than looking harder where it points.**
40. **Diagnostics interpretation:** never read `growth` without `x_at_max`; `dmp/int` means "≫1 is
    trouble", never "<1 is fine"; mass drift is an invariant **only** in a closed, unforced basin.
    (`ARCHITECTURE.md` §6)

---

## 8. Conventions for editing here

* **`BALFEM_models/` is the single source of mathematical truth.** `MODEL.md` mirrors its notation.
  If you change the maths, change both.
* **Notation bridge, the most common source of confusion.** LaTeX `M^V, 𝓜^V, 𝓖^V, A^V, K^V, 𝓐^V,
  𝓚^V, Φ, φ_j` ↔ code `Mmat, Mcal, Gcal, A, K, Acal, Kcal, Phi/D/C`, with the unit basis stored as
  `w_j = −φ_j_int`. Always cross-reference `MODEL.md` §2 when quoting tensor names.
* **If you add a `using X` to any `src/*.jl`, add it to `[deps]`.**
* **`building_files/` is gitignored** — nothing in it is version-controlled, so anything load-bearing
  belongs in the tracked `CLAUDE.md`, `README.md` or test files instead.
* **Run Julia via the `julia-mcp` tool.**
