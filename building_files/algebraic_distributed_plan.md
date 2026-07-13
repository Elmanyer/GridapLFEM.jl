# Distributed-memory port plan — serial/distributed algebraic LFE-M solver

**Goal.** Bring the algebraic package (`LFEM_2D/src/LFEModelAlg.jl`) to full feature parity with
the old solver (`../LFE-M_2D_solver/`) for **distributed-memory (MPI) execution**: partitioned
mesh/assembly, scalable Newton–Krylov solve, distributed VTK output including the reconstructed
w / total-pressure fields, cluster-ready example scripts. The old solver's distributed modules are
the template; every functionality is adapted to the stacked notation and workflow.

**Status: EXECUTED 2026-07-10 — validation results in §7.**

---

## 0. The structural win that shapes this whole port

The old solver needed **two** distributed paths: the Gridap `TransientFEOperator` path (linear core
only — its fused advection allocates ~1 GB/cell) and a hand-written **owned V⊗H θ-loop** with
manual DOF reconciliation for the nonlinear advection. The algebraic residual removes the reason
for that split: the stacked contraction integrand is cheap (validated ~3.6× faster / ~13× fewer
allocations than the fused per-layer form), so

> **ONE distributed path — Gridap `TransientFEOperator` + hand Jacobians + GMRES(Jacobi)+Newton —
> carries the FULL physics** (advection, linear slope pressure, `P_full`, nonlinear-pressure 6–8),
> serial and distributed, with the *same* residual/Jacobian code.

Verified prerequisites (GridapDistributed 0.4.13 source): `Operation` is forwarded for
`DistributedCellField` (unary/binary/varargs) — so every `alg_*` helper works on distributed
fields; FE bases and `TransientCellField`/multifield transients are wrapped per part; `CellField(f,
trian)` and `createvtk/createpvd` have distributed methods. Hence `residual_alg`,
`jacobian_u_alg`, `jacobian_u_t_alg` require **zero changes**.

## 1. Old → new functionality map

| Old solver artefact | New artefact | Adaptation |
|---|---|---|
| `timeloop2D_dist.jl :: build_horizontal_model_2D_distributed` | `timeloop_alg_dist.jl :: build_horizontal_model_alg_distributed` | verbatim (mesh is layout-agnostic) |
| `timeloop2D_dist.jl :: build_ode_solver_2D_distributed` (GMRES+Jacobi+NewtonSolver+θ) | `build_ode_solver_alg_distributed` | verbatim — the GridapSWE-pattern stack (LU is not scalable; `norm(PVector,Inf)` broken → own-values reduction) |
| `timeloop2D_dist.jl :: _eta_max_distributed_2D` | `eta_max_alg_distributed` | verbatim (`own_values` + `reduce(max,…;init=0.0)`) |
| `timeloop2D_dist.jl :: run_time_loop_2D_distributed` | `run_time_loop_alg_dist` | VTK block adapted: per-σ-node components extracted from the stacked fields via `alg_component` (`Operation(v->v[j])` — forwarded distributed) with the OLD field names `eta,u1x,u1y,…`; `createpvd(ranks,…)`, `i_am_main` guards, `MPI.Barrier`; gauges omitted (inter-rank point search) — diags carry `(t, eta_max)` |
| `utilities_lfem2D_dist.jl :: setup_and_run_lfem_distributed` (linear core only) | `utilities_alg_dist.jl :: setup_and_run_alg_distributed` | `with_mpi() do distribute … end` wrapper; vertical stage replicated per rank (tiny); **all flags supported distributed** (`advection, lin_pressure, P_full, nl_pressure68`) — no linear-core restriction, no V⊗H twin needed |
| `advection_vxh_dist_lfem2D.jl` (owned distributed V⊗H loop) | **not needed** | superseded by §0 |
| `horizontal2D.jl` MultiField eltype dispatch (`DistributedSingleFieldFESpace[…]` vs `SingleFieldFESpace[…]`) | `horizontal_alg.jl :: build_fe_spaces_alg` | same runtime `isa` trick, 3-element space vectors; keep ConsecutiveMultiFieldStyle (block style breaks Jacobi's `diag`) |
| `FEFunction(U, zeros(…))` IC (sequential-only) | `make_initial_conditions_alg(U, Nσ; eta0_func)` | `interpolate_everywhere([η₀, x->0⃗, x->0⃗], U)` — works sequential AND distributed (root rule #14); optional `eta0_func` enables IC-release problems (hump/soliton; those REQUIRE `x_wall_bc=true`) |
| `reconstruct_fields_lfem2D.jl :: build_field_recon / extra_field_cellfields` (w, total p per σ-node into VTK) | `reconstruct_alg.jl :: build_field_recon_alg / extra_field_cellfields_alg` | stacked rewrite — per level ℓ the modal sums collapse to THREE constant-vector contractions: `w_ℓ = −(a_ℓ⋅𝖺) + (b_ℓ⋅𝖻) − (c_ℓ⋅𝖲)` with `a_ℓ=φ(σ_ℓ), b_ℓ=σ_ℓφ(σ_ℓ), c_ℓ=φ_int(σ_ℓ)` and the residual's own building blocks `𝖺,𝖻,𝖲`; `p_ℓ = ρg(1−σ_ℓ)H − ρd²(π_ℓ⋅ΔDU/dt)` with `π_ℓ,j=∫_{σ_ℓ}^1φ_j_int` (second antiderivative BVP) and `u̇` by backward FD from tracked previous DOFs (`trial_space` passed explicitly — `get_fe_space` undefined distributed). Sanity: `p_nh|_{σ=1}=0` by construction; pure CellField algebra → identical code serial/distributed |
| `test_basic_lfem2D_distributed.jl` (4 ranks, linear core) + `test_basic_vxh_lfem2D_distributed.jl` (nonlinear) | `test/test_basic_alg_distributed.jl` | ONE test, 4 ranks (2×2): linear core AND fully nonlinear on the same 16×2 flume as `test_basic_alg.jl`; asserts no-NaN/bounded/wave + agreement with the sequential reference amplitudes (GMRES vs LU ⇒ tolerance 1e-3 rel) |
| `examples/distributed/` (`_dist_common.jl` + 4 run scripts + README/SLURM) | `examples/distributed/` (`_dist_common_alg.jl` + 4 run scripts + README/SLURM) | same env-var configuration pattern (`LFEM_*`), same four cases: plane wave, ring wave, IC hump (closed basin), bathymetry/shoaling (variable `d(x)`, `lin_pressure=true`); driver call renamed + new-physics flags exposed (`LFEM_PFULL`, `LFEM_NLP68`) |

## 2. Step-by-step execution order

| Step | Content |
|---|---|
| **D1** | Module plumbing: add `GridapDistributed / PartitionedArrays / MPI / GridapSolvers` to `LFEModelAlg.jl`; distributed-safe `build_fe_spaces_alg` (eltype dispatch); `interpolate_everywhere`-based ICs with `eta0_func` hook |
| **D2** | `reconstruct_alg.jl` (w / total-p σ-level fields, stacked contractions) + wire `write_w/write_pressure/rho` into the SEQUENTIAL driver and time loop (recon is layout-agnostic — build it once, use in both loops) |
| **D3** | `timeloop_alg_dist.jl` (mesh builder, GMRES+Jacobi+Newton factory, own-values eta_max, distributed time loop with VTK + recon) |
| **D4** | `utilities_alg_dist.jl :: setup_and_run_alg_distributed` (full-flag driver, `with_mpi`, replicated vertical stage, distributed IC) |
| **D5** | `test/test_basic_alg_distributed.jl` + sequential regression re-run (`test_basic_alg.jl` must still pass with the D1/D2 changes) |
| **D6** | `examples/distributed/` scripts + README (mpiexecjl + SLURM template) |
| **D7** | Local validation: sequential smoke; 4-rank MPI test; tiny-config distributed example smoke (validates env plumbing + distributed VTK with w/p fields end-to-end); docs + memory update |

## 3. Load-bearing conventions carried over (do NOT rediscover these)

1. Distributed solver stack: `ThetaMethod(NewtonSolver(GMRESSolver(m; Pr=JacobiLinearSolver(),
   rtol,…)), dt, θ)` — `NLSolver`/`LUSolver` are sequential-only at scale.
2. `norm(PVector, Inf)` is broken (PartitionedArrays 0.3.5) → `own_values` + `reduce(max,…)`.
3. Distributed IC: `interpolate_everywhere` (never `FEFunction(U, zeros)`).
4. Keep **ConsecutiveMultiFieldStyle** (BlockMultiFieldStyle breaks Jacobi's `diag`).
5. Launch with `~/.julia/bin/mpiexecjl` (system `mpiexec` → PMIx mismatch); `MPI_Finalize` prints a
   benign OFI error and exits 143 on this machine.
6. `LFEM_NX` divisible by `LFEM_PX`, `LFEM_NY` by `LFEM_PY`; `-n == PX·PY`.
7. All rank-0-only printing behind `i_am_main(ranks)`; `mkpath` on rank 0 + `MPI.Barrier`.
8. Julia buffers stdout when redirected — background runs monitored via explicit `flush`/file polling.
9. IC-release problems (hump/soliton) MUST set `x_wall_bc=true` (free x-walls + dispersion term =
   spurious-forcing instability).

## 4. Deliverable file map

```
LFEM_2D/
├── src/
│   ├── LFEModelAlg.jl            # + distributed deps, includes, exports        (D1)
│   ├── horizontal_alg.jl         # + distributed eltype dispatch                (D1)
│   ├── timeloop_alg.jl           # + recon wiring in sequential loop            (D2)
│   ├── utilities_alg.jl          # + write_w/write_pressure/rho/eta0_func       (D2)
│   ├── reconstruct_alg.jl        # NEW: w / total-p σ-level fields (stacked)    (D2)
│   ├── timeloop_alg_dist.jl      # NEW: distributed mesh/solver/time loop       (D3)
│   └── utilities_alg_dist.jl     # NEW: setup_and_run_alg_distributed           (D4)
├── test/
│   └── test_basic_alg_distributed.jl   # NEW: 4 ranks, linear + nonlinear       (D5)
└── examples/distributed/
    ├── _dist_common_alg.jl       # env-var plumbing (LFEM_*)                    (D6)
    ├── run_plane_wave_alg_dist.jl
    ├── run_ring_wave_alg_dist.jl
    ├── run_ic_hump_alg_dist.jl
    ├── run_bathymetry_alg_dist.jl
    └── README.md                 # launch guide + SLURM template
```

## 5. Risks & mitigations

- **`Operation` on distributed bases**: verified forwarded in GridapDistributed source; the
  4-rank test exercises every helper through residual AND Jacobian assembly.
- **GMRES convergence for the stacked system**: same operator as the validated old linear-core
  distributed runs plus the (well-conditioned) advection block; Newton tolerances identical to the
  old stack; monitored in the test.
- **Reconstruction `u_prev` handling distributed**: previous DOFs copied as PVector and re-wrapped
  with `FEFunction(trial_space, prev_vals)` (the old solver's proven pattern).
- **Per-rank JIT under MPI**: first distributed run compiles on every rank (~tens of minutes);
  tests use the tiny 16×2 mesh so compile dominates but wall time stays bounded.

## 6. Validation gates

1. Sequential regression: `test_basic_alg.jl` still 6/6 after D1/D2 (module additions must not
   disturb the serial path); plus a serial smoke with `write_w=write_pressure=true` checking
   `p_s1_000 ≈ 0` at the surface and `w_s0_000 ≈ 0` on the flat bed.
2. `test_basic_alg_distributed.jl` on 4 ranks: linear + nonlinear, bounded/no-NaN/wave, sequential
   agreement within GMRES tolerance.
3. Distributed example smoke (tiny env config, 2 ranks): script plumbing + distributed VTK pieces
   containing `eta,u*,w_s*,p_s*`.

## 7. Validation results (2026-07-10 execution)

| Gate | Result |
|---|---|
| sequential regression (`test_basic_alg.jl`) | **6/6 PASS** (post-D1/D2 module changes) |
| serial w/p field smoke | **PASS** — `w` at flat bed = 0 exactly, total `p` at surface = 0 exactly, bed `p ≈ ρgH` (+ small nh correction); VTK carries `eta, u{1..3}{x,y}, w_s{0.000,0.728,1.000}, p_s{…}` |
| `test_basic_alg_distributed.jl` (4 ranks, 2×2) | **6/6 PASS** — linear core max η = 0.0037095 (rel 4.6e-9 vs sequential LU), **fully nonlinear** max η = 0.0124478 (rel 1.7e-9); the GMRES+Jacobi+Newton distributed solve reproduces the sequential results to ~9 digits, both modes, 120 steps each |
| distributed example smoke (2 ranks, `run_plane_wave_alg_dist.jl`, tiny env config) | **PASS** — env plumbing + distributed VTK (pvtu + per-rank pieces) with `eta,u*x,u*y,w_s*,p_s*` |

Note: the distributed↔sequential agreement at ~1e-9 (vs the 2e-3 tolerance budgeted for
GMRES-vs-LU) confirms both the partitioned assembly of the stacked residual/Jacobians and the
solver stack; the fully nonlinear case exercises `Operation`-based contractions
(`alg_dx/alg_mul/alg_dc3/alg_outer`) through distributed residual AND Jacobian assembly.

## 8. Addendum (2026-07-13) — `nl_pressure_full` distributed mass solve

The pressure-completion pass (`algebraic_pressure_completion_plan.md`) added `nl_pressure_full`
(nonlinear-pressure components 1,2,4,5), whose frozen L²-projections `π𝖲, π𝖻` need one mass-matrix
solve per RHS per step. The first cut used sequential-only `lu()` and the distributed driver
rejected the flag. Closed the gap:

* **Problem**: `lu()` has no method for a partitioned `PSparseMatrix` — same reason the main
  time-stepping solver never uses `LUSolver` distributed.
* **Fix**: `build_nlp_ctx(...; distributed=true)` (`src/nlpressure_alg.jl`) builds
  `CGSolver(JacobiLinearSolver())` from GridapSolvers — the mass matrix is SPD and
  well-conditioned (unlike the advection-dominated Jacobian), so CG rather than GMRES is the
  natural choice — and reuses one `numerical_setup` every step.
* **A second, subtler bug surfaced first**: `assemble_vector(f, V)` and `allocate_in_domain(A)`
  are only *isomorphic* PRanges, not the same object (different ghost/assembly-cache layout);
  passing an `assemble_vector`-built RHS into `CGSolver`'s `solve!` throws
  `AssertionError: matching_ghost_indices(...)` inside `mul!`. Fix: allocate RHS/solution vectors
  FROM the matrix (`allocate_in_range(A)` / `allocate_in_domain(A)`) and assemble the RHS
  in-place into that buffer (`assemble_vector!`) — the pattern applies unchanged sequentially too.
* **Validated**: `test/test_nlpressure_alg_distributed.jl` (4 ranks, 2×2, tanh-bar case, ALL
  pressure flags on) matches the sequential (LU-projection) reference to **rel 4.6e-9** — the
  same order of agreement as the main distributed Newton solve vs sequential LU.
