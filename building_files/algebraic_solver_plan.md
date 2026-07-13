# Algebraic solver package — port plan (`LFEM_2D/src` + tests + examples)

**Goal.** Turn the validated single-file algebraic residual (`algebraic_lfem2D.jl`, 16/16 vs the
oracle) into a **self-contained solver package inside `LFEM_2D/`**, mirroring the module/test/example
structure of `../LFE-M_2D_solver/` but written entirely in the stacked algebraic style
(`[η, 𝖴x, 𝖴y]`, constant `TensorValue`/`ThirdOrderTensorValue` vertical tensors, loop-free
residual). The old solver is **not modified**; it remains the oracle.

**Status: EXECUTED 2026-07-10 — see §6 for validation results.**

---

## 1. Package layout (mirror of the old solver)

```
LFEM_2D/
├── src/
│   ├── LFEModelAlg.jl        # module entry: deps, Ex/Ey/E1_sig, includes, exports
│   ├── vertical_alg.jl       # σ-mesh + FULL vertical tensor assembly (ports vertical_lfem2D.jl;
│   │                         #   ADDS Pcal_ikj = ∫Θ_kj φᵢ_int dσ — nonlinear leading pressure)
│   ├── tensors_alg.jl        # constant-tensor constructors + Operation pointwise helpers
│   ├── horizontal_alg.jl     # 2D Cartesian mesh + stacked 3-field MultiFieldFESpace + wall BCs
│   ├── problem_alg.jl        # AlgebraicLFEM struct + loop-free residual + hand Jacobians
│   ├── timeloop_alg.jl       # ODE solver factory + time loop (VTK via per-component extraction)
│   └── utilities_alg.jl      # dispersion analysis, sponge, wavemakers, driver setup_and_run_alg
├── test/
│   ├── test_vertical_alg.jl     # fast: tensor identities, ΣΦ=1, P[3]=−B, Table-1 applicable kd
│   ├── test_primitives_alg.jl   # fast: constructors index order, ∂x/∂y, ⊗, double_contraction
│   ├── test_basic_alg.jl        # smoke: 16×2 flume, 3 periods, linearised + fully nonlinear
│   ├── test_dispersion_alg.jl   # FEM dispersion gate: kd=3, DFT phase, <3% error
│   └── test_equivalence_alg.jl  # ★ oracle test: virtual-work match vs LFEModel2D.residual_lfem
└── examples/
    ├── plane_wave_alg.jl     # long-crested plane wave (quick 8-period default; production in header)
    └── ring_wave_alg.jl      # point-source ring waves (quick 8-period default)
```

`algebraic_lfem2D.jl` + `test_algebraic_lfem2D.jl` (repo root of `LFEM_2D/`) remain as the
session-validated standalone prototype; the package supersedes them for all new work.

## 2. Port map — src modules

| Old (`../LFE-M_2D_solver/src/`) | New (`LFEM_2D/src/`) | Adaptation |
|---|---|---|
| `vertical2D.jl :: build_vertical_model` | `vertical_alg.jl` | copied verbatim (σ-map mesh) |
| `vertical_lfem2D.jl :: compute_antiderivative, assemble_vertical_tensors_lfem` | `vertical_alg.jl :: assemble_vertical_tensors_alg` | same math + quadrature; **adds `Pcal[i,k,j,c] = ∫Θ_kj[c] φᵢ_int dσ`** (already computed as the first Fubini piece of `Kcal` — stored instead of discarded); returns the same NamedTuple fields (`Mmat, Phi, Mcal, Gcal, A, K, P, Acal, Kcal, Pcal, B, …`) |
| *(new)* | `tensors_alg.jl` | `alg_to_vec/alg_to_tensor2/alg_to_tensor3` (column-major data, verified) + `alg_dx/alg_dy/alg_mul/alg_dot/alg_dc3/alg_outer/alg_vec2` Operation helpers |
| `horizontal2D.jl :: build_horizontal_model_2D` | `horizontal_alg.jl` | copied verbatim |
| `horizontal2D.jl :: build_fe_spaces_2D` (1+2Nσ scalar fields, per-component tags) | `horizontal_alg.jl :: build_fe_spaces_alg` (3 fields; `VectorValue{Nσ}` reffe) | wall BC = Dirichlet on the WHOLE stacked field, zero `VectorValue{Nσ}`; same tag sets incl. mandatory corners |
| `problem_lfem2D.jl :: LFEMProblemLFEM, residual_lfem, jacobian_*` | `problem_alg.jl :: AlgebraicLFEM, residual_alg, jacobian_u_alg, jacobian_u_t_alg` | the validated loop-free forms; flags `linearised/advection/lin_pressure/P_full/nl_pressure68`; full-Newton advection Jacobian; oracle IBP gravity `−(g/2)(H²−d²)(𝚽⋅DW)`; wavemaker `−∫q·src` |
| `timeloop2D.jl :: build_ode_operator, build_ode_solver, make_initial_conditions, run_time_loop` | `timeloop_alg.jl` | operator builds from `problem_alg`; solver factory copied; **VTK adapted**: per-σ-node components extracted from the stacked fields via `Operation(v -> v[j])`, written with the OLD field names (`eta, u1x, u1y, …`) so ParaView pipelines are unchanged |
| `utilities2D.jl :: find_wavenumber, dispersion_ratio, applicable_kd, make_sponge_2D, make_wavemaker_line/point` | `utilities_alg.jl` | copied; `dispersion_ratio` reads `vert.Mmat/vert.B/vert.Phi` (v2 names) |
| `utilities_lfem2D.jl :: setup_and_run_lfem` | `utilities_alg.jl :: setup_and_run_alg` | same kwargs + new flags; `LFE_DEFAULT_CBDY` table local |

**Notation adaptation rule** (applies everywhere): per-layer loops `for j in 1:N` + `field_sum`/
`mat_field_sum`/`mat_div_sum` → constant-tensor contractions (`𝗠⋅𝖴`, `𝚽⋅𝖴`,
`double_contraction(𝗧3, a⊗b)`); per-component `∇(f)⋅Ex` → `alg_dx(f)` (`e⋅∇f`, spatial index
first — load-bearing for `VectorValue{Nσ}` fields); `ix(j)/iy(j)` field indexing → `u[2]/u[3]`
component access only.

## 3. Port map — tests

| Old test | New test | Notes |
|---|---|---|
| `test_vertical_tensors_lfem.jl` (28/28) | `test_vertical_alg.jl` | subset: Mmat sym/PSD, ΣΦ=1, `φ_int(0)=0`, `φ_int(1)=Φ`, `P[:,:,3]=−B`, `Kcal`/`Pcal` Fubini split identity, applicable_kd vs Yang&Liu Table 1 (M=2→~10.8, M=3→~39.2) |
| *(new, from prototype)* | `test_primitives_alg.jl` | tensor-constructor index order + `∂x/∂y` orientation + `⊗`/`double_contraction` semantics |
| `test_basic_lfem2D.jl` | `test_basic_alg.jl` | same 16×2 flume config; runs linearised baseline AND fully nonlinear (advection on); no NaN, bounded, wave generated |
| `test_dispersion_lfem2D.jl` | `test_dispersion_alg.jl` | same kd=3 DFT-phase gate, <3% |
| *(the acceptance test)* | `test_equivalence_alg.jl` | loads BOTH modules (`using .LFEModel2D`, qualified `LFEModelAlg.*`); virtual-work equality on interpolated analytic states, 3 configs × 3 test sets, rel < 1e-10 |

## 4. Port map — examples

| Old | New | Notes |
|---|---|---|
| `plane_wave2D.jl` | `plane_wave_alg.jl` | same physics (d=3.5, T=1.6, A=0.001, line source); default = quick validation size (8 periods, 200×10); production values (50 periods, 400×15) in the header; `save_every` VTK on |
| `ring_wave2D.jl` | `ring_wave_alg.jl` | point source at basin centre, 4-side sponge; quick default (8 periods, 80×80), production (20 periods, 200×200) in header; radial gauges |

## 5. Execution order & gates

1. **P1** src modules; module loads (`include("src/LFEModelAlg.jl")`).
2. **P2** `test_vertical_alg.jl` + `test_primitives_alg.jl` (fast, no horizontal FEM JIT).
3. **P3** `test_equivalence_alg.jl` — the oracle gate (must be ~1e-10 before anything else runs).
4. **P4** `test_basic_alg.jl` (smoke, both modes).
5. **P5** `test_dispersion_alg.jl` (physics gate — exercises the R_P dispersion term end-to-end).
6. Examples committed as scripts (quick defaults); not run to completion (multi-hour), same policy
   as the old solver's production examples.

## 6. Validation results (2026-07-10 execution)

| Gate | Result |
|---|---|
| module load | OK (loads standalone, no LFEModel2D dependency) |
| `test_vertical_alg.jl` | **15/15 PASS** — identities incl. `P[:,:,3]=−B`, `Pcal−Kcal=∫σΘφᵢ≥0`; applicable kd: LFE-2 → 10.8, LFE-3 → 39.2 (Table 1) |
| `test_primitives_alg.jl` | **9/9 PASS** — tensor index order, trailing-2 `double_contraction`, `e⋅∇f` orientation, `⊗`, matvec/dot |
| `test_equivalence_alg.jl` | **10/10 PASS** — ported vertical tensors match oracle < 1e-13; virtual work rel ≤ 7e-15 across configs A/B/C |
| `test_basic_alg.jl` | **6/6 PASS** — 16×2 flume, 120 steps, linearised (max η 0.0037) and fully nonlinear (max η 0.0125 < 20A), wave generated |
| `test_dispersion_alg.jl` | **1/1 PASS** — kd=3, Cm=2.463 vs Ce=2.486, **err 0.90 %** (< 3 %; oracle: 1.03 %) |
| VTK output path | **OK** — snapshots + `solution.pvd` written; fields `eta,u1x,u1y,…` identical naming to the old solver |

## 7. Deferred → status update (2026-07-10, second pass)

- **w / total-pressure VTK field reconstruction** — **DONE** (`src/reconstruct_alg.jl`,
  `write_w`/`write_pressure` on both drivers; per-level modal sums collapse to three
  `alg_dot` contractions; sanity `w|bed=0`, `p|surface=0` exact). See
  `algebraic_distributed_plan.md`.
- **Distributed (MPI) twin** — **DONE** (`src/timeloop_alg_dist.jl`, `src/utilities_alg_dist.jl`,
  `test/test_basic_alg_distributed.jl` 6/6 on 4 ranks incl. FULLY NONLINEAR, `examples/distributed/`
  cluster scripts). One Gridap path carries all physics distributed — no V⊗H owned loop needed.
  See `algebraic_distributed_plan.md`.
- **Nonlinear pressure comps 1–5** (IBP ∇h / projection ∇H) and the `𝓟`-part of `R_P` (tensor
  `Pcal` IS already assembled by `vertical_alg.jl` — only the residual block remains). Still open.
- **AD Jacobian experiment** (`build_ode_operator_alg_ad` exists; compile cost unmeasured). Still open.
