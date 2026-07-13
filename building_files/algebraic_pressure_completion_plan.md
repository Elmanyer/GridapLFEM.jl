# Pressure-physics completion plan — the last residual blocks of the LFE-M model

**Goal.** Implement the remaining pressure terms of the corrected `main.tex` §8 residual in
`LFEModelAlg`, completing the model physics: the nonlinear slope-pressure components 1–5
(`𝓐/𝓚` halves) and the nonlinear part of the leading pressure (`𝓝⫶𝓟^V`). Then port the two
missing regression tests (mass conservation, sloshing period) and close with an AD-Jacobian probe.

**Status: EXECUTED — results in §6.**

---

## 0. Where each missing term stands (from main.tex §8, pressure-operator subsection)

Every nonlinear-pressure component `c` appears in THREE residual blocks, with different prefactors:

| block | prefactor | integral shape |
|---|---|---|
| bed-slope `𝓐` | `H ∂_αh` | `−∫ H∂_αh (W_α ⋅ (𝓐3[c] ⊡ N^(c)))` |
| surface-slope `𝓚` | `H ∂_αH` | `−∫ H∂_αH (W_α ⋅ (𝓚3[c] ⊡ N^(c)))` |
| leading `𝓟` (part of `R_P`) | `H²`, tested by `∇·V` | `−∫ H² (𝗣3[c] ⊡ N^(c)) ⋅ D_W` |

Admissibility classes (main.tex ground rules; `C⁰` spaces admit only FIRST derivatives of unknowns):

- **Native (c = 3, 6, 7, 8)** — first-order everywhere. c=6,7,8 are outer products of the building
  blocks `𝖺,𝖻,𝖲` (the `1/H` cancels one prefactor `H`); c=3's only second derivative is the
  ANALYTIC bed Hessian (`∂_a𝖺` differentiates `∇h` and first FE derivatives). Implementable in all
  three blocks, all paths (serial + distributed).
- **∇h-half of c = 1, 2, 4, 5 — EXACT integration by parts onto the test.** The prefactor
  `Ψ_c = H∂_αh (W_α ⋅ 𝓣3[c])` is smooth-in-the-bad-direction (test appears UNdifferentiated;
  `∂Ψ` produces first test derivatives + bed Hessian only). Works in the residual directly →
  serial AND distributed.
- **∇H-half and 𝓟-part of c = 1, 2, 4, 5 — irreducible** (`∂Ψ ⊃ ∂²η` for the `𝓚` half; `∂Ψ ⊃
  ∇(∇·v)` = second TEST derivatives for the `𝓟` part — IBP is unusable in both). Route:
  **frozen L²-projections** (project-then-differentiate, one step lag): per time step project
  `𝖲` and `𝖻` onto the velocity FE space (`π𝖲, π𝖻` — two SPD mass solves, factorised once), use
  `∂_a(π𝖲), ∂_a(π𝖻)` in the residual as frozen fields from the previous step. Consistent
  (`O(dt)` lag on an `O(A³)` term), quasi-Newton (excluded from the Jacobian, like the oracle's
  `𝓝` treatment). SEQUENTIAL path first (the old solver's nonlinear pressure was owned-loop-only
  too); the distributed driver rejects the flag with a clear error.

## 1. Exact-IBP algebra for the ∇h half (the error-prone part — derived once, tested at machine precision)

With `Ψ_c := H ∂_αh (W_α ⋅ 𝓐3[c]) ∈ TensorValue{Nσ,Nσ}` (test contracted into the FIRST index,
native), `(Ψ·u)_k = Σⱼ Ψ_{kj}u_j`, `(u·Ψ)_j = Σ_k u_kΨ_{kj}`, and `M := Σ_a (∂_a𝖲)⊗U_a`
(so `N¹ = −M`, `N² = M + 𝖲⊗D`):

```
−∫Ψ₁⊙N¹ = +∫ Σ_a (Ψ₁·U_a)⋅∂_a𝖲   →IBP→  −∫ 𝖲⋅[∂_x(Ψ₁·U_x)+∂_y(Ψ₁·U_y)] + ∮
−∫Ψ₂⊙N² = −∫ Σ_a (Ψ₂·U_a)⋅∂_a𝖲 − ∫Ψ₂⊙(𝖲⊗D)   (first piece IBP with sign flipped; second native)
−∫Ψ₄⊙N⁴ = −∫ Σ_a (U_a·Ψ₄)⋅∂_a𝖻   →IBP→  +∫ 𝖻⋅[∂_x(U_x·Ψ₄)+∂_y(U_y·Ψ₄)] − ∮
−∫Ψ₅⊙N⁵ = +∫ Σ_a (U_a·Ψ₅)⋅∂_a𝖲   →IBP→  −∫ 𝖲⋅[∂_x(U_x·Ψ₅)+∂_y(U_y·Ψ₅)] + ∮
−∫Ψ₃⊙N³ = +∫ Σ_a (U_a·Ψ₃)⋅∂_a𝖺   (DIRECT — bed Hessian analytic, no IBP)
```

Slot bookkeeping: c=1,2 carry the differentiated divergence in the **k** slot (`Ψ·U`), c=3,4,5 in
the **j** slot (`U·Ψ`) — the k/j asymmetry is precisely why the machine-precision identity gate G1
below uses a deliberately asymmetric state. Boundary integrals `∮ (g⋅q) n_a` vanish on solid walls
(`u·n=0` enters `q`) and behind sponges; dropped, consistent with every other flux. The IBP'd
derivatives `∂_a(Ψ·U_a)` differentiate the test ONCE (admissible), the state fields once, `H`,
`∂_αh` and the bed Hessian (analytic).

## 2. Flags & API

- `nl_pressure68` (existing) → **extended to the full native set** {3, 6, 7, 8} across all three
  blocks (`𝓐`, `𝓚`, `𝓟`). First-order, cheap, serial + distributed.
- `nl_pressure_full` (NEW) → adds c = 1, 2, 4, 5: `𝓐` half exact-IBP (§1), `𝓚` + `𝓟` halves via
  frozen projections. Implies a per-step projection context in the sequential time loop;
  distributed driver errors with guidance. Quasi-Newton (no Jacobian contribution — `O(A³)`).
- Struct additions: `nl_pressure_full::Bool`, `nlp_state::Base.RefValue{Any}` (frozen `π𝖲, π𝖻`
  FEFunctions; `nothing` before the first step ⇒ frozen blocks silently zero — exact from rest).

## 3. Deliverables

```
src/nlpressure_alg.jl        NEW: native-set block, ∇h exact-IBP block, frozen block,
                             projection context (mass matrix + LU, once) + per-step update
src/problem_alg.jl           struct fields + calls into the three blocks (flag-gated)
src/timeloop_alg.jl          nlp_ctx kwarg: update_nlp_state! after each accepted step
src/utilities_alg.jl         nl_pressure_full kwarg + ctx wiring
src/utilities_alg_dist.jl    accept & reject nl_pressure_full (clear error), nl_pressure68 OK
src/LFEModelAlg.jl           include + exports
test/test_nlpressure_alg.jl  gates G1–G3
test/test_conservation_alg.jl  port (closed basin, ∫η drift; nonlinear advection ON)
test/test_sloshing_alg.jl      port (standing-wave period vs LFE-M dispersion theory)
```

## 4. Validation gates

- **G1 (machine-precision IBP identity).** On a state built from polynomials of degree ≤ 2
  (Hessians globally constant ⇒ element-wise second derivatives are EXACT in Q2) with `u·n = 0`
  on the whole boundary (`u^x ∝ x(Lx−x)`, `u^y ∝ y(Ly−y)` ⇒ boundary terms vanish identically)
  and a quadratic bed: assemble the ∇h half both DIRECTLY (element-wise `∂_a𝖲`, exact for this
  state) and via the IBP form of §1 — require agreement ~1e-12. Asymmetric layer profiles to
  catch k/j slot swaps.
- **G2 (structural, mirrors old `test_nonlin_pressure_lfem2D` 4/4).** Full block scales as `A³`
  (residual-difference ratio 8 when A doubles); continuity rows untouched; ∇h halves vanish
  identically on a flat bed; `𝓟`-part active on flat bed (it has no slope prefactor).
- **G3 (dynamics).** Short sequential run over the tanh bar with ALL pressure flags on
  (`lin_pressure, P_full, nl_pressure68, nl_pressure_full`): bounded, no NaN, Newton converging.
- **G4 (regression).** `test_basic_alg` unchanged (new flags default off).
- **G5 (ports).** conservation drift < 1e-6; sloshing period error < 5%.
- **G6 (AD probe, time-boxed).** `build_ode_operator_alg_ad` on the 16×2 mesh, linear flags:
  measure first-step wall time; abort/record if impractical.

## 5. Execution order

N1 `nlpressure_alg.jl` native set + struct/flags → N2 exact-IBP block + **G1** → N3 frozen
projections + loop/driver wiring → N4 `test_nlpressure_alg.jl` (**G1–G3**) → N5 regression G4 →
N6 ports G5 → N7 AD probe G6 → N8 docs/memory.

## 6. Results (2026-07-11 execution)

| Gate | Result |
|---|---|
| G1 IBP identity | **PASS** — ‖direct − IBP‖/‖direct‖ = **4.0e-15** (asymmetric layers, quadratic bed+state, hand-derived exact second derivatives) |
| G2 structural | **PASS ×6** — ∇h block ≡ 0 on flat bed; continuity rows exactly zero (native + IBP blocks); scaling ratios **4.0001 / 8.0002 / 4.0001** (∇h-IBP → 4, ∇H-frozen → 8, 𝓟-frozen → 4) |
| G3 dynamics (tanh bar, ALL pressure flags: lin_pressure + P_full + nl_pressure68 + nl_pressure_full) | **PASS** — 40 steps, max η = 0.0069 m, bounded, no NaN, Newton converging |
| G4 regression | **PASS** — `test_basic_alg` 6/6, results byte-identical (0.00371 / 0.01245) |
| G5 conservation / sloshing | **PASS** — closed-basin mass drift **7.8e-16** (machine zero, nonlinear advection on); standing-wave period error **1.44 %** vs LFE-M dispersion theory (same value the old solver achieved) |
| G6 AD probe | **NOT VIABLE** — fails immediately with a `MethodError` in `Gridap.ODEs.time_derivative` during the multifield AD split (`TransientMultiFieldCellField` constructor missing for per-field-dualized fields, Gridap 0.19.11 `MultiFieldAutodiff.jl`). Not a compile-cost issue and independent of our residual's complexity: transient multifield AD with `∂t(u)` in the residual is structurally unsupported. Hand Jacobians remain the design (they are exact for everything but the flag-gated O(A³) pressure blocks, which are quasi-Newton like the oracle). |

**Distributed follow-up (2026-07-13, see `algebraic_distributed_plan.md` §8):** `nl_pressure_full`
was extended to run distributed — `build_nlp_ctx(...; distributed=true)` uses
`CGSolver(JacobiLinearSolver())` for the frozen-projection mass solves instead of the
sequential-only `lu()`, with RHS/solution vectors allocated from the matrix
(`allocate_in_range`/`allocate_in_domain` + `assemble_vector!` in place — an independently
assembled vector is only isomorphic to the matrix's PRange, not identical, and `solve!`'s
internal `mul!` asserts exact equality). Validated 4-rank vs sequential, rel 4.6e-9. Every
physics flag in this plan now runs on both paths.

**One implementation lesson worth keeping** (now encoded in `nlpressure_alg.jl` docstrings): `∇`
must NEVER be applied to an `Operation`-composed expression containing a TEST basis (block-array
`copyto!` is not implemented) — expand such derivatives by hand using linearity of the vertical
contraction in the test, `∂_a(W⋅𝓣) = (∂_aW)⋅𝓣`, and the collapse `Σ_a Ψ⋅∂_aU_a = Ψ⋅DU`; the same
policy is applied to state expressions (bed Hessian obtained once via `∇∇(d_cf)` and composed
afterwards).
