# Implementation plan — algebraic (loop-free) LFE-M residual

> **STATUS 2026-07-10: EXECUTED.** Implemented in `algebraic_lfem2D.jl` (P1–P4, P5 comps 6–8, P6
> driver) and validated by `test_algebraic_lfem2D.jl`: 16/16 (primitives + virtual-work equivalence
> vs `residual_lfem`, rel ≤ 7e-15, three flag configs), time integration matches the oracle to
> machine precision (linear 40 steps, nonlinear 20 steps), ~3.6× faster / ~13× fewer allocations
> than the fused per-layer oracle on the nonlinear run. Open: P5 comps 1–5, `Pcal`, AD experiment,
> dispersion benchmark, distributed twin.

Companion of `algebraic_residual_math.md` (the operator simplifications — read it first). Goal:
a **new residual function** that follows `main.tex` §8, evaluated with native Gridap tensor algebra,
**no per-layer MultiField decomposition and no vertical-index `for`-loops** in the residual — the
style the notebook `test_2HDmodel.ipynb` attempts but gets rank/dimension-wrong.

The enabling decision (math doc §0): **stack the layer index into the FE value type** — velocity is
two `VectorValue{Nσ}` fields `𝖴x,𝖴y`; the MultiField is `[H, 𝖴x, 𝖴y]` (3 fields), and the static
vertical arrays become constant `TensorValue`/`ThirdOrderTensorValue`.

---

## 0. Scope & phasing

| Phase | Content | Difficulty |
|-------|---------|-----------|
| **P1** | Stacked FE space + `to_vec/to_tensor2/to_tensor3` + `wdot3` helper + a unit test of the primitives (index-order, matvec, `⊙`) | low |
| **P2** | Algebraic residual: **mass, acceleration, gravity, leading pressure `R_P`** (all native, exact; math §6b — `R_P` is the dispersion and is MANDATORY) | low |
| **P3** | **Advection** (`𝗠3`,`𝗚3` via test-contracted `⊙`) | medium (1 rank-3 op) |
| **P4** | **Linear pressure** `L` (§6 math): three `VectorValue{Nσ}` + component matvecs (the same `L¹,L²,L³` already feed `R_P`) | medium |
| **P5** | **Non-linear pressure** `N`: comps **6–8** native; comps **1–5** behind a flag via IBP(∇h)/aux-field(∇H); + the `𝓟`-part of `R_P` (needs `Pcal` assembly) | high |
| **P6** | Wavemaker + sponge + BC adaptation; drivers; validation vs current solver | medium |

P1–P4 already deliver the full **validated linear-regime physics** (dispersion, sloshing, shoaling
of small waves) in the clean algebraic form — but only because `R_P` is in P2: **without `R_P` the
model is non-dispersive SWE** (2026-07-10 correction; math doc §6b). P5 is the genuinely hard,
`O(A³)` part — stage it.

---

## 1. FE spaces — the stacked layout

```julia
# vertical pre-compute (unchanged): σ-mesh → ϕ,φ(=φ_int),Φ and the static arrays
#   M2 (Nσ²), M3,G3 (Nσ³), A2,K2 (Nσ²×3), A3,K3 (Nσ³×8)   [as in the notebook / vertical_lfem2D.jl]
Nσ = num_free_dofs(U_phi)                    # = M·p+1

# horizontal spaces — THREE fields
reffe_H  = ReferenceFE(lagrangian, Float64,            orderH)          # H  (or η)
reffe_U  = ReferenceFE(lagrangian, VectorValue{Nσ,Float64}, orderU)     # stacked layer velocity
V_H  = FESpace(model, reffe_H;  conformity=:H1)
V_Ux = FESpace(model, reffe_U;  conformity=:H1 #=+ y-wall Dirichlet, see §6=#)
V_Uy = FESpace(model, reffe_U;  conformity=:H1)
X = MultiFieldFESpace([V_H, V_Ux, V_Uy])     # trial/test analogously
```

**Consequences / things that change from the per-layer solver:**

* Field access is `H=U[1]; 𝖴x=U[2]; 𝖴y=U[3]` (three, not `1+2Nσ`). `∂t(U)` gives `[Ht,∂t𝖴x,∂t𝖴y]`.
* **VTK / reconstruction**: a `VectorValue{Nσ}` field writes as an `Nσ`-component array. The
  `write_w`/`write_pressure` reconstruction (in `reconstruct_fields_lfem2D.jl`) currently reads
  per-layer scalar fields via `ix(j)/iy(j)`; it must instead read `𝖴x[j]`/`𝖴y[j]` components (a
  `VectorValue` component access) — trivial adaptation.
* **Solve unknown `H` vs `η`.** Keep `H=d+η` as in the current solver by making the first field `η`
  and using `H=d_cf+η` inside the residual (so rest state is `η≡0`; better conditioning and the
  gravity/pressure baselines match the validated solver). Decide once and be consistent.

## 2. Vertical constants (build once, outside the residual)

```julia
𝚽  = to_vec(Φvec)                               # VectorValue{Nσ}
𝗠  = to_tensor2(M2)                             # TensorValue{Nσ,Nσ}
𝗠3 = to_tensor3(M3)                             # ThirdOrderTensorValue{Nσ,Nσ,Nσ}, index [i,k,j]
𝗚3 = to_tensor3(remap_ikj(G3))                  # re-map notebook [i,j,k] → [i,k,j] (math §1)
A = ntuple(c->to_tensor2(getindex.(A2,c)), 3)   # A[1],A[2],A[3]  (component split of A2)
K = ntuple(c->to_tensor2(getindex.(K2,c)), 3)
P = ntuple(c->to_tensor2(Pmat[:,:,c]), 3)       # leading-pressure P^V (solver vert.P; P[3]=−B)
𝗔3 = ntuple(c->to_tensor3(remap(getindex.(A3,c))), 8)   # 𝓐^V per component
𝗞3 = ntuple(c->to_tensor3(remap(getindex.(K3,c))), 8)
# 𝗣3 (𝓟^V = ∫Θ_kj φᵢ_int, the nonlinear leading pressure) is NOT in vertical_lfem2D.jl yet —
# assemble alongside Kcal (same quadrature, no −σΘφᵢ part) when P5 is enabled.
```

`getindex.(A2,c)` peels the `c`-th component out of the `Nσ×Nσ` array of `VectorValue{3}`.
**Fix the `[i,k,j]` order** when filling `𝗠3/𝗚3/𝗔3/𝗞3` (math §1 warning; the notebook’s `G3` is
`[i,j,k]`).

## 3. Residual skeleton

```julia
ex = VectorValue(1.0,0.0); ey = VectorValue(0.0,1.0)
∂x(f) = ex⋅∇(f);  ∂y(f) = ey⋅∇(f)               # verify orientation once (math §0 ⚠)

function residual(t, U, V)
    Ut = ∂t(U)
    η, Ux, Uy   = U[1], U[2], U[3]
    ηt, Uxt, Uyt = Ut[1], Ut[2], Ut[3]
    q, Wx, Wy   = V[1], V[2], V[3]
    H  = d_cf + η                                # d_cf = CellField(d_func, Ωₕ);  ∇H = ∇d + ∇η

    # derived layer-vectors (VectorValue{Nσ})  — math §0
    DU  = ∂x(Ux) + ∂y(Uy)
    DW  = ∂x(Wx) + ∂y(Wy)                                  # test-divergence vector (math §6b)
    UgH = ∂x(H)*Ux + ∂y(H)*Uy
    Ugh = ∂x(d_cf)*Ux + ∂y(d_cf)*Uy
    S   = H*DU + UgH
    ū   = Operation(VectorValue)(𝚽⋅Ux, 𝚽⋅Uy)

    # P2  mass + acceleration + gravity + leading pressure R_P
    r  = ∫( q*ηt - H*(∇(q)⋅ū) )dΩ
    r += ∫( H*((Wx⋅(𝗠⋅Uxt)) + (Wy⋅(𝗠⋅Uyt))) )dΩ
    # gravity — ORACLE-MATCHING IBP form with hydrostatic baseline subtracted (math §4 note):
    #   linearised:  −∫ g·η·(𝚽⋅DW)      nonlinear:  −∫ (g/2)(H²−d²)·(𝚽⋅DW)
    r += ∫( -(0.5*g)*(H*H - d_cf*d_cf)*(𝚽⋅DW) )dΩ
    # R_P (math §6b) — the dispersion; MANDATORY. Assembled here as the NET −R_P contribution
    # (main.tex defines R_P = +∫H²(sP⋅DW) and subtracts it). Uses the L fields defined below:
    L1 = -(∂x(d_cf)*Uxt + ∂y(d_cf)*Uyt)
    L2 =   ∂x(H)*Uxt + ∂y(H)*Uyt
    L3 = -( H*(∂x(Uxt)+∂y(Uyt)) + L2 )
    #   P_full=false (oracle-exact): only the P³L³ product;  P_full=true: all three comps
    sP = P_full ? (P[1]⋅L1 + P[2]⋅L2 + P[3]⋅L3) : (P[3]⋅L3)
    r += ∫( -(H*H)*(sP⋅DW) )dΩ            # linearised branch: −∫ d²·((P[3]⋅L3lin)⋅DW), L3lin=−d·DUt

    # P3  advection   (math §5; double_contraction(𝓣,T2) = 𝓣⊡T2 → VectorValue{Nσ}, native)
    TMx = Ux⊗∂x(Ux) + Uy⊗∂y(Ux);  TMy = Ux⊗∂x(Uy) + Uy⊗∂y(Uy)
    TGx = S⊗Ux;                    TGy = S⊗Uy
    r += ∫( H*( double_contraction(𝗠3,TMx)⋅Wx + double_contraction(𝗠3,TMy)⋅Wy )
              + ( double_contraction(𝗚3,TGx)⋅Wx + double_contraction(𝗚3,TGy)⋅Wy ) )dΩ

    # P4  linear pressure   (math §6; reuses L1,L2,L3 from R_P above)
    πAx = (Wx⋅A[1])⋅L1 + (Wx⋅A[2])⋅L2 + (Wx⋅A[3])⋅L3      # scalars
    πAy = (Wy⋅A[1])⋅L1 + (Wy⋅A[2])⋅L2 + (Wy⋅A[3])⋅L3
    πKx = (Wx⋅K[1])⋅L1 + (Wx⋅K[2])⋅L2 + (Wx⋅K[3])⋅L3
    πKy = (Wy⋅K[1])⋅L1 + (Wy⋅K[2])⋅L2 + (Wy⋅K[3])⋅L3
    r -= ∫( H*( ∂x(d_cf)*πAx + ∂y(d_cf)*πAy + ∂x(H)*πKx + ∂y(H)*πKy ) )dΩ

    # P5  nonlinear pressure — comps 6–8 native; 1–5 flagged (see §5); + 𝓟-part of R_P
    # r -= nonlinear_pressure(...)

    # wavemaker source (continuity) + sponge (momentum) — §6
    r -= ∫( q*wm_src(t) )dΩ                                # MINUS, as in the oracle residual_lfem
    r += ∫( μ*((Wx⋅(𝗠⋅Ux)) + (Wy⋅(𝗠⋅Uy))) )dΩ
    return r
end
```

Notes: `∂x(H)=∂x(d_cf)+∂x(η)`; keep `d_cf` analytic so `∇d`, `∇²d` are exact (needed by comp. 3 of
`N`). `Operation(VectorValue)(…)` fuses two scalar CellFields into a `VectorValue{2}` (native).

## 4. `∂t`, `H`, bathymetry

* `∂t(U)` on the stacked MultiField yields `∂t` of each field — `Uxt,Uyt` are `VectorValue{Nσ}`.
  The effective mass operator (acceleration + the `L³` dispersion term, both `∝∂t`) is what the
  transient solver’s `jacobian_u_t` differentiates.
* `H=d+η`. Use `(H²−d²)=2dη+η²` if you switch gravity to the `½∇(H²)` energy form (current solver’s
  rest-state fix) — optional; the `gH∇η` form above already vanishes at rest.
* Variable bathymetry enters only through `d_cf`, `∇d_cf`, `∇²d_cf` — all analytic, no extra fields.

## 5. Non-linear pressure staging (P5)

* **Always-on, native (comps 6–8):** build the three `(k,j)` tensors `N6=-Ugh⊗S/H`, `N7=UgH⊗S/H`,
  `N8=-S⊗S/H`; contract `Σ_{c∈{6,7,8}} H(∂_αh)(Wα·𝗔3^{(c)})⊙N^{(c)} + H(∂_αH)(Wα·𝗞3^{(c)})⊙N^{(c)}`.
  First-order, cheap.
* **Flagged (comps 1–5):** second derivatives. Implement via the math-doc §7 routes:
  * **∇h half** — IBP onto the test (EXACT, uses analytic `∇²h`). Reuse the *proven* logic already in
    `advection_vxh_lfem2D.jl :: np_residual` (∇h branch) — port it to the stacked layout.
  * **∇H half** — irreducible `∂²η`. Prefer the **auxiliary-field** route (add `𝗤≈∇(∇·(H𝖴))` solved
    by `∫𝗤·φ=-∫𝖲(∇·φ)`), else reuse the current L²-projection branch. Keep behind `nonlin_pressure`.
* Rationale: `O(A³)`, only for finite-amplitude harmonics / strong shoaling; not needed for the
  linear benchmarks. Do **not** block P1–P4 on it.

## 6. BC / wavemaker / sponge / drivers

* **Solid wall** `u_j^y=0 ∀j` becomes a Dirichlet BC on the **whole** `𝖴y` field (all `Nσ`
  components zero) on the y-wall tags (incl. corners — mandatory, per root CLAUDE.md). Similarly
  `𝖴x=0` for `x_wall_bc`. One `dirichlet_tags`+`VectorValue{Nσ}`-zero per field; simpler than the
  current per-component tagging.
* **Wavemaker**: unchanged scalar source in continuity (`∫ q·S(x,t)`); §Wave-generation of the tex.
* **Sponge**: `∫ μ (Wx·(𝗠·Ux)+Wy·(𝗠·Uy))` (math form of `μ Σ_j M_ij u_j`).
* **Field output**: adapt `extra_field_cellfields` to read `Ux[j]`,`Uy[j]` components (§1).
* **Drivers**: a new `setup_and_run_lfem_algebraic` mirroring `setup_and_run_lfem`, building the
  stacked space and this residual; `TransientFEOperator(residual, X, Y)` (AD Jacobian — see §7).

## 7. Jacobian

* **Try Gridap AD first.** The notebook’s AD crash was **not** an AD limitation — it was the
  `U[2:end]` slice. With the stacked, well-typed residual, `TransientFEOperator(res, X, Y)` (AD) is
  worth trying; the residual is now a clean composition of native ops. Expect heavier compile for
  `ThirdOrderTensorValue` at `Nσ=7`.
* **Fallback**: hand-code `jacobian_u`/`jacobian_u_t`. The linear terms (P2, P4-`L`, sponge) are
  linear in the unknowns ⇒ their Jacobian equals the operator; advection (P3) is the only quadratic
  core (Picard/frozen-coefficient linearisation, as in the current V⊗H loop, is exact-in-applied-field
  and converges in 2–5 Newton iters).

## 8. Helper functions to write

| helper | signature | notes |
|--------|-----------|-------|
| `to_vec` | `Vector{Float64}→VectorValue{Nσ}` | |
| `to_tensor2` | `Matrix→TensorValue{Nσ,Nσ}` | |
| `to_tensor3` | `Array{,3}→ThirdOrderTensorValue{Nσ,Nσ,Nσ}` | fix `[i,k,j]` order |
| `∂x,∂y` | `CellField→CellField` | `e_{x,y}⋅∇` (orientation ✓ verified) |

**No contraction helper is needed** — the rank-3 double contraction `𝓣⊡𝖲` is native
`double_contraction(𝓣,𝖲)` (contracts trailing two indices → `VectorValue{Nσ}`; verified). Everything
in the residual is `⋅`,`⊗`,`⊙`,`double_contraction`,`∇` on `VectorValue{Nσ}`/`TensorValue{Nσ,Nσ}`/
`ThirdOrderTensorValue`. Only the constant-tensor constructors (`to_*`) are hand-written, at build
time.

## 9. Validation (must do before trusting it)

1. **Primitive unit test** (fast, no solve): on a small mesh, interpolate a known `[η,𝖴x,𝖴y]`; check
   `∂x/∂y` orientation, `𝗠⋅Uxt`, `⊗`, `⊙`, `double_contraction` against hand values.
2. **Residual equivalence** (the key test): assemble this residual **and** the current per-layer
   `residual_lfem` on the *same* FE state and assert agreement to `~1e-10` term-by-term (mass, acc,
   dispersion-`R_P`, grav, adv, lin-press). **Methodology that avoids any DOF-layout mapping**
   (the two MultiField layouts order DOFs differently — do NOT try to permute vectors):
   * Choose smooth analytic fields `η(x,y)`, `uⱼˣ(x,y)`, `uⱼʸ(x,y)` (and a second set for `u̇`);
     `interpolate_everywhere` them into **both** layouts → identical physical states (exact for
     polynomials of degree ≤ fe_order).
   * Build the transient states by hand: `TransientCellField(uh, (uht,))` (from `Gridap.ODEs`), so
     `∂t(u)` returns the chosen `u̇` state in both residuals.
   * Choose analytic **test** fields the same way, interpolate into both **test** spaces, and
     compare the scalars `r(v) = dot(get_free_dof_values(v_interp), assemble_vector(r, V))` —
     the virtual work of the residual on the same physical test function is layout-independent.
     Repeat for ~5 random polynomial/trig combos.
   * Match flags exactly: oracle `linearised/advection/lin_pressure` ↔ algebraic flags, gravity in
     the oracle's IBP form, `P_full=false` (oracle keeps only the `P³L³` dispersion product), same
     sponge/wavemaker functions (or zero them).
3. **Dispersion / propagation**: reuse the existing FEM benchmarks (`test_dispersion_lfem2D`,
   plane wave) on the new driver; compare phase speed to the validated solver.
4. Only then enable P5 and benchmark against Stokes harmonics / a shoaling case.

## 10. Risks & open questions (verify early)

1. **✓ RESOLVED — `∇` index order.** `∇` of a `VectorValue{Nσ}` field is `TensorValue{2,Nσ}`
   (spatial-first); `∂_x f = e_x⋅∇f` (verified: `(x,2y,x+y)⇒(1,0,1),(0,2,1)`). Use `e⋅∇f`.
2. **✓ RESOLVED — rank-3 contractions are native.** `double_contraction(𝓣,𝖲)` contracts the trailing
   two indices → `VectorValue{Nσ}`; `𝖶⋅𝓣` contracts the first index → `TensorValue{Nσ,Nσ}` (both
   verified). No helper. Store `𝓣` with `[i,k,j]` = `[output, u_k-layer, u_j-layer]`.
3. **`ThirdOrderTensorValue{7,7,7}`** (M=6): StaticArrays construction/AD cost (343 entries).
   Mitigation: contract the test in first (`𝖶⋅𝓣` keeps only a 2-tensor live) or per-`i` split.
   *Measure at P3.*
4. **AD through the stacked residual** compile time at `Nσ=7`. Fallback: hand Jacobian (§7).
5. **`Operation(VectorValue)` inside AD** — fine in current solver; confirm it survives the AD path.
6. **Second derivatives (P5)** — the `∇H`-half `∂²η` is genuinely irreducible; the auxiliary-field
   route is the clean fix but adds fields/coupling. Keep P5 optional.

## 11. Deliverable module layout (suggested)

```
src/
├── vertical_lfem2D.jl              # unchanged (builds M2/M3/G3/A2/K2/A3/K3)
├── algebraic_tensors_lfem2D.jl     # NEW: to_vec/to_tensor2/to_tensor3, wdot3, component split
├── horizontal_algebraic_lfem2D.jl  # NEW: stacked [H,𝖴x,𝖴y] MultiFieldFESpace + wall BC
├── problem_algebraic_lfem2D.jl     # NEW: residual() above (+ optional hand Jacobians)
└── utilities_algebraic_lfem2D.jl   # NEW: setup_and_run_lfem_algebraic driver
```

Keep the current per-layer solver as the **oracle** for the §9.2 equivalence test; the algebraic
solver is a parallel path, not a replacement, until it passes.
