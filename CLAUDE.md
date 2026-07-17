# CLAUDE.md — `GridapLFEM.jl/` (LFE-M derivation & the algebraic-residual project)

**What this folder is.** The *mathematical* home of the 2D LFE-M model: the LaTeX derivation
(`main.tex`), a Jupyter prototype (`test_2HDmodel.ipynb`), and a set of Markdown documents that
translate the derivation into a Gridap implementation. It is the design/paper side; the *running
solver* lives one level up in `../LFE-M_2D_solver/` (see that folder's `CLAUDE.md`).

**The forward task this folder exists to support:** implement a **new, fully algebraic (loop-free)
residual function** for the LFE-M model — one that follows `main.tex` §8 using native Gridap tensor
algebra instead of per-layer `for`-loop index contractions. `algebraic_residual_math.md` and
`algebraic_residual_plan.md` are the spec + plan for that task. **A new agent picking this up should
read those two files, then §5–§7 below, then start at Phase P1 of the plan.**

---

## 0. Orientation — the LFE-M model in one paragraph

LFE-M (Yang & Liu 2024, *JFM* 999 A32) is a depth-integrated, **non-hydrostatic** free-surface wave
model. The water column `σ∈[0,1]` (terrain-following) is discretised with `M` vertical finite
elements of order `p`, so the horizontal velocity is `u_h(x,σ,t)=Σ_j u_j(x,t) φ_j(σ)`. The vertical
velocity `w` and the non-hydrostatic pressure `p_nh` are eliminated **analytically** (no pressure
Poisson solve) into a set of small, precomputable **vertical σ-tensors**. What remains is a system of
`Nσ+2` coupled **2-D PDEs** in the unknowns `(H, u_0,…,u_{Nσ})` — solved on a horizontal FE mesh by
Gridap. `H=d+η` (still-water depth `d`, free surface `η`). This is a **(small dense vertical algebra)
⊗ (large sparse horizontal FE)** structure.

`Nσ` **convention (important):** in the v2 code `Nσ = num_free_dofs(U_phi) = M·p+1` = the number of
vertical nodes; velocity modes are 1-based `j=1..Nσ`, one per node. (Do **not** assume `Nσ=M·p`.)

---

## 1. Files in this folder

| File | What it is | Status |
|------|-----------|--------|
| `Project.toml` / `Manifest.toml` | **Not a registered package** (no `name`/`uuid`/`version` — intentional, `GridapLFEM` is loaded via `include()`+`using .GridapLFEM`, never `Pkg.add`). This is a standalone environment pinned to the SAME package versions as the parent `../Project.toml` (Gridap 0.19.11, GridapDistributed 0.4.13, GridapSolvers 0.6.2, PartitionedArrays 0.3.5, MPI 0.20.26 — 2026-07-13). If you add a new `using X` to any `src/*.jl` file, run `Pkg.add("X")` here too (`--project=GridapLFEM.jl`) or this environment silently falls out of sync and `--project=GridapLFEM.jl` breaks while the parent env keeps working. | pinned to parent |
| `main.tex` / `LFE_M_model_discretisation.pdf` | **Authoritative** step-by-step derivation, 3D Euler → σ-map → vertical FE → §8 Gridap global residual. **This is the reviewed `main_corrected.tex`, promoted 2026-07-10**: §8 now also contains the field-layout subsection (why `[H,𝖴x,𝖴y]`), the detailed 𝓛/𝓝 pressure-operator implementation subsection, and the wave-generation/sponge/BC subsection (integrated from `wavemaker_sponge_BC.tex`); §9 = full horizontal discretisation, completed & corrected (static tensors + state-weighted `C̄/S̄/P̄` operators = the V⊗H route). ⚠ Sign convention (this version): `R_P ≡ +∫H²[(𝓛:P^V)+(𝓝⫶𝓟^V)]·(∇·V)` and it is **subtracted** in the global sum, uniformly with `R_lin`/`R_nonlin` — the framed residual and all physics are unchanged. The PDF is stale until recompiled. | authoritative |
| `LFEM_Gridap.md` | Clean synthesis of `main.tex` §1–§8 (the derivation leading to the single scalar residual). Uses **`main.tex` notation exclusively**; §9 is a notation bridge to the solver code. | done |
| `wavemaker_sponge_BC.tex` | Source of the wavemaker/sponge/BC subsection — **now integrated into `main.tex` §8** (with the BC boundary-flux list modernised there); kept as the standalone original. | integrated |
| `algebraic_residual_math.md` | ★ Operator simplifications: how to write the §8 residual with native Gridap tensor ops, **no MultiField decomposition, no vertical-index loops**. The `L`/`N` pressure stacks, the leading-pressure `R_P` (§6b), IBP of second-derivative terms, verified Gridap operator table. | **spec — start here** |
| `algebraic_residual_plan.md` | ★ Phased implementation plan (P1–P6) for the new residual: FE-space redesign, tensor constants, residual skeleton, Jacobian, validation, risks. | **plan — then here** |
| `src/` (`GridapLFEM.jl` + 10 submodules) | ★★★ **The self-contained SERIAL + DISTRIBUTED algebraic solver PACKAGE** (`module GridapLFEM`): vertical tensors (incl. new `Pcal`), constant-tensor + Operation helpers, stacked spaces (distributed-safe MultiField dispatch), loop-free residual + hand Jacobians (the SAME code runs distributed — Operation is forwarded for `DistributedCellField`), θ time loops (sequential LU+Newton / distributed GMRES+Jacobi+Newton), per-component VTK (`eta,u1x,u1y,…` — old naming) **plus reconstructed `w_s<σ>`/`p_s<σ>` fields** (`reconstruct.jl`, `write_w`/`write_pressure`), drivers `setup_and_run` and `setup_and_run_distributed` (with_mpi; **full nonlinear physics distributed** — no V⊗H loop needed; `eta0_func` IC hook). No dependency on the old solver. | **VALIDATED 2026-07-10** |
| `test/` (6 tests) | `test_vertical` 15/15, `test_primitives` 9/9, `test_equivalence` 10/10 (oracle virtual-work, ≤7e-15), `test_basic` 6/6, `test_dispersion` 1/1 (kd=3 err 0.90%), `test_basic_distributed` (4 ranks 2×2, linear + FULLY NONLINEAR, sequential agreement). | **PASS** |
| `examples/` | `plane_wave.jl`, `ring_wave.jl` (sequential, quick defaults) + `examples/distributed/` — 4 env-configurable cluster scripts (plane wave, ring wave, IC hump closed basin, bathymetry/shoaling) + README with mpiexecjl/SLURM templates. | scripted |
| `algebraic_solver_plan.md` | The package port plan (module/test/example map, notation rules) + execution results. | executed |
| `algebraic_distributed_plan.md` | The distributed-memory port plan (old→new functionality map, GMRES stack conventions, reconstruction port) + validation results. | executed |
| `algebraic_pressure_completion_plan.md` | The pressure-physics completion plan: ALL eight 𝓝 components in all three blocks (native {3,6,7,8}; ∇h half {1,2,4,5} exact-IBP; ∇H/𝓟 halves {1,2,4,5} frozen L²-projections) + gates (IBP identity 4e-15; scaling 4/8/4; conservation 7.8e-16; sloshing 1.44%; AD ruled out — Gridap 0.19.11 transient-multifield-AD constructor bug). | executed |
| `algebraic_lfem2D.jl` | Standalone prototype of the residual (pre-package; validated 16/16). Superseded by `src/` — kept for reference. | superseded |
| `test_algebraic_lfem2D.jl` | Prototype validation script (16/16). Superseded by `test/`. | superseded |
| `test_2HDmodel.ipynb` | Prototype notebook. Its vertical pre-compute is good; **its `residual` is broken** (see §6). It is the *motivation* for the algebraic residual, not a working reference. | broken (reference only) |

Notation note: `LFEM_Gridap.md` and `main.tex` use `M^V, 𝓜^V, 𝓖^V, A^V, K^V, 𝓐^V, 𝓚^V, Φ, φ_j`;
the **solver code** uses different names (`Mmat, Mcal, Gcal, A, K, Acal, Kcal, Phi/D/C`, and stores
the unit basis as `w_j = −φ_j_int`). The bridge table is in `LFEM_Gridap.md` §9 and §4 below.

---

## 2. `main.tex` §8 — the target: one scalar residual

Gridap's `MultiFieldFESpace` solver wants a **single scalar** = total virtual work
`∫_Ω R·v`. `main.tex` §8 gives it (the framed `eq: multifield general residual`). With `U=[u_0…u_{Nσ}]`
(stacked velocities), `V=[v_0…v_{Nσ}]`, `q`/`H` the continuity test/trial:

```
Global Residual = ∫_Ω [ q ∂H/∂t − ∇q·(H U)·Φ                                    (mass)
        + ( H·M^V·U̇                                                             (acceleration)
          + H·F_M(U) + F_G(H,U)                                                  (advection)
          + gH∇η·Φ                                                              (gravity, η=H−h)
          − H[ ∇h·(L:A^V + N⫶𝓐^V) + ∇H·(L:K^V + N⫶𝓚^V) ] ) · V                 (lin+nonlin slope pressure)
        − H²[ (L:P^V) + (N⫶𝓟^V) ] · (∇·V) ] dΩ                                  (leading pressure = DISPERSION)
```
with `F_M(U)_i = Σ_kj 𝓜_ikj (u_k·∇u_j)`, `F_G_i = Σ_kj 𝓖_ikj (∇·[Hu_k]) u_j`;
`L_j = [−u̇_j·∇h, u̇_j·∇H, −∇·(Hu̇_j)]` (3 comp); `N_kj` = 8 comp (see §7 below / math doc §7);
`(∇·V)_i = ∇·v_i` (test-divergence vector); `P^V_ij = ∫θ_j φᵢ_int dσ` (3 comp, `P[3]=−B`),
`𝓟^V_ikj = ∫Θ_kj φᵢ_int dσ` (8 comp). **Sign convention (current `main.tex`):** ALL three pressure
blocks are defined as positive integrals and **subtracted** uniformly,
`Global = R_mass + R_Acc + R_Adv + R_Grav − R_P − R_lin − R_nonlin` with
`R_P = +∫H²[(L:P^V)+(N⫶𝓟^V)]·(∇·V)` — the net framed term (shown above with its minus) is what the
code assembles.
**⚠ 2026-07-10:** `R_P` was missing from `main.tex` §8 and all derived docs (the "boundary integral
vanishes" claim was wrong — only the boundary part of the IBP vanishes). It is the ONLY O(η)
non-hydrostatic term on a flat bed = the entire frequency dispersion; without it the model is
non-dispersive SWE. Now corrected everywhere. The wavemaker adds `−∫ q S(x,t)` to
mass; the sponge adds `+∫ μ (M^V U)·V` to momentum (`main.tex` §8, wave-generation subsection).

---

## 3. THE BIG IDEA for the algebraic residual (read this twice)

The whole project turns on **one decision**: *stack the vertical (layer) index into the FE value
type, not into a Julia array.*

* **Velocity = two vector-valued fields** `𝖴x, 𝖴y ∈ VectorValue{Nσ}` (all layers' x- resp.
  y-velocity). The MultiField is `[η, 𝖴x, 𝖴y]` — **3 fields**, not `1+2Nσ` scalar fields.
* The static vertical arrays become **constant tensors**: `M^V→TensorValue{Nσ,Nσ}`,
  `𝓜^V,𝓖^V→ThirdOrderTensorValue{Nσ,Nσ,Nσ}`, and the component-indexed `A^V/K^V/𝓐^V/𝓚^V` are **split
  per component** into 3 (resp. 8) such constants.
* Then **every layer sum `Σ_j`, `Σ_{kj}` is a matvec / tensor double-contraction** — no loops.
* The **spatial index is only 2-D** and is written out explicitly with `e_x=(1,0)`, `e_y=(0,1)`;
  `∂_x f ≡ e_x·∇f`, `∂_y f ≡ e_y·∇f`. That is *not* a loop.
* We touch the MultiField only for `η=U[1]`, `𝖴x=U[2]`, `𝖴y=U[3]` — **no `Nσ`-decomposition**.

This makes the residual well-typed *by construction*, which is exactly what the notebook lacked.

### Verified Gridap facts (checked live this session — rely on them)
1. `∇` of a `VectorValue{Nσ}` field = `TensorValue{2,Nσ}` (**spatial index first**,
   `(∇f)[d,j]=∂f_j/∂x_d`). So `∂_x f = e_x·∇f`, `∂_y f = e_y·∇f`. Verified: `f=(x,2y,x+y) ⇒
   e_x·∇f=(1,0,1)`, `e_y·∇f=(0,2,1)`. **Use `e·∇f`, not `∇f·e`.**
2. `double_contraction(𝓣::ThirdOrderTensorValue, S::TensorValue{Nσ,Nσ})` contracts the **trailing
   two** indices → `VectorValue{Nσ}` = `Σ_{k,j} 𝓣_{ikj} S_{kj}`. **Native — no helper needed.**
3. `W::VectorValue{Nσ} ⋅ 𝓣::ThirdOrderTensorValue` contracts the **first** index → `TensorValue{Nσ,Nσ}`.
   Native. (Alternative "contract the test in first" form.)
4. `⊗` (outer, `VectorValue{Nσ}⊗VectorValue{Nσ}→TensorValue{Nσ,Nσ}`), `⊙` (Frobenius `A:B→scalar`),
   `Operation(VectorValue)(a,b)` (build a `VectorValue{2}` from two scalar CellFields) — all native.

**Consequence: the residual needs ZERO custom contraction primitives.** The only hand-written helpers
are *build-time* constructors that turn the assembled Float arrays into the constant tensors.

### Variable / operator dictionary (algebraic residual)

| symbol | type | meaning | Gridap |
|--------|------|---------|--------|
| `η, H` | scalar CellField | free surface, total depth `H=d+η` | `U[1]`; `H=d_cf+η` |
| `𝖴x, 𝖴y` | `VectorValue{Nσ}` | stacked layer velocities (x, y comp) | `U[2]`, `U[3]` |
| `𝖶x, 𝖶y` | `VectorValue{Nσ}` | test functions | `V[2]`, `V[3]` |
| `d_cf` | scalar | still-water depth `d(x)`; `∇h=∇d`, `∇²h` analytic | `CellField(d_func,Ωₕ)` |
| `𝚽` | `VectorValue{Nσ}` const | depth-average weights `Φ_j` | `to_vec(Φvec)` |
| `𝗠` | `TensorValue{Nσ,Nσ}` const | vertical mass `M^V` | `to_tensor2(M2)` |
| `𝗠3,𝗚3` | `ThirdOrderTensorValue` const | advection `𝓜^V,𝓖^V`, index `[i,k,j]` | `to_tensor3(...)` |
| `A[c],K[c]` | 3×`TensorValue{Nσ,Nσ}` | linear-pressure `A^V,K^V` per component | component split |
| `P[c]` | 3×`TensorValue{Nσ,Nσ}` | leading-pressure `P^V` (`P[3]=−𝗕`, dispersion) | `to_tensor2(vert.P[:,:,c])` |
| `𝗔3[c],𝗞3[c]` | 8×`ThirdOrderTensorValue` | nonlinear-pressure `𝓐^V,𝓚^V` per component | component split |
| `DU`, `DW` | `VectorValue{Nσ}` | per-layer divergence of trial / test | `∂x(𝖴x)+∂y(𝖴y)`, `∂x(𝖶x)+∂y(𝖶y)` |
| `S` | `VectorValue{Nσ}` | `∇·(H u_j) = H·DU + u_j·∇H` | product rule |
| `ū, W̄` | `VectorValue{2}` | depth-avg velocity/test `(Φ·𝖴x, Φ·𝖴y)` | `Operation(VectorValue)(…)` |

**Index order rule (kills the "chaos"):** store `𝓜^V,𝓖^V,𝓐^V,𝓚^V` as `[i,k,j]` =
`[output/test layer, u_k / ∇·(Hu_k) layer, ∇u_j / u_j layer]`. The **notebook's `G3[i,j,k]` uses the
opposite last-two order** — re-map when filling the constant tensors.

---

## 4. The vertical σ-tensors (built once; same as the notebook / `vertical_lfem2D.jl`)

On the σ-mesh (`M` elements, order `p`, optimised nodes `c_bdy`), build `φ_j` (basis), `φ_j'`,
`varphi_j=∫_0^σ φ_j` (solved as a BVP `dφ/dσ=φ_j, φ(0)=0`, degree `p+1`), `Φ_j=∫_0^1 φ_j`. Then:

```
M2[i,j]   = ∫ φ_i φ_j                         → M^V     (vertical mass)
M3[i,j,k] = ∫ φ_i φ_j φ_k                      → 𝓜^V     (horizontal advection)
G3[i,j,k] = ∫ (σΦ_k − varphi_k) φ_j' φ_i       → 𝓖^V     (⚠ notebook index [i,j,k]!)
A2[i,j]   = ∫ φ_i θ_j            (θ_j 3-vec)    → A^V     (linear pressure, ∇h coupling)
K2[i,j]   = ∫ θ_j (varphi_i − σφ_i)            → K^V     (linear pressure, ∇H coupling)
A3[i,j,k] = ∫ Θ_kj φ_i          (Θ_kj 8-vec)   → 𝓐^V     (nonlinear pressure, ∇h)   ⚠ notebook leaves A3=0!
K3[i,j,k] = ∫ Θ_kj (varphi_i − σφ_i)           → 𝓚^V     (nonlinear pressure, ∇H)
θ_j  = [φ_j, σφ_j, varphi_j]                                   (linear shape vector)
Θ_kj = [σΦ_k φ_j, Φ_k varphi_j, φ_j φ_k, σφ_j φ_k, varphi_j φ_k,
        σΦ_j φ_k'−varphi_j φ_k',
        σΦ_j φ_k+σ²Φ_j φ_k'−varphi_j φ_k−σ varphi_j φ_k',
        σΦ_j φ_k−varphi_j φ_k]                                 (nonlinear shape tensor, 8 comp)
```
Solver-code names for these: `Mmat, Mcal, Gcal, A, K, Acal, Kcal`, **plus the leading-pressure
tensor `P[i,j,c] = ∫θ_j[c]·φᵢ_int dσ` (`P[:,:,3] = −B`, the dispersion carrier — REQUIRED by the
R_P term)** (in `../LFE-M_2D_solver/src/vertical_lfem2D.jl`, `assemble_vertical_tensors_lfem`), and
diagnostic `B = −∫ varphi_i varphi_j`. The nonlinear leading tensor `Pcal_ikj[c] = ∫Θ_kj[c]·φᵢ_int dσ`
is **not** in the solver yet (assemble like `Kcal` without the `−∫σΘφᵢ` part; only needed with the
nonlinear pressure). **The current solver's tensors are already validated (28/28 unit tests)** —
reuse them; the algebraic project only needs to *reshape* them into
`TensorValue`/`ThirdOrderTensorValue` constants (peel the component index, re-map to `[i,k,j]`).

---

## 5. The residual, term by term (algebraic form — the target code)

From `algebraic_residual_math.md` (§2–§7). `∂x(f)=e_x·∇(f)`, `∂y(f)=e_y·∇(f)`:

```julia
DU  = ∂x(Ux)+∂y(Uy);  DW = ∂x(Wx)+∂y(Wy)          # per-layer div of trial / TEST (VectorValue{Nσ})
UgH = ∂x(H)*Ux+∂y(H)*Uy;  Ugh = ∂x(d)*Ux+∂y(d)*Uy;  S = H*DU+UgH
ū   = Operation(VectorValue)(𝚽⋅Ux, 𝚽⋅Uy)

mass  : ∫ q*ηt − H*(∇(q)⋅ū)
acc   : ∫ H*( (Wx⋅(𝗠⋅Uxt)) + (Wy⋅(𝗠⋅Uyt)) )
grav  : − ∫ (g/2)*(H*H−d*d)*(𝚽⋅DW)                 # ORACLE IBP form (rest-state safe); lin: −∫g*η*(𝚽⋅DW)
disp  : − ∫ H²*(sP⋅DW)   with  sP = P[1]⋅L1+P[2]⋅L2+P[3]⋅L3   (R_P — MANDATORY, math §6b)
        (P_full=false → sP = P[3]⋅L3 only = oracle's B-term;  lin: −∫ d²*((P[3]⋅L3lin)⋅DW))
adv   : ∫ H*( double_contraction(𝗠3,TMx)⋅Wx + double_contraction(𝗠3,TMy)⋅Wy )
          + ( double_contraction(𝗚3,TGx)⋅Wx + double_contraction(𝗚3,TGy)⋅Wy )
        TMx=Ux⊗∂x(Ux)+Uy⊗∂y(Ux);  TMy=Ux⊗∂x(Uy)+Uy⊗∂y(Uy);  TGx=S⊗Ux; TGy=S⊗Uy
linP  : − ∫ H*( ∂x(d)*πAx + ∂y(d)*πAy + ∂x(H)*πKx + ∂y(H)*πKy )
        L1=−(∂x(d)*Uxt+∂y(d)*Uyt); L2=∂x(H)*Uxt+∂y(H)*Uyt; L3=−(H*(∂x(Uxt)+∂y(Uyt))+L2)
        πAx = (Wx⋅A[1])⋅L1+(Wx⋅A[2])⋅L2+(Wx⋅A[3])⋅L3   (πAy,πKx,πKy analogous)
nlP   : comps 6–8 native (below); comps 1–5 flagged (§7); + 𝓟-part of R_P (needs Pcal)
wm    : − ∫ q*wm_src(t)   (MINUS, as in oracle)      sponge: + ∫ μ*((Wx⋅(𝗠⋅Ux))+(Wy⋅(𝗠⋅Uy)))
```

**Linear pressure `L`** (the thing to get right): do **not** make a `VectorValue{3}` of stacked
objects (notebook's bug). Build its 3 components as 3 separate `VectorValue{Nσ}` fields `L1,L2,L3`,
and `L:A^V = A[1]·L1 + A[2]·L2 + A[3]·L3` (sum of matvecs). Fully native, first-order.

---

## 6. The non-linear pressure `N` — the hard part (staged)

`N_kj` (8 comp) as `(k,j)` `TensorValue{Nσ,Nσ}` fields:
* **Components 6,7,8 are first-order** (outer products) — always implementable:
  `N6=−Ugh⊗S/H`, `N7=UgH⊗S/H`, `N8=−S⊗S/H`.
* **Components 1–5 carry second derivatives** of the unknowns: `∂𝖲=∇(∇·(H𝖴))` ("gradient of a
  divergence") and `∂²η`. Not usable directly on `Q2`. Two routes (math doc §7):
  * **∇h half** → **integrate by parts onto the test function** (EXACT; uses analytic bed Hessian
    `∇²h`). The current solver already has this logic in `../LFE-M_2D_solver/src/advection_vxh_lfem2D.jl
    :: np_residual` (∇h branch) — port it.
  * **∇H half** → `∂²η` is **irreducible**. Prefer an **auxiliary-field / mixed** formulation (add
    `𝗤≈∇(∇·(H𝖴))` solved by `∫𝗤·ψ=−∫𝖲(∇·ψ)`, IBP), keeping everything first-order and AD-safe; else
    reuse the current L²-projection branch. Flag-gate it (`nonlin_pressure`); it is `O(A³)`, only
    needed for finite-amplitude harmonics / strong shoaling.

**Do not block the linear phases on the nonlinear pressure.** P1–P4 already give the full validated
linear-regime physics (dispersion, sloshing, small-amplitude shoaling) in clean algebraic form.

---

## 7. Why the notebook `residual` fails (so you don't repeat it)

1. `uj = U[2:end]` — illegal on `TransientMultiFieldCellField` (`lastindex` undefined) → the actual
   `MethodError`. **Fix:** stacked layout, read `U[2]=𝖴x`, `U[3]=𝖴y`.
2. `L(...)=Operation(VectorValue)(a,b,c)` with `a,b,c` stacked-over-layers → a `VectorValue{3}` of
   vectors (rank nonsense). **Fix:** 3 separate `VectorValue{Nσ}` (§5).
3. `L ⊙ A2`, `(∇(uj)')*M3*uj`, `∇(∇⋅(H*U))`, `∇((∇(H)')⋅U)` — rank-mismatched / 2nd-order.
   **Fix:** §5 (`double_contraction`) and §6 (IBP / aux field).
4. `A3` declared but **never filled** → nonlinear ∇h coefficient silently zero. **Fix:** assemble it.
5. Gravity `∇(H−h)*Φvec ⋅ vj` mixes a spatial vector with the layer vector with no defined product.
   **Fix:** §5 grav (`∇(H−d)⋅W̄`).

The stacked value-type model makes all of these well-typed by construction.

---

## 8. Implementation phases (from `algebraic_residual_plan.md`)

| Phase | Deliverable |
|-------|-------------|
| **P1** | Stacked FE space `[η,𝖴x,𝖴y]`; `to_vec/to_tensor2/to_tensor3` + component split; **primitive unit test** (verify `∂x/∂y`, `𝗠·U`, `⊗`, `double_contraction`) |
| **P2** | Residual: **mass + acceleration + gravity (oracle IBP form) + leading pressure `R_P`** (native, exact; `R_P` = the dispersion, MANDATORY) |
| **P3** | **Advection** (`double_contraction(𝗠3/𝗚3, T2)`) |
| **P4** | **Linear pressure `L`** (3 `VectorValue{Nσ}` + component matvecs; the same `L` fields feed `R_P`) |
| **P5** | **Nonlinear pressure `N`**: 6–8 native; 1–5 flagged (IBP ∇h / aux-field ∇H); + `𝓟`-part of `R_P` (assemble `Pcal`) |
| **P6** | wavemaker + sponge + wall BC (Dirichlet on whole `𝖴y`/`𝖴x` fields); driver `setup_and_run_lfem_algebraic`; validation |

**Jacobian:** try Gridap AD first — the notebook's AD crash was the `U[2:end]` slice, *not* an AD
limit; a clean stacked residual should AD. Fallback: hand `jacobian_u`/`jacobian_u_t` (linear terms =
operator; advection = Picard/frozen-coefficient, exact-in-applied-field). Watch AD compile cost for
`ThirdOrderTensorValue{7,7,7}` at `M=6`.

**THE acceptance test (do before trusting anything):** assemble the new algebraic residual **and** the
current per-layer `residual_lfem` on the *same* FE state and require agreement to `~1e-10`, term by
term. The current solver is the **oracle**. **Do not map DOF vectors between layouts** — instead
interpolate the same analytic trial/`u̇`/test fields into both layouts (`interpolate_everywhere`,
exact for polynomials ≤ fe_order), build `TransientCellField(uh,(uht,))` by hand, and compare the
scalars `dot(get_free_dof_values(v_interp), assemble_vector(r,V))` (virtual work — layout-independent).
Match flags: oracle gravity IBP form, `P_full=false` (oracle keeps only `P³L³`), same μ/src.

**Suggested module layout** (mirror the current solver, keep it as a parallel path):
`algebraic_tensors_lfem2D.jl` (constructors), `horizontal_algebraic_lfem2D.jl` (stacked space + BC),
`problem_algebraic_lfem2D.jl` (residual), `utilities_algebraic_lfem2D.jl` (driver).

---

## 9. The current running solver (`../LFE-M_2D_solver/`) — context & reusable assets

The **validated** per-layer solver (its own `CLAUDE.md` is the source of truth). Relevant to this
project:

* **Layout:** `[η, u₁ˣ,u₁ʸ, u₂ˣ,u₂ʸ, …]` (`1+2Nσ` scalar fields), `ix(j)=2j`, `iy(j)=2j+1`. The
  algebraic project replaces this with the 3-field stacked layout.
* **`vertical_lfem2D.jl`** — `assemble_vertical_tensors_lfem(M,p,c_bdy)` → the validated σ-tensors
  (`Mmat,Phi,Mcal,Gcal,A,K,Acal,Kcal,B,…`). **Reuse directly.** 28/28 unit tests.
* **`problem_lfem2D.jl`** — the per-layer `residual_lfem` (the **oracle** for the equivalence test)
  and hand Jacobians. Advection uses the single corrected `𝓖`-tensor (`Gcal`); the old `G1/G3` split
  was wrong.
* **`advection_vxh_lfem2D.jl`** — matrix-free V⊗H nonlinear advection + owned θ-loop
  (`setup_and_run_lfem_vxh`), and `np_residual` (the nonlinear-pressure ∇h/∇H treatment to port for
  P5). Distributed twin: `advection_vxh_dist_lfem2D.jl` (`setup_and_run_lfem_vxh_distributed`).
* **Field output (added this session):** `reconstruct_fields_lfem2D.jl` +
  `write_w`/`write_pressure` switches on all 4 drivers — reconstruct vertical velocity `w_s<σ>` and
  **total** pressure `p_s<σ>` fields into VTK. Key math: `w` = exact modal `w`-FE;
  `p = ρgH(1−σ)` (hydrostatic) `− ρ Σⱼ div(u̇ⱼ) d² Π³ⱼ` (nonhydro, linear flat-bed, `u̇` by backward
  FD). `Π³_j=∫_σ^1 φ_j_int` **vanishes at σ=1** (free-surface BC) — the old `reconstruct_pressure_2D`
  used `−φ_j_int` which does **not** vanish there (a known defect; left unpatched). The algebraic
  solver's output must adapt this to read `𝖴x[j]/𝖴y[j]` components.
* **Distributed examples:** `../LFE-M_2D_solver/examples/distributed/` (plane wave, ring wave, IC
  hump, bathymetry) — env-configurable, cluster-ready, write all fields, run at M=2 and M=6.
* **Key rules (root `CLAUDE.md`):** `fe_order≥2` (Q1 zeros the dispersion); solid-wall Dirichlet must
  include corner tags; IC problems need `x_wall_bc=true`; `A_wave≤0.001` for long stable runs;
  `B_stored=−B̃` sign is load-bearing; distributed uses `NewtonSolver(GMRESSolver(Pr=Jacobi))`; MPI via
  `~/.julia/bin/mpiexecjl` (system `mpiexec` → PMIx error); first full-FEM compile is tens of minutes;
  Julia buffers stdout to files (use `flush`); `MPI_Finalize` prints a benign OFI error, exits 143.
* **Stack:** Gridap 0.19.11, GridapDistributed 0.4.13, GridapSolvers 0.6.2, PartitionedArrays 0.3.5,
  MPI 0.20.26. Run Julia via the `julia-mcp` tool (not the CLI). Transient API:
  `TransientFEOperator(res,jac,jac_t,U,V)`, `res(t,u,v)` with `∂t(u)`, `TransientCellField` in
  `Gridap.ODEs`, `solve(solver,op,t0,tF,u0)` iterator yields `(t,uh)`.

---

## 10. Original task brief (the goal that generated the two spec files)

> *Analyse `test_2HDmodel.ipynb` (ignore the personal-library import errors, focus on the cell code).
> The notebook computes the residual with **algebraic operators** (matvec, dot, contractions) instead
> of `for`-loop index contractions, but has severe dimension/rank-matching trouble (gradients,
> divergences, `H*u_j` products…). Using `main.tex` §8 as the correct guideline and the current solver
> + notebook as examples, design a **new residual function** following the §8 derivation. Specifically
> study how to implement the linear stacked vector `L` and nonlinear matrix `N`, how to compute
> gradients of the fields **without decomposing the MultiField** and **without vertical-index
> for-loops**. Simplify the hard expressions (gradient-of-divergence → easier operators, product-rule
> expansions, operator identities), work around Gridap's missing operators (report needed helpers),
> and produce TWO documents: (1) the mathematical simplification of the complicated expressions into
> implementable Gridap expressions, and (2) the plan to implement the new residual.*

Those two documents are `algebraic_residual_math.md` and `algebraic_residual_plan.md`. The Gridap
operator uncertainties they raised were **resolved live** (see §3 verified facts) — no contraction
helper is needed.

**STATUS 2026-07-10 — EXECUTED AND VALIDATED.** `algebraic_lfem2D.jl` implements P1–P4, P5 comps
6–8, and P6 (driver). Validation (`test_algebraic_lfem2D.jl`, M=2/Nσ=3, Q2):
* **16/16 primitives + virtual-work equivalence** vs oracle `residual_lfem` — rel ≤ 7e-15 for
  (A) linear core, (B) nonlinear+advection, (C) sloped bed + linear pressure.
* **Time integration matches the oracle to machine precision**: 40-step linear run and 20-step
  fully-nonlinear (advection, full-Newton hand Jacobian) run, max gauge diff ≤ 4e-18.
* **Performance**: nonlinear 20-step run 171 s vs oracle 929 s wall (~3.6× faster runtime),
  27 GiB vs 352 GiB allocations (~13× less) — the fused per-layer advection cost collapses in the
  stacked form.
* New-physics flags (`P_full` — the `P¹L¹+P²L²` leading-pressure slope components the oracle lacks;
  `nl_pressure68`) assemble and integrate stably on a sloped bed.
* **During this work the leading-pressure/dispersion term `R_P` was found MISSING from `main.tex`
  §8** (and every doc derived from it) — corrected everywhere; see §2 warning and math doc §6b.

**PACKAGED 2026-07-10 (second pass).** The standalone prototype was promoted to the self-contained
package `src/GridapLFEM.jl` (module + tests + examples mirroring the old solver's structure; see
`algebraic_solver_plan.md`). All 41 package tests pass, including the FEM dispersion gate
(kd=3, err 0.90% — the restored `R_P` term working end-to-end) and the oracle equivalence.
`Pcal` (=∫Θφᵢ_int, the 𝓟 tensor of the nonlinear `R_P` part) is now assembled by
`assemble_vertical_tensors` and stored on the problem struct (`P3`), residual block pending.
`main.tex` §9 (horizontal discretisation) was completed and corrected the same day (momentum
subsection rewritten in the mass-continuity style; static `𝕋/𝔼/𝔹/𝕂` tensors + state-weighted
`M̄/D̄/C̄/S̄/P̄/Ā/K̄` operators — the latter are exactly the V⊗H production route; leading pressure
included; document compiles clean).

**PHYSICS COMPLETE (2026-07-11, fourth pass).** The full nonlinear pressure is implemented
(`src/nlpressure.jl` + flags `nl_pressure68` = native set {3,6,7,8} in all three blocks incl.
the `𝓟`-part of `R_P`, all paths; `nl_pressure_full` = comps {1,2,4,5}: ∇h half via EXACT IBP onto
the test — machine-verified 4e-15 — and ∇H/𝓟 halves via per-step frozen L²-projections `π𝖲,π𝖻`,
sequential loop). Regression tests ported: `test_conservation` (drift 7.8e-16, nonlinear
advection in a closed basin) and `test_sloshing` (period err 1.44%). AD Jacobians ruled out
for good: Gridap 0.19.11's multifield AD cannot dualize through `∂t(u)` (missing
`TransientMultiFieldCellField` constructor) — hand Jacobians are the design, not a workaround.
⚠ Coding rule (block arrays): never apply `∇` to an `Operation`-composed expression containing a
test basis — expand by hand via `∂_a(W⋅𝓣) = (∂_aW)⋅𝓣` (see `nlpressure.jl`).

**`nl_pressure_full` now runs distributed too (2026-07-13).** `build_nlp_ctx(...;
distributed=true)` solves the frozen-projection mass systems with `CGSolver(JacobiLinearSolver())`
(GridapSolvers; the mass matrix is SPD, so CG rather than GMRES) instead of the sequential-only
`lu()`. Gotcha fixed along the way: RHS/solution vectors for a distributed Krylov solve must be
allocated FROM the matrix (`allocate_in_range`/`allocate_in_domain`) and filled in place
(`assemble_vector!`) — an independently `assemble_vector`-built vector is only isomorphic to, not
identical to, the matrix's own PRange, and `mul!` asserts exact partition equality inside
`solve!`. `test/test_nlpressure_distributed.jl` (4 ranks, 2×2, tanh bar, ALL pressure flags)
matches the sequential reference to rel 4.6e-9. Every physics flag now runs both serial and
distributed.

**RUNTIME SOLVER MONITORING (2026-07-17).** New `src/monitor.jl` (serial + distributed):
* `SolverMonitor` — transparent `NonlinearSolver` wrapper (pass via `monitor=` to the two
  solver factories); harvests per step: Newton iterations, initial→final residual, convergence
  flag, last GMRES iteration count (distributed), nonlinear-solve wall time. Sources:
  `NLSolver` cache `.result` (NLsolve, `store_trace=true` now set) / GridapSolvers
  `NewtonSolver.log` (ConvergenceLog).
* `ResidualChecker` + `check_residuals` — every `check_every` steps the GOVERNING
  EQUATIONS are reassembled independently: (a) the θ-scheme discrete residual at
  `(t+θΔt, θu_{n+1}+(1−θ)u_n, (u_{n+1}−u_n)/Δt)` (exactly what ThetaMethod solves — must sit at
  the Newton tolerance, verified ~1e-13; prints WARN if > `check_tol`), (b) the instantaneous PDE
  residual at `(t_n, u_n, u̇_FD)` (= local time-discretisation error, O(Δt)). PVector-safe norms.
* Both time loops print: solver-config banner (solver type, tolerances, max iters, dt/steps),
  per-step line (`step, t, eta_max, NL its, r0→r, [conv], gmres, solve s, ETA`), timed VTK
  writes, non-convergence warnings, end-of-run summary (wall, s/step, %solve, Newton totals).
  Diags tuples gained `nl_iters, res_nl, t_solve`.
* Driver kwargs (both `setup_and_run*`): `nl_iter, nl_tol` (sequential — was hardcoded),
  `print_every=1` (step-based; legacy `print_dt` still honoured when passed — cluster scripts
  unchanged), `check_every=50` (0=off), `check_tol=1e-8`.
* Fixed along the way: commit a640ffc had `createvtk(...; cellfields=fields; append=false)` —
  a double-semicolon SYNTAX ERROR in both time loops (HEAD did not even load); now
  `cellfields=fields, append=false`.

Remaining (open): physical benchmarks at scale (Stokes harmonics / Dingemans bar on the cluster)
and a distributed-gauge utility.

---

## 11. Conventions for editing here

* Keep `main.tex` the single source of mathematical truth; `LFEM_Gridap.md` mirrors its notation.
  If you correct the math, update both, and note any solver-code discrepancy in `LFEM_Gridap.md` §9.
* When quoting solver tensor names vs paper names, always cross-reference (§4 table) — the
  paper/solver notation mismatch is the #1 source of confusion.
* The algebraic residual is a **parallel, unvalidated path** until it passes the §8 equivalence test;
  do not delete or modify the per-layer oracle to make it fit.
