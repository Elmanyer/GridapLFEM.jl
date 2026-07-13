# Algebraic (loop-free) LFE-M residual — operator simplifications for Gridap

**Purpose.** Turn the §8 global residual of `main.tex` into expressions that Gridap can evaluate
using only its native tensor algebra (`∇`, `⋅`, `⊙`, `⊗`, `tr`, `TensorValue`, `ThirdOrderTensorValue`)
— **without decomposing the velocity MultiField into `N_σ` per-layer fields and without Julia
`for`-loops over the vertical index** in the residual. This is the companion of
`algebraic_residual_plan.md` (the implementation plan) and follows the derivation in `main.tex` §8
(the framed global residual `eq: multifield general residual`) and `LFEM_Gridap.md` §8.

> **⚠ 2026-07-10 correction.** The §8 residual as originally written (and the first version of this
> document) **omitted the leading pressure term** `R_P` — the retained volume part of
> `−(1/ρ)∇ₕ(∫₀¹Hp_nh φᵢ dσ)`, which is the **entire flat-bed frequency dispersion** of the model
> (the running solver's validated `d²B`/`H²B` term, `P[:,:,3]=−B`). `main.tex`/`LFEM_Gridap.md`
> are now corrected; the stacked form of `R_P` is §6b below. **A residual without §6b is
> non-dispersive shallow water** — the linear benchmarks would silently fail.

Throughout, the vertical index runs `j,k = 0..N_σ` (code: `1..Nσ`, `Nσ = num_free_dofs(U_phi) = M·p+1`).
Spatial coordinates are `x=(x,y)`, `∇=∇_h=(∂_x,∂_y)`.

---

## 0. The one idea that removes every layer-loop

The notebook fails because it tries to hold the velocity as a *Julia vector of `N_σ` CellFields*
(`uj = U[2:end]`) and then do array algebra on it. `U[2:end]` is illegal on a
`TransientMultiFieldCellField`, and even where it works it hides `N_σ`-long loops and produces
rank-mismatched expressions (`∇(uj)`, `L ⊙ A2`, `∇(∇⋅(H*U))` …).

**Fix — stack the layer index into the FE *value* type, not into a Julia array.**
Represent the horizontal velocity as **two vector-valued fields**, each of value-dimension `N_σ`:

$$
\mathsf{U}_x(x,t)=\big(u_0^x,\dots,u_{N_\sigma}^x\big)\in\mathbb R^{N_\sigma},\qquad
\mathsf{U}_y(x,t)=\big(u_0^y,\dots,u_{N_\sigma}^y\big)\in\mathbb R^{N_\sigma},
$$

so the MultiField is just **three** fields `[H, 𝖴x, 𝖴y]` (scalar `H`, two `VectorValue{Nσ}`), *not*
`1+2N_σ` scalar fields. Then:

* the **vertical index becomes a genuine tensor axis** — the static vertical arrays become
  `TensorValue{Nσ,Nσ}` / `ThirdOrderTensorValue{Nσ,Nσ,Nσ}` constants, and every layer sum
  `Σ_j`, `Σ_{k,j}` becomes a matrix–vector product or a tensor double-contraction;
* the **spatial index is only 2-D** and is handled by writing the `x` and `y` pieces explicitly
  (dotting `∇` with `e_x=(1,0)`, `e_y=(0,1)`), which is *not* a loop;
* we touch the MultiField only to read `H = U[1]`, `𝖴x = U[2]`, `𝖴y = U[3]` — three components, no
  `N_σ`-decomposition.

This is the whole trick. Everything below is bookkeeping on top of it.

### Two primitive helpers (scalar **or** `VectorValue{Nσ}` argument — same code)

$$
\partial_x f \;\equiv\; e_x\!\cdot\!\nabla f,\qquad \partial_y f \;\equiv\; e_y\!\cdot\!\nabla f,
\qquad e_x=\mathrm{VectorValue}(1,0),\ e_y=\mathrm{VectorValue}(0,1).
$$

For a scalar `f`, `∇f∈VectorValue{2}` and `∂_x f` is a scalar. For `f=𝖴x∈VectorValue{Nσ}`, `∇f` is a
`TensorValue{2,Nσ}` and `∂_x f∈VectorValue{Nσ}` is the layer-vector of `x`-derivatives.

> **✓ Verified (this Gridap build).** `∇` of a `VectorValue{Nσ}` field is `TensorValue{2,Nσ}`
> (spatial index **first**, `(∇f)[d,j]=∂f_j/∂x_d`), so `e_x⋅∇f = ∂_x f` and `e_y⋅∇f = ∂_y f`
> (checked numerically: `f=(x,2y,x+y) ⇒ e_x⋅∇f=(1,0,1)`, `e_y⋅∇f=(0,2,1)`). Use `e⋅∇f`, **not**
> `∇f⋅e`.

### Derived layer-vectors (all `VectorValue{Nσ}`, built with native ops, no loops)

| Symbol | Meaning | Stacked expression |
|--------|---------|--------------------|
| `DU` | per-layer horizontal divergence `∇·u_j` | `∂_x 𝖴x + ∂_y 𝖴y` |
| `U∇H` | `u_j·∇H` | `(∂_x H) 𝖴x + (∂_y H) 𝖴y` |
| `U∇h` | `u_j·∇h` | `(∂_x h) 𝖴x + (∂_y h) 𝖴y` |
| `S` | `∇·(H u_j)` | `H·DU + U∇H`  *(product rule, see §7)* |

`H = d + η` (still-water depth `d(x)` known; `η` the free-surface unknown, `∇H=∇d+∇η`).

---

## 1. Splitting the vertical tensors by component

`main.tex` stores the pressure profiles as *vectors of components*: `θ_j` has 3, `Θ_kj` has 8. Keep
each component in its **own** constant tensor so contractions are plain matvecs.

| `main.tex` | shape | split into | Gridap constant |
|-----------|-------|-----------|-----------------|
| `M^V_{ij}` | `Nσ²` | — | `TensorValue{Nσ,Nσ}` `𝗠` |
| `Φ_j` | `Nσ` | — | `VectorValue{Nσ}` `𝚽` |
| `𝓜^V_{ikj}` | `Nσ³` | — | `ThirdOrderTensorValue{Nσ,Nσ,Nσ}` `𝗠𝟥` |
| `𝓖^V_{ikj}` | `Nσ³` | — | `ThirdOrderTensorValue` `𝗚𝟥` |
| `A^V_{ij}` (3-vec) | `Nσ²×3` | `A⁽¹⁾,A⁽²⁾,A⁽³⁾` | three `TensorValue{Nσ,Nσ}` |
| `K^V_{ij}` (3-vec) | `Nσ²×3` | `K⁽¹⁾,K⁽²⁾,K⁽³⁾` | three `TensorValue{Nσ,Nσ}` |
| `𝓐^V_{ikj}` (8-vec) | `Nσ³×8` | `𝗔⁽¹⁾…𝗔⁽⁸⁾` | eight `ThirdOrderTensorValue` |
| `𝓚^V_{ikj}` (8-vec) | `Nσ³×8` | `𝗞⁽¹⁾…𝗞⁽⁸⁾` | eight `ThirdOrderTensorValue` |

These are assembled once on the σ-mesh exactly as in the current solver / notebook (`M2,K2,A2,M3,G3,…`),
then **reshaped into `TensorValue`/`ThirdOrderTensorValue` constants** (component index peeled off).

> **Index convention (fix it now, it is the source of all the "chaos").** We store
> `𝓜^V_{ikj}`, `𝓖^V_{ikj}`, `𝓐^V_{ikj}`, `𝓚^V_{ikj}` with `i` = test/output layer (**first**),
> `k` = the `u_k`/`∇·(Hu_k)` layer, `j` = the `∇u_j`/`u_j` layer. The notebook’s `G3[i,j,k]`
> uses the *opposite* last-two order — do **not** copy it; re-map to `[i,k,j]` when filling the
> `ThirdOrderTensorValue`.

### The rank-3 contraction, done natively  (✓ both forms verified)

The double contraction `Σ_{k,j}𝓣_{ikj}𝖲_{kj}` over the trailing two indices is **native in Gridap**:

$$
\big(\boldsymbol{\mathcal T}\boxdot \mathsf S\big)_i=\sum_{k,j}\mathcal T_{ikj}\,\mathsf S_{kj}
\;=\;\texttt{double\_contraction(}\boldsymbol{\mathcal T},\mathsf S\texttt{)}\in\mathbb R^{N_\sigma},
$$

verified to contract the **last two** indices of a `ThirdOrderTensorValue` with a `TensorValue{Nσ,Nσ}`,
returning a `VectorValue{Nσ}`. So advection/pressure need **no helper** — build the field 2-tensor
`𝖲` (an outer product, §5/§7) and call `double_contraction`. (Equivalently, contracting the *test* in
first is also native: `𝖶 ⋅ 𝓣` contracts the **first** index → `TensorValue{Nσ,Nσ}`, then `⊙ 𝖲`
gives the same scalar. Use whichever reads cleaner.)

---

## 2. Mass continuity  (native, exact)

$$
R_{\text{mass}}=\int_{\Omega_h} q\,\partial_t H\;-\;\int_{\Omega_h}\nabla q\cdot(H\mathbf U)\cdot\boldsymbol\Phi
=\int_{\Omega_h} q\,\partial_t H\;-\;\int_{\Omega_h} H\,\nabla q\cdot\bar{\mathbf u},
\qquad \bar{\mathbf u}=\big(\boldsymbol\Phi\!\cdot\!\mathsf U_x,\ \boldsymbol\Phi\!\cdot\!\mathsf U_y\big).
$$

`𝚽·𝖴x` is a `VectorValue{Nσ}` dot → scalar; `ū` is the depth-averaged 2-D velocity, built with
`Operation(VectorValue)(𝚽⋅𝖴x, 𝚽⋅𝖴y)`. Gridap:
`∫( q*∂t(H) - H*(∇(q)⋅ū) )dΩ`. **No loops, no second derivatives.**

## 3. Acceleration  (native, exact)

$$
R_{\text{Acc}}=\int_{\Omega_h} H\,(\mathbf M^V\dot{\mathbf U})\cdot\mathbf V
=\int_{\Omega_h} H\big[(\mathsf W_x\!\cdot\!\mathsf M\,\dot{\mathsf U}_x)+(\mathsf W_y\!\cdot\!\mathsf M\,\dot{\mathsf U}_y)\big].
$$

`𝗠∈TensorValue{Nσ,Nσ}`, `∂t(𝖴x)∈VectorValue{Nσ}`. Gridap:
`∫( H*((Wx⋅(𝗠⋅∂t(𝖴x))) + (Wy⋅(𝗠⋅∂t(𝖴y)))) )dΩ`.

## 4. Gravity  (native, exact)

With `η=H−h`, `∇η=∇H−∇h`:

$$
R_{\text{Grav}}=\int_{\Omega_h} gH\,\nabla\eta\cdot\big(\boldsymbol\Phi\!\cdot\!\mathsf W_x,\ \boldsymbol\Phi\!\cdot\!\mathsf W_y\big).
$$

Gridap: `∫( g*H*(∇(H-h)⋅ Operation(VectorValue)(𝚽⋅Wx, 𝚽⋅Wy)) )dΩ`. **Reuse the depth-average
structure of §2 with the test.**

> **Oracle-matching form (recommended default).** The running solver does *not* use the direct
> `gH∇η·W̄` form: it uses the IBP'd energy form with the hydrostatic baseline subtracted,
> `−∫(g/2)(H²−d²)·(𝚽⋅DW)` (nonlinear) / `−∫gη·(𝚽⋅DW)` (linearised), with
> `DW = ∂x(Wx)+∂y(Wy)` the test-divergence vector of §6b. The `(H²−d²)=2dη+η²` baseline
> subtraction is load-bearing: with the boundary flux dropped (natural BC at open x-walls) the raw
> `H²` form leaves a spurious net hydrostatic force at rest (solver finding #6, immediate blow-up).
> The two forms differ only by a boundary integral — but for the **term-by-term equivalence test
> against the oracle you must use the oracle's IBP'd form**, otherwise gravity disagrees by exactly
> that boundary term on any state that is nonzero at the open walls.

## 5. Advection  (native modulo the one rank-3 op)

$$
R_{\text{Adv}}=\int_{\Omega_h}\big(H\,\boldsymbol{\mathcal F}_{\mathcal M}+\boldsymbol{\mathcal F}_{\mathcal G}\big)\cdot\mathbf V,\quad
[\boldsymbol{\mathcal F}_{\mathcal M}]_i=\sum_{k,j}\mathcal M_{ikj}(\mathbf u_k\!\cdot\!\nabla\mathbf u_j),\quad
[\boldsymbol{\mathcal F}_{\mathcal G}]_i=\sum_{k,j}\mathcal G_{ikj}\,\big(\nabla\!\cdot\![H\mathbf u_k]\big)\mathbf u_j.
$$

Build the two **field 2-tensors** (`TensorValue{Nσ,Nσ}`, `(k,j)` layout) per spatial component:

$$
\mathsf T^{\mathcal M,x}=\mathsf U_x\otimes(\partial_x\mathsf U_x)+\mathsf U_y\otimes(\partial_y\mathsf U_x),\quad
\mathsf T^{\mathcal M,y}=\mathsf U_x\otimes(\partial_x\mathsf U_y)+\mathsf U_y\otimes(\partial_y\mathsf U_y),
$$
$$
\mathsf T^{\mathcal G,x}=\mathsf S\otimes\mathsf U_x,\qquad
\mathsf T^{\mathcal G,y}=\mathsf S\otimes\mathsf U_y,\qquad \mathsf S=H\,\mathsf{DU}+\mathsf U\!\cdot\!\nabla H.
$$

Then `[𝓕_M]^x=\texttt{double\_contraction(}\mathsf{M3},\mathsf T^{\mathcal M,x}\texttt{)}\in\mathbb R^{N_\sigma}`
(native), and:

$$
R_{\text{Adv}}=\int_{\Omega_h} H\Big[(\mathsf{M3}\boxdot\mathsf T^{\mathcal M,x})\!\cdot\!\mathsf W_x+(\mathsf{M3}\boxdot\mathsf T^{\mathcal M,y})\!\cdot\!\mathsf W_y\Big]
+\Big[(\mathsf{G3}\boxdot\mathsf T^{\mathcal G,x})\!\cdot\!\mathsf W_x+(\mathsf{G3}\boxdot\mathsf T^{\mathcal G,y})\!\cdot\!\mathsf W_y\Big].
$$

`⊗` = outer product of two `VectorValue{Nσ}` → `TensorValue{Nσ,Nσ}` (native); `⊡` =
`double_contraction(ThirdOrderTensorValue, TensorValue)` → `VectorValue{Nσ}` (native, §1). **All
native, first-order in the unknowns, no loops.**

## 6. Linear pressure — the stacked vector `L`  (native, exact)

`main.tex`: `L_j=[-\dot{\mathbf u}_j\!\cdot\!\nabla h,\ \dot{\mathbf u}_j\!\cdot\!\nabla H,\ -\nabla\!\cdot\!(H\dot{\mathbf u}_j)]`.
**Do not build a `VectorValue{3}` of stacked things (the notebook’s mistake).** Build the *three
components* as three separate `VectorValue{Nσ}` layer-vectors of the time derivative
`\dot{\mathsf U}=∂t𝖴`:

$$
\boxed{\;
\mathsf L^{(1)}=-\big[(\partial_x h)\dot{\mathsf U}_x+(\partial_y h)\dot{\mathsf U}_y\big],\quad
\mathsf L^{(2)}=(\partial_x H)\dot{\mathsf U}_x+(\partial_y H)\dot{\mathsf U}_y,\quad
\mathsf L^{(3)}=-\big(H\,\dot{\mathsf{DU}}+\mathsf L^{(2)}\big)\;}
$$

with `\dot{\mathsf{DU}}=∂_x\dot{\mathsf U}_x+∂_y\dot{\mathsf U}_y`. (`L^{(3)}=−∇·(H\dot{\mathsf U})` via the
product rule of §7; note `\dot{\mathsf U}\!\cdot\!∇H = L^{(2)}` is reused.)

The double contraction `L:A^V` peels by component and becomes a **sum of three matvecs**:

$$
(\mathsf L\!:\!A^V)=\sum_{c=1}^{3}A^{(c)}\,\mathsf L^{(c)}\in\mathbb R^{N_\sigma},\qquad
(\mathsf L\!:\!K^V)=\sum_{c=1}^{3}K^{(c)}\,\mathsf L^{(c)}.
$$

Contract with the test up front (`(A^{(c)}L^{(c)})\!\cdot\!\mathsf W=(\mathsf W\!\cdot\!A^{(c)})\!\cdot\!\mathsf L^{(c)}`):

$$
R_{\text{lin}}=\int_{\Omega_h} H\Big[(\partial_x h)\,\pi^A_x+(\partial_y h)\,\pi^A_y+(\partial_x H)\,\pi^K_x+(\partial_y H)\,\pi^K_y\Big],\quad
\pi^A_{\{x,y\}}=\sum_{c}(\mathsf W_{\{x,y\}}\!\cdot\!A^{(c)})\!\cdot\!\mathsf L^{(c)},
$$

and `π^K` identically with `K^{(c)}`. **Everything is `VectorValue{Nσ}`/`TensorValue{Nσ,Nσ}`
matvecs and dots — fully native, first-order in the unknowns, no loops.** This is the clean answer to
“how to implement `L`”.

## 6b. Leading pressure `R_P` — the dispersion term (MANDATORY, native, exact)

The corrected `main.tex` §8 residual carries the retained volume part of
$-\frac1\rho\nabla_h\big(\int_0^1 Hp_{\text{nh}}\phi_i\,d\sigma\big)$. After the horizontal IBP onto
the test functions (boundary term vanishes on essential/solid-wall BCs), `main.tex` defines the
positive integral

$$
R_P=+\int_{\Omega_h}H^2\Big[(\mathsf L:\mathbf P^V)+(\mathsf N\therefore\boldsymbol{\mathcal P}^V)\Big]\cdot\big(\nabla_h\cdot\mathbf V\big)\,d\Omega_h,
\qquad
\mathbf P^V_{ij}=\int_0^1\theta_j\varphi_i\,d\sigma\ (3\text{-comp}),\quad
\boldsymbol{\mathcal P}^V_{ikj}=\int_0^1\Theta_{kj}\varphi_i\,d\sigma\ (8\text{-comp}),
$$

which is **subtracted** in the global sum, uniformly with the slope pressures
(`Global = … − R_P − R_lin − R_nonlin`); the **net contribution the code assembles is therefore
`−∫H²[…]·(∇·V)`**, exactly the framed term of `eq: multifield general residual`.

It **reuses the same `L¹,L²,L³` fields of §6** (and the same `N^{(c)}` of §7 for the nonlinear
part) — no new field algebra. The stacked form needs one extra derived object, the **test-divergence
layer-vector** `DW = ∂x(Wx) + ∂y(Wy) ∈ VectorValue{Nσ}` (same construction as `DU`):

```julia
DW = ∂x(Wx) + ∂y(Wy)                       # VectorValue{Nσ}, per-layer ∇·v_i
sP = P[1]⋅L1 + P[2]⋅L2 + P[3]⋅L3           # VectorValue{Nσ}: (L:P^V)_i  (3 matvecs, like §6)
r += ∫( -(H*H) * (sP⋅DW) )dΩ               # net −R_P contribution (R_P subtracted, main.tex conv.)
# nonlinear part (with the N^(c) of §7, flag-gated with P5):
# sPN = Σ_c double_contraction(𝗣3[c], N[c]);  r += ∫( -(H*H)*(sPN⋅DW) )dΩ
```

with `P[c] = to_tensor2(Pmat[:,:,c])` three constant `TensorValue{Nσ,Nσ}` (the solver's validated
`vert.P`; identity `P[:,:,3] = −B`) and `𝗣3[c]` the eight `ThirdOrderTensorValue` from
`Pcal_ikj = ∫Θ_kj φᵢ_int dσ` (assembled by the package's `vertical_alg.jl` — the first Fubini piece
of `Kcal`, i.e. `Kcal = Pcal − ∫σΘφᵢ`; still absent from the old solver's `vertical_lfem2D.jl`;
needed only when the nonlinear pressure is on).

**Flat-bed linear check** (equivalence with the running solver): `L¹=0`, `L²=O(η·u̇)`, and
`L³=−∇·(Hu̇)` against `P³=−B` gives exactly the oracle's `−∫∇·vᵢ H² Σⱼ B_ij ∇·(Hu̇ⱼ)`
(`linearised` branch: `d²·B·div(u̇)`). **Note the oracle keeps only the `P³L³` product** — the full
`P¹L¹+P²L²` pieces are new (they vanish on a flat bed at rest but contribute on slopes /
finite η). For the term-by-term equivalence test, gate them behind a flag (`P_full=false` matches
the oracle exactly; `P_full=true` is the complete, more consistent physics).

## 7. Non-linear pressure — the matrix `N`  (the hard part)

`main.tex` `N_{kj}` has 8 components; as `(k,j)` `TensorValue{Nσ,Nσ}` fields:

| c | `N^{(c)}` (main.tex) | stacked form | order |
|---|----------------------|--------------|-------|
| 1 | `-u_j·∇(∇·[Hu_k])` | `-[(∂_x𝖲)⊗𝖴x+(∂_y𝖲)⊗𝖴y]` | **2nd** (∂𝖲) |
| 2 | `∇·((∇·[Hu_k])u_j)` | `(∂_x𝖲)⊗𝖴x+(∂_y𝖲)⊗𝖴y + 𝖲⊗\mathsf{DU}` | **2nd** (∂𝖲) |
| 3 | `-u_k·∇(u_j·∇h)` | `-[𝖴x⊗∂_x(\mathsf{U∇h})+𝖴y⊗∂_y(\mathsf{U∇h})]` | **2nd** (∂²h known + ∂𝖴) |
| 4 | `u_k·∇(u_j·∇H)` | `𝖴x⊗∂_x(\mathsf{U∇H})+𝖴y⊗∂_y(\mathsf{U∇H})` | **2nd** (∂²η ‼) |
| 5 | `-u_k·∇(∇·[Hu_j])` | `-[𝖴x⊗∂_x𝖲+𝖴y⊗∂_y𝖲]` | **2nd** (∂𝖲) |
| 6 | `-\tfrac1H(∇·[Hu_j])(u_k·∇h)` | `-\tfrac1H\,\mathsf{U∇h}\otimes 𝖲` | **1st ✅** |
| 7 | `\tfrac1H(∇·[Hu_j])(u_k·∇H)` | `\tfrac1H\,\mathsf{U∇H}\otimes 𝖲` | **1st ✅** |
| 8 | `-\tfrac1H(∇·[Hu_j])(∇·[Hu_k])` | `-\tfrac1H\,𝖲\otimes 𝖲` | **1st ✅** |

(`𝖲=∇·(H𝖴)` stacked, §0. `∂_x𝖲` etc. are `∇` of the **already-first-derivative** `𝖲`, i.e. second
derivatives of the unknowns; likewise `∂_x(\mathsf{U∇H})` contains `∂²η`.)

The whole block is `R_{\text{nonlin}}=\int H\big[\nabla h\,(N\!:\!\mathcal A)+\nabla H\,(N\!:\!\mathcal K)\big]\!\cdot\!\mathbf V`,
and `(N:\mathcal A)=\sum_{c=1}^{8}\mathcal A^{(c)}\boxdot N^{(c)}`; contracting the test in gives, per
component `c` and spatial direction `α∈{x,y}`:
$$
\int_{\Omega_h} H\,(\partial_\alpha h)\,(\mathsf W_\alpha\!\cdot\!\mathsf{A3}^{(c)})\!:\!N^{(c)}
\;+\;H\,(\partial_\alpha H)\,(\mathsf W_\alpha\!\cdot\!\mathsf{K3}^{(c)})\!:\!N^{(c)} .
$$

**Components 6–8 are directly implementable** (first order, native `⊗`/`⊙`, one `W·A3` op).

**Components 1–5 carry second derivatives of the unknowns** — `∂𝖲=∇(∇·(H𝖴))` (“gradient of a
divergence”) and `∂²η`. These are *not* directly usable with `Q2` elements. Two routes:

### 7a. Integrate the derivative onto the test function (works for the `∇h` half)
Each 2nd-order piece has the shape `∫ H(∂_α h)(\text{coef})\,\big(u\!\cdot\!\nabla g\big)` with
`g∈{𝖲, \mathsf{U∇h}}` a **first-derivative** scalar-per-layer. IBP moves `∇` off `g`:
$$
\int_{\Omega_h}\!\! \Psi\,\big(u\!\cdot\!\nabla g\big)
=-\int_{\Omega_h}\!\! g\,\nabla\!\cdot\!(\Psi\,u)
=-\int_{\Omega_h}\!\! g\,\big(u\!\cdot\!\nabla\Psi+\Psi\,\nabla\!\cdot\!u\big)+\oint_{\partial\Omega_h}\!\! g\,\Psi\,(u\!\cdot\!n),
$$
where `Ψ` gathers `H(∂_α h)(\mathsf W_α\!\cdot\!\mathsf A3^{(c)})…` (contains the **test** and, for
comp. 3, the *known* bed Hessian `∂²h`). Result: `g` stays first-order, the new `∇Ψ` differentiates
the **test** and `h` — all admissible. The boundary term vanishes on solid walls / essential BC.
This reproduces the current solver’s **“∇h half: analytic expansion + IBP onto the test (EXACT)”**.

### 7b. The `∇H` half is irreducible → auxiliary field or projection
For the `∇H` half, `Ψ` contains `∂_α H=∂_α η`; IBP’ing produces `∇Ψ⊃∂²η`, the Hessian of the FE
*unknown* `η` — no cancellation. Options, in order of rigor:

1. **Mixed / auxiliary field (recommended, keeps everything 1st-order & AD-safe).** Introduce an
   extra vector unknown `𝗤_k ≈ ∇(∇·(Hu_k))` — i.e. add one `VectorValue{Nσ}`-per-spatial-dim field
   solved by its own weak equation `∫ 𝗤·φ = -∫ 𝖲 (∇·φ)` (IBP). Then all `∂𝖲`/`∂²η` in `N^{(1,2,4,5)}`
   are replaced by the field `𝗤` at **first order**, and the residual stays native + AD-differentiable.
2. **`L²` projection of `𝖲`/`∇𝖲`** onto the FE space before differentiating (the current solver’s
   “∇H half: L2 projection” — sound, `O(A³)` on a flat bed, but a separate solve per step).
3. **Defer** the nonlinear pressure entirely (flag it off) — it is `O(A³)`, only needed for
   finite-amplitude harmonics / strong shoaling, and the current solver keeps it flag-gated.

**Recommendation:** ship the algebraic residual with components **6–8** always on (cheap, native) and
**1–5** behind a flag using route 7b-1 (auxiliary field) or reuse the validated per-component
projection code; see the plan.

---

## 8. Operator identities & Gridap availability

**Product-rule identities used** (all exact, reduce operator order):

$$
\nabla\!\cdot\!(H\mathbf u_j)=H\,\nabla\!\cdot\!\mathbf u_j+\mathbf u_j\!\cdot\!\nabla H,\qquad
\nabla(\mathbf u_j\!\cdot\!\nabla h)=(\nabla\mathbf u_j)^{\!\top}\nabla h+(\nabla^2h)\,\mathbf u_j,
$$
$$
\nabla\!\cdot\!(g\,\mathbf u)=\mathbf u\!\cdot\!\nabla g+g\,\nabla\!\cdot\!\mathbf u
\quad(\text{IBP driver, §7a}),\qquad gH\nabla H=\tfrac12\nabla(H^2)\ (\text{optional gravity}).
$$

**Gridap operator map:**

| need | Gridap | status |
|------|--------|--------|
| `∂_x f`, `∂_y f` (scalar or `VectorValue{Nσ}`) | `e_x⋅∇(f)` / `e_y⋅∇(f)` | native (verify index order once) |
| depth-average `𝚽·𝖴x` | `𝚽 ⋅ 𝖴x` (`VectorValue{Nσ}` dot) | native |
| `𝗠·𝖴x` (matvec) | `𝗠 ⋅ 𝖴x` | native |
| outer product `a⊗b` | `a ⊗ b` | native (`TensorValue{Nσ,Nσ}`) |
| Frobenius `A:B` → scalar | `A ⊙ B` (`double_contraction`) | native |
| build `VectorValue{2}` from 2 scalar CellFields | `Operation(VectorValue)(a,b)` | native |
| `𝓣 ⊡ 𝖲` (`ThirdOrder`⊙`Tensor` → `Vector`, last-two) | `double_contraction(𝓣,𝖲)` | **native ✓** (contracts trailing 2 idx) |
| `𝖶 ⋅ 𝓣` (`Vector`·`ThirdOrder` → `Tensor`, first idx) | `𝖶 ⋅ 𝓣` | **native ✓** |
| `∇(∇·(H𝖴))` = `∂𝖲` | nested `∇`/`∇⋅` | **avoid** (2nd order) → §7a IBP or §7b aux field |

**No residual-level contraction helper is needed** — the rank-3 double contraction is native (verified).
The only build-time helpers are the constant-tensor constructors:

* `to_tensor3(A::Array{Float64,3}) -> ThirdOrderTensorValue{Nσ,Nσ,Nσ}` and
  `to_tensor2(::Matrix) -> TensorValue{Nσ,Nσ}`, `to_vec(::Vector) -> VectorValue{Nσ}` — build the
  constant tensors from the assembled `M2/M3/G3/A2/K2/A3/K3` (peel the component index, re-map
  `[i,j,k]→[i,k,j]` per §1).

> **StaticArrays size note.** `ThirdOrderTensorValue{Nσ,Nσ,Nσ}` has `Nσ³` entries (`M=2⇒27`,
> `M=6,p=1⇒Nσ=7⇒343`). These are *constants*; fine to store, but constructing/【AD-through】 them can
> stress StaticArrays for `Nσ=7`. If it bites, fall back to the `wdot3` helper form (keeps only
> `TensorValue{Nσ,Nσ}` live) or split the `⊡` per output-`i` — measure first.

---

## 9. Why the notebook residual fails (for reference)

1. **`uj = U[2:end]`** — `getindex(::TransientMultiFieldCellField, 2:end)` / `lastindex` undefined ⇒
   the exact `MethodError` in the last cell. The stacked layout removes the need (read `U[2]`,`U[3]`).
2. **`L(...)=Operation(VectorValue)(a,b,c)`** with `a,b,c` themselves stacked-over-layers ⇒ a
   `VectorValue{3}` whose entries are vectors: rank nonsense. Fix: three separate `VectorValue{Nσ}`
   `L^{(1..3)}` (§6).
3. **`L ⊙ A2`** with `A2` an `Nσ×Nσ` Julia array of `VectorValue{3}` and `L` a `VectorValue{3}` —
   dimensions don’t contract. Fix: component split + matvec (§6).
4. **`(∇(uj)') * M3 * uj`**, **`∇(∇⋅(H*U))`**, **`∇((∇(H)')⋅U)`** — rank-3×field products and
   gradient-of-divergence: ill-typed and 2nd-order. Fix: §5 (test-contracted `⊙`) and §7 (IBP / aux).
5. **`A3` never filled** in the build loop (declared, left zero) — the nonlinear-pressure `∇h`
   coefficient is silently zero. Must assemble `A3` (=`𝓐^V`) alongside `K3`.
6. **Gravity `∇(H-h)*Φvec ⋅ vj`** mixes a spatial `VectorValue{2}` with the layer vector `Φ` with no
   defined product. Fix: §4.

All of these are dimension/rank-matching errors that the **stacked value-type** model (Ux,Uy ∈
`VectorValue{Nσ}` + component-split constant tensors) makes well-typed by construction.
