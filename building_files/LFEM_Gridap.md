# LFE-M Model — Theoretical Derivation and the Gridap Global-Residual Implementation

**Scope.** This document reconstructs, step by step, the full mathematical derivation of the
depth-integrated, non-hydrostatic **Linear-Finite-Element multi-layer (LFE-M)** model in a 2D
horizontal layout $\Omega_h\subset\mathbb{R}^2$ (Yang & Liu, *J. Fluid Mech.* **999**, A32, 2024),
and traces how that derivation collapses into the **single scalar weak-form residual** that is
handed to `Gridap`'s `MultiFieldFESpace` solver.

It follows the LaTeX reference `main.tex` — in particular the vertical-projection algebra (end of
§6), the assembled semi-discrete system (§7), and the Gridap global-residual assembly (§8) — and is
cross-checked against the solver documentation (`LFE-M_2D_solver/2DHmodel.md`,
`LFE-M_2D_solver/CLAUDE.md`, `IMPLEMENTATION.md`). The horizontal FE bookkeeping of the *fully*
discrete algebraic system (LaTeX §9, completed and corrected 2026-07-10 — static tensors +
state-weighted `M̄/D̄/C̄/S̄/P̄` operators, the latter being exactly the solver's V⊗H route) is
deliberately **not** covered here: in the actual solver that second discretisation is performed
automatically by `Gridap`, and the user supplies only the continuous-in-$\mathbf{x}$ weak form
derived below.

The central idea to keep in mind throughout:

> **LFE-M = (small dense vertical algebra) $\otimes$ (large sparse horizontal FE assembly).**
> The water column is eliminated analytically into a handful of pre-computable $\sigma$-tensors;
> what remains is a *system of $N_\sigma+2$ coupled 2D PDEs* in the unknowns $\big(H,\ \mathbf{u}_0,
> \dots,\mathbf{u}_{N_\sigma}\big)$, which `Gridap` assembles natively.

---

## Notation

| Symbol | Meaning |
|--------|---------|
| $\mathbf{x}=(x,y)^T$, $z$ | horizontal coordinates and physical vertical coordinate |
| $h(\mathbf{x})$, $\eta(\mathbf{x},t)$ | still-water bed depth, free-surface elevation |
| $H=\eta+h$ | total water depth (the top-layer unknown) |
| $\sigma=(z+h)/H\in[0,1]$ | terrain-following (sigma) vertical coordinate |
| $\mathbf{u}_h=(u,v)^T$, $w$ | horizontal / physical vertical velocity |
| $\omega\equiv H\,D\sigma/Dt$ | transformed (mesh-relative) vertical velocity |
| $p_{\text{nh}}$ | non-hydrostatic (dynamic) pressure; $p_{\text{nh},0}=p_{\text{nh}}\vert_{\sigma=0}$ |
| $\nabla_h=(\partial_x,\partial_y)^T$ | horizontal gradient |
| $\phi_j(\sigma)$ | vertical Lagrange FE basis of order $p$, $j=0,\dots,N_\sigma$ |
| $\varphi_j(\sigma)=\int_0^\sigma\phi_j\,d\sigma'$ | vertical **antiderivative** (unit) basis |
| $\Phi_j=\varphi_j(1)=\int_0^1\phi_j\,d\sigma$ | depth-integrated weight ($\sum_j\Phi_j=1$) |
| $\mathbf{u}_j(\mathbf{x},t)$ | nodal horizontal velocity at vertical DOF $j$ (a **2D field**) |

**Index convention.** $i,j,k$ index the $N_\sigma+1$ vertical DOFs; $j=0$ is the seabed ($\sigma=0$),
$j=N_\sigma$ the free surface ($\sigma=1$). For $M$ vertical elements of order $p$, $N_\sigma=Mp$.

---

## 1. 3D Governing Equations and Boundary Conditions

The starting point is the incompressible, inviscid Euler system, split into horizontal and vertical
parts. With $\nabla=(\nabla_h,\partial_z)^T$:

$$
\text{Mass:}\qquad \nabla_h\cdot\mathbf{u}_h+\frac{\partial w}{\partial z}=0
$$

$$
\text{Horizontal momentum:}\qquad
\frac{\partial\mathbf{u}_h}{\partial t}+(\mathbf{u}_h\cdot\nabla_h)\mathbf{u}_h+w\frac{\partial\mathbf{u}_h}{\partial z}
=-\frac{1}{\rho}\nabla_h p
$$

$$
\text{Vertical momentum:}\qquad
\frac{\partial w}{\partial t}+\mathbf{u}_h\cdot\nabla_h w+w\frac{\partial w}{\partial z}
=-\frac{1}{\rho}\frac{\partial p}{\partial z}-g
$$

closed by the **kinematic** boundary conditions at the moving surface and the fixed bed:

$$
z=\eta:\quad w=\frac{\partial\eta}{\partial t}+\mathbf{u}_h\cdot\nabla_h\eta,
\qquad\qquad
z=-h:\quad w=-\mathbf{u}_h\cdot\nabla_h h .
$$

---

## 2. The $\sigma$-Coordinate Transformation

### 2.1 The map and the chain rule

To map the irregular, time-dependent water column onto the fixed reference slab $\sigma\in[0,1]$:

$$
\sigma(\mathbf{x},z,t)=\frac{z+h(\mathbf{x})}{H(\mathbf{x},t)} .
$$

The differential operators transform as

$$
\frac{\partial}{\partial t}\Big|_z=\frac{\partial}{\partial t}\Big|_\sigma-\frac{\sigma}{H}\frac{\partial H}{\partial t}\frac{\partial}{\partial\sigma},
\qquad
\nabla_h|_z=\nabla_h|_\sigma+\frac{1}{H}\big(\nabla_h h-\sigma\nabla_h H\big)\frac{\partial}{\partial\sigma},
\qquad
\frac{\partial}{\partial z}=\frac{1}{H}\frac{\partial}{\partial\sigma}.
$$

### 2.2 The transformed vertical velocity $\omega$

Define the **mesh-relative** vertical velocity $\omega\equiv H\,D\sigma/Dt$. Expanding the material
derivative of $\sigma$ gives the key identity

$$
\boxed{\;\omega\equiv w-\sigma\frac{\partial H}{\partial t}+\mathbf{u}_h\cdot\big(\nabla_h h-\sigma\nabla_h H\big)\;}
\tag{$\omega$-def}
$$

$\omega$ measures flow *through the moving mesh layers*, and it is the object that makes the whole
transformation clean.

### 2.3 Transformed governing equations

Applying the chain-rule operators and grouping every term proportional to $\partial_\sigma(\cdot)$
into $\omega$ (via $\omega$-def), the Euler system condenses to

$$
\frac{\partial H}{\partial t}+\nabla_h\cdot(H\mathbf{u}_h)+\frac{\partial\omega}{\partial\sigma}=0
\tag{M}
$$

$$
\frac{\partial\mathbf{u}_h}{\partial t}+(\mathbf{u}_h\cdot\nabla_h)\mathbf{u}_h+\frac{\omega}{H}\frac{\partial\mathbf{u}_h}{\partial\sigma}+g\nabla_h\eta
=-\frac{1}{\rho}\nabla_h p_{\text{nh}}-\frac{1}{\rho H}\big(\nabla_h h-\sigma\nabla_h H\big)\frac{\partial p_{\text{nh}}}{\partial\sigma}
\tag{H}
$$

$$
\frac{\partial w}{\partial t}+\mathbf{u}_h\cdot\nabla_h w+\frac{\omega}{H}\frac{\partial w}{\partial\sigma}
=-\frac{1}{\rho H}\frac{\partial p_{\text{nh}}}{\partial\sigma}
\tag{V}
$$

The mass equation (M) follows by substituting
$\partial_\sigma w+\partial_\sigma\mathbf{u}_h\cdot(\nabla_h h-\sigma\nabla_h H)=\partial_\sigma\omega+\partial_t H+\mathbf{u}_h\cdot\nabla_h H$
(obtained by differentiating $\omega$-def in $\sigma$) into the transformed continuity equation.

The **pressure split** $p=\underbrace{\rho g(\eta-z)}_{\text{hydrostatic}}+p_{\text{nh}}$ cancels
gravity in (V) and replaces $-g\nabla_h\eta$ by the hydrostatic term already shown in (H).

### 2.4 Clean boundary conditions

Evaluating $\omega$-def at $\sigma=1$ and $\sigma=0$ and inserting the two kinematic BCs makes the
right-hand side cancel identically, leaving

$$
\boxed{\;\omega\big|_{\sigma=0}=0,\qquad \omega\big|_{\sigma=1}=0\;}
$$

**This is the pivot of the whole method.** Because no fluid crosses the top/bottom mesh layers,
vertical transport is purely *internal* to the mesh, the boundary terms of every vertical
integration-by-parts vanish, and — as shown in §4–§5 — both $\omega$ and $p_{\text{nh}}$ can be
solved *analytically* in closed form. Neither $w$ nor $p_{\text{nh}}$ is ever an independent unknown.

---

## 3. The Vertical Layer Finite Element Approximation

Slice $\sigma\in[0,1]$ into $M$ layers and interpolate the horizontal velocity with a **single
scalar vertical basis** $\phi_j(\sigma)$ of order $p$:

$$
\boxed{\;\mathbf{u}_h(\mathbf{x},\sigma,t)=\sum_{j=0}^{N_\sigma}\mathbf{u}_j(\mathbf{x},t)\,\phi_j(\sigma)
=\sum_{j=0}^{N_\sigma}\begin{pmatrix}u_j(\mathbf{x},t)\\ v_j(\mathbf{x},t)\end{pmatrix}\phi_j(\sigma)\;}
\tag{u-FE}
$$

The $\mathbf{u}_j$ are the **only velocity unknowns** — purely 2D fields, one per vertical DOF.
The antiderivative (unit) basis $\varphi_j$ and its endpoint value $\Phi_j$ (defined in the notation
table) are the single most reused vertical objects: every column-integrated quantity below is
written in terms of $\phi_j$, $\varphi_j$, $\sigma\varphi_j$, $\Phi_j$.

---

## 4. Closed-Form Vertical Velocity

### 4.1 $\omega$ from continuity

Integrate (M) from the bed with $\omega|_0=0$, then insert (u-FE) and pull the $\sigma$-independent
horizontal fields out of the integral:

$$
\omega=-\int_0^\sigma\!\Big[\partial_t H+\nabla_h\cdot(H\mathbf{u}_h)\Big]d\sigma'
\;\Longrightarrow\;
\boxed{\;\omega(\mathbf{x},\sigma,t)=-\sigma\frac{\partial H}{\partial t}-\sum_{j}\nabla_h\cdot(H\mathbf{u}_j)\,\varphi_j(\sigma)\;}
\tag{$\omega$-raw}
$$

### 4.2 Depth-integrated continuity (DIC)

Integrate (M) over the *whole* column $[0,1]$; the $\partial_\sigma\omega$ term becomes
$[\omega]_0^1=0$ by the clean BCs, giving a prognostic equation for $H$:

$$
\boxed{\;\frac{\partial H}{\partial t}=-\sum_{j}\nabla_h\cdot(H\mathbf{u}_j)\,\Phi_j\;}
\tag{DIC}
$$

### 4.3 Invariant-form $\omega$

Substituting (DIC) back into ($\omega$-raw) eliminates the explicit $\partial_t H$ and yields the
form used everywhere downstream:

$$
\boxed{\;\omega=\sum_{j}\nabla_h\cdot(H\mathbf{u}_j)\big(\sigma\Phi_j-\varphi_j\big)\;}
\tag{$\omega$-FE}
$$

### 4.4 Physical vertical velocity $w$

Recovered from $\omega$-def by substituting ($\omega$-raw) and (u-FE):

$$
\boxed{\;w(\mathbf{x},\sigma,t)=\sum_{j}\Big[-\mathbf{u}_j\cdot\nabla_h h\,\phi_j
+\mathbf{u}_j\cdot\nabla_h H\,\sigma\phi_j-\nabla_h\cdot(H\mathbf{u}_j)\,\varphi_j\Big]\;}
\tag{w-FE}
$$

$w$ is **diagnostic only** — evaluated in post-processing when the physical vertical velocity is
needed (`reconstruction2D.jl`), never solved for.

---

## 5. Analytical Elimination of the Non-Hydrostatic Pressure

### 5.1 The vertical-momentum closure

Isolating $p_{\text{nh}}$ in the vertical momentum equation (V):

$$
\frac{\partial p_{\text{nh}}}{\partial\sigma}
=-\rho H\left(\frac{\partial w}{\partial t}+\mathbf{u}_h\cdot\nabla_h w+\frac{\omega}{H}\frac{\partial w}{\partial\sigma}\right).
\tag{$\partial_\sigma p$}
$$

Every ingredient on the right is now a *known* function of $\mathbf{u}_j$, $H$ and $h$ through
(w-FE) and ($\omega$-FE). We expand the three derivatives of $w$.

**Time derivative.** With stationary bathymetry ($\partial_t h=0$) and using (DIC) to remove the
$\partial_t H$ that the product rule generates:

$$
\frac{\partial w}{\partial t}=\sum_{j}\Big[-\dot{\mathbf{u}}_j\cdot\nabla_h h\,\phi_j
+\dot{\mathbf{u}}_j\cdot\nabla_h H\,\sigma\phi_j-\nabla_h\cdot(H\dot{\mathbf{u}}_j)\,\varphi_j\Big]
+\text{(quadratic terms from } \nabla_h\partial_t H).
$$

where $\dot{\mathbf{u}}_j\equiv\partial_t\mathbf{u}_j$.

**Vertical derivative** ($\partial_\sigma$ acts only on the profiles, since $\mathbf{u}_j,h,H$ are
horizontal):

$$
\frac{\partial w}{\partial\sigma}=\sum_{j}\Big[-\mathbf{u}_j\cdot\nabla_h h\,\phi'_j
+\mathbf{u}_j\cdot\nabla_h H\,(\phi_j+\sigma\phi'_j)-\nabla_h\cdot(H\mathbf{u}_j)\,\phi_j\Big],
$$

which in $(\partial_\sigma p)$ is multiplied by $\omega/H$ using ($\omega$-FE) — producing purely
quadratic (double-sum in $k,j$) contributions.

**Horizontal advection** $\mathbf{u}_h\cdot\nabla_h w$: apply $\nabla_h$ to (w-FE) (profiles pass
through), then contract with (u-FE) — again purely quadratic.

### 5.2 Linear package $(\theta_j,\mathcal{L}_j)$

Collect all terms carrying a **single velocity time-derivative** $\dot{\mathbf{u}}_j$. Separate the
$\sigma$-profiles (in $\theta_j$) from the horizontal operators (in $\mathcal{L}_j$):

$$
\theta_j(\sigma)=\begin{bmatrix}\phi_j\\[1mm]\sigma\phi_j\\[1mm]\varphi_j\end{bmatrix},
\qquad
\mathcal{L}_j(\mathbf{u}_j,H)=\begin{bmatrix}
-\dot{\mathbf{u}}_j\cdot\nabla_h h\\[1mm]
\ \ \dot{\mathbf{u}}_j\cdot\nabla_h H\\[1mm]
-\nabla_h\cdot(H\dot{\mathbf{u}}_j)
\end{bmatrix},
\qquad
\left(-\frac{1}{\rho H}\frac{\partial p_{\text{nh}}}{\partial\sigma}\right)_{\text{lin}}=\sum_j\mathcal{L}_j\cdot\theta_j .
$$

> **Sign convention (load-bearing).** *All* signs live in $\mathcal{L}_j$; the profile vector
> $\theta_j$ is kept strictly positive. This is what makes the assembled matrices $\mathbf{A}^V$,
> $\mathbf{K}^V$ recognizable standard objects; putting a sign in $\theta_j$ silently corrupts them.

### 5.3 Nonlinear package $(\Theta_{kj},\mathcal{N}_{kj})$

Collect all **quadratic** (double-sum) terms. The eight $\sigma$-profiles go into $\Theta_{kj}$, the
eight matching horizontal operators into $\mathcal{N}_{kj}$:

$$
\Theta_{kj}(\sigma)=\begin{bmatrix}
\sigma\Phi_k\phi_j\\[1mm]
\Phi_k\varphi_j\\[1mm]
\phi_j\phi_k\\[1mm]
\sigma\phi_j\phi_k\\[1mm]
\varphi_j\phi_k\\[1mm]
\sigma\Phi_j\phi'_k-\varphi_j\phi'_k\\[1mm]
\sigma\Phi_j\phi_k+\sigma^2\Phi_j\phi'_k-\varphi_j\phi_k-\sigma\varphi_j\phi'_k\\[1mm]
\sigma\Phi_j\phi_k-\varphi_j\phi_k
\end{bmatrix},
\qquad
\mathcal{N}_{kj}=\begin{bmatrix}
-\mathbf{u}_j\cdot\nabla_h\!\big(\nabla_h\cdot[H\mathbf{u}_k]\big)\\[1mm]
\ \ \nabla_h\cdot\!\big((\nabla_h\cdot[H\mathbf{u}_k])\,\mathbf{u}_j\big)\\[1mm]
-\mathbf{u}_k\cdot\nabla_h(\mathbf{u}_j\cdot\nabla_h h)\\[1mm]
\ \ \mathbf{u}_k\cdot\nabla_h(\mathbf{u}_j\cdot\nabla_h H)\\[1mm]
-\mathbf{u}_k\cdot\nabla_h\!\big(\nabla_h\cdot[H\mathbf{u}_j]\big)\\[1mm]
-\tfrac{1}{H}[\nabla_h\cdot(H\mathbf{u}_j)][\mathbf{u}_k\cdot\nabla_h h]\\[1mm]
\ \ \tfrac{1}{H}[\nabla_h\cdot(H\mathbf{u}_j)][\mathbf{u}_k\cdot\nabla_h H]\\[1mm]
-\tfrac{1}{H}[\nabla_h\cdot(H\mathbf{u}_j)][\nabla_h\cdot(H\mathbf{u}_k)]
\end{bmatrix},
\qquad
\left(-\frac{1}{\rho H}\frac{\partial p_{\text{nh}}}{\partial\sigma}\right)_{\text{nl}}=\sum_{k,j}\mathcal{N}_{kj}\cdot\Theta_{kj}.
$$

(Note: entries 6–8 of $\Theta_{kj}$ contain $\phi'_k$ and are *sign-changing*, not strictly
positive — the "all positive" annotation in the older `2DHmodel.md §4` is inaccurate here; the
signs still belong to the profiles by construction and the implementation is faithful.)

### 5.4 Closed-form pressure

Combining, $(\partial_\sigma p)$ becomes
$\partial_\sigma p_{\text{nh}}=-\rho H\big(\sum_j\mathcal{L}_j\cdot\theta_j+\sum_{k,j}\mathcal{N}_{kj}\cdot\Theta_{kj}\big)$.
Integrating **downward** from the surface, where the atmospheric condition gives
$p_{\text{nh}}|_{\sigma=1}=0$:

$$
\boxed{\;p_{\text{nh}}(\mathbf{x},\sigma,t)=\rho H\sum_j\mathcal{L}_j\cdot\Pi_j(\sigma)
+\rho H\sum_{k,j}\mathcal{N}_{kj}\cdot\Lambda_{kj}(\sigma)\;}
\tag{p-FE}
$$

with the **upward integrals** (equivalently solved as the BVPs $\Pi_j'=-\theta_j,\ \Pi_j(1)=0$ and
$\Lambda_{kj}'=-\Theta_{kj},\ \Lambda_{kj}(1)=0$):

$$
\Pi_j(\sigma)=\int_\sigma^1\theta_j(\sigma')\,d\sigma',\qquad
\Lambda_{kj}(\sigma)=\int_\sigma^1\Theta_{kj}(\sigma')\,d\sigma'.
$$

> **Core feature — no pressure Poisson solve.** Once the horizontal operators $\mathcal{L}_j$,
> $\mathcal{N}_{kj}$ are known, $p_{\text{nh}}$ is a *closed-form polynomial in $\sigma$*. This is
> the central computational advantage of LFE-M over projection-method non-hydrostatic solvers.

---

## 6. Vertical Projection and the Origin of the Structural Matrices

To eliminate $\sigma$ entirely, insert (u-FE), ($\omega$-FE) and (p-FE) into the horizontal momentum
equation (H), multiply by $H$, then apply the **Galerkin weighted-residual method in $\sigma$**:
multiply by each basis $\phi_i(\sigma)$ and integrate over $[0,1]$. This closes the system with
exactly $N_\sigma+1$ momentum equations. Starting from the $H$-scaled equation

$$
H\frac{\partial\mathbf{u}_h}{\partial t}+H\mathbf{u}_h\cdot\nabla_h\mathbf{u}_h+\omega\frac{\partial\mathbf{u}_h}{\partial\sigma}+gH\nabla_h\eta
=-\frac{1}{\rho}\Big[H\nabla_h p_{\text{nh}}+\frac{\partial p_{\text{nh}}}{\partial\sigma}\big(\nabla_h h-\sigma\nabla_h H\big)\Big],
$$

we treat each term in $\int_0^1(\cdots)\phi_i\,d\sigma$.

### 6.1 Acceleration $\to$ mass matrix $\mathbf{M}^V$

$$
\mathbf{Acc}_i^V=\int_0^1 H\,\partial_t\mathbf{u}_h\,\phi_i\,d\sigma
=\sum_j H\,\dot{\mathbf{u}}_j\underbrace{\int_0^1\phi_i\phi_j\,d\sigma}_{\mathbf{M}^V_{ij}}
=\sum_j H\,\dot{\mathbf{u}}_j\,\mathbf{M}^V_{ij},
\qquad
\boxed{\mathbf{M}^V_{ij}\equiv\int_0^1\phi_i\phi_j\,d\sigma}.
$$

### 6.2 Advection $\to$ tensors $\boldsymbol{\mathcal{M}}^V$ and $\boldsymbol{\mathcal{G}}^V$

Split $\mathbf{Adv}_i^V=\int_0^1[H\mathbf{u}_h\cdot\nabla_h\mathbf{u}_h+\omega\,\partial_\sigma\mathbf{u}_h]\phi_i\,d\sigma$.
The horizontal part (insert u-FE twice) isolates a triple-product mass tensor; the transformed
vertical part (insert $\omega$-FE and u-FE) isolates a gradient tensor:

$$
\mathbf{Adv}_i^V=\sum_{k,j}\Big[H\,\boldsymbol{\mathcal{M}}^V_{ikj}\,(\mathbf{u}_k\cdot\nabla_h\mathbf{u}_j)
+(\nabla_h\cdot[H\mathbf{u}_k])\,\mathbf{u}_j\,\boldsymbol{\mathcal{G}}^V_{ikj}\Big],
$$

$$
\boxed{\boldsymbol{\mathcal{M}}^V_{ikj}\equiv\int_0^1\phi_i\phi_k\phi_j\,d\sigma},
\qquad
\boxed{\boldsymbol{\mathcal{G}}^V_{ikj}\equiv\int_0^1\big(\sigma\Phi_k-\varphi_k\big)\phi'_j\,\phi_i\,d\sigma}.
$$

> The **entire** transformed-vertical-advection block collapses to the *single* tensor
> $\boldsymbol{\mathcal{G}}^V$ contracting $(\nabla_h\cdot[H\mathbf{u}_k])\,\mathbf{u}_j$. (An earlier
> derivation that split this into $-G^{(1)}+(G^{(2)}+G^{(3)})$ with fictitious extra tensors was
> incorrect; see the migration note in §9.)

### 6.3 Gravity $\to$ depth-integrated weight $\Phi_i$

$$
\mathbf{Grav}_i^V=\int_0^1 gH\nabla_h\eta\,\phi_i\,d\sigma=gH\nabla_h\eta\,\Phi_i,
\qquad \Phi_i=\int_0^1\phi_i\,d\sigma.
$$

### 6.4 Pressure RHS — double integration by parts and the seabed cancellation

$$
\mathbf{RHS}_i^V=\underbrace{-\frac{1}{\rho}\int_0^1 H(\nabla_h p_{\text{nh}})\phi_i\,d\sigma}_{\text{horizontal gradient}}
\underbrace{-\frac{1}{\rho}\int_0^1\frac{\partial p_{\text{nh}}}{\partial\sigma}\big(\nabla_h h-\sigma\nabla_h H\big)\phi_i\,d\sigma}_{\text{vertical gradient}}.
$$

**Horizontal-gradient term** — integrate by parts *horizontally* using
$H\nabla_h p_{\text{nh}}=\nabla_h(Hp_{\text{nh}})-p_{\text{nh}}\nabla_h H$. Since the $\sigma$-limits
are fixed, $\nabla_h$ passes outside the vertical integral, producing the gradient of the
column-integrated pressure moment $\mathcal{P}_i\equiv\int_0^1 Hp_{\text{nh}}\phi_i\,d\sigma$.
**This gradient term must be retained.** Under the later horizontal Galerkin weighting,
$\int_{\Omega_h}\nabla_h\mathcal{P}_i\cdot\mathbf{v}_i\,d\Omega_h
=-\int_{\Omega_h}\mathcal{P}_i(\nabla_h\cdot\mathbf{v}_i)\,d\Omega_h+\oint\mathcal{P}_i\mathbf{v}_i\cdot\mathbf{n}$
— *only the boundary integral vanishes*; the volume part survives and is the **leading
non-hydrostatic pressure force**, the sole $O(\eta)$ non-hydrostatic term on a flat bed and the
carrier of the model's entire frequency dispersion (see §6.6). With
$\nabla_h(H\phi_i)=\phi_i\nabla_h H$ (as $\nabla_h\phi_i=0$):

$$
\text{Horizontal gradient}=-\frac{1}{\rho}\nabla_h\mathcal{P}_i+\frac{1}{\rho}\int_0^1 p_{\text{nh}}\,\phi_i\,\nabla_h H\,d\sigma.
$$

**Vertical-gradient term** — integrate by parts *vertically* with
$\Gamma=(\nabla_h h-\sigma\nabla_h H)\phi_i$. The surface boundary vanishes ($p_{\text{nh}}|_{\sigma=1}=0$);
the seabed boundary survives only for the bottom DOF because $\phi_i(0)=\phi_{i,0}=\delta_{i0}$:

$$
\text{Vertical gradient}=\frac{1}{\rho}\nabla_h h\,p_{\text{nh},0}\,\phi_{i,0}
+\frac{1}{\rho}\int_0^1 p_{\text{nh}}\Big[(\nabla_h h-\sigma\nabla_h H)\phi'_i-\phi_i\nabla_h H\Big]d\sigma.
$$

**Cancellation.** Adding the two, the $+\phi_i\nabla_h H$ from the horizontal part exactly cancels
the $-\phi_i\nabla_h H$ from the vertical product rule, leaving

$$
\mathbf{RHS}_i^V=-\frac{1}{\rho}\nabla_h\mathcal{P}_i+\frac{1}{\rho}\nabla_h h\,p_{\text{nh},0}\,\phi_{i,0}
+\frac{1}{\rho}\int_0^1 p_{\text{nh}}\big(\nabla_h h-\sigma\nabla_h H\big)\phi'_i\,d\sigma .
$$

Now insert the closed-form pressure (p-FE) into the integral (the $\rho$ cancels; $H$, $\mathcal{L}_j$,
$\mathcal{N}_{kj}$ factor out) and integrate by parts once more in $\sigma$ on $\Pi_j$ and
$\Lambda_{kj}$, using $\Pi_j'=-\theta_j$, $\Lambda_{kj}'=-\Theta_{kj}$ and $\partial_\sigma\Gamma$.
The new seabed boundary terms sum (via p-FE evaluated at $\sigma=0$) to *exactly* $p_{\text{nh},0}/\rho$,
which **cancels the standalone seabed term** $\tfrac1\rho\nabla_h h\,p_{\text{nh},0}\phi_{i,0}$. Inserting (p-FE)
into the retained pressure moment gives
$\mathcal{P}_i=\rho H^2\big[\sum_j\mathcal{L}_j\cdot\mathbf{P}^V_{ij}+\sum_{k,j}\mathcal{N}_{kj}\cdot\boldsymbol{\mathcal{P}}^V_{ikj}\big]$
with the **leading-pressure tensors** $\mathbf{P}^V_{ij}\equiv\int_0^1\Pi_j\phi_i\,d\sigma$ (3-comp) and
$\boldsymbol{\mathcal{P}}^V_{ikj}\equiv\int_0^1\Lambda_{kj}\phi_i\,d\sigma$ (8-comp). What remains is
the clean semi-discrete RHS:

$$
\mathbf{RHS}_i^V=-\nabla_h\Big(H^2\Big[\sum_j\mathcal{L}_j\cdot\mathbf{P}^V_{ij}+\sum_{k,j}\mathcal{N}_{kj}\cdot\boldsymbol{\mathcal{P}}^V_{ikj}\Big]\Big)
+H\sum_j\Big[\nabla_h h\,(\mathcal{L}_j\cdot\mathbf{A}^V_{ij})+\nabla_h H\,(\mathcal{L}_j\cdot\mathbf{K}^V_{ij})\Big]
+H\sum_{k,j}\Big[\nabla_h h\,(\mathcal{N}_{kj}\cdot\boldsymbol{\mathcal{A}}^V_{ikj})+\nabla_h H\,(\mathcal{N}_{kj}\cdot\boldsymbol{\mathcal{K}}^V_{ikj})\Big],
$$

$$
\boxed{\mathbf{A}^V_{ij}\equiv\int_0^1\phi_i\,\theta_j\,d\sigma},\qquad
\boxed{\mathbf{K}^V_{ij}\equiv\int_0^1\big(\Pi_j-\sigma\theta_j\big)\phi_i\,d\sigma},
$$
$$
\boxed{\boldsymbol{\mathcal{A}}^V_{ikj}\equiv\int_0^1\Theta_{kj}\,\phi_i\,d\sigma},\qquad
\boxed{\boldsymbol{\mathcal{K}}^V_{ikj}\equiv\int_0^1\big(\Lambda_{kj}-\sigma\Theta_{kj}\big)\phi_i\,d\sigma}.
$$

Here $\mathbf{A}^V,\boldsymbol{\mathcal{A}}^V$ are the **bed-slope ($\nabla_h h$)** couplings and
$\mathbf{K}^V,\boldsymbol{\mathcal{K}}^V$ the **surface-slope ($\nabla_h H$)** couplings.

### 6.5 Fubini simplification of $\mathbf{K}^V$ and $\boldsymbol{\mathcal{K}}^V$

The definitions of $\mathbf{K}^V,\boldsymbol{\mathcal{K}}^V$ still contain the *upward* integrals
$\Pi_j,\Lambda_{kj}$. Swapping the order of the resulting double integral over the triangle
$\{0\le\sigma\le\tau\le1\}$ (Fubini) turns the inner $\int_\sigma^1$ into a plain factor
$\varphi_i(\tau)=\int_0^\tau\phi_i$:

$$
\int_0^1\Big(\int_\sigma^1\theta_j(\tau)\,d\tau\Big)\phi_i(\sigma)\,d\sigma
=\int_0^1\theta_j(\tau)\,\varphi_i(\tau)\,d\tau ,
$$

so $\Pi_j,\Lambda_{kj}$ **need never be formed**:

$$
\boxed{\mathbf{K}^V_{ij}=\int_0^1\theta_j\big(\varphi_i-\sigma\phi_i\big)\,d\sigma},\qquad
\boxed{\boldsymbol{\mathcal{K}}^V_{ikj}=\int_0^1\Theta_{kj}\big(\varphi_i-\sigma\phi_i\big)\,d\sigma},
$$
$$
\boxed{\mathbf{P}^V_{ij}=\int_0^1\theta_j\,\varphi_i\,d\sigma},\qquad
\boxed{\boldsymbol{\mathcal{P}}^V_{ikj}=\int_0^1\Theta_{kj}\,\varphi_i\,d\sigma}.
$$

Every vertical tensor is now a plain $[0,1]$ quadrature of low-degree polynomial products. Two
useful identities: $\mathbf{K}^V_{ij}=\mathbf{P}^V_{ij}-\int_0^1\sigma\theta_j\phi_i\,d\sigma$, and
$\mathbf{P}^V_{ij}[3]=\int_0^1\varphi_j\varphi_i\,d\sigma=-B_{ij}$ — the third component of the
leading-pressure tensor *is* (minus) the classical flat-bed dispersion matrix.

### 6.6 The leading pressure term is the dispersion (do not drop it)

On a flat bed ($\nabla_h h=0$, $\nabla_h H=\nabla_h\eta=O(\eta)$) the $\mathbf{A}^V/\mathbf{K}^V$
and $\boldsymbol{\mathcal{A}}^V/\boldsymbol{\mathcal{K}}^V$ blocks are $O(\eta^2)$ or smaller, so
the retained $-\frac1\rho\nabla_h\mathcal{P}_i$ is the **only** $O(\eta)$ non-hydrostatic force.
Linearising ($H\to d$, $\mathcal{L}^{(3)}_j\to-d\,\nabla_h\cdot\dot{\mathbf{u}}_j$, and one factor of
$H$ dropped if the momentum is not $H$-scaled), it reduces to the validated
$-\int\nabla_h\cdot\mathbf{v}_i\ d^2\sum_jB_{ij}\nabla_h\cdot\dot{\mathbf{u}}_j$ dispersion term of
the running solver. **An earlier version of `main.tex` (and of this document) dropped this term,
claiming the total gradient becomes a vanishing boundary integral — that is wrong (only the
boundary part of the horizontal IBP vanishes) and would leave a non-dispersive shallow-water
model. Corrected 2026-07-10.**

---

## 7. The Final Vertically-Discretised (Semi-Discrete) System

Assembling $\mathbf{Acc}_i^V+\mathbf{Adv}_i^V+\mathbf{Grav}_i^V=\mathbf{RHS}_i^V$ together with (DIC)
gives the closed system of $N_\sigma+2$ equations, still continuous in $\mathbf{x}$:

**Mass continuity:**
$$
\frac{\partial H}{\partial t}+\sum_j\nabla_h\cdot(H\mathbf{u}_j)\,\Phi_j=0.
$$

**Horizontal momentum at vertical layer $i$:**
$$
\sum_j H\dot{\mathbf{u}}_j\mathbf{M}^V_{ij}
+\sum_{k,j}\Big[H\boldsymbol{\mathcal{M}}^V_{ikj}(\mathbf{u}_k\cdot\nabla_h\mathbf{u}_j)+(\nabla_h\cdot[H\mathbf{u}_k])\mathbf{u}_j\boldsymbol{\mathcal{G}}^V_{ikj}\Big]
+gH\nabla_h\eta\,\Phi_i
$$
$$
=-\nabla_h\Big(H^2\Big[\sum_j\mathcal{L}_j\cdot\mathbf{P}^V_{ij}+\sum_{k,j}\mathcal{N}_{kj}\cdot\boldsymbol{\mathcal{P}}^V_{ikj}\Big]\Big)
+H\sum_j\Big[\nabla_h h(\mathcal{L}_j\cdot\mathbf{A}^V_{ij})+\nabla_h H(\mathcal{L}_j\cdot\mathbf{K}^V_{ij})\Big]
+H\sum_{k,j}\Big[\nabla_h h(\mathcal{N}_{kj}\cdot\boldsymbol{\mathcal{A}}^V_{ikj})+\nabla_h H(\mathcal{N}_{kj}\cdot\boldsymbol{\mathcal{K}}^V_{ikj})\Big].
$$

### Pre-computable static vertical tensors

| Tensor | Definition | Order | Role |
|--------|-----------|-------|------|
| $\mathbf{M}^V_{ij}$ | $\int_0^1\phi_i\phi_j\,d\sigma$ | 2 | vertical mass (acceleration) |
| $\Phi_j$ | $\int_0^1\phi_j\,d\sigma$ | 1 | depth-average weight ($\sum\Phi_j=1$) |
| $\boldsymbol{\mathcal{M}}^V_{ikj}$ | $\int_0^1\phi_i\phi_k\phi_j\,d\sigma$ | 3 | horizontal advection |
| $\boldsymbol{\mathcal{G}}^V_{ikj}$ | $\int_0^1(\sigma\Phi_k-\varphi_k)\phi'_j\phi_i\,d\sigma$ | 3 | transformed vertical advection |
| $\mathbf{A}^V_{ij}$ | $\int_0^1\phi_i\,\theta_j\,d\sigma$ (3-vector) | 2 | linear pressure, bed slope $\nabla_h h$ |
| $\mathbf{K}^V_{ij}$ | $\int_0^1\theta_j(\varphi_i-\sigma\phi_i)\,d\sigma$ (3-vector) | 2 | linear pressure, surface slope $\nabla_h H$ |
| $\boldsymbol{\mathcal{A}}^V_{ikj}$ | $\int_0^1\Theta_{kj}\phi_i\,d\sigma$ (8-vector) | 3 | non-linear pressure, $\nabla_h h$ |
| $\boldsymbol{\mathcal{K}}^V_{ikj}$ | $\int_0^1\Theta_{kj}(\varphi_i-\sigma\phi_i)\,d\sigma$ (8-vector) | 3 | non-linear pressure, $\nabla_h H$ |
| $\mathbf{P}^V_{ij}$ | $\int_0^1\theta_j\,\varphi_i\,d\sigma$ (3-vector); $\mathbf{P}^V[3]=-B$ | 2 | **leading pressure / dispersion** |
| $\boldsymbol{\mathcal{P}}^V_{ikj}$ | $\int_0^1\Theta_{kj}\,\varphi_i\,d\sigma$ (8-vector) | 3 | leading pressure, non-linear part |

With $N_\sigma\lesssim5$ these are dense arrays no larger than $5\times5\times5\times8$ — computed
once by exact Gauss quadrature on the $\sigma$-mesh and reused at every horizontal quadrature point,
every time step, every Newton iteration.

---

## 8. The Gridap Implementation

### 8.1 Multi-field compact form

After the vertical projection the model is a system of $N_\sigma+2$ PDEs in the 2D unknowns
$\{H,\mathbf{u}_0,\dots,\mathbf{u}_{N_\sigma}\}$. The **second** (horizontal) FE discretisation is
handled *under the hood* by `Gridap`: the user supplies only the continuous-in-$\mathbf{x}$ weak
form. Stack the velocity trial/test fields,

$$
\mathbf{U}=\begin{bmatrix}\mathbf{u}_0&\cdots&\mathbf{u}_{N_\sigma}\end{bmatrix}^T,\qquad
\mathbf{V}=\begin{bmatrix}\mathbf{v}_0&\cdots&\mathbf{v}_{N_\sigma}\end{bmatrix}^T,
$$

and prepend $H$ (trial) and $q$ (test) to form the `MultiFieldFESpace` layout actually used in code:

$$
\boldsymbol{\xi}=\begin{bmatrix}H&\mathbf{u}_0&\cdots&\mathbf{u}_{N_\sigma}\end{bmatrix}^T,\qquad
\mathbf{Q}=\begin{bmatrix}q&\mathbf{v}_0&\cdots&\mathbf{v}_{N_\sigma}\end{bmatrix}^T.
$$

### 8.2 The single global scalar residual

For a system of residuals $\mathbf{R}=[R_0,\dots,R_{N_\sigma+1}]^T=\mathbf{0}$, `Gridap`'s
`MultiFieldFESpace` solver expects **one** combined scalar — the total virtual work — obtained by
weighting each equation with its own test function and summing:

$$
\text{Global Residual}=\sum_{\alpha}\int_{\Omega_h}R_\alpha v_\alpha\,d\Omega_h=\int_{\Omega_h}\mathbf{R}\cdot\mathbf{v}\,d\Omega_h .
$$

`Gridap` recovers the individual layer equations internally by taking the functional derivative of
this scalar with respect to each $v_\alpha$. The task therefore reduces to writing each term of §7 as
a $\int_{\Omega_h}(\cdots)\cdot\mathbf{V}$ (or $\int_{\Omega_h}(\cdots)q$) contribution, **integrating
by parts** every divergence / $\nabla_h H$ term so only first derivatives of the basis remain.

### 8.3 Residual contributions, term by term

**Mass continuity** — weight (DIC-form) by $q$, integrate by parts the divergence
($\nabla_h\cdot(qH\mathbf{u}_j)$; the boundary integral vanishes under essential BC / solid wall):

$$
R_{\text{mass}}=\int_{\Omega_h}q\,\frac{\partial H}{\partial t}\,d\Omega_h
-\Big[\int_{\Omega_h}\nabla_h q\cdot(H\mathbf{U})\,d\Omega_h\Big]\cdot\boldsymbol{\Phi},
\qquad
\boldsymbol{\Phi}=[\Phi_0\ \cdots\ \Phi_{N_\sigma}]^T .
$$

**Acceleration** — with $\dot{\mathbf{U}}=\partial_t\mathbf{U}$:

$$
R_{\text{Acc}}=\int_{\Omega_h}H\,(\mathbf{M}^V\dot{\mathbf{U}})\cdot\mathbf{V}\,d\Omega_h .
$$

**Advection** — define the two vector-valued flux fields whose $i$-th components carry the $k,j$ sums,

$$
\big[\boldsymbol{\mathcal{F}}_{\mathcal{M}}(\mathbf{U})\big]_i=\sum_{k,j}\boldsymbol{\mathcal{M}}^V_{ikj}(\mathbf{u}_k\cdot\nabla_h\mathbf{u}_j),
\qquad
\big[\boldsymbol{\mathcal{F}}_{\mathcal{G}}(H,\mathbf{U})\big]_i=\sum_{k,j}\boldsymbol{\mathcal{G}}^V_{ikj}(\nabla_h\cdot[H\mathbf{u}_k])\mathbf{u}_j,
$$

$$
R_{\text{Adv}}=\int_{\Omega_h}\Big(H\,\boldsymbol{\mathcal{F}}_{\mathcal{M}}(\mathbf{U})+\boldsymbol{\mathcal{F}}_{\mathcal{G}}(H,\mathbf{U})\Big)\cdot\mathbf{V}\,d\Omega_h .
$$

**Gravity:**

$$
R_{\text{Grav}}=\int_{\Omega_h}gH\nabla_h\eta\,\boldsymbol{\Phi}\cdot\mathbf{V}\,d\Omega_h .
$$

**Linear pressure** — stack $\boldsymbol{\mathcal{L}}=[\mathcal{L}_0\ \cdots\ \mathcal{L}_{N_\sigma}]^T$;
the $j$-sums become double contractions ($:$) with the vertical tensors:

$$
R_{\text{lin}}=\int_{\Omega_h}H\Big[\nabla_h h\,(\boldsymbol{\mathcal{L}}:\mathbf{A}^V)+\nabla_h H\,(\boldsymbol{\mathcal{L}}:\mathbf{K}^V)\Big]\cdot\mathbf{V}\,d\Omega_h .
$$

**Non-linear pressure** — stack $\boldsymbol{\mathcal{N}}=[\mathcal{N}_{kj}]$; the $k,j$-sums become
triple contractions ($\therefore$):

$$
R_{\text{nonlin}}=\int_{\Omega_h}H\Big[\nabla_h h\,(\boldsymbol{\mathcal{N}}\therefore\boldsymbol{\mathcal{A}}^V)+\nabla_h H\,(\boldsymbol{\mathcal{N}}\therefore\boldsymbol{\mathcal{K}}^V)\Big]\cdot\mathbf{V}\,d\Omega_h .
$$

**Leading pressure (dispersion)** — the retained $-\frac1\rho\nabla_h\mathcal{P}_i$ is, like the
slope pressures, a right-hand-side term: define its weighted contribution
$R_P = -\sum_i\int\nabla_h(H^2[\cdots]_i)\cdot\mathbf{v}_i$ and integrate by parts horizontally
(boundary term vanishes on essential/solid-wall BCs), moving the derivative onto the
test-divergence vector $\nabla_h\cdot\mathbf{V}\equiv[\nabla_h\cdot\mathbf{v}_0,\dots]^T$:

$$
R_{P}=+\int_{\Omega_h}H^2\Big[(\boldsymbol{\mathcal{L}}:\mathbf{P}^V)+(\boldsymbol{\mathcal{N}}\therefore\boldsymbol{\mathcal{P}}^V)\Big]\cdot\big(\nabla_h\cdot\mathbf{V}\big)\,d\Omega_h .
$$

$R_P$ is **mandatory** (it is the flat-bed dispersion, via $\mathbf{P}^V[3]=-B$ acting on
$\mathcal{L}^{(3)}$); like $R_{\text{lin}}$/$R_{\text{nonlin}}$ it is **subtracted** in the global
residual (LHS $-$ RHS, uniformly for all pressure blocks — the `main.tex` convention since the
2026-07-10 revision).

### 8.4 The assembled global residual

Since the pressure sits on the RHS of the momentum equation, all three pressure contributions enter
with a minus sign. Summing everything:

$$
\text{Global Residual}=R_{\text{mass}}+R_{\text{Acc}}+R_{\text{Adv}}+R_{\text{Grav}}-R_{P}-R_{\text{lin}}-R_{\text{nonlin}},
$$

$$
\boxed{
\begin{aligned}
\text{Global Residual}=\int_{\Omega_h}\Bigg[\ & q\,\frac{\partial H}{\partial t}-\big(\nabla_h q\cdot(H\mathbf{U})\big)\cdot\boldsymbol{\Phi}
+\Big(H\,\mathbf{M}^V\dot{\mathbf{U}}+H\,\boldsymbol{\mathcal{F}}_{\mathcal{M}}(\mathbf{U})+\boldsymbol{\mathcal{F}}_{\mathcal{G}}(H,\mathbf{U})\\[1mm]
&+\,gH\nabla_h\eta\,\boldsymbol{\Phi}
-H\big[\nabla_h h\,(\boldsymbol{\mathcal{L}}:\mathbf{A}^V+\boldsymbol{\mathcal{N}}\therefore\boldsymbol{\mathcal{A}}^V)
+\nabla_h H\,(\boldsymbol{\mathcal{L}}:\mathbf{K}^V+\boldsymbol{\mathcal{N}}\therefore\boldsymbol{\mathcal{K}}^V)\big]\Big)\cdot\mathbf{V}\\[1mm]
&-\,H^2\Big[(\boldsymbol{\mathcal{L}}:\mathbf{P}^V)+(\boldsymbol{\mathcal{N}}\therefore\boldsymbol{\mathcal{P}}^V)\Big]\cdot\big(\nabla_h\cdot\mathbf{V}\big)\ \Bigg]\,d\Omega_h
\end{aligned}}
$$

This scalar — a function of $(\boldsymbol{\xi},\partial_t\boldsymbol{\xi},\mathbf{Q})$ — is exactly
what the residual routine returns to `Gridap`'s `TransientFEOperator`. The horizontal FE assembly,
the Jacobians, and the Newton solve are then performed by `Gridap`; in this solver the spatial and
mass Jacobians are additionally supplied by hand (`jacobian_u`, `jacobian_u_t`) for speed and to
avoid AD compile pathologies on the nested multi-field residual.

### 8.5 Computing the unit vertical basis $\varphi_j$

The antiderivative basis $\varphi_j$ is obtained not by symbolic integration but by solving its
defining BVP $\varphi_j'=\phi_j,\ \varphi_j(0)=0$ as a small 1D FE problem on the $\sigma$-mesh (if
$\phi_j$ has degree $p$, then $\varphi_j$ has degree $p+1$):

$$
\int_0^1\frac{d\xi}{d\sigma}\frac{d\varphi_j}{d\sigma}\,d\sigma=\int_0^1\frac{d\xi}{d\sigma}\phi_j\,d\sigma,\qquad\forall\,\xi .
$$

From $\varphi_j$ (and $\Phi_j=\varphi_j(1)$, $\phi'_j$) all eight vertical tensors of §7 are assembled
by exact quadrature. (In the solver this same object is stored as the *unit vertical velocity* with
the opposite sign — see the notation-bridge note in §9 — but the derivation above uses only $\varphi_j$,
exactly as in `main.tex`.)

---

## 9. Mapping to the Solver Code

**Everything above (§1–§8) uses `main.tex` notation exclusively.** The solver source and some older
companion docs use different symbols for the *same* objects; the table and notes below are the only
place those appear, precisely so the derivation stays unambiguous. Watch out in particular for:

- **$\varphi_j$ (this document / `main.tex`) $\;=\;-w_j$ (solver).** `main.tex` uses the antiderivative
  basis $\varphi_j=\int_0^\sigma\phi_j$; the solver stores the *unit vertical velocity*
  $w_j=-\varphi_j$ (from $w_j'=-\phi_j,\ w_j(0)=0$). The dispersion matrix is then
  $B_{ij}=-\int w_i w_j=-\int\varphi_i\varphi_j\le0$ — $B$ does **not** appear in `main.tex` at all.
- **$\mathbf{M}^V_{ij}$ is the vertical *mass* matrix here.** An older derivation
  (`test/2DHmodel_final.md`) confusingly named the mass matrix $A_{ij}$; in `main.tex`/this document
  the symbol $\mathbf{A}^V_{ij}$ is instead the *linear-pressure bed-slope* tensor $\int\phi_i\theta_j$.
  Do not conflate the two.
- **$\Phi_j$ (depth weight)** is called `D`, `C`, or `Phi` in the code.

The reference (`v2`) implementation assembles the §7 tensors in
`LFE-M_2D_solver/src/vertical_lfem2D.jl` (`assemble_vertical_tensors_lfem`) and evaluates the §8
residual in `problem_lfem2D.jl` (fused `MultiFieldFESpace` form) and, at scale, in the matrix-free
`advection_vxh_lfem2D.jl` (the V$\otimes$H route). Tensor-name correspondence:

| Derivation symbol | Code field | Notes |
|-------------------|-----------|-------|
| $\mathbf{M}^V_{ij}$ | `Mmat` | vertical mass; validated symmetric PSD |
| $\Phi_j$ | `Phi` (= `D` = `C`) | $\sum\Phi_j=1$ |
| $\boldsymbol{\mathcal{M}}^V_{ikj}$ | `Mcal` | horizontal advection |
| $\boldsymbol{\mathcal{G}}^V_{ikj}$ | `Gcal` | **whole** $\omega$-transport block (single tensor) |
| $\mathbf{A}^V_{ij}$ (3-vec) | `A` | bed-slope $\nabla_h h$ linear pressure |
| $\mathbf{K}^V_{ij}$ (3-vec) | `K` | surface-slope $\nabla_h H$ linear pressure (Fubini form) |
| $\boldsymbol{\mathcal{A}}^V_{ikj}$ (8-vec) | `Acal` | non-linear pressure, $\nabla_h h$ |
| $\boldsymbol{\mathcal{K}}^V_{ikj}$ (8-vec) | `Kcal` | non-linear pressure, $\nabla_h H$ (Fubini form) |
| $\Pi_j[3]=\int\theta_j$ carrier | `P` (with `P[:,:,3] = −B`) | leading-pressure / dispersion carrier |
| $B_{ij}=-\int\varphi_i\varphi_j$ | `B` | diagnostic dispersion matrix (flat-bed $d^2B$ term) |

**Load-bearing conventions and gotchas** (carried over from the solver docs):

1. **Signs live in $\mathcal{L}_j,\mathcal{N}_{kj}$**, never in the profiles $\theta_j,\Theta_{kj}$.
2. **$B$ is stored negative** ($B=-\tilde B$); the explicit $(-1)$ in the dispersion term is
   load-bearing. On a flat bed the leading pressure reduces to the validated
   $-\int\partial\phi_i\,d^2\sum_j B_{ij}\,\nabla_h\cdot\mathbf{u}_j$ dispersion term.
3. **`fe_order` $\ge 2$** in the horizontal FE space — linear elements zero the $B$-matrix
   contribution and disable all non-hydrostatic physics.
4. **Integrate by parts** every divergence / $\nabla_h H$ term before handing the integrand to
   `Gridap` (keeps operators sparse; boundary integrals vanish under essential/solid-wall BCs).
5. **Separation of variables** is exact: the horizontal operators $\mathcal{L}_j,\mathcal{N}_{kj}$ and
   the fluxes $\boldsymbol{\mathcal{F}}_\mathcal{M},\boldsymbol{\mathcal{F}}_\mathcal{G}$ carry the
   $\mathbf{x}$-dependence; the $\sigma$-tensors are constant and pre-computed once.

**Migration note (corrected vertical advection).** The transformed vertical advection is the *single*
tensor $\boldsymbol{\mathcal{G}}^V_{ikj}=\int(\sigma\Phi_k-\varphi_k)\phi'_j\phi_i$ contracting
$(\nabla_h\cdot[H\mathbf{u}_k])\,\mathbf{u}_j$. An earlier derivation that split it into
$-G^{(1)}+(G^{(2)}+G^{(3)})$ (with a spurious factor 2, a seabed-slope $\mathbf{u}_k\cdot\nabla_h h$
term that does not belong, and fictitious $G^{(4)},G^{(5)}$) is **superseded** — those tensors are
removed. Likewise the older `2DHmodel.md §5` — and, until 2026-07-10, `main.tex` §6/§8 and this
document — omitted the leading $-\tfrac1\rho\nabla_h\mathcal{P}_i$ contribution (wrongly claiming
the total-gradient term becomes a vanishing boundary integral). It is **not** folded into
$\mathbf{A}^V/\mathbf{K}^V$: it is a separate, mandatory term carried by
$\mathbf{P}^V_{ij}=\int\theta_j\varphi_i$ (code `P`, `P[:,:,3]=−B`) and, in the nonlinear part, by
$\boldsymbol{\mathcal{P}}^V_{ikj}=\int\Theta_{kj}\varphi_i$ (code: not yet assembled — assemble as
`Pcal` via the same Fubini form). §6.4/§6.6/§8.3 above now derive it correctly as $R_P$.

---

### Summary of core features

- **No pressure Poisson solve** — $p_{\text{nh}}$ is a closed-form $\sigma$-polynomial (§5).
- **$w$ and $\omega$ are diagnostic** — only $H$ and $\mathbf{u}_j$ are solved.
- **Clean BCs** $\omega|_0=\omega|_1=0$ make vertical transport internal and enable the analytic
  eliminations.
- **Separable structure** — every term factorises into a small vertical tensor $\times$ large sparse
  horizontal FE operator; the model assembles like $N_\sigma+2$ coupled 2D PDEs.
- **One scalar residual** $\int_{\Omega_h}\mathbf{R}\cdot\mathbf{v}\,d\Omega_h$ is all `Gridap` needs;
  it differentiates internally to recover every layer equation.
