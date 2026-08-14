# ==============================================================
#  mms.jl — ANALYTIC (verification) MMS, Stage 1: linear core on a flat bed
#
#  Implements ValidationTests.tex §"Analytic (verification) MMS". Plan and full
#  rationale: building_files/MMS_ANALYTIC_PLAN.md.
#
#  ############################################################################
#  #  INDEPENDENCE IS THE WHOLE POINT.                                        #
#  #  Nothing in this file may call global_residual, jacobian_u, or any        #
#  #  assembly helper from src/problem.jl. It uses assemble_vertical_tensors   #
#  #  (separately validated against Yang & Liu's applicable-kd table) and      #
#  #  nothing else from the solver. If the forcing were derived from the       #
#  #  residual, an error E in the residual would appear identically in the     #
#  #  forcing and cancel — which is precisely why test_selfconsistency.jl      #
#  #  cannot detect a wrong residual and this file can. Gate G5 of             #
#  #  test_mms_forcing.jl enforces this by grepping this source file.          #
#  ############################################################################
#
#  STRONG FORM UNDER TEST (linearised, flat bed, constant d):
#      𝓛_η(u) = ∂ₜη + d Σⱼ Φⱼ (∇·uⱼ)
#      𝓛ᵢ(u) = Σⱼ Mᵢⱼ ∂ₜuⱼ + g Φᵢ ∇η + d² Σⱼ Bᵢⱼ ∇(∇·∂ₜuⱼ)
#
#  SIGN OF B. The solver's linear dispersion term is −∫d²(Bv·DU̇)⋅DW, i.e.
#  −∫d²ΣᵢⱼBᵢⱼ(∇·u̇ⱼ)(∇·vᵢ), which is the integration by parts of +d²Bᵢⱼ∇(∇·u̇ⱼ).
#  So `vert.B` is exactly Bᵢⱼ above, with NO sign flip. This is asserted by the
#  eigenmode gate (G1), not by the argument — a sign error there makes the
#  forcing fail to vanish on an exact solution, loudly.
#
#  TWO INDEPENDENT IMPLEMENTATIONS, deliberately:
#    * `mms_forcing_stage1`   — the closed form, by hand (fast; used by all runs)
#    * `strong_residual_stage1` — a generic evaluator of 𝓛(u*) for ANY analytic
#      field, derivatives by ForwardDiff
#  Their agreement (gate G2) is a check of both, and the generic one also
#  evaluates the eigenmode gate and generalises to Stages 2–4.
# ==============================================================

# ---------------------------------------------------------------------------
#  The manufactured field
# ---------------------------------------------------------------------------
"""
    MMSField(Nσ; a_eta, alpha, beta, phi, psi, omega, Lx, Ly)

The Stage-1 manufactured field of `ValidationTests.tex` `eq: mms ustar`, with
`kₓ=π/Lx`, `k_y=π/Ly`:

    η*      = a_η cos(ωt) Cx Cy
    u*ˣⱼ    = αⱼ sin(ωt+φⱼ) Sx Cy
    u*ʸⱼ    = βⱼ cos(ωt−ψⱼ) Cx Sy

`Sx` vanishes at `x=0,Lx` and `Sy` at `y=0,Ly`, so the velocity satisfies the
solid-wall conditions EXACTLY (including at the corner tags, where both factors
vanish) and `η*` is unconstrained, as it must be. All three boundary integrals
produced by the residual's integrations by parts then vanish identically and no
boundary forcing is needed — the test is a pure interior-operator check.

`αⱼ,βⱼ,φⱼ,ψⱼ` default to values **asymmetric in j**, so that a swapped k/j tensor
slot cannot cancel. The dependence is trigonometric, not polynomial, on purpose:
`Q2` interpolates a degree-2 polynomial exactly, which would return machine
precision at every `h` and measure no rate at all.

`ω` is a free constant here — it is NOT tied to the dispersion relation (only the
eigenmode of `eigenmode_field` requires ω = k·Cm(k)).
"""
struct MMSField
    Nσ     :: Int
    a_eta  :: Float64
    alpha  :: Vector{Float64}
    beta   :: Vector{Float64}
    phi    :: Vector{Float64}
    psi    :: Vector{Float64}
    omega  :: Float64
    kx     :: Float64
    ky     :: Float64
end

function MMSField(Nσ::Int; a_eta::Float64 = 0.8,
                  alpha  :: Union{Nothing,Vector{Float64}} = nothing,
                  beta   :: Union{Nothing,Vector{Float64}} = nothing,
                  phi    :: Union{Nothing,Vector{Float64}} = nothing,
                  psi    :: Union{Nothing,Vector{Float64}} = nothing,
                  omega  :: Float64 = 1.3,
                  Lx     :: Float64 = 1.0,
                  Ly     :: Float64 = 1.0,
                  ky     :: Union{Nothing,Float64} = nothing)
    #  ky = 0.0 gives the QUASI-1D field: Cy≡1, Sy≡0, hence u*ʸ ≡ 0 identically.
    #  That satisfies the y-walls trivially, removes all 𝖴y dynamics, and makes the
    #  solution y-invariant — which is how a 1-D horizontal problem is posed in a
    #  structurally 2-D solver. Refine nx only; refining ny changes nothing.
    α = alpha === nothing ? [1.0 + 0.25j  for j in 1:Nσ] : alpha
    β = beta  === nothing ? [0.7 - 0.16j  for j in 1:Nσ] : beta
    φ = phi   === nothing ? [0.30j        for j in 1:Nσ] : phi
    ψ = psi   === nothing ? [0.5 - 0.20j  for j in 1:Nσ] : psi
    all(v -> length(v) == Nσ, (α, β, φ, ψ)) ||
        error("MMSField: alpha/beta/phi/psi must each have length Nσ = $Nσ")
    return MMSField(Nσ, a_eta, α, β, φ, ψ, omega, pi/Lx,
                    ky === nothing ? pi/Ly : ky)
end

# Scalar component evaluators — generic in the number type so ForwardDiff can
# differentiate them (x, y, t may arrive as Dual numbers).
mms_eta(f::MMSField, x, y, t) =
    f.a_eta * cos(f.omega*t) * cos(f.kx*x) * cos(f.ky*y)
mms_ux(f::MMSField, x, y, t, j::Int) =
    f.alpha[j] * sin(f.omega*t + f.phi[j]) * sin(f.kx*x) * cos(f.ky*y)
mms_uy(f::MMSField, x, y, t, j::Int) =
    f.beta[j]  * cos(f.omega*t - f.psi[j]) * cos(f.kx*x) * sin(f.ky*y)

"""
    field_callables(f::MMSField) → (eta, ux, uy)

The field as the generic `(x,y,t)` / `(x,y,t,j)` interface consumed by
`strong_residual_stage1`. Any analytic field exposing these three closures works
— which is how the plane-wave eigenmode reuses the same evaluator.
"""
field_callables(f::MMSField) = (
    eta = (x, y, t)    -> mms_eta(f, x, y, t),
    ux  = (x, y, t, j) -> mms_ux(f, x, y, t, j),
    uy  = (x, y, t, j) -> mms_uy(f, x, y, t, j),
)

"""
    mms_initial_fields(f) → (eta0, ux0, uy0)

Callables in Gridap's `x::VectorValue{2}` form for `u*(t)`, used for the initial
condition and for the exact reference in the error norms.
"""
mms_exact_eta(f::MMSField, t::Float64) = x -> mms_eta(f, x[1], x[2], t)
mms_exact_ux(f::MMSField, t::Float64)  =
    x -> VectorValue(ntuple(j -> mms_ux(f, x[1], x[2], t, j), f.Nσ))
mms_exact_uy(f::MMSField, t::Float64)  =
    x -> VectorValue(ntuple(j -> mms_uy(f, x[1], x[2], t, j), f.Nσ))

# ---------------------------------------------------------------------------
#  Closed-form forcing  (ValidationTests.tex eq: mms Seta / Sx / Sy)
# ---------------------------------------------------------------------------
"""
    mms_forcing_stage1(f::MMSField, vert, d, g) → (Seta, Sx, Sy)

The Stage-1 forcing `𝓢 = 𝓛(u*)` in closed form:

    Dⱼ(t) = αⱼkₓ sin(ωt+φⱼ) + βⱼk_y cos(ωt−ψⱼ)        (so ∇·u*ⱼ = Dⱼ CxCy)
    Ḋⱼ(t) = ω[αⱼkₓ cos(ωt+φⱼ) − βⱼk_y sin(ωt−ψⱼ)]
    Πᵢ(t) = g Φᵢ a_η cos(ωt) + d² Σⱼ Bᵢⱼ Ḋⱼ(t)

    S_η = CxCy[ −a_η ω sin(ωt) + d Σⱼ Φⱼ Dⱼ(t) ]
    Sˣᵢ = SxCy[  ω Σⱼ Mᵢⱼ αⱼ cos(ωt+φⱼ) − kₓ Πᵢ(t) ]
    Sʸᵢ = −CxSy[ ω Σⱼ Mᵢⱼ βⱼ sin(ωt−ψⱼ) + k_y Πᵢ(t) ]

Every quantity is a prescribed constant or a Stage-1 vertical tensor. Returns
`(x,t)` callables in Gridap's `VectorValue` form, ready for `prob.mms_src`.
"""
function mms_forcing_stage1(f::MMSField, vert, d::Float64, g::Float64)
    Φ = vert.Phi; M = vert.Mmat; B = vert.B; N = f.Nσ
    N == length(Φ) ||
        error("mms_forcing_stage1: field Nσ=$N does not match vert (Nσ=$(length(Φ)))")
    ω, kx, ky = f.omega, f.kx, f.ky
    d2 = d*d

    Dj(t)  = ntuple(j -> f.alpha[j]*kx*sin(ω*t + f.phi[j]) +
                         f.beta[j] *ky*cos(ω*t - f.psi[j]), N)
    Ddj(t) = ntuple(j -> ω*(f.alpha[j]*kx*cos(ω*t + f.phi[j]) -
                            f.beta[j] *ky*sin(ω*t - f.psi[j])), N)
    # Πᵢ(t) = gΦᵢa_η cos ωt + d²Σⱼ Bᵢⱼ Ḋⱼ
    function Pi(t)
        Dd = Ddj(t)
        return ntuple(i -> g*Φ[i]*f.a_eta*cos(ω*t) +
                           d2*sum(B[i,j]*Dd[j] for j in 1:N), N)
    end

    function Seta(x, t)
        D = Dj(t)
        return cos(kx*x[1])*cos(ky*x[2]) *
               ( -f.a_eta*ω*sin(ω*t) + d*sum(Φ[j]*D[j] for j in 1:N) )
    end
    #  ⚠ MOMENTUM FORCING CARRIES A FACTOR d (2026-08-14).
    #  The solver's linear momentum is the h-WEIGHTED form of LinearModel.tex
    #  `eq: linearised system momentum` (Σⱼ h Mᵢⱼ u̇ⱼ + g h Φᵢ∇η + ∇(h²𝓛·P)), not the
    #  h-divided form these expressions were originally derived from. On a flat bed
    #  h ≡ d is constant, so the two differ by exactly the constant factor d and the
    #  forcing rescales by d. Continuity is unaffected (it was never h-divided).
    #  A constant rescaling cannot change a convergence ORDER — the Stage-1 rates are
    #  unchanged — but omitting it would make u* stop being the exact solution.
    #  See building_files/MMS_VARBED_PLAN.md §0.A.
    function Sx(x, t)
        Π = Pi(t); s = sin(kx*x[1])*cos(ky*x[2])
        return VectorValue(ntuple(i ->
            d*s*( ω*sum(M[i,j]*f.alpha[j]*cos(ω*t + f.phi[j]) for j in 1:N)
                  - kx*Π[i] ), N))
    end
    function Sy(x, t)
        Π = Pi(t); c = cos(kx*x[1])*sin(ky*x[2])
        return VectorValue(ntuple(i ->
            -d*c*( ω*sum(M[i,j]*f.beta[j]*sin(ω*t - f.psi[j]) for j in 1:N)
                   + ky*Π[i] ), N))
    end
    return (Seta = Seta, Sx = Sx, Sy = Sy)
end

# ---------------------------------------------------------------------------
#  Generic strong-form evaluator (ForwardDiff) — the cross-check
# ---------------------------------------------------------------------------
"""
    strong_residual_stage1(cbs, vert, d, g, x, y, t) → (Lη, Lx, Ly)

Evaluate `𝓛(u*)` at a point for ANY analytic field given as the `(eta, ux, uy)`
callable interface, taking every derivative with ForwardDiff. Independent coding
of the same operator as `mms_forcing_stage1`, used to cross-check it (G2) and to
evaluate the eigenmode gate (G1), where the result must vanish.

`Lx`/`Ly` are returned as plain `Vector{Float64}` of length Nσ.
"""
function strong_residual_stage1(cbs, vert, d::Float64, g::Float64,
                                x::Float64, y::Float64, t::Float64)
    Φ = vert.Phi; M = vert.Mmat; B = vert.B; N = length(Φ)
    d2 = d*d

    # NOTE: named `d_t`/`d_x`/`d_y`, NOT `∂t` — `∂t` is exported by Gridap and
    # shadowing it inside this function would be a trap for later edits.
    d_t(fn) = ForwardDiff.derivative(τ -> fn(τ), t)
    d_x(fn) = ForwardDiff.derivative(ξ -> fn(ξ), x)
    d_y(fn) = ForwardDiff.derivative(υ -> fn(υ), y)

    # ∂ₜη and ∇η
    dt_eta = d_t(τ -> cbs.eta(x, y, τ))
    dx_eta = d_x(ξ -> cbs.eta(ξ, y, t))
    dy_eta = d_y(υ -> cbs.eta(x, υ, t))

    # ∇·uⱼ  and  ∂ₜuⱼ
    div_u  = [ d_x(ξ -> cbs.ux(ξ, y, t, j)) + d_y(υ -> cbs.uy(x, υ, t, j)) for j in 1:N ]
    dt_ux  = [ d_t(τ -> cbs.ux(x, y, τ, j)) for j in 1:N ]
    dt_uy  = [ d_t(τ -> cbs.uy(x, y, τ, j)) for j in 1:N ]

    # ∇(∇·∂ₜuⱼ) — nested AD: the divergence of the time derivative (`_divdt`),
    # then its spatial gradient. Three levels of Dual nesting; ForwardDiff tags
    # keep them distinct.
    ddx_divdt = [ ForwardDiff.derivative(ξ -> _divdt(cbs, ξ, y, t, j), x) for j in 1:N ]
    ddy_divdt = [ ForwardDiff.derivative(υ -> _divdt(cbs, x, υ, t, j), y) for j in 1:N ]

    Lη = dt_eta + d*sum(Φ[j]*div_u[j] for j in 1:N)
    Lx = [ sum(M[i,j]*dt_ux[j] for j in 1:N) + g*Φ[i]*dx_eta +
           d2*sum(B[i,j]*ddx_divdt[j] for j in 1:N) for i in 1:N ]
    Ly = [ sum(M[i,j]*dt_uy[j] for j in 1:N) + g*Φ[i]*dy_eta +
           d2*sum(B[i,j]*ddy_divdt[j] for j in 1:N) for i in 1:N ]
    return Lη, Lx, Ly
end

# ∇·(∂ₜuⱼ) at (x,y,t), generic in x/y so the outer ForwardDiff pass can nest.
_divdt(cbs, x, y, t, j::Int) =
    ForwardDiff.derivative(ξ -> ForwardDiff.derivative(τ -> cbs.ux(ξ, y, τ, j), t), x) +
    ForwardDiff.derivative(υ -> ForwardDiff.derivative(τ -> cbs.uy(x, υ, τ, j), t), y)

# ---------------------------------------------------------------------------
#  The free check: a plane-wave eigenmode, on which 𝓛(u*) ≡ 0
# ---------------------------------------------------------------------------
"""
    model_celerity(vert, d, g, k) → Cm

Model phase speed from the discrete dispersion relation
`Cm² = g d Φᵀ(M − (kd)²B)⁻¹Φ`, derived by substituting a plane wave into the
Stage-1 strong form. `M − (kd)²B = M + (kd)²|B| ≻ 0` since `B ≤ 0`.
"""
function model_celerity(vert, d::Float64, g::Float64, k::Float64)
    Φ = vert.Phi; M = vert.Mmat; B = vert.B
    kd = k*d
    Cm2 = g*d*(Φ' * ((M - kd^2 .* B) \ Φ))
    Cm2 > 0 || error("model_celerity: non-positive Cm² = $Cm2 at kd=$kd")
    return sqrt(Cm2)
end

"""
    standing_mode(vert, d, g; n=1, Lx=1.0, eta_hat=1.0) → (cbs, ω, û, k)

An **exact, unforced** solution of the Stage-1 strong form that ALSO satisfies the
solid-wall conditions, so it can be run in a closed basin without reflection:

    η*  = η̂ cos(kx) cos(ωt),   u*ˣⱼ = ûⱼ sin(kx) sin(ωt),   u*ʸⱼ = 0,   k = nπ/Lx
    û   = (g k η̂ / ω) (M − (kd)²B)⁻¹ Φ,      ω = k·Cm(k)

`sin(kx)` vanishes at `x = 0, Lx` for integer `n`, and `u*ʸ ≡ 0` satisfies the
y-walls trivially. Substituting into `𝓛` gives the SAME eigenvalue relation as
the travelling wave (a standing mode is the superposition of ±k), so this is the
model's own dispersion relation made visible as a time-dependent solution.

This is what makes MODEL validation possible as distinct from CODE verification:
with the forcing off, the discrete solution must converge to this exact wave, and
a wrong dispersion relation shows up as a phase drift that does not converge away.
"""
function standing_mode(vert, d::Float64, g::Float64;
                       n::Int = 1, Lx::Float64 = 1.0, eta_hat::Float64 = 1.0)
    Φ = vert.Phi; M = vert.Mmat; B = vert.B
    k  = n*pi/Lx
    Cm = model_celerity(vert, d, g, k)
    ω  = k*Cm
    û  = (g*k*eta_hat/ω) .* ((M - (k*d)^2 .* B) \ Φ)
    cbs = (
        eta = (x, y, t)    -> eta_hat*cos(k*x)*cos(ω*t),
        ux  = (x, y, t, j) -> û[j]*sin(k*x)*sin(ω*t),
        uy  = (x, y, t, j) -> zero(promote_type(typeof(x), typeof(y), typeof(t))),
    )
    return cbs, ω, û, k
end

"""
    exact_cfs(cbs, Nσ, t) → (eta, ux, uy)

Wrap a generic `(x,y,t)` callable field as Gridap-form `x::VectorValue{2}`
closures at a fixed time, for initial conditions and L² error references.
"""
exact_cfs(cbs, Nσ::Int, t::Float64) = (
    x -> cbs.eta(x[1], x[2], t),
    x -> VectorValue(ntuple(j -> cbs.ux(x[1], x[2], t, j), Nσ)),
    x -> VectorValue(ntuple(j -> cbs.uy(x[1], x[2], t, j), Nσ)),
)

"""
    eigenmode_callables(vert, d, g, k; eta_hat=1.0) → (cbs, ω, û)

An exact TRAVELLING solution of the Stage-1 strong form:

    η*  = η̂ cos(kx − ωt),   u*ˣⱼ = ûⱼ cos(kx − ωt),   u*ʸⱼ = 0
    û   = (g k η̂ / ω) (M − (kd)²B)⁻¹ Φ,     ω = k·Cm(k)

`𝓛(u*) ≡ 0` identically, so the computed forcing must vanish to round-off. This
is the free check of `ValidationTests.tex`: it validates the forcing routine
before any FE solve is run, and it is what pins the sign convention of `B`.
(It does NOT satisfy solid walls — use `standing_mode` for a run in a basin.)
"""
function eigenmode_callables(vert, d::Float64, g::Float64, k::Float64;
                             eta_hat::Float64 = 1.0)
    Φ = vert.Phi; M = vert.Mmat; B = vert.B
    kd = k*d
    Cm = model_celerity(vert, d, g, k)
    ω  = k*Cm
    û  = (g*k*eta_hat/ω) .* ((M - kd^2 .* B) \ Φ)
    cbs = (
        eta = (x, y, t)    -> eta_hat*cos(k*x - ω*t),
        ux  = (x, y, t, j) -> û[j]*cos(k*x - ω*t),
        uy  = (x, y, t, j) -> zero(promote_type(typeof(x), typeof(y), typeof(t))),
    )
    return cbs, ω, û
end
