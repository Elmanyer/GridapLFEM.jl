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
    bathymetry_field(; d0=1.0, a_b=0.2, kbx=1.0, kby=0.0) → h(x,y)

Smooth analytic bathymetry for the variable-bed MMS:
`h = d0 (1 + a_b sin(kbx x) cos(kby y))`, so `h ∈ [(1−a_b)d0, (1+a_b)d0] > 0` for `a_b < 1`.

Trigonometric (not polynomial) so it is not in the FE space. `kby=0` gives a y-invariant bed for
the quasi-1D case. Derivatives are taken by AD where needed, so no hand-coded `∇h`/`∇²h` is exposed
— one fewer place for the two to disagree.
"""
bathymetry_field(; d0::Float64 = 1.0, a_b::Float64 = 0.2,
                   kbx::Float64 = 1.0, kby::Float64 = 0.0) =
    (x, y) -> d0*(1 + a_b*sin(kbx*x)*cos(kby*y))

# ===========================================================================
#  THE PARENT EVALUATOR — all four models as restrictions of one strong form
#
#  ValidationTests.tex eq: mms strong general. Models 1–4 (§subsec: mms model1
#  … model4) are restrictions of it under EXACTLY the substitutions
#  resolve_physics performs on the solver:
#
#      flat_bed=true   →  ∇h ≡ 0  (one control point, as in global_residual)
#      regime=:linear  →  H→h, ∇H→∇h, drop 𝓕_M, 𝓕_G, drop all 𝓝
#      nl_pressure     →  :none ⇒ 𝓝≡0 | :native ⇒ {3,6,7,8} | :full ⇒ all 8
#
#  Writing them as ONE evaluator rather than six is not tidiness: it is what
#  makes "the forcing and the solver are restrictions of the same parent" a
#  property of the code instead of a claim in a comment.
#
#  ALL derivatives by ForwardDiff on the ANALYTIC u* and h — never on the
#  residual. Components {1,2,4,5} of 𝓝 carry second derivatives of the
#  unknowns and the leading-pressure term differentiates them once more, so
#  the 𝓟 block is a THIRD derivative of u*. That is exact here and only here:
#  the solver must integrate by parts or freeze-project those same terms.
# ===========================================================================

"Component sets of 𝓝 selected by `nl_pressure` (ValidationTests.tex §subsec: mms model3)."
function _nl_components(nl_pressure::Symbol)
    nl_pressure === :none   && return ()
    nl_pressure === :native && return (3, 6, 7, 8)
    nl_pressure === :full   && return (1, 2, 3, 4, 5, 6, 7, 8)
    error("_nl_components: nl_pressure must be :none, :native or :full (got :$nl_pressure)")
end

"""
    strong_residual_model(cbs, vert, hfun, g, x, y, t;
                          regime, flat_bed, nl_pressure) → (Lη, Lx, Ly)

`𝓛(u*)` for **any** of the four models, evaluated at one point. `cbs` is the
`(eta, ux, uy)` callable triple of [`field_callables`](@ref); `hfun(x,y)` the
analytic bathymetry.

    𝓛_η = ∂ₜη + Σⱼ Φⱼ ∇·(H uⱼ)
    𝓛_i = H Σⱼ Mᵢⱼ u̇ⱼ + H 𝓕_M,i + 𝓕_G,i + g H Φᵢ ∇η
          + ∇[ H²( (𝓛:P)ᵢ + (𝓝⫶𝓟)ᵢ ) ]
          − H[ ∇h( (𝓛:A)ᵢ + (𝓝⫶𝓐)ᵢ ) + ∇H( (𝓛:K)ᵢ + (𝓝⫶𝓚)ᵢ ) ]

with `H = h+η` (`= h` when linear), `∇H = ∇h+∇η` (`= ∇h` when linear),
`𝓛ⱼ = [−u̇ⱼ·∇h, u̇ⱼ·∇H, −∇·(H u̇ⱼ)]`, `𝓝ₖⱼ` the eight components of
`eq: def Nkj nh pressure derivative`, and

    𝓕_M,i = Σₖⱼ 𝓜ᵢₖⱼ (uₖ·∇uⱼ),      𝓕_G,i = Σₖⱼ 𝓖ᵢₖⱼ (∇·[Huₖ]) uⱼ .

Tensor index order is `[i,k,j]` throughout, matching `assemble_vertical_tensors`.
"""
function strong_residual_model(cbs, vert, hfun, g::Float64,
                               x::Float64, y::Float64, t::Float64;
                               regime::Symbol      = :linear,
                               flat_bed::Bool      = true,
                               nl_pressure::Symbol = :none)
    Φ  = vert.Phi;  M  = vert.Mmat;  N = length(Φ)
    P  = vert.P;    A  = vert.A;     K = vert.K
    Pc = vert.Pcal; Ac = vert.Acal;  Kc = vert.Kcal
    Mc = vert.Mcal; Gc = vert.Gcal

    lin   = regime === :linear
    lin && nl_pressure !== :none && error(
        "strong_residual_model: nl_pressure=:$nl_pressure requires regime=:nonlinear " *
        "(a linear model carries no quadratic pressure) — mirrors resolve_physics.")
    comps = lin ? () : _nl_components(nl_pressure)
    useN  = !isempty(comps)

    #  ⚠ KNOWN LIMITATION — nl_pressure ≠ :none is NOT yet usable (2026-08-16).
    #  The 𝓝 components {1,2,4,5} carry second derivatives of u*, and the leading
    #  pressure differentiates the result once more, so the 𝓟 block is a THIRD
    #  derivative — three nested ForwardDiff levels. ForwardDiff decides which
    #  perturbation is outermost by TAG PRECEDENCE, which is a property of the tag
    #  TYPES (a deterministic but arbitrary ordering), not of the order the calls
    #  are written in. For this call tree the outer spatial tag orders BELOW the
    #  inner ones, so `partials` returns a Dual instead of a scalar and the result
    #  is silently mis-nested (perturbation confusion) rather than merely slow.
    #  Reproduced with both `derivative`×2 and a single `gradient`.
    #  The fix is to remove the nesting, not to work around the symptom: supply the
    #  spatial derivatives of u* and h ANALYTICALLY (they are elementary for
    #  MMSField and bathymetry_field) so the evaluator contains ONE AD level.
    #  See building_files/MMS_NONLINEAR_PLAN.md §"Risks".
    #  Refusing loudly here rather than returning a wrong forcing — a wrong forcing
    #  would show up as a collapsed convergence rate and look like a solver defect.
    useN && error(
        "strong_residual_model: nl_pressure=:$nl_pressure is not yet available. " *
        "The 𝓝 blocks need three nested ForwardDiff levels and hit a tag-precedence " *
        "inversion (see the comment at this line). regime=:nonlinear with " *
        "nl_pressure=:none IS available and verified. Refusing rather than " *
        "returning a silently mis-nested forcing.")

    # --- scalar building blocks, generic in the number type (AD-able) ---------
    hv(ξ, υ)          = hfun(ξ, υ)
    ηv(ξ, υ, τ)       = cbs.eta(ξ, υ, τ)
    uv(ξ, υ, τ, j, a) = a == 1 ? cbs.ux(ξ, υ, τ, j) : cbs.uy(ξ, υ, τ, j)
    Hv(ξ, υ, τ)       = lin ? hv(ξ, υ) : hv(ξ, υ) + ηv(ξ, υ, τ)
    ut(ξ, υ, τ, j, a) = ForwardDiff.derivative(s -> uv(ξ, υ, s, j, a), τ)

    #  ∇h — the SINGLE control point for flat_bed, exactly as in global_residual
    dh(ξ, υ, a) = flat_bed ? zero(promote_type(typeof(ξ), typeof(υ))) :
                  (a == 1 ? ForwardDiff.derivative(p -> hv(p, υ), ξ) :
                            ForwardDiff.derivative(p -> hv(ξ, p), υ))
    dη(ξ, υ, τ, a) = a == 1 ? ForwardDiff.derivative(p -> ηv(p, υ, τ), ξ) :
                              ForwardDiff.derivative(p -> ηv(ξ, p, τ), υ)
    dH(ξ, υ, τ, a) = lin ? dh(ξ, υ, a) : dh(ξ, υ, a) + dη(ξ, υ, τ, a)

    #  s_j = ∇·(H uⱼ)  and  ∇·(H u̇ⱼ)
    sflux(ξ, υ, τ, j) = ForwardDiff.derivative(p -> Hv(p, υ, τ)*uv(p, υ, τ, j, 1), ξ) +
                        ForwardDiff.derivative(p -> Hv(ξ, p, τ)*uv(ξ, p, τ, j, 2), υ)
    sdot(ξ, υ, τ, j)  = ForwardDiff.derivative(p -> Hv(p, υ, τ)*ut(p, υ, τ, j, 1), ξ) +
                        ForwardDiff.derivative(p -> Hv(ξ, p, τ)*ut(ξ, p, τ, j, 2), υ)

    #  𝓛ⱼ = [−u̇ⱼ·∇h, u̇ⱼ·∇H, −∇·(H u̇ⱼ)]
    function Lvec(ξ, υ, τ, j)
        ugh = ut(ξ,υ,τ,j,1)*dh(ξ,υ,1)      + ut(ξ,υ,τ,j,2)*dh(ξ,υ,2)
        ugH = lin ? ugh :
              ut(ξ,υ,τ,j,1)*dH(ξ,υ,τ,1)    + ut(ξ,υ,τ,j,2)*dH(ξ,υ,τ,2)
        return (-ugh, ugH, -sdot(ξ, υ, τ, j))
    end

    #  𝓝ₖⱼ — the eight quadratic components (eq: def Nkj nh pressure derivative)
    function Nvec(ξ, υ, τ, k, j)
        Z   = zero(promote_type(typeof(ξ), typeof(υ), typeof(τ)))
        H   = Hv(ξ, υ, τ)
        sj  = sflux(ξ, υ, τ, j)
        sk  = sflux(ξ, υ, τ, k)
        ukx = uv(ξ,υ,τ,k,1); uky = uv(ξ,υ,τ,k,2)
        ujx = uv(ξ,υ,τ,j,1); ujy = uv(ξ,υ,τ,j,2)

        c1 = (1 in comps) ?
             -(ujx*ForwardDiff.derivative(p -> sflux(p,υ,τ,k), ξ) +
               ujy*ForwardDiff.derivative(p -> sflux(ξ,p,τ,k), υ)) : Z
        c2 = (2 in comps) ?
             ( ForwardDiff.derivative(p -> sflux(p,υ,τ,k)*uv(p,υ,τ,j,1), ξ) +
               ForwardDiff.derivative(p -> sflux(ξ,p,τ,k)*uv(ξ,p,τ,j,2), υ) ) : Z
        #  aⱼ = uⱼ·∇h ,  bⱼ = uⱼ·∇H
        aj(p, q) = uv(p,q,τ,j,1)*dh(p,q,1)   + uv(p,q,τ,j,2)*dh(p,q,2)
        bj(p, q) = uv(p,q,τ,j,1)*dH(p,q,τ,1) + uv(p,q,τ,j,2)*dH(p,q,τ,2)
        c3 = (3 in comps && !flat_bed) ?
             -(ukx*ForwardDiff.derivative(p -> aj(p,υ), ξ) +
               uky*ForwardDiff.derivative(p -> aj(ξ,p), υ)) : Z
        c4 = (4 in comps) ?
             ( ukx*ForwardDiff.derivative(p -> bj(p,υ), ξ) +
               uky*ForwardDiff.derivative(p -> bj(ξ,p), υ) ) : Z
        c5 = (5 in comps) ?
             -(ukx*ForwardDiff.derivative(p -> sflux(p,υ,τ,j), ξ) +
               uky*ForwardDiff.derivative(p -> sflux(ξ,p,τ,j), υ)) : Z
        c6 = (6 in comps && !flat_bed) ?
             -(sj/H)*(ukx*dh(ξ,υ,1) + uky*dh(ξ,υ,2)) : Z
        c7 = (7 in comps) ?
              (sj/H)*(ukx*dH(ξ,υ,τ,1) + uky*dH(ξ,υ,τ,2)) : Z
        c8 = (8 in comps) ? -(sj/H)*sk : Z
        return (c1, c2, c3, c4, c5, c6, c7, c8)
    end

    #  the leading-pressure potential  Ψᵢ = H²[ (𝓛:P)ᵢ + (𝓝⫶𝓟)ᵢ ]
    function Ψ(ξ, υ, τ, i)
        s = zero(promote_type(typeof(ξ), typeof(υ), typeof(τ)))
        for j in 1:N
            L = Lvec(ξ, υ, τ, j)
            s += P[i,j,1]*L[1] + P[i,j,2]*L[2] + P[i,j,3]*L[3]
        end
        if useN
            for k in 1:N, j in 1:N
                Nc = Nvec(ξ, υ, τ, k, j)
                for c in comps
                    s += Nc[c]*Pc[i,k,j,c]
                end
            end
        end
        return Hv(ξ, υ, τ)^2 * s
    end

    # ---- assemble at the point ------------------------------------------------
    H   = Hv(x, y, t)
    dhx = dh(x, y, 1);      dhy = dh(x, y, 2)
    dHx = dH(x, y, t, 1);   dHy = dH(x, y, t, 2)

    #  continuity:  ∂ₜη + Σⱼ Φⱼ ∇·(H uⱼ)
    Lη = ForwardDiff.derivative(s -> ηv(x, y, s), t)
    for j in 1:N
        Lη += Φ[j]*sflux(x, y, t, j)
    end

    dx_eta = dη(x, y, t, 1);  dy_eta = dη(x, y, t, 2)
    Lxv = zeros(Float64, N);  Lyv = zeros(Float64, N)

    #  precompute the per-(k,j) quantities the momentum loop reuses
    svals = [sflux(x, y, t, j) for j in 1:N]
    Nall  = useN ? [Nvec(x, y, t, k, j) for k in 1:N, j in 1:N] : nothing

    for i in 1:N
        acc_x = sum(M[i,j]*ut(x,y,t,j,1) for j in 1:N)
        acc_y = sum(M[i,j]*ut(x,y,t,j,2) for j in 1:N)

        #  ∇Ψ by a SINGLE gradient over [x,y], not two independent `derivative` calls.
        #  Ψ itself contains inner ForwardDiff derivatives (𝓛 carries ∇·(H u̇), 𝓝 carries
        #  second derivatives), so this is nested AD. Two separate outer `derivative`
        #  calls create two independent tags, and ForwardDiff's tag PRECEDENCE — not the
        #  order the calls are written in — decides which perturbation ends up outermost.
        #  For the y-closure that ordering inverts, and `partials` then returns a Dual
        #  instead of a Float64. One `gradient` uses ONE outer tag for both components,
        #  so the nesting order is fixed and correct for x and y alike.
        gΨ     = ForwardDiff.gradient(v -> Ψ(v[1], v[2], t, i), [x, y])
        gradΨx = gΨ[1]
        gradΨy = gΨ[2]

        #  slope packages:  (𝓛:A)+(𝓝⫶𝓐)  and  (𝓛:K)+(𝓝⫶𝓚)
        sA = 0.0; sK = 0.0
        for j in 1:N
            L = Lvec(x, y, t, j)
            sA += A[i,j,1]*L[1] + A[i,j,2]*L[2] + A[i,j,3]*L[3]
            sK += K[i,j,1]*L[1] + K[i,j,2]*L[2] + K[i,j,3]*L[3]
        end
        if useN
            for k in 1:N, j in 1:N
                Nc = Nall[k,j]
                for c in comps
                    sA += Nc[c]*Ac[i,k,j,c]
                    sK += Nc[c]*Kc[i,k,j,c]
                end
            end
        end

        #  advection (nonlinear only)
        advx = 0.0; advy = 0.0
        if !lin
            for k in 1:N, j in 1:N
                ukx = uv(x,y,t,k,1); uky = uv(x,y,t,k,2)
                gjx = ForwardDiff.derivative(p -> uv(p,y,t,j,1), x)*ukx +
                      ForwardDiff.derivative(p -> uv(x,p,t,j,1), y)*uky
                gjy = ForwardDiff.derivative(p -> uv(p,y,t,j,2), x)*ukx +
                      ForwardDiff.derivative(p -> uv(x,p,t,j,2), y)*uky
                advx += H*Mc[i,k,j]*gjx + Gc[i,k,j]*svals[k]*uv(x,y,t,j,1)
                advy += H*Mc[i,k,j]*gjy + Gc[i,k,j]*svals[k]*uv(x,y,t,j,2)
            end
        end

        Lxv[i] = H*acc_x + advx + g*H*Φ[i]*dx_eta + gradΨx - H*(dhx*sA + dHx*sK)
        Lyv[i] = H*acc_y + advy + g*H*Φ[i]*dy_eta + gradΨy - H*(dhy*sA + dHy*sK)
    end
    return Lη, Lxv, Lyv
end

"""
    strong_residual_linear(cbs, vert, hfun, g, x, y, t) → (Lη, Lx, Ly)

`𝓛(u*)` for the **linearised model over ARBITRARY bathymetry** `hfun(x,y)`
(`LinearModel.tex` `eq: linearised system momentum`), every derivative by ForwardDiff:

    𝓛_η = ∂ₜη + Σⱼ Φⱼ ∇·(h uⱼ)
    𝓛_i = h Σⱼ Mᵢⱼ u̇ⱼ + g h Φᵢ ∇η + ∇( h² Σⱼ 𝓛ⱼ·Pᵢⱼ ) − h ∇h Σⱼ 𝓛ⱼ·(Aᵢⱼ+Kᵢⱼ)
    𝓛ⱼ  = [ −u̇ⱼ·∇h , u̇ⱼ·∇h , −∇·(h u̇ⱼ) ]

**Derived by AD on the analytic expressions, NOT by hand.** With variable `h` the
`∇(h²Σ𝓛·P)` term expands into a large product-rule tree; a hand slip there would produce a wrong
forcing, a collapsed rate, and the appearance of a solver defect. AD removes that risk while keeping
the forcing fully independent of `problem.jl` — which is the property the MMS depends on.

Setting `hfun ≡ const` reduces this to the Stage-1 operator times `h` (gate G-flat).
"""
strong_residual_linear(cbs, vert, hfun, g::Float64,
                       x::Float64, y::Float64, t::Float64) =
    strong_residual_model(cbs, vert, hfun, g, x, y, t;
                          regime = :linear, flat_bed = false, nl_pressure = :none)

#  The original hand-written linear evaluator, superseded by the parent above and
#  retained (unused) only as the reference the refactor was checked against.
function _strong_residual_linear_legacy(cbs, vert, hfun, g::Float64,
                                x::Float64, y::Float64, t::Float64)
    Φ = vert.Phi; M = vert.Mmat; N = length(Φ)
    P = vert.P; A = vert.A; K = vert.K       # (N,N,3) each

    # --- scalar building blocks, generic in the number type (AD-able) ---------
    hv(ξ, υ)       = hfun(ξ, υ)
    dt_u(ξ,υ,τ,j,c)= c == 1 ? ForwardDiff.derivative(s -> cbs.ux(ξ,υ,s,j), τ) :
                              ForwardDiff.derivative(s -> cbs.uy(ξ,υ,s,j), τ)
    # ∇·(h u̇ⱼ)
    div_hudot(ξ,υ,j) =
        ForwardDiff.derivative(a -> hv(a,υ)*dt_u(a,υ,t,j,1), ξ) +
        ForwardDiff.derivative(b -> hv(ξ,b)*dt_u(ξ,b,t,j,2), υ)
    # 𝓛ⱼ components at a point
    function Lvec(ξ, υ, j)
        dhx = ForwardDiff.derivative(a -> hv(a,υ), ξ)
        dhy = ForwardDiff.derivative(b -> hv(ξ,b), υ)
        ugh = dt_u(ξ,υ,t,j,1)*dhx + dt_u(ξ,υ,t,j,2)*dhy
        return (-ugh, ugh, -div_hudot(ξ,υ,j))
    end
    # scalar  Sᵢ(ξ,υ) = h² Σⱼ 𝓛ⱼ·Pᵢⱼ   (the leading-pressure potential)
    function Spot(ξ, υ, i)
        h2 = hv(ξ,υ)^2
        s  = zero(promote_type(typeof(ξ), typeof(υ)))
        for j in 1:N
            L = Lvec(ξ,υ,j)
            s += P[i,j,1]*L[1] + P[i,j,2]*L[2] + P[i,j,3]*L[3]
        end
        return h2*s
    end

    h   = hv(x,y)
    dhx = ForwardDiff.derivative(a -> hv(a,y), x)
    dhy = ForwardDiff.derivative(b -> hv(x,b), y)

    # continuity: ∂ₜη + Σⱼ Φⱼ ∇·(h uⱼ)
    dt_eta = ForwardDiff.derivative(s -> cbs.eta(x,y,s), t)
    Lη = dt_eta
    for j in 1:N
        divhu = ForwardDiff.derivative(a -> hv(a,y)*cbs.ux(a,y,t,j), x) +
                ForwardDiff.derivative(b -> hv(x,b)*cbs.uy(x,b,t,j), y)
        Lη += Φ[j]*divhu
    end

    dx_eta = ForwardDiff.derivative(a -> cbs.eta(a,y,t), x)
    dy_eta = ForwardDiff.derivative(b -> cbs.eta(x,b,t), y)
    Lxv = zeros(Float64, N); Lyv = zeros(Float64, N)
    for i in 1:N
        acc_x = sum(M[i,j]*dt_u(x,y,t,j,1) for j in 1:N)
        acc_y = sum(M[i,j]*dt_u(x,y,t,j,2) for j in 1:N)
        gradSx = ForwardDiff.derivative(a -> Spot(a,y,i), x)
        gradSy = ForwardDiff.derivative(b -> Spot(x,b,i), y)
        sAK = 0.0
        for j in 1:N
            L = Lvec(x,y,j)
            sAK += (A[i,j,1]+K[i,j,1])*L[1] + (A[i,j,2]+K[i,j,2])*L[2] +
                   (A[i,j,3]+K[i,j,3])*L[3]
        end
        Lxv[i] = h*acc_x + g*h*Φ[i]*dx_eta + gradSx - h*dhx*sAK
        Lyv[i] = h*acc_y + g*h*Φ[i]*dy_eta + gradSy - h*dhy*sAK
    end
    return Lη, Lxv, Lyv
end

"""
    mms_forcing(field, vert, hfun, g; regime, flat_bed, nl_pressure) → (Seta, Sx, Sy)

**THE single entry point — the forcing is selected by the SAME symbols that select the solver
model**, so a forcing/model mismatch is unrepresentable rather than merely discouraged. Feeding a
`flat_bed=false` solver with flat-bed forcing would produce a silently wrong convergence rate that
looks like a solver defect; this signature makes that impossible.

| `regime` | `flat_bed` | `nl_pressure` | forcing | doc |
|---|---|---|---|---|
| `:linear` | `true`  | `:none` | Stage-1 closed form (fast) | §subsec: mms model1 |
| `:linear` | `false` | `:none` | variable-bed, AD | §subsec: mms model2 |
| `:nonlinear` | `true`  | any | nonlinear flat-bed, AD | §subsec: mms model3 |
| `:nonlinear` | `false` | any | full model, AD | §subsec: mms model4 |

All AD paths go through the single parent [`strong_residual_model`](@ref).

**Memoisation.** The three returned closures are evaluated by Gridap at the *same* quadrature point
consecutively, and each would otherwise trigger a full (and, with `𝓝`, third-derivative) evaluation.
A one-entry cache keyed on `(x,y,t)` collapses three evaluations into one — worth ≈3× on the
nonlinear tiers, which is the difference between a study that runs in minutes and one that does not.
"""
function mms_forcing(field::MMSField, vert, hfun, g::Float64;
                     regime::Symbol = :linear, flat_bed::Bool = true,
                     nl_pressure::Symbol = :none)
    regime in (:linear, :nonlinear) ||
        error("mms_forcing: regime must be :linear or :nonlinear (got :$regime)")
    regime === :linear && nl_pressure !== :none && error(
        "mms_forcing: nl_pressure=:$nl_pressure requires regime=:nonlinear " *
        "(mirrors resolve_physics, so forcing and solver cannot disagree).")

    if flat_bed
        d0 = hfun(0.0, 0.0)
        # guard: a flat_bed=true forcing over a non-constant bed is a silent-wrong-answer trap
        for (px,py) in ((0.31,0.17),(1.13,0.87),(0.66,1.02))
            isapprox(hfun(px,py), d0; rtol=1e-12) || error(
                "mms_forcing: flat_bed=true but hfun varies (h(0,0)=$d0, h($px,$py)=$(hfun(px,py))). " *
                "This mismatch would void the convergence rate — pass flat_bed=false.")
        end
        #  Model 1 keeps its hand-written closed form: it is fast, and its agreement
        #  with the AD parent is itself a gate (test_mms_forcing.jl G2).
        regime === :linear && return mms_forcing_stage1(field, vert, d0, g)
    end

    cbs = field_callables(field)
    N   = field.Nσ

    #  H = h + η* must stay positive: components {6,7,8} of 𝓝 divide by it, and every
    #  term is H-weighted. Checked ONCE here, not at every quadrature point.
    if regime === :nonlinear
        hmin = minimum(hfun(px, py) for px in range(0, 2; length=9),
                                        py in range(0, 2; length=9))
        hmin - field.a_eta > 0 || error(
            "mms_forcing: H = h+η* would reach $(hmin - field.a_eta) ≤ 0 " *
            "(min h ≈ $hmin, a_eta = $(field.a_eta)). The nonlinear models divide by H; " *
            "reduce a_eta (≤ h_min/3 recommended) or raise the depth.")
    end

    #  one-entry memo: Gridap evaluates Seta/Sx/Sy at the same point consecutively
    last_key = Ref((NaN, NaN, NaN))
    last_val = Ref{Any}(nothing)
    function eval_at(x, t)
        key = (x[1], x[2], t)
        if key !== last_key[]
            last_val[] = strong_residual_model(cbs, vert, hfun, g, x[1], x[2], t;
                                               regime = regime, flat_bed = flat_bed,
                                               nl_pressure = nl_pressure)
            last_key[] = key
        end
        return last_val[]
    end

    Seta = (x, t) -> eval_at(x, t)[1]
    Sx   = (x, t) -> (L = eval_at(x, t); VectorValue(ntuple(i -> L[2][i], N)))
    Sy   = (x, t) -> (L = eval_at(x, t); VectorValue(ntuple(i -> L[3][i], N)))
    return (Seta = Seta, Sx = Sx, Sy = Sy)
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
