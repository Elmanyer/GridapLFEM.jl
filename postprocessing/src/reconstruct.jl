# ==============================================================
#  reconstruct.jl — vertical-velocity w(σ) and pressure p(σ) profiles
#                   FROM THE STORED VELOCITY MODES (no solver needed)
#
#  The solver can write w_s<σ>/p_s<σ> fields directly (src/reconstruct.jl), but
#  only at the Nσ vertical Lagrange nodes and only if write_w/write_pressure were
#  on. This module reconstructs the SAME quantities purely in postprocessing —
#  from the horizontal velocity-mode fields (u1x,u1y,u2x,…) — so a profile can be
#  obtained at ANY σ, on a run that did not store them, and cross-checked against
#  the solver's own w_s/p_s (they agree to the finite-difference/FE gradient
#  discretisation).  Math ported verbatim from ../../src/reconstruct.jl:
#
#     w(σ) = −(a(σ)·𝖺) + (b(σ)·𝖻) − (c(σ)·𝖲)                    [full, any bed]
#       a_j(σ)=φⱼ(σ),  b_j(σ)=σφⱼ(σ),  c_j(σ)=φⱼ_int(σ)          (σ-basis, per mode j)
#       𝖺_j=u_j·∇h,  𝖻_j=u_j·∇H,  𝖲_j=H(∂ₓu_jˣ+∂_yu_jʸ)+𝖻_j     (per mode, at the point)
#
#     p(σ)   = ρg(1−σ)H            [hydrostatic, exact]
#            − ρd² (π(σ)·dDU/dt)   [non-hydrostatic; π_j(σ)=∫_σ¹ φⱼ_int, →0 at σ=1]
#     p_nh(σ)= the non-hydrostatic term alone (for the Airy cosh comparison)
#
#  The σ-basis (φⱼ, φⱼ_int, πⱼ) is rebuilt analytically from (c_bdy, p) with
#  exact Gauss–Legendre quadrature — no Gridap dependency. Spatial derivatives at
#  the station come from central differences on the regularised node grid; the
#  pressure's ∂ₜ(∇·u) uses a backward difference across snapshots.
# ==============================================================

# ---- σ-basis: piecewise-Lagrange on the optimised σ-mesh ---------------------
"Piecewise-Lagrange basis on the σ-mesh (element boundaries `c_bdy`, order `p`)."
struct SigmaBasis
    c_bdy       :: Vector{Float64}
    p           :: Int
    gnodes      :: Vector{Vector{Int}}       # per element: global node indices
    local_nodes :: Vector{Vector{Float64}}   # per element: node σ-positions
    Nσ          :: Int
end

"""
    sigma_basis(c_bdy; p=1) -> SigmaBasis

Build the BALFE-M vertical basis. Nodes = `p+1` equispaced points per element,
deduplicated at shared boundaries, ordered by ascending σ — matching Gridap's
DOF order for the `p=1` optimised-node meshes used by P1LFE-2/3/4 (mode `j` ↔
σ-node `j`). For `p>1` the interior-node order should be verified against the
solver's `phi_fns`.
"""
function sigma_basis(c_bdy::AbstractVector; p::Int=1)
    nelem = length(c_bdy) - 1
    gnodes = Vector{Vector{Int}}(undef, nelem)
    local_nodes = Vector{Vector{Float64}}(undef, nelem)
    ids = Dict{Float64,Int}(); nxt = 0
    for e in 1:nelem
        a, b = c_bdy[e], c_bdy[e+1]
        locs = [a + (b - a) * m / p for m in 0:p]
        gids = Int[]
        for σ in locs
            key = round(σ; digits=12)
            haskey(ids, key) || (nxt += 1; ids[key] = nxt)
            push!(gids, ids[key])
        end
        gnodes[e] = gids; local_nodes[e] = locs
    end
    return SigmaBasis(collect(float.(c_bdy)), p, gnodes, local_nodes, nxt)
end

# 5-point Gauss–Legendre on [-1,1] (exact to degree 9 ⇒ exact for all BALFE-M p≤3)
const _GL_X = (-0.9061798459386640, -0.5384693101056831, 0.0,
                0.5384693101056831,  0.9061798459386640)
const _GL_W = ( 0.2369268850561891,  0.4786286704993665, 0.5688888888888889,
                0.4786286704993665,  0.2369268850561891)

"Which element contains σ (last element whose left node ≤ σ)."
function _elem_of(b::SigmaBasis, σ)
    e = 1
    for k in 1:length(b.c_bdy)-1
        b.c_bdy[k] ≤ σ + 1e-12 && (e = k)
    end
    return e
end

"φⱼ(σ): global nodal Lagrange basis for mode j."
function phi(b::SigmaBasis, j::Int, σ::Real)
    e = _elem_of(b, σ)
    gid = b.gnodes[e]; m = findfirst(==(j), gid)
    m === nothing && return 0.0
    x = b.local_nodes[e]; v = 1.0
    @inbounds for r in eachindex(x)
        r == m && continue
        v *= (σ - x[r]) / (x[m] - x[r])
    end
    return v
end

"∫ φⱼ over a single smooth panel [a,b] (5-pt Gauss, exact for the Lagrange poly)."
function _panel_int(f, a, b)
    h = (b - a) / 2; c = (a + b) / 2; s = 0.0
    @inbounds for q in 1:5
        s += _GL_W[q] * f(c + h * _GL_X[q])
    end
    return h * s
end

"Integrate `f` over [a,b], splitting at the σ-mesh breakpoints (kinks of φ)."
function _piecewise_int(b::SigmaBasis, f, a, z)
    a ≥ z && return 0.0
    brk = filter(x -> a < x < z, b.c_bdy)
    pts = vcat(a, brk, z); s = 0.0
    for k in 1:length(pts)-1
        s += _panel_int(f, pts[k], pts[k+1])
    end
    return s
end

"φⱼ_int(σ) = ∫₀^σ φⱼ dσ'  (the unit vertical basis antiderivative; φⱼ_int(0)=0)."
phi_int(b::SigmaBasis, j::Int, σ::Real) = _piecewise_int(b, s -> phi(b, j, s), 0.0, σ)

"πⱼ(σ) = ∫_σ¹ φⱼ_int dσ'  (surface-vanishing pressure moment; πⱼ(1)=0)."
pi3(b::SigmaBasis, j::Int, σ::Real) = _piecewise_int(b, s -> phi_int(b, j, s), σ, 1.0)

"Unit vertical velocity wⱼ(σ) = −φⱼ_int(σ)  (bottom BC wⱼ(0)=0)."
unit_w(b::SigmaBasis, j::Int, σ::Real) = -phi_int(b, j, σ)

# ---- point-wise gradient on the node grid ------------------------------------
_idx_near(xs, x) = clamp(argmin(abs.(xs .- x)), 1, length(xs))

"(value, ∂/∂x, ∂/∂y) of a field COLUMN at (x,y) by central differences on the grid."
function _grad_at_col(col::AbstractVector, gv::GridView, x, y)
    i = _idx_near(gv.xs, x); j = _idx_near(gv.ys, y)
    Nx = length(gv.xs); Ny = length(gv.ys)
    ip = min(i+1, Nx); im = max(i-1, 1)
    jp = min(j+1, Ny); jm = max(j-1, 1)
    val  = col[gv.idx[i, j]]
    dvdx = (col[gv.idx[ip, j]] - col[gv.idx[im, j]]) / (gv.xs[ip] - gv.xs[im])
    dvdy = (col[gv.idx[i, jp]] - col[gv.idx[i, jm]]) / (gv.ys[jp] - gv.ys[jm])
    return val, dvdx, dvdy
end

_depth_at(d::Real, x, y) = (float(d), 0.0, 0.0)
function _depth_at(d, x, y)                     # d is a callable d(x,y)
    h = 1e-4
    return (float(d(x, y)),
            (d(x+h, y) - d(x-h, y)) / (2h),
            (d(x, y+h) - d(x, y-h)) / (2h))
end

# ---- the reconstruction ------------------------------------------------------
"""
    reconstruct_profile(sim, x, y; kind=:w, c_bdy, p=1, depth,
                        rho=1025.0, g=9.81, ω=nothing, it=nothing, nσ=61) -> SigmaProfile

Reconstruct the vertical profile of `kind` at station (x,y) from the stored
velocity-mode fields `u{j}x,u{j}y` (j=1..Nσ). `kind`:
  `:w`   vertical velocity w(σ),
  `:p`   total pressure p(σ) = hydrostatic + non-hydrostatic,
  `:pnh` non-hydrostatic pressure only (for the Airy cosh comparison).
`c_bdy`/`p` define the σ-mesh (the solver's `M`/`c_bdy`); `depth` is the
still-water depth `d` as a Number (flat bed) or a callable `d(x,y)` (∇h by FD).

Profile value per σ-level: the DFT first-harmonic amplitude at `ω` if given
(the natural comparison to Airy shapes); the peak steady amplitude if `ω` is
`nothing`; or the instantaneous value at snapshot `it` if `it` is given.
Requires a regularised grid (`load_simulation` calls `regularize!`).
"""
function reconstruct_profile(sim::WaveSimulation, x::Real, y::Real;
                             kind::Symbol=:w, c_bdy::AbstractVector, p::Int=1,
                             depth, rho::Real=1025.0, g::Real=9.81,
                             ω=nothing, it=nothing, nσ::Int=61)
    gv = sim.grid
    gv === nothing && error("reconstruct_profile needs a regularised grid (regularize!)")
    kind in (:w, :p, :pnh) || error("kind must be :w, :p or :pnh")
    basis = sigma_basis(c_bdy; p=p); Nσ = basis.Nσ
    for j in 1:Nσ
        (haskey(sim.fields, "u$(j)x") && haskey(sim.fields, "u$(j)y")) ||
            error("simulation lacks mode field u$(j)x/u$(j)y (need Nσ=$Nσ modes)")
    end

    σgrid = collect(range(0.0, 1.0; length=nσ))
    PHI  = [phi(basis, j, σ)        for σ in σgrid, j in 1:Nσ]
    PHIs = [σgrid[ℓ] * PHI[ℓ, j]     for ℓ in 1:nσ,  j in 1:Nσ]
    PHIi = [phi_int(basis, j, σ)    for σ in σgrid, j in 1:Nσ]
    PI3  = [pi3(basis, j, σ)        for σ in σgrid, j in 1:Nσ]

    dval, ddx, ddy = _depth_at(depth, x, y)
    its = it === nothing ? collect(1:nsnapshots(sim)) : [Int(it)]
    W = zeros(nσ, length(its))
    DUprev = nothing; tprev = 0.0

    for (n, itn) in enumerate(its)
        η, dηx, dηy = _grad_at_col(sim.fields["eta"][:, itn], gv, x, y)
        H = dval + η; dHx = ddx + dηx; dHy = ddy + dηy
        ux = zeros(Nσ); uy = zeros(Nσ); DU = zeros(Nσ)
        for j in 1:Nσ
            uxj, duxj, _   = _grad_at_col(sim.fields["u$(j)x"][:, itn], gv, x, y)
            uyj, _,  duyj  = _grad_at_col(sim.fields["u$(j)y"][:, itn], gv, x, y)
            ux[j] = uxj; uy[j] = uyj; DU[j] = duxj + duyj
        end
        𝖺 = ddx .* ux .+ ddy .* uy         # u·∇h
        𝖻 = dHx .* ux .+ dHy .* uy         # u·∇H
        𝖲 = H .* DU .+ 𝖻                    # ∇·(Hu)

        if kind === :w
            for ℓ in 1:nσ
                W[ℓ, n] = -dot(view(PHI, ℓ, :), 𝖺) +
                           dot(view(PHIs, ℓ, :), 𝖻) -
                           dot(view(PHIi, ℓ, :), 𝖲)
            end
        else
            if kind === :p
                for ℓ in 1:nσ
                    W[ℓ, n] = rho * g * (1.0 - σgrid[ℓ]) * H
                end
            end
            if DUprev !== nothing
                dt = sim.times[itn] - tprev
                dDU = (DU .- DUprev) ./ dt
                for ℓ in 1:nσ
                    W[ℓ, n] += -rho * dval^2 * dot(view(PI3, ℓ, :), dDU)
                end
            end
            DUprev = DU; tprev = sim.times[itn]
        end
    end

    ts = sim.times[its]
    val = if it !== nothing
        W[:, 1]
    elseif ω !== nothing
        [amplitude_at(ts, W[ℓ, :], ω) for ℓ in 1:nσ]
    else
        [steady_amplitude(ts, W[ℓ, :]) for ℓ in 1:nσ]
    end
    return SigmaProfile(σgrid, collect(val), kind === :pnh ? :p : kind)
end

"Convenience: reconstruct the vertical-velocity profile w(σ) (see `reconstruct_profile`)."
reconstruct_w(sim, x, y; kwargs...) = reconstruct_profile(sim, x, y; kind=:w, kwargs...)

"Convenience: reconstruct the non-hydrostatic pressure profile p_nh(σ)."
reconstruct_pressure(sim, x, y; kwargs...) = reconstruct_profile(sim, x, y; kind=:pnh, kwargs...)
