# ==============================================================
#  reconstruct_alg.jl — w / total-pressure VTK fields (stacked layout)
#
#  Port of ../../LFE-M_2D_solver/src/reconstruct_fields_lfem2D.jl to the
#  algebraic package. Turns the stacked solution (η, 𝖴x, 𝖴y) into extra 2D
#  cellfields — the physical vertical velocity w(x,σ_ℓ) and the TOTAL pressure
#  p(x,σ_ℓ) — at the N_dof vertical Lagrange σ-nodes, appended to VTK snapshots.
#
#  In the stacked layout the per-mode sums of the old solver collapse to THREE
#  constant-vector contractions per level (the residual's own building blocks):
#
#     w_ℓ = −(a_ℓ ⋅ 𝖺) + (b_ℓ ⋅ 𝖻) − (c_ℓ ⋅ 𝖲)
#           a_ℓ = φ(σ_ℓ),  b_ℓ = σ_ℓ φ(σ_ℓ),  c_ℓ = φ_int(σ_ℓ)   ∈ ℝ^{Nσ}
#           𝖺 = u·∇h, 𝖻 = u·∇H, 𝖲 = ∇·(Hu)  (stacked layer-vectors)
#     p_ℓ = ρ g (1−σ_ℓ) H                       [hydrostatic, exact]
#         − ρ d² (π_ℓ ⋅ ΔDU/dt)                 [non-hydrostatic, linear flat-bed]
#           π_ℓ[j] = ∫_{σ_ℓ}^1 φ_j_int dσ'  ⇒  p_nh|_{σ=1} = 0  (surface BC)
#           ΔDU/dt = backward-FD div(u̇) from the previous step's DOFs
#
#  Pure CellField algebra (alg_dot contractions) → identical code for the
#  sequential and the distributed (GridapDistributed) triangulations.
# ==============================================================

"VTK-safe suffix for a σ-level, e.g. 0.728 → \"0_728\"."
recon_level_str_alg(σ::Float64) = replace(@sprintf("%.3f", σ), "." => "_")

"""
    build_field_recon_alg(vert, d_func, g; rho=1025.0, write_w=false, write_pressure=false)

Precompute the per-level constant contraction vectors from the vertical stage
(`assemble_vertical_tensors_alg` NamedTuple). σ-levels = the N_dof vertical
Lagrange nodes (one per velocity mode). Returns `nothing` if both switches are
off; else a NamedTuple consumed by `extra_field_cellfields_alg`.
"""
function build_field_recon_alg(vert, d_func, g::Float64;
                               rho::Float64 = 1025.0,
                               write_w::Bool = false,
                               write_pressure::Bool = false)
    (write_w || write_pressure) || return nothing

    N_dof = vert.N_dof
    p     = vert.p
    c_bdy = vert.c_bdy

    # σ-levels = vertical Lagrange nodes: p+1 equispaced points per element, dedup.
    nelem  = length(c_bdy) - 1
    levels = Float64[]
    for e in 1:nelem
        a, b = c_bdy[e], c_bdy[e+1]
        for m in 0:p
            push!(levels, a + (b - a) * m / p)
        end
    end
    levels = sort(unique(round.(levels; digits = 12)))
    L = length(levels)
    @assert L == N_dof "reconstruction σ-levels ($L) must equal N_dof ($N_dof)"

    phi_fns     = vert.phi_fns
    phi_int_fns = vert.phi_int_fns

    # constant contraction vectors per level (VectorValue{Nσ})
    avec = Vector{Any}(undef, L)   # a_ℓ[j] = φⱼ(σ_ℓ)
    bvec = Vector{Any}(undef, L)   # b_ℓ[j] = σ_ℓ φⱼ(σ_ℓ)
    cvec = Vector{Any}(undef, L)   # c_ℓ[j] = φⱼ_int(σ_ℓ)
    for (ℓ, σ) in enumerate(levels)
        pt = VectorValue(σ)
        aj = Float64[phi_fns[j](pt) for j in 1:N_dof]
        avec[ℓ] = alg_to_vec(aj)
        bvec[ℓ] = alg_to_vec(σ .* aj)
        cvec[ℓ] = alg_to_vec(Float64[phi_int_fns[j](pt) for j in 1:N_dof])
    end

    # π_ℓ[j] = ∫_{σ_ℓ}^1 φⱼ_int dσ'  via the second antiderivative ξⱼ (ξⱼ' = φⱼ_int,
    # ξⱼ(0)=0, degree p+2): π_ℓ[j] = ξⱼ(1) − ξⱼ(σ_ℓ). Only needed for pressure.
    pivec = Vector{Any}(undef, L)
    if write_pressure
        reffe2 = ReferenceFE(lagrangian, Float64, p + 2)
        V2  = FESpace(vert.sigma_model, reffe2; conformity = :H1, dirichlet_tags = ["tag_1"])
        U2  = TrialFESpace(V2, 0.0)
        dS2 = Measure(vert.sigma_trian, 3 * p + 6)
        pt1 = VectorValue(1.0)
        Pi3 = zeros(Float64, L, N_dof)
        for j in 1:N_dof
            xi_j = compute_antiderivative_alg(phi_int_fns[j], V2, U2, dS2)
            xi1  = xi_j(pt1)
            for (ℓ, σ) in enumerate(levels)
                Pi3[ℓ, j] = xi1 - xi_j(VectorValue(σ))
            end
        end
        for ℓ in 1:L
            pivec[ℓ] = alg_to_vec(Pi3[ℓ, :])
        end
    end

    return (levels = levels, L = L, N_dof = N_dof,
            avec = avec, bvec = bvec, cvec = cvec, pivec = pivec,
            d_func = d_func, g = g, rho = rho,
            write_w = write_w, write_pressure = write_pressure)
end

"""
    extra_field_cellfields_alg(u_n, u_prev, dt, recon, trian) -> Vector{Pair{String,Any}}

Extra VTK cellfields (`w_s<σ>` and/or total pressure `p_s<σ>` per σ-level) from
the current stacked solution `u_n = [η,𝖴x,𝖴y]`. `u_prev` (previous step, same
space) feeds the u̇ backward FD of the non-hydrostatic pressure; pass `nothing`
on the first step to zero that contribution. Works on sequential and
distributed triangulations alike.
"""
function extra_field_cellfields_alg(u_n, u_prev, dt::Float64, recon, trian)
    fields = Pair{String,Any}[]
    recon === nothing && return fields

    d_cf = CellField(recon.d_func, trian)
    η    = u_n[1]
    Ux   = u_n[2]
    Uy   = u_n[3]
    H    = d_cf + η

    dhx = alg_dx(d_cf);      dhy = alg_dy(d_cf)
    dHx = dhx + alg_dx(η);   dHy = dhy + alg_dy(η)

    # stacked building blocks (same objects as the residual)
    DU = alg_dx(Ux) + alg_dy(Uy)
    a  = dhx*Ux + dhy*Uy                       # u·∇h
    b  = dHx*Ux + dHy*Uy                       # u·∇H
    S  = H*DU + b                              # ∇·(Hu)

    if recon.write_w
        for ℓ in 1:recon.L
            σ  = recon.levels[ℓ]
            wℓ = (-1.0)*alg_dot(recon.avec[ℓ], a) +
                        alg_dot(recon.bvec[ℓ], b) +
                 (-1.0)*alg_dot(recon.cvec[ℓ], S)
            push!(fields, "w_s$(recon_level_str_alg(σ))" => wℓ)
        end
    end

    if recon.write_pressure
        have_dot = u_prev !== nothing
        DUdot = have_dot ?
            (DU - (alg_dx(u_prev[2]) + alg_dy(u_prev[3]))) * (1.0/dt) : nothing
        d2 = d_cf * d_cf
        for ℓ in 1:recon.L
            σ  = recon.levels[ℓ]
            pℓ = (recon.rho * recon.g * (1.0 - σ)) * H
            if have_dot
                pℓ = pℓ + (-recon.rho) * (d2 * alg_dot(recon.pivec[ℓ], DUdot))
            end
            push!(fields, "p_s$(recon_level_str_alg(σ))" => pℓ)
        end
    end

    return fields
end
