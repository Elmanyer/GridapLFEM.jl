# ==============================================================
#  vertical_alg.jl — Stage 1: σ-mesh and the full LFE-M vertical tensor set
#
#  Port of ../../LFE-M_2D_solver/src/vertical_lfem2D.jl (validated 28/28)
#  with ONE addition: the nonlinear leading-pressure tensor
#      Pcal[i,k,j,c] = ∫₀¹ Θ_kj[c] φᵢ_int dσ
#  (the first Fubini piece of Kcal, previously discarded), required by the
#  𝓟-part of the leading pressure term R_P (main.tex §8, corrected).
#
#  Building blocks (CellFields on the σ-mesh):
#    φⱼ(σ)      Lagrange basis, degree p                (j = 1..N_dof)
#    φ'ⱼ(σ)     dφⱼ/dσ
#    φⱼ_int(σ)  ∫₀^σ φⱼ dσ'  (POSITIVE antiderivative; BVP dF/dσ=+φⱼ, F(0)=0)
#    Φⱼ         φⱼ_int(1) = ∫₀¹ φⱼ dσ   (depth weight; ΣΦ = 1)
#
#  Linear pressure profiles   θⱼ  = [φⱼ, σφⱼ, φⱼ_int]                (3 comp)
#  Nonlinear pressure profiles Θₖⱼ (8 comp; see main.tex §5)
#  Fubini identity: ∫₀¹ φᵢ(σ) [∫_σ¹ f(τ)dτ] dσ = ∫₀¹ f(τ) φᵢ_int(τ) dτ  —
#  so no upward integral Π/Λ is ever formed.
# ==============================================================

"""
    build_vertical_model_alg(M, c_bdy)

1D Gridap CartesianDiscreteModel for σ ∈ [0,1]: `M` elements with boundaries
at `c_bdy` (length M+1), via a piecewise-linear coordinate map.
"tag_1" = σ=0 (seabed), "tag_2" = σ=1 (free surface).
"""
function build_vertical_model_alg(M::Int, c_bdy::Vector{Float64})
    @assert length(c_bdy) == M + 1
    @assert c_bdy[1] ≈ 0.0 && c_bdy[end] ≈ 1.0

    function sigma_map(x)
        t = x[1]
        for k in 1:M
            a = (k-1) / M
            b = k / M
            if t <= b + 1e-14
                local_t = (t - a) / (b - a)
                return VectorValue(c_bdy[k] + local_t * (c_bdy[k+1] - c_bdy[k]))
            end
        end
        return VectorValue(c_bdy[end])
    end

    return CartesianDiscreteModel((0.0, 1.0), M; map=sigma_map)
end

"""
    compute_antiderivative_alg(phi_j_fn, V_F, U_F, dS)

φⱼ_int = ∫₀^σ φⱼ dσ' as a degree-(p+1) FEFunction via the BVP
dF/dσ = +φⱼ, F(0)=0 (H1-seminorm Galerkin — exact, the trial space contains
the antiderivative).
"""
function compute_antiderivative_alg(phi_j_fn, V_F, U_F, dS)
    a(F, v) = ∫((∇(F) ⋅ E1_sig) * (∇(v) ⋅ E1_sig)) * dS
    l(v)    = ∫((phi_j_fn) * (∇(v) ⋅ E1_sig)) * dS
    op      = AffineFEOperator(a, l, U_F, V_F)
    return solve(op)
end

"""
    assemble_vertical_tensors_alg(M, p, c_bdy) → NamedTuple

Full LFE-M vertical static tensor set. Fields:

  N_dof            number of vertical DOFs (= M·p + 1)
  Phi   (N)        depth weights ∫φⱼ                        (ΣΦ = 1)
  Mmat  (N×N)      vertical mass ∫φᵢφⱼ
  Mcal  (N×N×N)    horizontal advection ∫φᵢφₖφⱼ             (index [i,k,j])
  Gcal  (N×N×N)    vertical (ω) advection ∫(σΦₖ−φₖ_int)φ'ⱼφᵢ (index [i,k,j])
  A     (N×N×3)    linear pressure, bed slope    ∫φᵢθⱼ
  K     (N×N×3)    linear pressure, surface slope ∫θⱼ(φᵢ_int−σφᵢ)  (Fubini)
  P     (N×N×3)    LEADING pressure ∫θⱼφᵢ_int; P[:,:,3] = −B (dispersion)
  Acal  (N×N×N×8)  nonlinear pressure, bed slope  ∫Θₖⱼφᵢ    (index [i,k,j,c])
  Kcal  (N×N×N×8)  nonlinear pressure, surface    ∫Θₖⱼ(φᵢ_int−σφᵢ)
  Pcal  (N×N×N×8)  nonlinear LEADING pressure ∫Θₖⱼφᵢ_int    (NEW vs old solver)
  B     (N×N)      dispersion matrix −∫φᵢ_int φⱼ_int ≤ 0
  + σ-mesh/FE objects (sigma_model, phi_fns, phi_int_fns, dphi_fns, …)
"""
function assemble_vertical_tensors_alg(M::Int, p::Int, c_bdy::Vector{Float64})
    @assert length(c_bdy) == M + 1
    @assert c_bdy[1] ≈ 0.0 && c_bdy[end] ≈ 1.0

    sigma_model = build_vertical_model_alg(M, c_bdy)
    sigma_trian = Triangulation(sigma_model)

    reffe_phi = ReferenceFE(lagrangian, Float64, p)
    reffe_int = ReferenceFE(lagrangian, Float64, p + 1)

    V_phi = FESpace(sigma_model, reffe_phi; conformity=:H1)
    U_phi = TrialFESpace(V_phi)
    V_int = FESpace(sigma_model, reffe_int; conformity=:H1, dirichlet_tags=["tag_1"])
    U_int = TrialFESpace(V_int, 0.0)

    N_dof = num_free_dofs(U_phi)

    # Integrands reach degree ~3p+2 (triple products with φ_int of degree p+1);
    # 3p+4 is exact for all of them.
    dS = Measure(sigma_trian, 3*p + 4)

    sigma_cf = CellField(x -> x[1], sigma_trian)

    # ---- basis, derivative, antiderivative -----------------------------------
    phi_fns     = Vector{Any}(undef, N_dof)
    dphi_fns    = Vector{Any}(undef, N_dof)
    phi_int_fns = Vector{Any}(undef, N_dof)
    for j in 1:N_dof
        e_j = zeros(Float64, N_dof); e_j[j] = 1.0
        phi_fns[j]  = FEFunction(U_phi, e_j)
        dphi_fns[j] = ∇(phi_fns[j]) ⋅ E1_sig
    end
    for j in 1:N_dof
        phi_int_fns[j] = compute_antiderivative_alg(phi_fns[j], V_int, U_int, dS)
    end

    # ---- depth weights Φⱼ -----------------------------------------------------
    Phi = Float64[ sum(∫(phi_fns[j]) * dS) for j in 1:N_dof ]

    # ---- Mmat -----------------------------------------------------------------
    Mmat = zeros(Float64, N_dof, N_dof)
    for i in 1:N_dof, j in 1:N_dof
        Mmat[i,j] = sum(∫(phi_fns[i] * phi_fns[j]) * dS)
    end

    # ---- advection tensors Mcal, Gcal ([i,k,j]) --------------------------------
    Mcal = zeros(Float64, N_dof, N_dof, N_dof)
    Gcal = zeros(Float64, N_dof, N_dof, N_dof)
    for i in 1:N_dof, k in 1:N_dof
        sPk_minus_intk = sigma_cf * Phi[k] - phi_int_fns[k]
        for j in 1:N_dof
            Mcal[i,k,j] = sum(∫(phi_fns[i] * phi_fns[k] * phi_fns[j]) * dS)
            Gcal[i,k,j] = sum(∫(sPk_minus_intk * dphi_fns[j] * phi_fns[i]) * dS)
        end
    end

    # ---- linear pressure package: θⱼ → A, K, P ---------------------------------
    A = zeros(Float64, N_dof, N_dof, 3)
    K = zeros(Float64, N_dof, N_dof, 3)
    P = zeros(Float64, N_dof, N_dof, 3)
    for i in 1:N_dof
        phi_i = phi_fns[i]
        int_i = phi_int_fns[i]
        for j in 1:N_dof
            theta = ( phi_fns[j], sigma_cf*phi_fns[j], phi_int_fns[j] )
            for c in 1:3
                tc = theta[c]
                A[i,j,c] = sum(∫(phi_i * tc) * dS)
                P[i,j,c] = sum(∫(tc * int_i) * dS)                    # leading pressure
                K[i,j,c] = P[i,j,c] - sum(∫(sigma_cf * tc * phi_i) * dS)   # Fubini split
            end
        end
    end

    # ---- nonlinear pressure package: Θₖⱼ → Acal, Kcal, Pcal ([i,k,j,c]) --------
    Acal = zeros(Float64, N_dof, N_dof, N_dof, 8)
    Kcal = zeros(Float64, N_dof, N_dof, N_dof, 8)
    Pcal = zeros(Float64, N_dof, N_dof, N_dof, 8)
    for i in 1:N_dof
        phi_i = phi_fns[i]
        int_i = phi_int_fns[i]
        for k in 1:N_dof
            phi_k  = phi_fns[k]
            dphi_k = dphi_fns[k]
            Phk    = Phi[k]
            for j in 1:N_dof
                phi_j = phi_fns[j]
                int_j = phi_int_fns[j]
                Phj   = Phi[j]
                sPj_minus_intj = sigma_cf*Phj - int_j
                Theta = (
                    sigma_cf*Phk*phi_j,                                   # 1
                    Phk*int_j,                                            # 2
                    phi_j*phi_k,                                          # 3
                    sigma_cf*phi_j*phi_k,                                 # 4
                    int_j*phi_k,                                          # 5
                    sPj_minus_intj*dphi_k,                                # 6
                    sigma_cf*Phj*phi_k + sigma_cf*sigma_cf*Phj*dphi_k
                        - int_j*phi_k - sigma_cf*int_j*dphi_k,            # 7
                    sPj_minus_intj*phi_k,                                 # 8
                )
                for c in 1:8
                    Tc = Theta[c]
                    Acal[i,k,j,c] = sum(∫(Tc * phi_i) * dS)
                    Pcal[i,k,j,c] = sum(∫(Tc * int_i) * dS)          # leading pressure
                    Kcal[i,k,j,c] = Pcal[i,k,j,c] -
                                    sum(∫(sigma_cf * Tc * phi_i) * dS)   # Fubini split
                end
            end
        end
    end

    # ---- dispersion matrix B = −∫φᵢ_int φⱼ_int (≤ 0); identity P[:,:,3] = −B ---
    B = zeros(Float64, N_dof, N_dof)
    for i in 1:N_dof, j in 1:N_dof
        B[i,j] = -sum(∫(phi_int_fns[i] * phi_int_fns[j]) * dS)
    end

    return (
        N_dof = N_dof,
        Phi   = Phi,
        Mmat  = Mmat,
        Mcal  = Mcal,  Gcal = Gcal,
        A     = A,     K    = K,     P    = P,
        Acal  = Acal,  Kcal = Kcal,  Pcal = Pcal,
        B     = B,
        sigma_model = sigma_model, sigma_trian = sigma_trian,
        phi_fns = phi_fns, dphi_fns = dphi_fns, phi_int_fns = phi_int_fns,
        U_phi = U_phi, V_phi = V_phi, U_int = U_int, V_int = V_int,
        M = M, p = p, c_bdy = c_bdy,
    )
end
