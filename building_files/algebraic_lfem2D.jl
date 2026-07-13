# ==============================================================
#  algebraic_lfem2D.jl — Loop-free (algebraic) LFE-M 2D residual
#
#  Implements the corrected main.tex §8 global residual (including the
#  leading-pressure / dispersion term R_P, restored 2026-07-10) with the
#  STACKED value-type layout of algebraic_residual_math.md:
#
#    MultiField = [η, 𝖴x, 𝖴y]     (3 fields; 𝖴x,𝖴y ∈ VectorValue{Nσ})
#
#  Every vertical (layer) sum is a native tensor contraction:
#    Σⱼ M_ij u̇ⱼ                = 𝗠 ⋅ 𝖴t              (matvec)
#    Σ_kj 𝓜_ikj (u_k·∇u_j)     = double_contraction(𝗠3, Ux⊗∂xUx + Uy⊗∂yUx)
#    Σ_c Σⱼ P_ij^(c) L_j^(c)   = Σ_c P[c] ⋅ L(c)     (3 matvecs)
#  No per-layer for-loops, no MultiField decomposition beyond u[1],u[2],u[3].
#
#  Terms match the oracle `residual_lfem` (problem_lfem2D.jl) exactly, flag
#  by flag (verified by test_algebraic_lfem2D.jl):
#    mass, acceleration, dispersion R_P (lin: d²B·div(u̇); nonlin: H²B·∇·(Hu̇);
#    P_full=true adds the full P¹L¹+P²L² leading-pressure components),
#    gravity (oracle IBP form, hydrostatic baseline subtracted),
#    sponge, wavemaker (−∫q·src), advection (𝓜/𝓖 block),
#    linear pressure (A/K package), nonlinear pressure comps 6–8 (native).
#
#  Requires the LFEModel2D module to be loaded first (vertical tensors,
#  mesh/sponge/wavemaker helpers, oracle for the equivalence test):
#      include("../LFE-M_2D_solver/src/LFEModel2D.jl"); using .LFEModel2D
#      using Gridap, Gridap.ODEs, LinearAlgebra, Printf
#      include("algebraic_lfem2D.jl")
# ==============================================================

# ----------------------------------------------------------
#  Constant-tensor constructors (build time only)
#  Gridap TensorValue data is column-major (first index fastest):
#    TensorValue{N,N}(data...)  →  T[i,j] = data[i + (j-1)N]
#    ThirdOrderTensorValue{N,N,N}(data...) → T[i,j,k] = data[i+(j-1)N+(k-1)N²]
#  (verified by the primitives test)
# ----------------------------------------------------------

alg_to_vec(v::AbstractVector{Float64}) = VectorValue{length(v),Float64}(v...)

function alg_to_tensor2(M::AbstractMatrix{Float64})
    N = size(M, 1); @assert size(M, 2) == N
    data = ntuple(k -> M[(k-1) % N + 1, (k-1) ÷ N + 1], N * N)
    return TensorValue{N,N,Float64}(data...)
end

function alg_to_tensor3(T::AbstractArray{Float64,3})
    N = size(T, 1); @assert size(T, 2) == N && size(T, 3) == N
    data = ntuple(N^3) do k
        k0 = k - 1
        i = k0 % N + 1
        j = (k0 ÷ N) % N + 1
        l = k0 ÷ (N * N) + 1
        T[i, j, l]
    end
    return ThirdOrderTensorValue{N,N,N,Float64}(data...)
end

# ----------------------------------------------------------
#  Pointwise-algebra helpers (Operation wrappers; constants are captured
#  in closures — allowed, only CellFields must not be closed over in loops)
# ----------------------------------------------------------

const ALG_EX = VectorValue(1.0, 0.0)
const ALG_EY = VectorValue(0.0, 1.0)

"∂x of a scalar or VectorValue{Nσ} CellField: e_x ⋅ ∇f (spatial index first)."
alg_dx(f) = Operation(g -> ALG_EX ⋅ g)(∇(f))
alg_dy(f) = Operation(g -> ALG_EY ⋅ g)(∇(f))

"Constant TensorValue{N,N} ⋅ VectorValue{N}-CellField → VectorValue{N}-CellField."
alg_mul(A::TensorValue, u) = Operation(x -> A ⋅ x)(u)

"Constant VectorValue{N} ⋅ VectorValue{N}-CellField → scalar CellField."
alg_dot(a::VectorValue, u) = Operation(x -> a ⋅ x)(u)

"double_contraction(constant ThirdOrderTensorValue, TensorValue-CellField) → VectorValue-CellField.
Contracts the TRAILING two indices: result_i = Σ_{k,j} T[i,k,j] S[k,j]."
alg_dc3(T::ThirdOrderTensorValue, S) =
    Operation(s -> Gridap.TensorValues.double_contraction(T, s))(S)

"Outer product of two VectorValue{N}-CellFields → TensorValue{N,N}-CellField, (a⊗b)[k,j]=a[k]b[j]."
alg_outer(a, b) = Operation(Gridap.TensorValues.outer)(a, b)

"Fuse two scalar CellFields into a VectorValue{2}-CellField."
alg_vec2(a, b) = Operation(VectorValue)(a, b)

# ----------------------------------------------------------
#  Problem bundle
# ----------------------------------------------------------

"""
    AlgebraicLFEM

Coefficient bundle for the stacked algebraic residual. All vertical tensors are
stored as constant `TensorValue`/`ThirdOrderTensorValue` (index order `[i,k,j]`
= [test layer, u_k layer, u_j layer] — the solver's `Mcal/Gcal/Acal/Kcal` are
already in this order, no remap needed).
"""
struct AlgebraicLFEM{PV,MV,BV,PT,AT,KT,M3T,G3T,A3T,K3T}
    g            :: Float64
    d_func       :: Function
    Nσ           :: Int
    Φ            :: PV     # VectorValue{Nσ}                depth weights
    Mv           :: MV     # TensorValue{Nσ,Nσ}             vertical mass
    Bv           :: BV     # TensorValue{Nσ,Nσ}             dispersion B ≤ 0 (= −P[:,:,3])
    P            :: PT     # NTuple{3,TensorValue}          leading pressure P^V
    Av           :: AT     # NTuple{3,TensorValue}          linear pressure (∇h)
    Kv           :: KT     # NTuple{3,TensorValue}          linear pressure (∇H)
    M3           :: M3T    # ThirdOrderTensorValue          horizontal advection 𝓜
    G3           :: G3T    # ThirdOrderTensorValue          vertical advection 𝓖
    A3           :: A3T    # NTuple{8,ThirdOrderTensorValue} nonlinear pressure 𝓐
    K3           :: K3T    # NTuple{8,ThirdOrderTensorValue} nonlinear pressure 𝓚
    linearised   :: Bool   # drop H on Acc/Grav, dispersion in d²B form (oracle baseline)
    advection    :: Bool
    lin_pressure :: Bool
    P_full       :: Bool   # true → full P¹L¹+P²L²+P³L³ leading pressure (oracle keeps P³L³ only)
    nl_pressure68:: Bool   # native nonlinear-pressure components 6–8
    mu_sponge    :: Function
    wm_src       :: Function
end

"""
    build_algebraic_problem(vert; g, d_func, kwargs...) → AlgebraicLFEM

Reshape the validated `assemble_vertical_tensors_lfem` NamedTuple into constant
Gridap tensors and bundle the runtime flags.
"""
function build_algebraic_problem(vert;
        g            :: Float64  = 9.81,
        d_func       :: Function = (x -> 3.5),
        linearised   :: Bool     = false,
        advection    :: Bool     = true,
        lin_pressure :: Bool     = false,
        P_full       :: Bool     = false,
        nl_pressure68:: Bool     = false,
        mu_sponge    :: Function = (x -> 0.0),
        wm_src       :: Function = ((x, t) -> 0.0))
    Φ  = alg_to_vec(vert.Phi)
    Mv = alg_to_tensor2(vert.Mmat)
    Bv = alg_to_tensor2(vert.B)
    P  = ntuple(c -> alg_to_tensor2(vert.P[:, :, c]), 3)
    Av = ntuple(c -> alg_to_tensor2(vert.A[:, :, c]), 3)
    Kv = ntuple(c -> alg_to_tensor2(vert.K[:, :, c]), 3)
    M3 = alg_to_tensor3(vert.Mcal)
    G3 = alg_to_tensor3(vert.Gcal)
    A3 = ntuple(c -> alg_to_tensor3(vert.Acal[:, :, :, c]), 8)
    K3 = ntuple(c -> alg_to_tensor3(vert.Kcal[:, :, :, c]), 8)
    return AlgebraicLFEM(g, d_func, vert.N_dof, Φ, Mv, Bv, P, Av, Kv, M3, G3,
                         A3, K3, linearised, advection, lin_pressure, P_full,
                         nl_pressure68, mu_sponge, wm_src)
end

# ----------------------------------------------------------
#  Stacked FE spaces
# ----------------------------------------------------------

"""
    build_fe_spaces_algebraic(model, fe_order, Nσ; y_wall_bc, x_wall_bc) → (U, V)

Three-field MultiFieldFESpace `[η, 𝖴x, 𝖴y]` with `𝖴x,𝖴y` VectorValue{Nσ}-valued.
Solid-wall Dirichlet acts on the WHOLE stacked field (all Nσ components zero);
corner tags are mandatory (root CLAUDE.md rule 5).
"""
function build_fe_spaces_algebraic(model, fe_order::Int, Nσ::Int;
                                   y_wall_bc::Bool = true, x_wall_bc::Bool = false)
    reffe_eta = ReferenceFE(lagrangian, Float64, fe_order)
    reffe_U   = ReferenceFE(lagrangian, VectorValue{Nσ,Float64}, fe_order)
    zvv       = VectorValue(ntuple(_ -> 0.0, Nσ)...)
    y_tags = y_wall_bc ? ["tag_1","tag_2","tag_3","tag_4","tag_5","tag_6"] : String[]
    x_tags = x_wall_bc ? ["tag_1","tag_2","tag_3","tag_4","tag_7","tag_8"] : String[]

    V_eta = FESpace(model, reffe_eta; conformity=:H1)
    U_eta = TrialFESpace(V_eta)
    V_Ux  = isempty(x_tags) ? FESpace(model, reffe_U; conformity=:H1) :
                              FESpace(model, reffe_U; conformity=:H1, dirichlet_tags=x_tags)
    U_Ux  = isempty(x_tags) ? TrialFESpace(V_Ux) : TrialFESpace(V_Ux, zvv)
    V_Uy  = isempty(y_tags) ? FESpace(model, reffe_U; conformity=:H1) :
                              FESpace(model, reffe_U; conformity=:H1, dirichlet_tags=y_tags)
    U_Uy  = isempty(y_tags) ? TrialFESpace(V_Uy) : TrialFESpace(V_Uy, zvv)

    U = MultiFieldFESpace([U_eta, U_Ux, U_Uy])
    V = MultiFieldFESpace([V_eta, V_Ux, V_Uy])
    return U, V
end

# ----------------------------------------------------------
#  The loop-free residual (main.tex §8, corrected)
# ----------------------------------------------------------

"""
    residual_algebraic(t, u, v, prob, trian, dO)

Single scalar Gridap residual for the stacked layout. `u` is a
TransientCellField (`∂t(u)` available); `u[1]=η`, `u[2]=𝖴x`, `u[3]=𝖴y`.
"""
function residual_algebraic(t::Real, u, v, prob::AlgebraicLFEM, trian, dO)
    ut = ∂t(u)
    η,  Ux,  Uy  = u[1], u[2], u[3]
    ηt, Uxt, Uyt = ut[1], ut[2], ut[3]
    q,  Wx,  Wy  = v[1], v[2], v[3]

    g   = prob.g
    lin = prob.linearised
    d_cf   = CellField(prob.d_func, trian)
    src_cf = CellField(x -> prob.wm_src(x, t), trian)
    mu_cf  = CellField(prob.mu_sponge, trian)
    H = d_cf + η

    # derived scalar / layer-vector fields (math doc §0/§6b)
    dhx = alg_dx(d_cf);        dhy = alg_dy(d_cf)          # bed slope ∇h
    dHx = dhx + alg_dx(η);     dHy = dhy + alg_dy(η)       # ∇H = ∇d + ∇η
    DW  = alg_dx(Wx) + alg_dy(Wy)                          # test-divergence vector
    DUt = alg_dx(Uxt) + alg_dy(Uyt)                        # Σ layer div(u̇)
    ub  = alg_vec2(alg_dot(prob.Φ, Ux), alg_dot(prob.Φ, Uy))  # depth-avg velocity ū

    # ---- mass continuity + wavemaker source --------------------------------
    r = ∫( q*ηt - H*(∇(q) ⋅ ub) - q*src_cf ) * dO

    # ---- acceleration -------------------------------------------------------
    accx = alg_mul(prob.Mv, Uxt); accy = alg_mul(prob.Mv, Uyt)
    r = lin ? r + ∫( (Wx ⋅ accx) + (Wy ⋅ accy) ) * dO :
              r + ∫( H*(Wx ⋅ accx) + H*(Wy ⋅ accy) ) * dO

    # ---- gravity (oracle IBP form; hydrostatic baseline subtracted) ---------
    PhiDW = alg_dot(prob.Φ, DW)
    r = lin ? r + ∫( (-g)*η*PhiDW ) * dO :
              r + ∫( (-0.5*g)*(H*H - d_cf*d_cf)*PhiDW ) * dO

    # ---- leading pressure R_P (dispersion — MANDATORY; math doc §6b) --------
    if lin
        # flat-linearised: −∫ d² (B·DUt)·DW
        r = r + ∫( (-1.0)*(d_cf*d_cf)*((alg_mul(prob.Bv, DUt)) ⋅ DW) ) * dO
    else
        UgHt = dHx*Uxt + dHy*Uyt                           # u̇ⱼ·∇H stacked
        if prob.P_full
            L1 = (-1.0)*(dhx*Uxt + dhy*Uyt)
            L2 = UgHt
            L3 = (-1.0)*(H*DUt + UgHt)
            sP = alg_mul(prob.P[1], L1) + alg_mul(prob.P[2], L2) + alg_mul(prob.P[3], L3)
            r  = r + ∫( (-1.0)*(H*H)*(sP ⋅ DW) ) * dO
        else
            # oracle form: −∫ H² (B·∇·(Hu̇))·DW    (only the P³L³ product)
            r = r + ∫( (-1.0)*(H*H)*((alg_mul(prob.Bv, H*DUt + UgHt)) ⋅ DW) ) * dO
        end
    end

    # ---- sponge --------------------------------------------------------------
    r = r + ∫( mu_cf*((Wx ⋅ alg_mul(prob.Mv, Ux)) + (Wy ⋅ alg_mul(prob.Mv, Uy))) ) * dO

    # ---- nonlinear advection (𝓜/𝓖 block) ------------------------------------
    if prob.advection
        DU  = alg_dx(Ux) + alg_dy(Uy)
        UgH = dHx*Ux + dHy*Uy
        S   = H*DU + UgH                                   # ∇·(H u_j) stacked
        TMx = alg_outer(Ux, alg_dx(Ux)) + alg_outer(Uy, alg_dy(Ux))
        TMy = alg_outer(Ux, alg_dx(Uy)) + alg_outer(Uy, alg_dy(Uy))
        advx = alg_dc3(prob.M3, TMx);  advy = alg_dc3(prob.M3, TMy)
        gvx  = alg_dc3(prob.G3, alg_outer(S, Ux))
        gvy  = alg_dc3(prob.G3, alg_outer(S, Uy))
        r = r + ∫( H*(advx ⋅ Wx) + H*(advy ⋅ Wy) + (gvx ⋅ Wx) + (gvy ⋅ Wy) ) * dO
    end

    # ---- linear non-hydrostatic pressure (A/K slope package) -----------------
    if prob.lin_pressure
        UgHt = dHx*Uxt + dHy*Uyt
        L1 = (-1.0)*(dhx*Uxt + dhy*Uyt)
        L2 = UgHt
        L3 = (-1.0)*(H*DUt + UgHt)
        LA = alg_mul(prob.Av[1], L1) + alg_mul(prob.Av[2], L2) + alg_mul(prob.Av[3], L3)
        LK = alg_mul(prob.Kv[1], L1) + alg_mul(prob.Kv[2], L2) + alg_mul(prob.Kv[3], L3)
        r = r + ∫( (-1.0)*H*( dhx*(Wx ⋅ LA) + dHx*(Wx ⋅ LK)
                            + dhy*(Wy ⋅ LA) + dHy*(Wy ⋅ LK) ) ) * dO
    end

    # ---- nonlinear pressure, first-order components 6–8 (native) -------------
    #  N6=−(1/H)Ugh⊗S, N7=(1/H)UgH⊗S, N8=−(1/H)S⊗S; the residual's H prefactor
    #  cancels the 1/H, so we contract M_c = H·N_c directly.
    if prob.nl_pressure68
        DU  = alg_dx(Ux) + alg_dy(Uy)
        UgH = dHx*Ux + dHy*Uy
        Ugh = dhx*Ux + dhy*Uy
        S   = H*DU + UgH
        M6 = (-1.0)*alg_outer(Ugh, S)
        M7 = alg_outer(UgH, S)
        M8 = (-1.0)*alg_outer(S, S)
        NA = alg_dc3(prob.A3[6], M6) + alg_dc3(prob.A3[7], M7) + alg_dc3(prob.A3[8], M8)
        NK = alg_dc3(prob.K3[6], M6) + alg_dc3(prob.K3[7], M7) + alg_dc3(prob.K3[8], M8)
        r = r + ∫( (-1.0)*( dhx*(Wx ⋅ NA) + dHx*(Wx ⋅ NK)
                          + dhy*(Wy ⋅ NA) + dHy*(Wy ⋅ NK) ) ) * dO
    end

    return r
end

# ----------------------------------------------------------
#  Hand Jacobians (mirror the oracle's quasi-Newton choices:
#  H frozen in the u̇-terms; lin_pressure and nl_pressure68 omitted)
# ----------------------------------------------------------

"∂R/∂u̇ — effective mass operator (acceleration + R_P dispersion)."
function jacobian_u_t_algebraic(t::Real, u, dut, v, prob::AlgebraicLFEM, trian, dO)
    η = u[1]
    dηt, dUxt, dUyt = dut[1], dut[2], dut[3]
    q, Wx, Wy = v[1], v[2], v[3]
    lin  = prob.linearised
    d_cf = CellField(prob.d_func, trian)
    H = d_cf + η

    r = ∫( q*dηt ) * dO
    accx = alg_mul(prob.Mv, dUxt); accy = alg_mul(prob.Mv, dUyt)
    r = lin ? r + ∫( (Wx ⋅ accx) + (Wy ⋅ accy) ) * dO :
              r + ∫( H*(Wx ⋅ accx) + H*(Wy ⋅ accy) ) * dO

    DW   = alg_dx(Wx) + alg_dy(Wy)
    dDUt = alg_dx(dUxt) + alg_dy(dUyt)
    if lin
        r = r + ∫( (-1.0)*(d_cf*d_cf)*((alg_mul(prob.Bv, dDUt)) ⋅ DW) ) * dO
    else
        dhx = alg_dx(d_cf); dhy = alg_dy(d_cf)
        dHx = dhx + alg_dx(η); dHy = dhy + alg_dy(η)
        dUgHt = dHx*dUxt + dHy*dUyt
        if prob.P_full
            dL1 = (-1.0)*(dhx*dUxt + dhy*dUyt)
            dL2 = dUgHt
            dL3 = (-1.0)*(H*dDUt + dUgHt)
            sP  = alg_mul(prob.P[1], dL1) + alg_mul(prob.P[2], dL2) + alg_mul(prob.P[3], dL3)
            r   = r + ∫( (-1.0)*(H*H)*(sP ⋅ DW) ) * dO
        else
            r = r + ∫( (-1.0)*(H*H)*((alg_mul(prob.Bv, H*dDUt + dUgHt)) ⋅ DW) ) * dO
        end
    end
    return r
end

"∂R/∂u — continuity + gravity + sponge + (nonlinear Acc η-term) + full advection derivative."
function jacobian_u_algebraic(t::Real, u, du, v, prob::AlgebraicLFEM, trian, dO)
    η,  Ux,  Uy  = u[1], u[2], u[3]
    dη, dUx, dUy = du[1], du[2], du[3]
    q,  Wx,  Wy  = v[1], v[2], v[3]
    g = prob.g; lin = prob.linearised
    d_cf  = CellField(prob.d_func, trian)
    mu_cf = CellField(prob.mu_sponge, trian)
    H = d_cf + η

    ub  = alg_vec2(alg_dot(prob.Φ, Ux),  alg_dot(prob.Φ, Uy))
    dub = alg_vec2(alg_dot(prob.Φ, dUx), alg_dot(prob.Φ, dUy))

    # continuity
    r = ∫( (-1.0)*( (∇(q) ⋅ ub)*dη + H*(∇(q) ⋅ dub) ) ) * dO

    # gravity
    DW = alg_dx(Wx) + alg_dy(Wy)
    PhiDW = alg_dot(prob.Φ, DW)
    r = lin ? r + ∫( (-g)*dη*PhiDW ) * dO :
              r + ∫( (-g)*H*dη*PhiDW ) * dO

    # nonlinear-Acc η derivative (needs current u̇)
    if !lin
        ut = ∂t(u)
        r = r + ∫( dη*(Wx ⋅ alg_mul(prob.Mv, ut[2])) + dη*(Wy ⋅ alg_mul(prob.Mv, ut[3])) ) * dO
    end

    # sponge
    r = r + ∫( mu_cf*((Wx ⋅ alg_mul(prob.Mv, dUx)) + (Wy ⋅ alg_mul(prob.Mv, dUy))) ) * dO

    # full advection derivative
    if prob.advection
        dhx = alg_dx(d_cf); dhy = alg_dy(d_cf)
        dHx = dhx + alg_dx(η); dHy = dhy + alg_dy(η)
        DU  = alg_dx(Ux) + alg_dy(Uy)
        UgH = dHx*Ux + dHy*Uy
        S   = H*DU + UgH
        dDU = alg_dx(dUx) + alg_dy(dUy)
        dS  = dη*DU + H*dDU + (alg_dx(dη)*Ux + alg_dy(dη)*Uy) + (dHx*dUx + dHy*dUy)

        TMx  = alg_outer(Ux, alg_dx(Ux)) + alg_outer(Uy, alg_dy(Ux))
        TMy  = alg_outer(Ux, alg_dx(Uy)) + alg_outer(Uy, alg_dy(Uy))
        dTMx = alg_outer(dUx, alg_dx(Ux)) + alg_outer(Ux, alg_dx(dUx)) +
               alg_outer(dUy, alg_dy(Ux)) + alg_outer(Uy, alg_dy(dUx))
        dTMy = alg_outer(dUx, alg_dx(Uy)) + alg_outer(Ux, alg_dx(dUy)) +
               alg_outer(dUy, alg_dy(Uy)) + alg_outer(Uy, alg_dy(dUy))
        dTGx = alg_outer(dS, Ux) + alg_outer(S, dUx)
        dTGy = alg_outer(dS, Uy) + alg_outer(S, dUy)

        r = r + ∫( dη*((alg_dc3(prob.M3, TMx)) ⋅ Wx) + dη*((alg_dc3(prob.M3, TMy)) ⋅ Wy)
                 + H*((alg_dc3(prob.M3, dTMx)) ⋅ Wx) + H*((alg_dc3(prob.M3, dTMy)) ⋅ Wy)
                 + ((alg_dc3(prob.G3, dTGx)) ⋅ Wx) + ((alg_dc3(prob.G3, dTGy)) ⋅ Wy) ) * dO
    end

    return r
end

"TransientFEOperator with the hand Jacobians."
function build_ode_operator_algebraic(prob::AlgebraicLFEM, U, V, trian, dO)
    r  = (t, u, v)      -> residual_algebraic(t, u, v, prob, trian, dO)
    j  = (t, u, du, v)  -> jacobian_u_algebraic(t, u, du, v, prob, trian, dO)
    jt = (t, u, dut, v) -> jacobian_u_t_algebraic(t, u, dut, v, prob, trian, dO)
    return TransientFEOperator(r, j, jt, U, V)
end

"AD-Jacobian variant (experiment; expected to compile much faster than the
per-layer fused residual since the expression tree has only 3 fields)."
function build_ode_operator_algebraic_ad(prob::AlgebraicLFEM, U, V, trian, dO)
    r = (t, u, v) -> residual_algebraic(t, u, v, prob, trian, dO)
    return TransientFEOperator(r, U, V)
end

# ----------------------------------------------------------
#  Driver
# ----------------------------------------------------------

const ALG_DEFAULT_CBDY = Dict(
    1 => [0.0, 1.0],
    2 => [0.0, 0.728, 1.0],
    3 => [0.0, 0.726, 0.925, 1.0],
    4 => [0.0, 0.745, 0.923, 0.977, 1.0],
)

"""
    setup_and_run_lfem_algebraic(; kwargs...) → (diags, vert, prob)

Mirror of `setup_and_run_lfem` for the stacked algebraic residual.
Reuses the module's sponge/wavemaker/mesh/time-loop helpers.
"""
function setup_and_run_lfem_algebraic(;
    M            :: Int     = 2,
    p_vert       :: Int     = 1,
    c_bdy                   = nothing,
    domain                  = ((0.0, 60.0), (0.0, 10.0)),
    partition    :: Tuple   = (120, 20),
    fe_order     :: Int     = 2,
    d_val        :: Float64 = 3.5,
    g            :: Float64 = 9.81,
    T_wave       :: Float64 = 1.6,
    A_wave       :: Float64 = 0.001,
    x_wm         :: Float64 = 12.0,
    y_wm                    = nothing,
    sponge_wL    :: Float64 = 12.0,
    sponge_wR    :: Float64 = 12.0,
    sponge_wB    :: Float64 = 0.0,
    sponge_wT    :: Float64 = 0.0,
    mu_max       :: Float64 = 5.0,
    T_final      :: Float64 = 12.8,
    dt           :: Float64 = 0.02,
    theta        :: Float64 = 0.5,
    output_dir   :: String  = joinpath(@__DIR__, "output", "algebraic_out"),
    save_every   :: Int     = 0,
    gauges                  = [],
    y_wall_bc    :: Bool    = true,
    x_wall_bc    :: Bool    = false,
    linearised   :: Bool    = false,
    advection    :: Bool    = true,
    lin_pressure :: Bool    = false,
    P_full       :: Bool    = false,
    nl_pressure68:: Bool    = false,
    d_func                  = nothing,
    use_ad       :: Bool    = false,
    show_trace   :: Bool    = false,
)
    if isnothing(c_bdy)
        c_bdy = get(ALG_DEFAULT_CBDY, M, collect(LinRange(0.0, 1.0, M + 1)))
    end

    println("=== Vertical FE problem (algebraic LFE-M) ===")
    vert = assemble_vertical_tensors_lfem(M, p_vert, c_bdy)
    @printf("  Nσ=%d   ΣΦ=%.6f\n", vert.N_dof, sum(vert.Phi))

    println("\n=== 2D Horizontal FE problem (stacked [η,𝖴x,𝖴y]) ===")
    model, trian = build_horizontal_model_2D(domain, partition)
    dO = Measure(trian, 2*fe_order + 2)
    U, V = build_fe_spaces_algebraic(model, fe_order, vert.N_dof;
                                     y_wall_bc=y_wall_bc, x_wall_bc=x_wall_bc)
    @printf("  Fields: 3 (η + 2 stacked VectorValue{%d})  free DOFs: %d\n",
            vert.N_dof, num_free_dofs(U))

    omega  = 2.0*pi/T_wave
    k_wave = find_wavenumber(omega, d_val, g)
    @printf("  Wave: λ=%.2f m, kd=%.2f\n", 2pi/k_wave, k_wave*d_val)

    sponge = make_sponge_2D(domain, sponge_wL, sponge_wR, sponge_wB, sponge_wT, mu_max)
    wm = isnothing(y_wm) ? make_wavemaker_line(x_wm, A_wave, T_wave, k_wave) :
                           make_wavemaker_point(x_wm, Float64(y_wm), A_wave, T_wave)
    dfn = isnothing(d_func) ? (x -> d_val) : d_func

    prob = build_algebraic_problem(vert; g=g, d_func=dfn,
        linearised=linearised, advection=advection, lin_pressure=lin_pressure,
        P_full=P_full, nl_pressure68=nl_pressure68, mu_sponge=sponge, wm_src=wm)

    op = use_ad ? build_ode_operator_algebraic_ad(prob, U, V, trian, dO) :
                  build_ode_operator_algebraic(prob, U, V, trian, dO)
    solver = build_ode_solver(dt; solver_type=:theta, theta=theta, show_trace=show_trace)
    u0     = make_initial_conditions(U)

    println("\n=== Time loop (algebraic) ===")
    diags = run_time_loop(op, solver, u0, 0.0, T_final;
                          output_dir=output_dir, save_every=0,
                          trian=trian, N_dof=vert.N_dof,
                          print_dt=max(dt, T_final/50.0), gauges=gauges)
    return diags, vert, prob
end
