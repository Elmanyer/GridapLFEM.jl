# ==============================================================
#  problem.jl — LFEMProblem problem + loop-free residual + hand Jacobians
#
#  Implements the corrected main.tex §8 global residual (INCLUDING the
#  leading-pressure/dispersion term R_P) in the stacked layout. Terms match
#  the oracle `residual_lfem` (../../LFE-M_2D_solver/src/problem_lfem2D.jl)
#  flag by flag; validated to machine precision (test_equivalence.jl).
#
#  Sign/form conventions (load-bearing):
#    * gravity: oracle IBP energy form  −∫(g/2)(H²−d²)(𝚽⋅DW)  — the (H²−d²)
#      baseline subtraction keeps the discrete rest state force-free at open
#      walls (old-solver finding #6);
#    * dispersion R_P: −∫ d²(𝗕⋅DUt)⋅DW (linearised) / −∫ H²(𝗕⋅∇·(Hu̇))⋅DW
#      (nonlinear, P_full=false = oracle) / full P¹L¹+P²L²+P³L³ (P_full=true);
#      𝗕 = B_stored ≤ 0 and the (−1) factors are load-bearing;
#    * wavemaker enters continuity with a MINUS: −∫ q·S(x,t);
#    * pressure slope packages are subtracted (they sit on the momentum RHS).
# ==============================================================

"""
    LFEMProblem

Coefficient bundle for the stacked algebraic residual. All vertical tensors
are constant `TensorValue`/`ThirdOrderTensorValue` (index order `[i,k,j]` =
[test layer, u_k layer, u_j layer] — matches `assemble_vertical_tensors`
storage directly, no remap).
"""
struct LFEMProblem{PV,MV,BV,PT,AT,KT,M3T,G3T,A3T,K3T,P3T}
    g            :: Float64
    d_func       :: Function          # still-water depth d(x,y)
    Nσ           :: Int
    Φ            :: PV                # VectorValue{Nσ}   depth weights
    Mv           :: MV                # TensorValue       vertical mass
    Bv           :: BV                # TensorValue       dispersion B ≤ 0 (= −P[:,:,3])
    P            :: PT                # NTuple{3,TensorValue}  leading pressure P^V
    Av           :: AT                # NTuple{3,TensorValue}  linear pressure (∇h)
    Kv           :: KT                # NTuple{3,TensorValue}  linear pressure (∇H)
    M3           :: M3T               # ThirdOrderTensorValue  advection 𝓜
    G3           :: G3T               # ThirdOrderTensorValue  advection 𝓖
    A3           :: A3T               # NTuple{8,ThirdOrderTensorValue}  NL pressure 𝓐
    K3           :: K3T               # NTuple{8,ThirdOrderTensorValue}  NL pressure 𝓚
    P3           :: P3T               # NTuple{8,ThirdOrderTensorValue}  NL leading 𝓟 (reserved)
    linearised   :: Bool              # drop H-weights; d²B dispersion (validated baseline)
    advection    :: Bool              # nonlinear advection block
    lin_pressure :: Bool              # A/K slope pressure package
    P_full       :: Bool              # full P¹L¹+P²L²+P³L³ in R_P (false = oracle's P³L³)
    nl_pressure68:: Bool              # nonlinear pressure, NATIVE first-order set c∈{3,6,7,8}
                                      #   (𝓐/𝓚 slope halves + 𝓟 leading part; all paths)
    nl_pressure_full :: Bool          # + comps c∈{1,2,4,5}: 𝓐 half exact-IBP; 𝓚/𝓟 halves via
                                      #   frozen L²-projections (sequential loop; O(A³))
    nlp_state    :: Base.RefValue{Any} # frozen (π𝖲, π𝖻) FEFunctions; nothing before step 1
    mu_sponge    :: Function
    wm_src       :: Function
end

"""
    build_problem(vert; g, d_func, flags..., mu_sponge, wm_src) → LFEMProblem

Reshape the `assemble_vertical_tensors` NamedTuple into constant Gridap
tensors and bundle the runtime flags.
"""
function build_problem(vert;
        g            :: Float64  = 9.81,
        d_func       :: Function = (x -> 3.5),
        linearised   :: Bool     = false,
        advection    :: Bool     = true,
        lin_pressure :: Bool     = false,
        P_full       :: Bool     = false,
        nl_pressure68:: Bool     = false,
        nl_pressure_full :: Bool = false,
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
    P3 = ntuple(c -> alg_to_tensor3(vert.Pcal[:, :, :, c]), 8)
    return LFEMProblem(g, d_func, vert.N_dof, Φ, Mv, Bv, P, Av, Kv, M3, G3,
                         A3, K3, P3, linearised, advection, lin_pressure,
                         P_full, nl_pressure68, nl_pressure_full,
                         Ref{Any}(nothing), mu_sponge, wm_src)
end

"""
    global_residual(t, u, v, prob, trian, dO)

Single scalar Gridap residual, stacked layout. `u` is a TransientCellField
(`∂t(u)` available); `u[1]=η`, `u[2]=𝖴x`, `u[3]=𝖴y`. No per-layer loops.
"""
function global_residual(t::Real, u, v, prob::LFEMProblem, trian, dO)
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

    # derived scalar / layer-vector fields
    dhx = alg_dx(d_cf);        dhy = alg_dy(d_cf)              # bed slope ∇h
    dHx = dhx + alg_dx(η);     dHy = dhy + alg_dy(η)           # ∇H = ∇d + ∇η
    DW  = alg_dx(Wx) + alg_dy(Wy)                              # test-divergence vector
    DUt = alg_dx(Uxt) + alg_dy(Uyt)                            # layer div(u̇)
    ub  = alg_vec2(alg_dot(prob.Φ, Ux), alg_dot(prob.Φ, Uy))   # depth-averaged velocity

    # ---- mass continuity + wavemaker source ----------------------------------
    r = ∫( q*ηt - H*(∇(q) ⋅ ub) - q*src_cf ) * dO

    # ---- acceleration ----------------------------------------------------------
    accx = alg_mul(prob.Mv, Uxt); accy = alg_mul(prob.Mv, Uyt)
    r = lin ? r + ∫( (Wx ⋅ accx) + (Wy ⋅ accy) ) * dO :
              r + ∫( H*(Wx ⋅ accx) + H*(Wy ⋅ accy) ) * dO

    # ---- gravity (oracle IBP form; hydrostatic baseline subtracted) ------------
    PhiDW = alg_dot(prob.Φ, DW)
    r = lin ? r + ∫( (-g)*η*PhiDW ) * dO :
              r + ∫( (-0.5*g)*(H*H - d_cf*d_cf)*PhiDW ) * dO

    # ---- leading pressure R_P (dispersion — MANDATORY) -------------------------
    if lin
        r = r + ∫( (-1.0)*(d_cf*d_cf)*((alg_mul(prob.Bv, DUt)) ⋅ DW) ) * dO
    else
        UgHt = dHx*Uxt + dHy*Uyt                               # u̇ⱼ·∇H stacked
        if prob.P_full
            L1 = (-1.0)*(dhx*Uxt + dhy*Uyt)
            L2 = UgHt
            L3 = (-1.0)*(H*DUt + UgHt)
            sP = alg_mul(prob.P[1], L1) + alg_mul(prob.P[2], L2) + alg_mul(prob.P[3], L3)
            r  = r + ∫( (-1.0)*(H*H)*(sP ⋅ DW) ) * dO
        else
            r = r + ∫( (-1.0)*(H*H)*((alg_mul(prob.Bv, H*DUt + UgHt)) ⋅ DW) ) * dO
        end
    end

    # ---- sponge -----------------------------------------------------------------
    r = r + ∫( mu_cf*((Wx ⋅ alg_mul(prob.Mv, Ux)) + (Wy ⋅ alg_mul(prob.Mv, Uy))) ) * dO

    # ---- nonlinear advection (𝓜/𝓖 block) ---------------------------------------
    if prob.advection
        DU  = alg_dx(Ux) + alg_dy(Uy)
        UgH = dHx*Ux + dHy*Uy
        S   = H*DU + UgH                                       # ∇·(H u_j) stacked
        TMx = alg_outer(Ux, alg_dx(Ux)) + alg_outer(Uy, alg_dy(Ux))
        TMy = alg_outer(Ux, alg_dx(Uy)) + alg_outer(Uy, alg_dy(Uy))
        advx = alg_dc3(prob.M3, TMx);  advy = alg_dc3(prob.M3, TMy)
        gvx  = alg_dc3(prob.G3, alg_outer(S, Ux))
        gvy  = alg_dc3(prob.G3, alg_outer(S, Uy))
        r = r + ∫( H*(advx ⋅ Wx) + H*(advy ⋅ Wy) + (gvx ⋅ Wx) + (gvy ⋅ Wy) ) * dO
    end

    # ---- linear non-hydrostatic pressure (A/K slope package) --------------------
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

    # ---- nonlinear pressure (nlpressure.jl) ---------------------------------
    #  nl_pressure68:   native first-order set c∈{3,6,7,8}, all three blocks
    #                   (𝓐/𝓚 slope halves + 𝓟 leading part).
    #  nl_pressure_full: + c∈{1,2,4,5}: 𝓐 half via EXACT IBP onto the test;
    #                   𝓚/𝓟 halves via the frozen projections π𝖲,π𝖻 (previous
    #                   step; zero before the first step — exact from rest).
    if prob.nl_pressure68 || prob.nl_pressure_full
        DU  = alg_dx(Ux) + alg_dy(Uy)
        UgH = dHx*Ux + dHy*Uy
        Ugh = dhx*Ux + dhy*Uy
        S   = H*DU + UgH
        r = r + nlp_native_contrib(prob, d_cf, η, H, dhx, dhy, dHx, dHy,
                                   Ux, Uy, Wx, Wy, DW, Ugh, UgH, S, DU, dO)
        if prob.nl_pressure_full
            r = r + nlp_gradh_contrib(prob, d_cf, η, H, dhx, dhy,
                                      Ux, Uy, Wx, Wy, Ugh, UgH, S, DU, dO)
            st = prob.nlp_state[]
            if st !== nothing
                N1, N2, N4, N5 = nlp_frozen_N(Ux, Uy, S, DU, st.piS, st.pib)
                r = r + nlp_gradH_frozen_contrib(prob, H, dHx, dHy, Wx, Wy,
                                                 N1, N2, N4, N5, dO)
                r = r + nlp_P_frozen_contrib(prob, H, DW, N1, N2, N4, N5, dO)
            end
        end
    end

    return r
end

# ----------------------------------------------------------
#  Hand Jacobians (oracle-matching quasi-Newton choices: H frozen in the
#  u̇ terms; lin_pressure and nl_pressure68 omitted; advection FULL Newton)
# ----------------------------------------------------------

"∂R/∂u̇ — effective mass operator (acceleration + R_P dispersion)."
function jacobian_u_t(t::Real, u, dut, v, prob::LFEMProblem, trian, dO)
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

"∂R/∂u — continuity + gravity + sponge + (nonlinear Acc η-term) + FULL advection derivative."
function jacobian_u(t::Real, u, du, v, prob::LFEMProblem, trian, dO)
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

"TransientFEOperator with the hand Jacobians (default; fast)."
function build_ode_operator(prob::LFEMProblem, U, V, trian, dO)
    r  = (t, u, v)      -> global_residual(t, u, v, prob, trian, dO)
    j  = (t, u, du, v)  -> jacobian_u(t, u, du, v, prob, trian, dO)
    jt = (t, u, dut, v) -> jacobian_u_t(t, u, dut, v, prob, trian, dO)
    return TransientFEOperator(r, j, jt, U, V)
end

"AD-Jacobian variant (experimental; the 3-field stacked residual has a much
smaller expression tree than the old per-layer fused residual)."
function build_ode_operator_ad(prob::LFEMProblem, U, V, trian, dO)
    r = (t, u, v) -> global_residual(t, u, v, prob, trian, dO)
    return TransientFEOperator(r, U, V)
end
