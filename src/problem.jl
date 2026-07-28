# ==============================================================
#  problem.jl — the LFEMProblem bundle + the loop-free residual + hand Jacobians
#
#  This file assembles the single scalar residual of §8 of the derivation — the
#  total virtual work ∫_Ω R·v that Gridap's MultiField solver drives to zero at
#  each stage. `LFEMProblem` gathers the precomputed vertical tensors (as Gridap
#  constants) and the physics flags; `global_residual` evaluates the weak form;
#  `jacobian_u`/`jacobian_u_t` provide the exact spatial and effective-mass
#  Jacobians so Newton needs no finite differencing or automatic differentiation.
#
#  Sign/form conventions (load-bearing — the assembly relies on them):
#    * gravity uses the integrated-by-parts energy form −∫(g/2)(H²−d²)(𝚽⋅DW);
#      subtracting the still-water baseline (H²−d²) makes the discrete rest state
#      exactly force-free, so an undisturbed surface at an open wall stays at rest;
#    * dispersion R_P: −∫ d²(𝗕⋅DUt)⋅DW (linearised) / −∫ H²(𝗕⋅∇·(Hu̇))⋅DW
#      (nonlinear) / the full P¹L¹+P²L²+P³L³ slope decomposition (P_full=true);
#      𝗕 ≤ 0 (the stored dispersion tensor) and the explicit (−1) factors give the
#      term its correct sign — this term IS the frequency dispersion of the model;
#    * the wavemaker enters continuity with a minus, −∫ q·S(x,t);
#    * the pressure slope packages sit on the momentum right-hand side (subtracted).
# ==============================================================

"""
    LFEMProblem

Coefficient bundle carried through the time loop and consumed by the residual.
It holds the vertical tensors as Gridap constants (built once by
`assemble_vertical_tensors`, reshaped in `build_problem`), the still-water depth
`d(x,y)`, the source/sponge/relaxation profiles, and the physics flags that
switch individual residual terms on or off. Third-order tensors use the index
order `[i,k,j]` = [test layer, u_k layer, u_j layer], so contraction over the
trailing two indices directly yields the mode-i momentum contribution.
"""
struct LFEMProblem{PV,MV,BV,PT,AT,KT,M3T,G3T,A3T,K3T,P3T}
    g            :: Float64
    h_bathy       :: Function          # still-water depth d(x,y)
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
    P3           :: P3T               # NTuple{8,ThirdOrderTensorValue}  NL leading 𝓟
    linearised   :: Bool              # linear regime: drop H-weights, use d²B dispersion
    advection    :: Bool              # nonlinear advection block
    lin_pressure :: Bool              # A/K linear slope-pressure package (needs a sloped bed)
    P_full       :: Bool              # keep all three slope components P¹L¹+P²L²+P³L³ in R_P
                                      #   (false = the P³L³ dispersion carrier alone)
    nl_pressure68:: Bool              # nonlinear pressure, native first-order set c∈{3,6,7,8}
                                      #   (𝓐/𝓚 slope halves + 𝓟 leading part; all paths)
    nl_pressure_full :: Bool          # + comps c∈{1,2,4,5}: 𝓐 half by exact IBP; 𝓚/𝓟 halves via
                                      #   per-step frozen L²-projections (finite-amplitude, O(A³))
    nlp_state    :: Base.RefValue{Any} # frozen (π𝖲, π𝖻) FEFunctions; nothing before the first step
    mu_sponge    :: Function
    wm_src       :: Function
    relax_bc     :: Bool              # generation/absorption relaxation zone (Dirichlet inflow)
    relax_mu     :: Function          # zone profile μ_g(x) (quadratic, max at the boundary)
    relax_tg     :: Any               # incident_fields NamedTuple (eta, ux, uy) or nothing
end

"""
    build_problem(vert; g, h_bathy, flags..., mu_sponge, wm_src) → LFEMProblem

Reshape the `assemble_vertical_tensors` NamedTuple into constant Gridap
tensors and bundle the runtime flags.
"""
function build_problem(vert;
        g            :: Float64  = g,
        h_bathy       :: Function = (x -> 3.5),
        linearised   :: Bool     = false,
        advection    :: Bool     = true,
        lin_pressure :: Bool     = false,
        P_full       :: Bool     = false,
        nl_pressure68:: Bool     = false,
        nl_pressure_full :: Bool = false,
        mu_sponge    :: Function = (x -> 0.0),
        wm_src       :: Function = ((x, t) -> 0.0),
        relax_bc     :: Bool     = false,
        relax_mu     :: Function = (x -> 0.0),
        relax_tg                 = nothing)
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
    relax_bc && relax_tg === nothing &&
        error("build_problem: relax_bc=true requires relax_tg (incident_fields NamedTuple)")
    return LFEMProblem(g, h_bathy, vert.N_dof, Φ, Mv, Bv, P, Av, Kv, M3, G3,
                         A3, K3, P3, linearised, advection, lin_pressure,
                         P_full, nl_pressure68, nl_pressure_full,
                         Ref{Any}(nothing), mu_sponge, wm_src,
                         relax_bc, relax_mu, relax_tg)
end

"""
    global_residual(t, u, v, prob, trian, dΩh)

Single scalar Gridap residual, stacked layout. `u` is a TransientCellField
(`∂t(u)` available); `u[1]=η`, `u[2]=𝖴x`, `u[3]=𝖴y`. No per-layer loops.
"""
function global_residual(t::Real, u, v, prob::LFEMProblem, trian, dΩh)
    ut = ∂t(u)
    η,  Ux,  Uy  = u[1], u[2], u[3]
    ηt, Uxt, Uyt = ut[1], ut[2], ut[3]
    q,  Wx,  Wy  = v[1], v[2], v[3]

    g   = prob.g
    lin = prob.linearised
    d_cf   = CellField(prob.h_bathy, trian)
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
    r = ∫( q*ηt - H*(∇(q) ⋅ ub) - q*src_cf ) * dΩh

    # ---- acceleration ----------------------------------------------------------
    accx = alg_mul(prob.Mv, Uxt); accy = alg_mul(prob.Mv, Uyt)
    r = lin ? r + ∫( (Wx ⋅ accx) + (Wy ⋅ accy) ) * dΩh :
              r + ∫( H*(Wx ⋅ accx) + H*(Wy ⋅ accy) ) * dΩh

    # ---- gravity (integrated-by-parts energy form; rest-state baseline removed) -
    PhiDW = alg_dot(prob.Φ, DW)
    r = lin ? r + ∫( (-g)*η*PhiDW ) * dΩh :
              r + ∫( (-0.5*g)*(H*H - d_cf*d_cf)*PhiDW ) * dΩh

    # ---- leading pressure R_P (dispersion — MANDATORY) -------------------------
    if lin
        r = r + ∫( (-1.0)*(d_cf*d_cf)*((alg_mul(prob.Bv, DUt)) ⋅ DW) ) * dΩh
    else
        UgHt = dHx*Uxt + dHy*Uyt                               # u̇ⱼ·∇H stacked
        if prob.P_full
            L1 = (-1.0)*(dhx*Uxt + dhy*Uyt)
            L2 = UgHt
            L3 = (-1.0)*(H*DUt + UgHt)
            sP = alg_mul(prob.P[1], L1) + alg_mul(prob.P[2], L2) + alg_mul(prob.P[3], L3)
            r  = r + ∫( (-1.0)*(H*H)*(sP ⋅ DW) ) * dΩh
        else
            r = r + ∫( (-1.0)*(H*H)*((alg_mul(prob.Bv, H*DUt + UgHt)) ⋅ DW) ) * dΩh
        end
    end

    # ---- sponge -----------------------------------------------------------------
    r = r + ∫( mu_cf*((Wx ⋅ alg_mul(prob.Mv, Ux)) + (Wy ⋅ alg_mul(prob.Mv, Uy))) ) * dΩh

    # ---- generation/absorption relaxation zone (Dirichlet inflow) ---------------
    #  Relax the state toward the incident wave in a zone adjacent to the
    #  generation boundary: absorbs outgoing deviations, enforces the incident
    #  field (classical relaxation-zone practice). Linear in u.
    if prob.relax_bc
        tg     = prob.relax_tg
        mug_cf = CellField(prob.relax_mu, trian)
        eta_i  = CellField(x -> tg.eta(x, t), trian)
        ux_i   = CellField(x -> tg.ux(x, t),  trian)
        uy_i   = CellField(x -> tg.uy(x, t),  trian)
        r = r + ∫( mug_cf*q*(η - eta_i)
                 + mug_cf*((Wx ⋅ alg_mul(prob.Mv, Ux - ux_i))
                         + (Wy ⋅ alg_mul(prob.Mv, Uy - uy_i))) ) * dΩh
    end

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
        r = r + ∫( H*(advx ⋅ Wx) + H*(advy ⋅ Wy) + (gvx ⋅ Wx) + (gvy ⋅ Wy) ) * dΩh
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
                            + dhy*(Wy ⋅ LA) + dHy*(Wy ⋅ LK) ) ) * dΩh
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
                                   Ux, Uy, Wx, Wy, DW, Ugh, UgH, S, DU, dΩh)
        if prob.nl_pressure_full
            r = r + nlp_gradh_contrib(prob, d_cf, η, H, dhx, dhy,
                                      Ux, Uy, Wx, Wy, Ugh, UgH, S, DU, dΩh)
            st = prob.nlp_state[]
            if st !== nothing
                N1, N2, N4, N5 = nlp_frozen_N(Ux, Uy, S, DU, st.piS, st.pib)
                r = r + nlp_gradH_frozen_contrib(prob, H, dHx, dHy, Wx, Wy,
                                                 N1, N2, N4, N5, dΩh)
                r = r + nlp_P_frozen_contrib(prob, H, DW, N1, N2, N4, N5, dΩh)
            end
        end
    end

    return r
end

# ----------------------------------------------------------
#  Hand Jacobians. Splitting ∂R/∂u̇ (effective mass) from ∂R/∂u (spatial) lets
#  the time integrator form its per-stage system J = ∂R/∂u + (1/aΔt)∂R/∂u̇
#  directly. The advection block is differentiated in full (quadratic Newton);
#  the slope-pressure packages are linearised with H frozen in the u̇-carrying
#  terms — a quasi-Newton choice that keeps the Jacobian sparse while retaining
#  fast, robust convergence.
# ----------------------------------------------------------

"∂R/∂u̇ — effective mass operator (acceleration + R_P dispersion)."
function jacobian_u_t(t::Real, u, dut, v, prob::LFEMProblem, trian, dΩh)
    η = u[1]
    dηt, dUxt, dUyt = dut[1], dut[2], dut[3]
    q, Wx, Wy = v[1], v[2], v[3]
    lin  = prob.linearised
    d_cf = CellField(prob.h_bathy, trian)
    H = d_cf + η

    r = ∫( q*dηt ) * dΩh
    accx = alg_mul(prob.Mv, dUxt); accy = alg_mul(prob.Mv, dUyt)
    r = lin ? r + ∫( (Wx ⋅ accx) + (Wy ⋅ accy) ) * dΩh :
              r + ∫( H*(Wx ⋅ accx) + H*(Wy ⋅ accy) ) * dΩh

    DW   = alg_dx(Wx) + alg_dy(Wy)
    dDUt = alg_dx(dUxt) + alg_dy(dUyt)
    if lin
        r = r + ∫( (-1.0)*(d_cf*d_cf)*((alg_mul(prob.Bv, dDUt)) ⋅ DW) ) * dΩh
    else
        dhx = alg_dx(d_cf); dhy = alg_dy(d_cf)
        dHx = dhx + alg_dx(η); dHy = dhy + alg_dy(η)
        dUgHt = dHx*dUxt + dHy*dUyt
        if prob.P_full
            dL1 = (-1.0)*(dhx*dUxt + dhy*dUyt)
            dL2 = dUgHt
            dL3 = (-1.0)*(H*dDUt + dUgHt)
            sP  = alg_mul(prob.P[1], dL1) + alg_mul(prob.P[2], dL2) + alg_mul(prob.P[3], dL3)
            r   = r + ∫( (-1.0)*(H*H)*(sP ⋅ DW) ) * dΩh
        else
            r = r + ∫( (-1.0)*(H*H)*((alg_mul(prob.Bv, H*dDUt + dUgHt)) ⋅ DW) ) * dΩh
        end
    end
    return r
end

"∂R/∂u — continuity + gravity + sponge + (nonlinear Acc η-term) + FULL advection derivative."
function jacobian_u(t::Real, u, du, v, prob::LFEMProblem, trian, dΩh)
    η,  Ux,  Uy  = u[1], u[2], u[3]
    dη, dUx, dUy = du[1], du[2], du[3]
    q,  Wx,  Wy  = v[1], v[2], v[3]
    g = prob.g; lin = prob.linearised
    d_cf  = CellField(prob.h_bathy, trian)
    mu_cf = CellField(prob.mu_sponge, trian)
    H = d_cf + η

    ub  = alg_vec2(alg_dot(prob.Φ, Ux),  alg_dot(prob.Φ, Uy))
    dub = alg_vec2(alg_dot(prob.Φ, dUx), alg_dot(prob.Φ, dUy))

    # continuity
    r = ∫( (-1.0)*( (∇(q) ⋅ ub)*dη + H*(∇(q) ⋅ dub) ) ) * dΩh

    # gravity
    DW = alg_dx(Wx) + alg_dy(Wy)
    PhiDW = alg_dot(prob.Φ, DW)
    r = lin ? r + ∫( (-g)*dη*PhiDW ) * dΩh :
              r + ∫( (-g)*H*dη*PhiDW ) * dΩh

    # nonlinear-Acc η derivative (needs current u̇)
    if !lin
        ut = ∂t(u)
        r = r + ∫( dη*(Wx ⋅ alg_mul(prob.Mv, ut[2])) + dη*(Wy ⋅ alg_mul(prob.Mv, ut[3])) ) * dΩh
    end

    # sponge
    r = r + ∫( mu_cf*((Wx ⋅ alg_mul(prob.Mv, dUx)) + (Wy ⋅ alg_mul(prob.Mv, dUy))) ) * dΩh

    # relaxation zone (linear in u — exact derivative)
    if prob.relax_bc
        mug_cf = CellField(prob.relax_mu, trian)
        r = r + ∫( mug_cf*q*dη
                 + mug_cf*((Wx ⋅ alg_mul(prob.Mv, dUx))
                         + (Wy ⋅ alg_mul(prob.Mv, dUy))) ) * dΩh
    end

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
                 + ((alg_dc3(prob.G3, dTGx)) ⋅ Wx) + ((alg_dc3(prob.G3, dTGy)) ⋅ Wy) ) * dΩh
    end

    return r
end

"TransientFEOperator with the hand Jacobians (default; fast)."
function build_ode_operator(prob::LFEMProblem, U, V, trian, dΩh)
    r  = (t, u, v)      -> global_residual(t, u, v, prob, trian, dΩh)
    j  = (t, u, du, v)  -> jacobian_u(t, u, du, v, prob, trian, dΩh)
    jt = (t, u, dut, v) -> jacobian_u_t(t, u, dut, v, prob, trian, dΩh)
    return TransientFEOperator(r, j, jt, U, V)
end

"Operator variant that lets Gridap build the Jacobians by automatic
differentiation of `global_residual` instead of using the hand Jacobians —
useful for cross-checking. The three-field stacked residual keeps the AD
expression tree small enough to be practical."
function build_ode_operator_ad(prob::LFEMProblem, U, V, trian, dΩh)
    r = (t, u, v) -> global_residual(t, u, v, prob, trian, dΩh)
    return TransientFEOperator(r, U, V)
end
