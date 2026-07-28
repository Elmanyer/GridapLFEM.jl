# ==============================================================
#  test_nlpressure.jl — FULL nonlinear pressure validation (gates G1–G3)
#
#  G1  MACHINE-PRECISION IBP identity (∇h half, comps 1,2,4,5): on a state of
#      polynomials of degree ≤ 2 (all second derivatives globally constant ⇒
#      hand-derivable exactly) with u·n = 0 on the whole boundary (boundary
#      terms vanish identically) and a quadratic bed, the exact-IBP form must
#      equal the direct form built from ANALYTIC second derivatives.
#      Deliberately asymmetric layer coefficients to catch k/j slot swaps.
#  G2  Structural: ∇h half ≡ 0 on a flat bed; continuity (q) rows untouched by
#      every block; amplitude scaling of each block at small A
#      (∇h-IBP ratio→4, ∇H-frozen ratio→8, 𝓟-frozen ratio→4).
#  G3  Dynamics: short run over a tanh submerged bar with ALL pressure flags
#      on — bounded, no NaN.
#
#  RUN:  julia --project=. GridapLFEM.jl/test/test_nlpressure.jl
# ==============================================================

if !isdefined(Main, :GridapLFEM)
    include(joinpath(@__DIR__, "..", "src", "GridapLFEM.jl"))
end
using .GridapLFEM
using Gridap
using Gridap.ODEs
using LinearAlgebra, Printf

println("=" ^ 60)
println("  test_nlpressure.jl — full nonlinear pressure (G1–G3)")
println("=" ^ 60)

n_pass = 0; n_fail = 0
function check(name, cond)
    global n_pass, n_fail
    if cond; println("  PASS  $name"); n_pass += 1
    else;    println("  FAIL  $name"); n_fail += 1; end
end

# ---- common setup -------------------------------------------------------------
vert = assemble_vertical_tensors(2, 1, [0.0, 0.728, 1.0])
Nσ   = vert.N_dof                      # = 3
Lx, Ly = 4.0, 2.0
model, trian = build_horizontal_model(((0.0,Lx),(0.0,Ly)), (8,4))
dO = Measure(trian, 14)                # integrates every polynomial term exactly
U, V = build_fe_spaces(model, 2, Nσ; y_wall_bc=:open)
nq = num_free_dofs(V[1])               # continuity-row count

# polynomial state (all degree ≤ 2; u·n = 0 on the whole boundary)
ax = [0.030, -0.011, 0.021];  ay = [0.017, 0.026, -0.013]     # asymmetric layers
e1, e2, e3, e4, e5 = 0.004, -0.003, 0.002, 0.006, -0.005
d0, b1, b2 = 3.5, -0.012, 0.008
ujx(j) = x -> ax[j]*x[1]*(Lx - x[1])
ujy(j) = x -> ay[j]*x[2]*(Ly - x[2])
eta_f  = x -> e1*x[1]^2 + e2*x[1]*x[2] + e3*x[2]^2 + e4*x[1] + e5*x[2]
d_fun  = x -> d0 + b1*x[1]^2 + b2*x[1]*x[2]
stackf(fs) = x -> VectorValue(ntuple(j -> fs[j](x), Nσ)...)

uh = interpolate_everywhere([eta_f, stackf([ujx(j) for j in 1:Nσ]),
                                    stackf([ujy(j) for j in 1:Nσ])], U)

"assemble a test-linear contribution (fields prebuilt from uh) into a vector"
function assemble_block(blockfun, uh, dfun)
    η = uh[1]; Ux = uh[2]; Uy = uh[3]
    d_cf = CellField(dfun, trian)
    H   = d_cf + η
    dhx = alg_dx(d_cf); dhy = alg_dy(d_cf)
    dHx = dhx + alg_dx(η); dHy = dhy + alg_dy(η)
    DU  = alg_dx(Ux) + alg_dy(Uy)
    af  = dhx*Ux + dhy*Uy
    bf  = dHx*Ux + dHy*Uy
    S   = H*DU + bf
    return assemble_vector(v -> blockfun(d_cf,η,Ux,Uy,H,dhx,dhy,dHx,dHy,DU,af,bf,S,
                                         v[1], v[2], v[3],
                                         alg_dx(v[2]) + alg_dy(v[3])), V)
end

# ==============================================================
println("\n-- G1: exact-IBP identity (∇h half, comps 1,2,4,5) --")
# ==============================================================

# hand-derived exact derivatives of the polynomial state
Hx(x)  = 2*(e1+b1)*x[1] + (e2+b2)*x[2] + e4
Hy(x)  = (e2+b2)*x[1] + 2*e3*x[2] + e5
Hxx    = 2*(e1+b1);  Hxy = (e2+b2);  Hyy = 2*e3
Hval(x)  = d_fun(x) + eta_f(x)
DUj(x,j) = ax[j]*(Lx - 2*x[1]) + ay[j]*(Ly - 2*x[2])
bj(x,j)  = Hx(x)*ujx(j)(x) + Hy(x)*ujy(j)(x)
Sj(x,j)  = Hval(x)*DUj(x,j) + bj(x,j)
dxbj(x,j) = Hxx*ujx(j)(x) + Hx(x)*ax[j]*(Lx-2*x[1]) + Hxy*ujy(j)(x)
dybj(x,j) = Hxy*ujx(j)(x) + Hyy*ujy(j)(x) + Hy(x)*ay[j]*(Ly-2*x[2])
dxSj(x,j) = Hx(x)*DUj(x,j) - 2*ax[j]*Hval(x) + dxbj(x,j)
dySj(x,j) = Hy(x)*DUj(x,j) - 2*ay[j]*Hval(x) + dybj(x,j)

vecf(f) = x -> VectorValue(ntuple(j -> f(x,j), Nσ)...)
Ux_an  = CellField(stackf([ujx(j) for j in 1:Nσ]), trian)
Uy_an  = CellField(stackf([ujy(j) for j in 1:Nσ]), trian)
S_an   = CellField(vecf(Sj),  trian)
DU_an  = CellField(vecf(DUj), trian)
dxS_an = CellField(vecf(dxSj), trian)
dyS_an = CellField(vecf(dySj), trian)
dxb_an = CellField(vecf(dxbj), trian)
dyb_an = CellField(vecf(dybj), trian)
H_an   = CellField(Hval, trian)
dhx_an = CellField(x -> 2*b1*x[1] + b2*x[2], trian)
dhy_an = CellField(x -> b2*x[1], trian)

vert_prob = build_problem(vert; g=9.81, h_bathy=d_fun)

# direct form: −∫ H ∂_αh (W_α ⋅ (𝓐3[c] ⊡ N_c^exact)),  c ∈ {1,2,4,5}
N1_an = (-1.0)*(alg_outer(dxS_an, Ux_an) + alg_outer(dyS_an, Uy_an))
N2_an = (-1.0)*N1_an + alg_outer(S_an, DU_an)
N4_an = alg_outer(Ux_an, dxb_an) + alg_outer(Uy_an, dyb_an)
N5_an = (-1.0)*(alg_outer(Ux_an, dxS_an) + alg_outer(Uy_an, dyS_an))
NA_an = alg_dc3(vert_prob.A3[1], N1_an) + alg_dc3(vert_prob.A3[2], N2_an) +
        alg_dc3(vert_prob.A3[4], N4_an) + alg_dc3(vert_prob.A3[5], N5_an)
r_direct = assemble_vector(v -> ∫( (-1.0)*H_an*( dhx_an*(v[2] ⋅ NA_an)
                                               + dhy_an*(v[3] ⋅ NA_an) ) )*dO, V)

# IBP form (the implemented residual block) on the interpolated FE state
r_ibp = assemble_block((d_cf,η,Ux,Uy,H,dhx,dhy,dHx,dHy,DU,af,bf,S,q,Wx,Wy,DW) ->
            nlp_gradh_contrib(vert_prob, d_cf, η, H, dhx, dhy, Ux, Uy, Wx, Wy, af, bf, S, DU, dO),
        uh, d_fun)

rel = norm(r_direct - r_ibp) / norm(r_direct)
@printf("  ‖direct − IBP‖/‖direct‖ = %.3e   (‖direct‖ = %.3e)\n", rel, norm(r_direct))
check("G1: exact-IBP identity at machine precision (rel < 1e-10)", rel < 1e-10)

# ==============================================================
println("\n-- G2: structural checks --")
# ==============================================================

d_flat = x -> 3.5

# ∇h half vanishes identically on a flat bed
r_flat = assemble_block((d_cf,η,Ux,Uy,H,dhx,dhy,dHx,dHy,DU,af,bf,S,q,Wx,Wy,DW) ->
            nlp_gradh_contrib(vert_prob, d_cf, η, H, dhx, dhy, Ux, Uy, Wx, Wy, af, bf, S, DU, dO),
        uh, d_flat)
check("G2: ∇h-IBP block ≡ 0 on flat bed", norm(r_flat) < 1e-13)

# continuity rows untouched by every block
r_nat = assemble_block((d_cf,η,Ux,Uy,H,dhx,dhy,dHx,dHy,DU,af,bf,S,q,Wx,Wy,DW) ->
            nlp_native_contrib(vert_prob, d_cf, η, H, dhx, dhy, dHx, dHy, Ux, Uy, Wx, Wy, DW,
                               af, bf, S, DU, dO),
        uh, d_fun)
check("G2: continuity rows zero (native block)",  norm(r_nat[1:nq]) < 1e-14)
check("G2: continuity rows zero (∇h-IBP block)",  norm(r_ibp[1:nq]) < 1e-14)

# amplitude scaling at small A (blocks isolated; frozen projections from the SAME state)
function scaled_state(s)
    interpolate_everywhere([x -> s*eta_f(x),
                            stackf([x -> s*ujx(j)(x) for j in 1:Nσ]),
                            stackf([x -> s*ujy(j)(x) for j in 1:Nσ])], U)
end
ctx = build_nlp_ctx(model, 2, Nσ, trian, dO)

function block_norms(s, dfun)
    uh_s = scaled_state(s)
    update_nlp_state!(vert_prob, ctx, uh_s)
    st = vert_prob.nlp_state[]
    r_h = assemble_block((d_cf,η,Ux,Uy,H,dhx,dhy,dHx,dHy,DU,af,bf,S,q,Wx,Wy,DW) ->
            nlp_gradh_contrib(vert_prob, d_cf, η, H, dhx, dhy, Ux, Uy, Wx, Wy, af, bf, S, DU, dO),
          uh_s, dfun)
    r_K = assemble_block((d_cf,η,Ux,Uy,H,dhx,dhy,dHx,dHy,DU,af,bf,S,q,Wx,Wy,DW) ->
            begin
                N1,N2,N4,N5 = nlp_frozen_N(Ux, Uy, S, DU, st.piS, st.pib)
                nlp_gradH_frozen_contrib(vert_prob, H, dHx, dHy, Wx, Wy, N1,N2,N4,N5, dO)
            end, uh_s, dfun)
    r_P = assemble_block((d_cf,η,Ux,Uy,H,dhx,dhy,dHx,dHy,DU,af,bf,S,q,Wx,Wy,DW) ->
            begin
                N1,N2,N4,N5 = nlp_frozen_N(Ux, Uy, S, DU, st.piS, st.pib)
                nlp_P_frozen_contrib(vert_prob, H, DW, N1,N2,N4,N5, dO)
            end, uh_s, dfun)
    return norm(r_h), norm(r_K), norm(r_P)
end

s1 = 1e-3
h1, K1f, P1f = block_norms(s1,  d_fun)     # sloped bed for the ∇h block
h2, K2f, P2f = block_norms(2s1, d_fun)
_,  K1, P1   = block_norms(s1,  d_flat)    # flat bed for the ∇H / 𝓟 blocks
_,  K2, P2   = block_norms(2s1, d_flat)
@printf("  ratios: ∇h-IBP %.4f (→4)   ∇H-frozen %.4f (→8)   𝓟-frozen %.4f (→4)\n",
        h2/h1, K2/K1, P2/P1)
check("G2: ∇h-IBP block scales ~A² (ratio 4 ± 0.05)",   abs(h2/h1 - 4) < 0.05)
check("G2: ∇H-frozen block scales ~A³ (ratio 8 ± 0.1)", abs(K2/K1 - 8) < 0.1)
check("G2: 𝓟-frozen block scales ~A² (ratio 4 ± 0.05)", abs(P2/P1 - 4) < 0.05)

# ==============================================================
println("\n-- G3: dynamics over a submerged bar, ALL pressure flags on --")
# ==============================================================

bar(x) = 3.5 - 0.5*1.5*(tanh((x[1]-20.0)/4.0) - tanh((x[1]-32.0)/4.0))
diags, _, _ = setup_and_run(
    M=2, h_val=3.5, T_wave=2.0, A_wave=0.001,
    domain=((0.0,60.0),(0.0,2.0)), partition=(60,2), p_horizontal=2,
    x_wm=8.0, sponge_wL=8.0, sponge_wR=8.0, mu_max=30.0,
    T_final=2.0, dt=0.05, h_bathy=bar,
    linearised=false, advection=true,
    lin_pressure=true, P_full=true, nl_pressure68=true, nl_pressure_full=true,
    gauges=[(26.0,1.0)], save_every=0)
emax = maximum(d.eta_max for d in diags)
@printf("  bar run: steps=%d  max η=%.5f m\n", length(diags), emax)
check("G3: bounded over the bar (max η < 20A)", emax < 0.02)
check("G3: no NaN", all(!isnan(d.eta_max) for d in diags))

println()
println("=" ^ 60)
@printf("  Results: %d PASS,  %d FAIL\n", n_pass, n_fail)
println("=" ^ 60)
n_fail > 0 ? error("test_nlpressure: $n_fail failed!") :
             println("  Full nonlinear pressure validated.")
