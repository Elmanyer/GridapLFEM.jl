# ==============================================================
#  test_algebraic_lfem2D.jl — Validation of the loop-free algebraic residual
#
#  Part 1  Primitives: constant-tensor constructors (index order),
#          ∂x/∂y orientation on VectorValue{Nσ} fields, outer product,
#          native double_contraction semantics.
#  Part 2  THE acceptance test: virtual-work equivalence of
#          residual_algebraic vs the oracle residual_lfem on identical
#          interpolated states/test functions (no DOF mapping), for
#          three flag configurations (linear core / nonlinear+advection /
#          sloped bed + linear pressure).
#
#  Run from the repo root:
#    julia --project=. GridapLFEM.jl/test_algebraic_lfem2D.jl     (or via julia-mcp)
# ==============================================================

if !isdefined(Main, :LFEModel2D)
    include(joinpath(@__DIR__, "..", "LFE-M_2D_solver", "src", "LFEModel2D.jl"))
end
using .LFEModel2D
using Gridap
using Gridap.ODEs
using Gridap.TensorValues
using LinearAlgebra
using Printf
using Test

include(joinpath(@__DIR__, "algebraic_lfem2D.jl"))

npass = 0; nfail = 0
function check(name, ok)
    global npass, nfail
    if ok; npass += 1; @printf("  PASS  %s\n", name)
    else;  nfail += 1; @printf("  FAIL  %s\n", name)
    end
    ok
end

# ==============================================================
println("=== Part 1: primitives ===")
# ==============================================================

# 1a. constant-tensor constructors preserve index order
let N = 3
    Mtest = reshape(collect(1.0:9.0), N, N)
    T2 = alg_to_tensor2(Mtest)
    check("to_tensor2 index order", all(T2[i,j] == Mtest[i,j] for i in 1:N, j in 1:N))

    Ttest = reshape(collect(1.0:27.0), N, N, N)
    T3 = alg_to_tensor3(Ttest)
    check("to_tensor3 index order", all(T3[i,j,k] == Ttest[i,j,k] for i in 1:N, j in 1:N, k in 1:N))

    # native double_contraction contracts TRAILING two indices
    Stest = reshape(collect(2.0:2.0:18.0), N, N)
    S2 = alg_to_tensor2(Stest)
    dc = Gridap.TensorValues.double_contraction(T3, S2)
    ref = [sum(Ttest[i,k,j]*Stest[k,j] for k in 1:N, j in 1:N) for i in 1:N]
    check("double_contraction trailing-2 semantics",
          all(abs(dc[i]-ref[i]) < 1e-12 for i in 1:N))

    # vector ⋅ third-order contracts FIRST index
    w = VectorValue(1.0, 2.0, 3.0)
    wT = w ⋅ T3
    refT = [sum(w[i]*Ttest[i,k,j] for i in 1:N) for k in 1:N, j in 1:N]
    check("W ⋅ ThirdOrder first-index semantics",
          all(abs(wT[k,j]-refT[k,j]) < 1e-12 for k in 1:N, j in 1:N))
end

# 1b. ∂x/∂y orientation + outer product on FE fields
let
    model = CartesianDiscreteModel((0.0,1.0,0.0,1.0), (4,4))
    trian = Triangulation(model)
    reffe = ReferenceFE(lagrangian, VectorValue{3,Float64}, 2)
    Vs = FESpace(model, reffe; conformity=:H1)
    f = interpolate_everywhere(x -> VectorValue(x[1], 2*x[2], x[1]+x[2]), Vs)
    pt = VectorValue(0.3, 0.6)
    dxf = alg_dx(f)(pt); dyf = alg_dy(f)(pt)
    check("∂x orientation (x,2y,x+y) → (1,0,1)", norm(dxf - VectorValue(1.0,0.0,1.0)) < 1e-12)
    check("∂y orientation (x,2y,x+y) → (0,2,1)", norm(dyf - VectorValue(0.0,2.0,1.0)) < 1e-12)

    g2 = interpolate_everywhere(x -> VectorValue(1.0+x[1], x[2], 2.0-x[1]), Vs)
    ab = alg_outer(f, g2)(pt)
    fa = f(pt); gb = g2(pt)
    check("outer (a⊗b)[k,j]=a[k]b[j]",
          all(abs(ab[k,j] - fa[k]*gb[j]) < 1e-12 for k in 1:3, j in 1:3))
end

# ==============================================================
println("\n=== Part 2: equivalence vs oracle residual_lfem ===")
# ==============================================================

M_v, p_v = 2, 1
c_bdy = [0.0, 0.728, 1.0]
vert  = assemble_vertical_tensors_lfem(M_v, p_v, c_bdy)
Nσ    = vert.N_dof
g_phys = 9.81

domain    = ((0.0, 10.0), (0.0, 5.0))
partition = (6, 4)
fe_order  = 2
model, trian = build_horizontal_model_2D(domain, partition)
dO = Measure(trian, 2*fe_order + 2)

# no Dirichlet constraints → every DOF free, interpolation states identical
U_lay, V_lay = build_fe_spaces_2D(model, fe_order, Nσ; y_wall_bc=false)
U_alg, V_alg = build_fe_spaces_algebraic(model, fe_order, Nσ; y_wall_bc=false)

# --- analytic states (smooth, nonzero at boundaries) --------------------------
eta_f  = x -> 0.02*cos(0.4*x[1])*cos(0.5*x[2]) + 0.005*x[1]/10.0
ujx_f  = [x -> 0.03*cos(0.3*x[1] + 0.1*j)*sin(0.4*x[2] + 0.2*j) + 0.004*j        for j in 1:Nσ]
ujy_f  = [x -> 0.02*sin(0.25*x[1] - 0.15*j)*cos(0.35*x[2]) + 0.003*x[2]/5.0*j    for j in 1:Nσ]
etat_f = x -> 0.01*sin(0.3*x[1])*cos(0.6*x[2])
ujxt_f = [x -> 0.02*sin(0.45*x[1] + 0.3*j)*cos(0.2*x[2]) + 0.002*j               for j in 1:Nσ]
ujyt_f = [x -> 0.015*cos(0.5*x[1])*sin(0.3*x[2] + 0.1*j) + 0.001*j*x[1]/10.0     for j in 1:Nσ]

# --- analytic test functions ---------------------------------------------------
q_fs   = [x -> cos(0.7*x[1])*cos(0.3*x[2]),
          x -> 0.5 + 0.1*x[1] - 0.04*x[2]^2,
          x -> sin(0.5*x[1] + 0.4*x[2])]
vjx_fs = [[x -> sin(0.6*x[1] + 0.2*j)*cos(0.5*x[2]) + 0.05*j      for j in 1:Nσ],
          [x -> 0.3*x[1]/10.0 + 0.2*cos(0.8*x[2] + j)             for j in 1:Nσ],
          [x -> cos(0.35*x[1])*cos(0.25*x[2] + 0.3*j)             for j in 1:Nσ]]
vjy_fs = [[x -> cos(0.4*x[1])*sin(0.7*x[2] + 0.1*j) + 0.02*j      for j in 1:Nσ],
          [x -> 0.15*sin(0.3*x[1]*j) + 0.1                        for j in 1:Nσ],
          [x -> sin(0.45*x[1] + 0.5*x[2])*0.5 + 0.03*j            for j in 1:Nσ]]

stackx(fs) = x -> VectorValue(ntuple(j -> fs[j](x), Nσ)...)

# interpolate states into both layouts
lay_fields  = Any[eta_f];  [push!(lay_fields, ujx_f[j], ujy_f[j])   for j in 1:Nσ]
lay_fieldst = Any[etat_f]; [push!(lay_fieldst, ujxt_f[j], ujyt_f[j]) for j in 1:Nσ]
uh_lay  = interpolate_everywhere(lay_fields,  U_lay)
uth_lay = interpolate_everywhere(lay_fieldst, U_lay)
uh_alg  = interpolate_everywhere([eta_f,  stackx(ujx_f),  stackx(ujy_f)],  U_alg)
uth_alg = interpolate_everywhere([etat_f, stackx(ujxt_f), stackx(ujyt_f)], U_alg)

tu_lay = Gridap.ODEs.TransientCellField(uh_lay, (uth_lay,))
tu_alg = Gridap.ODEs.TransientCellField(uh_alg, (uth_alg,))

# physical setup shared by both problems
sponge = make_sponge_2D(domain, 2.0, 2.0, 0.0, 0.0, 5.0)
omega  = 2.0*pi/1.6
k_wave = find_wavenumber(omega, 3.5, g_phys)
wm     = make_wavemaker_line(3.0, 0.001, 1.6, k_wave)
t_eval = 0.37

d_flat  = x -> 3.5
d_slope = x -> 3.5 - 0.02*x[1] + 0.01*x[2]

configs = [
    (name="A: linear core (lin=true, adv=false, linP=false, flat)",
     lin=true,  adv=false, linp=false, dfn=d_flat),
    (name="B: nonlinear + advection (lin=false, adv=true, linP=false, flat)",
     lin=false, adv=true,  linp=false, dfn=d_flat),
    (name="C: sloped bed + linear pressure (lin=false, adv=true, linP=true)",
     lin=false, adv=true,  linp=true,  dfn=d_slope),
]

for cfg in configs
    println("\n-- config $(cfg.name)")
    prob_lay = LFEMProblemLFEM(g_phys, cfg.dfn,
        vert.Mmat, vert.Phi, vert.B, vert.Mcal, vert.Gcal, vert.A, vert.K,
        Nσ, cfg.lin, cfg.linp, cfg.adv, sponge, wm)
    prob_alg = build_algebraic_problem(vert; g=g_phys, d_func=cfg.dfn,
        linearised=cfg.lin, advection=cfg.adv, lin_pressure=cfg.linp,
        P_full=false, nl_pressure68=false, mu_sponge=sponge, wm_src=wm)

    r_lay = assemble_vector(v -> residual_lfem(t_eval, tu_lay, v, prob_lay, trian, dO), V_lay)
    r_alg = assemble_vector(v -> residual_algebraic(t_eval, tu_alg, v, prob_alg, trian, dO), V_alg)

    for s in 1:3
        v_fields = Any[q_fs[s]]
        [push!(v_fields, vjx_fs[s][j], vjy_fs[s][j]) for j in 1:Nσ]
        vh_lay = interpolate_everywhere(v_fields, V_lay)
        vh_alg = interpolate_everywhere([q_fs[s], stackx(vjx_fs[s]), stackx(vjy_fs[s])], V_alg)
        w_lay = dot(get_free_dof_values(vh_lay), r_lay)
        w_alg = dot(get_free_dof_values(vh_alg), r_alg)
        rel = abs(w_lay - w_alg) / max(abs(w_lay), 1e-14)
        ok = check(@sprintf("virtual work, test set %d  (lay=%.10e  alg=%.10e  rel=%.2e)",
                            s, w_lay, w_alg, rel), rel < 1e-10)
    end
end

# ==============================================================
@printf("\n=== RESULT: %d passed, %d failed ===\n", npass, nfail)
# ==============================================================
nfail == 0 || error("equivalence test failed")
println("ALL TESTS PASSED")
