# ==============================================================
#  test_equivalence_alg.jl — THE acceptance test (oracle equivalence)
#
#  Assembles the package residual `LFEModelAlg.residual_alg` AND the old
#  per-layer oracle `LFEModel2D.residual_lfem` on identical physical states
#  and compares virtual works — no DOF-layout mapping:
#    * identical analytic trial/u̇/test fields interpolated into BOTH layouts,
#    * TransientCellField built by hand,
#    * scalar r(v) = dot(free_dofs(v_interp), assemble_vector(r, V)) compared.
#  Three flag configs × three test sets; requires rel < 1e-10.
#
#  Loads BOTH modules (oracle qualified) — heavier than the other tests.
#  RUN:  julia --project=. LFEM_2D/test/test_equivalence_alg.jl
# ==============================================================

if !isdefined(Main, :LFEModel2D)
    include(joinpath(@__DIR__, "..", "..", "LFE-M_2D_solver", "src", "LFEModel2D.jl"))
end
if !isdefined(Main, :LFEModelAlg)
    include(joinpath(@__DIR__, "..", "src", "LFEModelAlg.jl"))
end
using .LFEModel2D                      # oracle (unqualified: residual_lfem, LFEMProblemLFEM, …)
import .LFEModelAlg as ALG             # package under test (qualified)
using Gridap
using Gridap.ODEs
using LinearAlgebra, Printf

println("=" ^ 60)
println("  test_equivalence_alg.jl — package vs oracle virtual work")
println("=" ^ 60)

n_pass = 0; n_fail = 0
function check(name, cond)
    global n_pass, n_fail
    if cond; println("  PASS  $name"); n_pass += 1
    else;    println("  FAIL  $name"); n_fail += 1; end
end

M_v, p_v = 2, 1
c_bdy  = [0.0, 0.728, 1.0]
g_phys = 9.81

vert_o = assemble_vertical_tensors_lfem(M_v, p_v, c_bdy)   # oracle tensors
vert_a = ALG.assemble_vertical_tensors_alg(M_v, p_v, c_bdy)
Nσ = vert_a.N_dof
check("vertical tensors match oracle (Mmat,Phi,B,Mcal,Gcal,A,K,P,Acal,Kcal)",
      norm(vert_a.Mmat - vert_o.Mmat) < 1e-13 && norm(vert_a.Phi - vert_o.Phi) < 1e-13 &&
      norm(vert_a.B - vert_o.B) < 1e-13 && norm(vert_a.Mcal - vert_o.Mcal) < 1e-13 &&
      norm(vert_a.Gcal - vert_o.Gcal) < 1e-13 && norm(vert_a.A - vert_o.A) < 1e-13 &&
      norm(vert_a.K - vert_o.K) < 1e-13 && norm(vert_a.P - vert_o.P) < 1e-13 &&
      norm(vert_a.Acal - vert_o.Acal) < 1e-13 && norm(vert_a.Kcal - vert_o.Kcal) < 1e-13)

domain    = ((0.0, 10.0), (0.0, 5.0))
partition = (6, 4)
fe_order  = 2
model, trian = build_horizontal_model_2D(domain, partition)
dO = Measure(trian, 2*fe_order + 2)

# unconstrained spaces → all DOFs free, interpolated states identical
U_lay, V_lay = build_fe_spaces_2D(model, fe_order, Nσ; y_wall_bc=false)
U_alg, V_alg = ALG.build_fe_spaces_alg(model, fe_order, Nσ; y_wall_bc=false)

# ---- analytic states / test functions -----------------------------------------
eta_f  = x -> 0.02*cos(0.4*x[1])*cos(0.5*x[2]) + 0.005*x[1]/10.0
ujx_f  = [x -> 0.03*cos(0.3*x[1] + 0.1*j)*sin(0.4*x[2] + 0.2*j) + 0.004*j        for j in 1:Nσ]
ujy_f  = [x -> 0.02*sin(0.25*x[1] - 0.15*j)*cos(0.35*x[2]) + 0.003*x[2]/5.0*j    for j in 1:Nσ]
etat_f = x -> 0.01*sin(0.3*x[1])*cos(0.6*x[2])
ujxt_f = [x -> 0.02*sin(0.45*x[1] + 0.3*j)*cos(0.2*x[2]) + 0.002*j               for j in 1:Nσ]
ujyt_f = [x -> 0.015*cos(0.5*x[1])*sin(0.3*x[2] + 0.1*j) + 0.001*j*x[1]/10.0     for j in 1:Nσ]

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

lay_fields  = Any[eta_f];  [push!(lay_fields, ujx_f[j], ujy_f[j])   for j in 1:Nσ]
lay_fieldst = Any[etat_f]; [push!(lay_fieldst, ujxt_f[j], ujyt_f[j]) for j in 1:Nσ]
uh_lay  = interpolate_everywhere(lay_fields,  U_lay)
uth_lay = interpolate_everywhere(lay_fieldst, U_lay)
uh_alg  = interpolate_everywhere([eta_f,  stackx(ujx_f),  stackx(ujy_f)],  U_alg)
uth_alg = interpolate_everywhere([etat_f, stackx(ujxt_f), stackx(ujyt_f)], U_alg)

tu_lay = Gridap.ODEs.TransientCellField(uh_lay, (uth_lay,))
tu_alg = Gridap.ODEs.TransientCellField(uh_alg, (uth_alg,))

sponge = make_sponge_2D(domain, 2.0, 2.0, 0.0, 0.0, 5.0)
omega  = 2.0*pi/1.6
k_wave = find_wavenumber(omega, 3.5, g_phys)
wm     = make_wavemaker_line(3.0, 0.001, 1.6, k_wave)
t_eval = 0.37

d_flat  = x -> 3.5
d_slope = x -> 3.5 - 0.02*x[1] + 0.01*x[2]

configs = [
    (name="A: linear core (lin, no adv, no linP, flat)",
     lin=true,  adv=false, linp=false, dfn=d_flat),
    (name="B: nonlinear + advection (flat)",
     lin=false, adv=true,  linp=false, dfn=d_flat),
    (name="C: sloped bed + linear pressure",
     lin=false, adv=true,  linp=true,  dfn=d_slope),
]

for cfg in configs
    println("\n-- config $(cfg.name)")
    prob_lay = LFEMProblemLFEM(g_phys, cfg.dfn,
        vert_o.Mmat, vert_o.Phi, vert_o.B, vert_o.Mcal, vert_o.Gcal, vert_o.A, vert_o.K,
        Nσ, cfg.lin, cfg.linp, cfg.adv, sponge, wm)
    prob_alg = ALG.build_problem_alg(vert_a; g=g_phys, d_func=cfg.dfn,
        linearised=cfg.lin, advection=cfg.adv, lin_pressure=cfg.linp,
        P_full=false, nl_pressure68=false, mu_sponge=sponge, wm_src=wm)

    r_lay = assemble_vector(v -> residual_lfem(t_eval, tu_lay, v, prob_lay, trian, dO), V_lay)
    r_alg = assemble_vector(v -> ALG.residual_alg(t_eval, tu_alg, v, prob_alg, trian, dO), V_alg)

    for s in 1:3
        v_fields = Any[q_fs[s]]
        [push!(v_fields, vjx_fs[s][j], vjy_fs[s][j]) for j in 1:Nσ]
        vh_lay = interpolate_everywhere(v_fields, V_lay)
        vh_alg = interpolate_everywhere([q_fs[s], stackx(vjx_fs[s]), stackx(vjy_fs[s])], V_alg)
        w_lay = dot(get_free_dof_values(vh_lay), r_lay)
        w_alg = dot(get_free_dof_values(vh_alg), r_alg)
        rel = abs(w_lay - w_alg) / max(abs(w_lay), 1e-14)
        check(@sprintf("virtual work, test set %d  (rel=%.2e)", s, rel), rel < 1e-10)
    end
end

println()
println("=" ^ 60)
@printf("  Results: %d PASS,  %d FAIL\n", n_pass, n_fail)
println("=" ^ 60)
n_fail > 0 ? error("test_equivalence_alg: $n_fail failed!") :
             println("  Package residual is oracle-equivalent.")
