# ==============================================================
#  test_mms.jl — UNSTEADY, NONLINEAR Manufactured-Solution recovery
#
#  Residual-based MMS with a genuinely UNSTEADY, finite-amplitude manufactured
#  solution marched over MANY Crank–Nicolson steps, so it validates that the
#  NONLINEAR time-dependent flow is solved correctly (not just the linear
#  operator, not just a single step).
#
#  Method. Pick a smooth space-time field u*(x,t) — degree-2 in space (exactly
#  Q2-representable; velocity vanishing on the closed walls), oscillating in
#  time — over a CURVED bed (∇h,∇²h ≠ 0, analytic). Define the per-step forcing
#  by assembling the solver's own residual on u*. At step n (u_n = u*(tₙ) from
#  the previous step) the θ-scheme step equation
#     R(t_{n+θ}, u_{n+θ}, u̇) − f_n = 0,   f_n = R(t_{n+θ}, u*_{n+θ}, u̇*_scheme)
#  is solved with a Newton loop using the solver's OWN hand Jacobians
#     J = θ·jacobian_u + (1/Δt)·jacobian_u_t.
#  The SCHEME-consistent u̇*_scheme=(u*_{n+1}−u*_n)/Δt in the forcing makes the
#  recovery EXACT (machine precision) at every step — a growing error over the
#  run would expose an inconsistency in the nonlinear assembly, the Jacobians,
#  or the multi-step bookkeeping.
#
#  Physics. Fully nonlinear: advection ON, full leading pressure (P_full), and
#  the native nonlinear-pressure set 𝓝∈{3,6,7,8} (nl_pressure68). The MMS is
#  finite-amplitude on a sloped bed, so these terms are a SIGNIFICANT fraction
#  of the residual (verified below). The remaining 𝓝∈{1,2,4,5} frozen-projection
#  components are O(A³) corrections validated to machine precision by
#  test_nlpressure.jl (exact-IBP identity) and exercised at scale by
#  test/cluster/cluster_mms.jl; set env MMS_FULL=1 to include them here too
#  (much heavier first-compile).
#
#  Gates: (1) machine-precision recovery over the whole unsteady run; (2) the
#  nonlinear terms are a SIGNIFICANT fraction of the residual (genuinely
#  nonlinear); (3) the native nonlinear-pressure block is computed (nonzero).
#
#  RUN:  julia --project=. GridapLFEM.jl/test/test_mms.jl
# ==============================================================

if !isdefined(Main, :GridapLFEM)
    include(joinpath(@__DIR__, "..", "src", "GridapLFEM.jl"))
end
using .GridapLFEM
using Gridap
using Gridap.ODEs
using LinearAlgebra, Printf

FULL = lowercase(get(ENV, "MMS_FULL", "0")) in ("1","true","yes")

println("=" ^ 66)
println("  test_mms.jl — unsteady NONLINEAR manufactured-solution recovery",
        FULL ? "  [FULL 𝓝]" : "")
println("=" ^ 66)

n_pass = 0; n_fail = 0
check(name, cond, extra="") = (global n_pass, n_fail;
    cond ? (println("  PASS  $name $extra"); n_pass += 1) :
           (println("  FAIL  $name $extra"); n_fail += 1))

# ---- mesh, spaces, curved bed --------------------------------------------------
g = 9.81; Lx, Ly = 2.0, 1.0; d0 = 1.0
vert = assemble_vertical_tensors(2, 1, [0.0, 0.728, 1.0]); Nσ = vert.N_dof
model, trian = build_horizontal_model(((0.0,Lx),(0.0,Ly)), (6,4)); feo = 2
dO = Measure(trian, 2*feo + 2)
U, V = build_fe_spaces(model, feo, Nσ; y_wall_bc=true, x_wall_bc=true)   # closed basin
d_func(x) = d0 - 0.15*x[1] - 0.05*x[1]^2 + 0.08*x[1]*x[2]                # ∇h,∇²h ≠ 0

# ---- FINITE-AMPLITUDE unsteady manufactured solution ---------------------------
# velocity O(0.3), η O(0.2) over d0=1 ⇒ advection & nonlinear pressure non-trivial
T  = 2.0; ω = 2π/T
ax = [0.30, -0.12, 0.20]; ay = [0.18, 0.25, -0.14]                       # asymmetric modes
ec = (0.20, -0.12, 0.10, 0.15, -0.08, 0.05)                             # η spatial coeffs
etas(x,t)  = cos(ω*t)*(ec[1]*x[1]^2 + ec[2]*x[1]*x[2] + ec[3]*x[2]^2 + ec[4]*x[1] + ec[5]*x[2] + ec[6])
uxs(j,x,t) = sin(ω*t + 0.3j)*ax[j]*x[1]*(Lx - x[1])          # 0 at x=0,Lx  (Ux walls)
uys(j,x,t) = cos(ω*t - 0.2j)*ay[j]*x[2]*(Ly - x[2])          # 0 at y=0,Ly  (Uy walls)
stackf(fs) = x -> VectorValue(ntuple(j -> fs[j](x), Nσ)...)
ustar(t)   = interpolate_everywhere(
    [x -> etas(x,t), stackf([x -> uxs(j,x,t) for j in 1:Nσ]),
                     stackf([x -> uys(j,x,t) for j in 1:Nσ])], U)
ustar_dofs(t) = get_free_dof_values(ustar(t))

# ---- problem: fully nonlinear (advection + full leading + native 𝓝) ------------
prob = build_problem(vert; g=g, d_func=d_func,
    linearised=false, advection=true, lin_pressure=true, P_full=true,
    nl_pressure68=true, nl_pressure_full=FULL,
    mu_sponge=x -> 0.0, wm_src=(x,t) -> 0.0)
ctx = FULL ? build_nlp_ctx(model, feo, Nσ, trian, dO) : nothing

dt = 0.06; θ = 0.5; N = 12                            # unsteady CN march (≈0.36 period)
TCF(vec, dvec) = TransientCellField(FEFunction(U, vec), (FEFunction(U, dvec),))

# ================================================================
println("\n-- unsteady nonlinear march ($N steps) --")
# ================================================================
un = ustar_dofs(0.0)                                  # start from u*(0)
FULL && update_nlp_state!(prob, ctx, FEFunction(U, un))
errs = Float64[]; nl_iters_tot = 0
for n in 0:N-1
    global un, nl_iters_tot
    tn = n*dt; tθ = tn + θ*dt
    us1 = ustar_dofs(tn + dt)
    umid_star = un .+ θ.*(us1 .- un)                  # θ-point of the manufactured soln
    udot_star = (us1 .- un) ./ dt                     # scheme-consistent u̇* ⇒ exact recovery
    f = assemble_vector(v -> global_residual(tθ, TCF(umid_star, udot_star), v, prob, trian, dO), V)
    u1 = copy(un)
    for it in 1:15
        umid = un .+ θ.*(u1 .- un); udot = (u1 .- un)./dt
        tu = TCF(umid, udot)
        r  = assemble_vector(v -> global_residual(tθ, tu, v, prob, trian, dO), V) .- f
        Ju = assemble_matrix((du,v) -> jacobian_u(  tθ, tu, du, v, prob, trian, dO), U, V)
        Jt = assemble_matrix((du,v) -> jacobian_u_t(tθ, tu, du, v, prob, trian, dO), U, V)
        δ  = (θ .* Ju .+ (1.0/dt) .* Jt) \ (-r)
        u1 = u1 .+ δ; nl_iters_tot += 1
        norm(δ)/max(norm(us1),1e-14) < 1e-12 && break
    end
    push!(errs, norm(u1 .- us1)/norm(us1))
    un = u1                                           # = u*(t_{n+1}) to machine precision
    FULL && update_nlp_state!(prob, ctx, FEFunction(U, un))
end
@printf("  steps=%d  Newton iters=%d (%.2f/step)  max rel err over run = %.3e\n",
        N, nl_iters_tot, nl_iters_tot/N, maximum(errs))
@printf("  err(step 1)=%.2e  err(mid)=%.2e  err(last)=%.2e\n",
        errs[1], errs[N÷2], errs[end])
check("unsteady nonlinear recovery: machine precision over the whole run (max rel < 1e-8)",
      maximum(errs) < 1e-8)

# ================================================================
println("\n-- nonlinearity diagnostics (on u* at t=T/2) --")
# ================================================================
tm = 0.5*N*dt; um = ustar_dofs(tm)
udm = (ustar_dofs(tm + 1e-4) .- ustar_dofs(tm - 1e-4)) ./ 2e-4     # ∂ₜu* (FD)
tu_m = TCF(um, udm)
R_full = assemble_vector(v -> global_residual(tm, tu_m, v, prob, trian, dO), V)

prob_lin = build_problem(vert; g=g, d_func=d_func, linearised=true, advection=false,
                         mu_sponge=x -> 0.0, wm_src=(x,t) -> 0.0)
R_lin = assemble_vector(v -> global_residual(tm, tu_m, v, prob_lin, trian, dO), V)
nl_frac = norm(R_full .- R_lin) / norm(R_full)
@printf("  ‖R_full − R_linear‖ / ‖R_full‖ = %.1f%%  (nonlinear content of the residual)\n",
        100*nl_frac)
check("MMS is genuinely nonlinear (nonlinear residual fraction > 15%)", nl_frac > 0.15,
      "($(round(100*nl_frac,digits=1))%)")

# native nonlinear-pressure block is computed (nonzero)
um_fe = FEFunction(U, um); ηm = um_fe[1]; Uxm = um_fe[2]; Uym = um_fe[3]
dcf = CellField(d_func, trian); Hm = dcf + ηm
dhx = alg_dx(dcf); dhy = alg_dy(dcf); dHx = dhx + alg_dx(ηm); dHy = dhy + alg_dy(ηm)
DUm = alg_dx(Uxm) + alg_dy(Uym); afm = dhx*Uxm + dhy*Uym; bfm = dHx*Uxm + dHy*Uym
Sm  = Hm*DUm + bfm
r_nat = assemble_vector(v -> nlp_native_contrib(prob, dcf, ηm, Hm, dhx, dhy, dHx, dHy,
                             Uxm, Uym, v[2], v[3], alg_dx(v[2])+alg_dy(v[3]),
                             afm, bfm, Sm, DUm, dO), V)
@printf("  native nonlinear-pressure block ‖·‖ = %.3e  (𝓝 comps {3,6,7,8} active)\n", norm(r_nat))
check("native nonlinear-pressure block is computed (nonzero)", norm(r_nat) > 1e-9)

println()
println("=" ^ 66)
@printf("  Results: %d PASS,  %d FAIL\n", n_pass, n_fail)
println("=" ^ 66)
n_fail > 0 ? error("test_mms: $n_fail failed!") :
             println("  Unsteady nonlinear flow solved correctly (u* recovered over the run).")
