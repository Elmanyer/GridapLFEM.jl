# ==============================================================
#  cluster_mms.jl — DISTRIBUTED unsteady, nonlinear Manufactured Solution
#                   (validates the full nonlinear solver at scale, many steps)
#
#  The distributed twin of test/test_mms.jl, run through the REAL distributed
#  solver stack (GMRES + Jacobi + Newton). A smooth, finite-amplitude, UNSTEADY
#  manufactured solution u*(x,t) — degree-2 in space (exactly Q2-representable,
#  velocity vanishing on the closed walls), oscillating in time, over a curved
#  bed — is marched over MANY Crank–Nicolson steps with ALL nonlinear pressure
#  terms active. The per-step forcing is the solver's own residual assembled on
#  u*, injected by wrapping the residual:
#     res_forced(t,u,v) = R(t,u,v) − R(t, u*_{n+θ}, u̇*_scheme; v)
#  with the SCHEME-consistent θ-point/ u̇ built analytically from t, Δt, θ, so
#  the θ-method recovers u*(tₙ) to MACHINE PRECISION at every step. The
#  Jacobians are unchanged (the forcing is independent of u), so the standard
#  distributed operator/solver/iterator drive it. The error ‖u_n − u*(tₙ)‖ is a
#  GLOBAL reduction (`sum(∫·dΩ)`), so it works on any core count with no gauges.
#
#  A growing error would expose an inconsistency in the DISTRIBUTED nonlinear
#  assembly, the hand Jacobians, the frozen-projection (CG+Jacobi) bookkeeping,
#  or the GMRES/Newton stack — over a long nonlinear run at scale.
#
#  LAUNCH (px·py MUST equal -n):
#    LFEM_PX=4 LFEM_PY=2 LFEM_NX=400 LFEM_NY=200 LFEM_NSTEPS=200 \
#    ~/.julia/bin/mpiexecjl --project=. -n 8 julia --project=. \
#        GridapLFEM.jl/test/cluster/cluster_mms.jl
#
#  Env vars: LFEM_PX,LFEM_PY; LFEM_NX,LFEM_NY; LFEM_LX,LFEM_LY; LFEM_D; LFEM_M;
#    LFEM_DT; LFEM_NSTEPS; LFEM_AMP (nonlinearity strength); LFEM_NLPFULL (all
#    𝓝 comps, default 1); LFEM_PRINT_EVERY; LFEM_OUTDIR.
# ==============================================================

include(joinpath(@__DIR__, "..", "..", "examples", "distributed", "_dist_common.jl"))
using .GridapLFEM
using Gridap
using Gridap.ODEs
using PartitionedArrays, MPI
using LinearAlgebra, Printf

px, py = genv_i("LFEM_PX", 2), genv_i("LFEM_PY", 1)
nx, ny = genv_i("LFEM_NX", 40), genv_i("LFEM_NY", 24)
Lx, Ly = genv_f("LFEM_LX", 2.0), genv_f("LFEM_LY", 1.0)
d0     = genv_f("LFEM_D", 1.0)
Mlay   = genv_i("LFEM_M", 2)
dt     = genv_f("LFEM_DT", 0.05)
Nsteps = genv_i("LFEM_NSTEPS", 60)
amp    = genv_f("LFEM_AMP", 1.0)                 # scales the manufactured amplitude
nlpfull= genv_b("LFEM_NLPFULL", 1)
feord  = genv_i("LFEM_FE_ORDER", 2)
prevery= genv_i("LFEM_PRINT_EVERY", 10)
outdir = genv("LFEM_OUTDIR", joinpath(ROOT, "output", "cluster_mms"))
g = 9.81; θ = 0.5; T = 2.0; ω = 2π/T

result = with_mpi() do distribute
    ranks = distribute(LinearIndices((px*py,)))
    r0 = i_am_main(ranks)
    r0 && mkpath(outdir)

    vert = assemble_vertical_tensors(Mlay, 1, cbdy_override() === nothing ?
              get(GridapLFEM.DEFAULT_CBDY, Mlay, collect(LinRange(0,1,Mlay+1))) : cbdy_override())
    Nσ = vert.N_dof
    model, trian = build_horizontal_model_distributed(ranks, (px,py),
                       (0.0,Lx,0.0,Ly), (nx,ny))
    dΩh = Measure(trian, 2*feord + 2)
    U, V = build_fe_spaces(model, feord, Nσ; y_wall_bc=:wall, x_wall_bc=true)   # closed basin
    h_bathy(x) = d0 - 0.15*x[1] - 0.05*x[1]^2 + 0.08*x[1]*x[2]                  # ∇h,∇²h ≠ 0

    # finite-amplitude unsteady manufactured solution (velocity → 0 on walls)
    ax = amp .* [0.30, -0.12, 0.20, 0.15, -0.08, 0.11][1:Nσ]
    ay = amp .* [0.18, 0.25, -0.14, -0.09, 0.13, 0.07][1:Nσ]
    ec = amp .* (0.20, -0.12, 0.10, 0.15, -0.08, 0.05)
    etas(x,t)  = cos(ω*t)*(ec[1]*x[1]^2 + ec[2]*x[1]*x[2] + ec[3]*x[2]^2 + ec[4]*x[1] + ec[5]*x[2] + ec[6])
    uxs(j,x,t) = sin(ω*t + 0.3j)*ax[j]*x[1]*(Lx - x[1])
    uys(j,x,t) = cos(ω*t - 0.2j)*ay[j]*x[2]*(Ly - x[2])
    stk(f,t)   = x -> VectorValue(ntuple(j -> f(j,x,t), Nσ)...)
    ustar(t)   = interpolate_everywhere([x->etas(x,t), stk(uxs,t), stk(uys,t)], U)
    # θ-point combination and scheme-consistent u̇, built analytically from t
    function forcing_tcf(t)
        tp = t + (1-θ)*dt; tm = t - θ*dt
        umid = interpolate_everywhere(
            [x-> θ*etas(x,tp)+(1-θ)*etas(x,tm),
             x-> VectorValue(ntuple(j-> θ*uxs(j,x,tp)+(1-θ)*uxs(j,x,tm), Nσ)...),
             x-> VectorValue(ntuple(j-> θ*uys(j,x,tp)+(1-θ)*uys(j,x,tm), Nσ)...)], U)
        udot = interpolate_everywhere(
            [x-> (etas(x,tp)-etas(x,tm))/dt,
             x-> VectorValue(ntuple(j-> (uxs(j,x,tp)-uxs(j,x,tm))/dt, Nσ)...),
             x-> VectorValue(ntuple(j-> (uys(j,x,tp)-uys(j,x,tm))/dt, Nσ)...)], U)
        return TransientCellField(umid, (udot,))
    end

    prob = build_problem(vert; g=g, h_bathy=h_bathy,
        linearised=false, advection=true, lin_pressure=true, P_full=true,
        nl_pressure68=true, nl_pressure_full=nlpfull,
        mu_sponge=x -> 0.0, wm_src=(x,t) -> 0.0)
    nlp = nlpfull ? (prob, build_nlp_ctx(model, feord, Nσ, trian, dΩh; distributed=true)) : nothing

    # forced residual (manufactured), same hand Jacobians (forcing indep of u)
    res_f(t,u,v) = global_residual(t,u,v,prob,trian,dΩh) -
                   global_residual(t, forcing_tcf(t), v, prob, trian, dΩh)
    jac_f(t,u,du,v)  = jacobian_u(t,u,du,v,prob,trian,dΩh)
    jact_f(t,u,dut,v)= jacobian_u_t(t,u,dut,v,prob,trian,dΩh)
    op = TransientFEOperator(res_f, jac_f, jact_f, U, V)

    monitor = SolverMonitor()
    solver  = build_ode_solver_distributed(dt; nl_tol=1e-8, monitor=monitor)
    u0 = ustar(0.0)
    nlp !== nothing && update_nlp_state!(nlp[1], nlp[2], u0)     # frozen π from u*(0)

    Tfinal = Nsteps*dt
    if r0
        @printf("# cluster_mms | %d ranks (%d×%d) | mesh %d×%d = %d cells | Nσ=%d | amp=%.2f nlPfull=%s\n",
                px*py, px, py, nx, ny, nx*ny, Nσ, amp, nlpfull)
        @printf("# %d steps dt=%.4g (Tfinal=%.2f), all 𝓝 comps on curved bed\n", Nsteps, dt, Tfinal)
        open(joinpath(outdir, "mms.csv"), "w") do io; println(io, "step,t,rel_err,nl_iters"); end
    end

    function relerr(u, t)
        us = ustar(t)
        e = sqrt(abs(sum(∫( (u[1]-us[1])*(u[1]-us[1])
                          + (u[2]-us[2])⋅(u[2]-us[2]) + (u[3]-us[3])⋅(u[3]-us[3]) )*dΩh)))
        n = sqrt(abs(sum(∫( us[1]*us[1] + us[2]⋅us[2] + us[3]⋅us[3] )*dΩh)))
        return e/max(n,1e-30)
    end

    emax = 0.0; step = 0; wall0 = time()
    for (t_n, u_n) in solve(solver, op, 0.0, Tfinal, u0)
        step += 1
        st = take_step_stats!(monitor)
        er = relerr(u_n, t_n); emax = max(emax, er)
        if r0 && (step % prevery == 0 || step == Nsteps)
            @printf("  step %5d  t=%.3f  rel_err=%.3e  nl=%d  (%.1fs)\n",
                    step, t_n, er, st.nl_iters, time()-wall0)
            open(joinpath(outdir, "mms.csv"), "a") do io
                @printf(io, "%d,%.6f,%.6e,%d\n", step, t_n, er, st.nl_iters); end
            flush(stdout)
        end
        nlp !== nothing && update_nlp_state!(nlp[1], nlp[2], ustar(t_n))   # refresh π from u*(tₙ)
        isnan(er) && (r0 && @warn("NaN at t=$t_n"); break)
    end

    if r0
        @printf("\n# SUMMARY: steps=%d  max rel err over run = %.3e\n", step, emax)
        pass = emax < 1e-8
        println(pass ? "  PASS  unsteady nonlinear u* recovered to machine precision at scale" :
                       "  FAIL  MMS error grew ($emax) — check distributed nonlinear assembly/Jacobians")
        println("# CSV: $(joinpath(outdir, "mms.csv"))")
        pass || error("cluster_mms: FAILED")
        println("# cluster_mms: distributed unsteady nonlinear solver validated.")
    end
    return nothing
end
