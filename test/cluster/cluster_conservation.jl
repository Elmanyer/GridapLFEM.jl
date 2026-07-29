# ==============================================================
#  cluster_conservation.jl — DISTRIBUTED long-run conservation & stability
#                            (big problem, many time steps, multi-node)
#
#  Validates that the solver stays CORRECT at scale and over LONG runs: in a
#  closed, inviscid basin (all walls solid, NO wavemaker, NO sponge) with an
#  initial free-surface hump, the depth-integrated continuity equation conserves
#  total mass ∫η EXACTLY (test ψ≡const ⇒ d/dt∫η=0), while the (hydrostatic) wave
#  energy stays bounded and the surface amplitude does not blow up. All three
#  diagnostics are GLOBAL reductions (`sum(∫·dΩ)` reduces across ranks in
#  GridapDistributed), so they need no point gauges and scale to any core count.
#
#  Records a CSV time series (t, mass, energy, etaL2, Newton iters) for offline
#  plotting/tabulation, and asserts the mass drift stays at solver tolerance and
#  the amplitude bounded over the WHOLE run. Fully nonlinear by default.
#
#  LAUNCH (px·py MUST equal -n):
#    LFEM_PX=8 LFEM_PY=1 LFEM_NX=2000 LFEM_NY=200 LFEM_STEPS=4000 \
#    ~/.julia/bin/mpiexecjl --project=. -n 8 julia --project=. \
#        GridapLFEM.jl/test/cluster/cluster_conservation.jl
#
#  Env vars: LFEM_PX,LFEM_PY (grid); LFEM_NX,LFEM_NY (mesh); LFEM_LX,LFEM_LY
#    (domain); LFEM_D (depth); LFEM_A (hump amplitude); LFEM_DT; LFEM_STEPS or
#    LFEM_TFINAL; LFEM_M (layers); LFEM_LINEARISED,LFEM_ADVECTION,LFEM_NLP68;
#    LFEM_PRINT_EVERY; LFEM_OUTDIR.
# ==============================================================

include(joinpath(@__DIR__, "..", "..", "examples", "distributed", "_dist_common.jl"))
using .GridapLFEM
using Gridap
using PartitionedArrays, MPI
using LinearAlgebra, Printf

# ---- configuration (environment) ----------------------------------------------
px, py = genv_i("LFEM_PX", 2), genv_i("LFEM_PY", 1)
nx, ny = genv_i("LFEM_NX", 200), genv_i("LFEM_NY", 40)
Lx, Ly = genv_f("LFEM_LX", 40.0), genv_f("LFEM_LY", 8.0)
d0     = genv_f("LFEM_D", 1.0)
A0     = genv_f("LFEM_A", 0.01)
Mlay   = genv_i("LFEM_M", 2)
dt     = genv_f("LFEM_DT", 0.02)
nsteps = genv_i("LFEM_STEPS", 2000)                          # long run by default
feord  = genv_i("LFEM_FE_ORDER", 2)
prevery= genv_i("LFEM_PRINT_EVERY", 20)
outdir = genv("LFEM_OUTDIR", joinpath(ROOT, "output", "cluster_conservation"))
g      = 9.81
Tfinal = haskey(ENV, "LFEM_TFINAL") ? genv_f("LFEM_TFINAL", 0.0) : nsteps*dt

result = with_mpi() do distribute
    ranks = distribute(LinearIndices((px*py,)))
    r0 = i_am_main(ranks)
    r0 && mkpath(outdir)

    vert = assemble_vertical_tensors(Mlay, 1, cbdy_override() === nothing ?
              get(GridapLFEM.DEFAULT_CBDY, Mlay, collect(LinRange(0,1,Mlay+1))) : cbdy_override())
    Nσ = vert.N_dof
    Mt = alg_to_tensor2(vert.Mmat); Bt = alg_to_tensor2(vert.B)

    model, trian = build_horizontal_model_distributed(ranks, (px,py),
                       (0.0,Lx,0.0,Ly), (nx,ny))
    dΩh = Measure(trian, 2*feord + 2)
    U, V = build_fe_spaces(model, feord, Nσ; y_wall_bc=:wall, x_wall_bc=true)  # closed basin

    prob = build_problem(vert; g=g, h_bathy=x -> d0,
        regime=regime_sym(), nl_pressure=nl_pressure_sym(), flat_bed=flat_bed_flag(),   # constant depth → flat bed
        mu_sponge=x -> 0.0, wm_src=(x,t) -> 0.0)

    # initial free-surface hump at the basin centre (u=0); x_wall_bc=true is set
    xc, yc, wd = Lx/2, Ly/2, max(Lx,Ly)/20
    eta0(x) = A0*exp(-(((x[1]-xc)^2 + (x[2]-yc)^2))/wd^2)
    u0 = make_initial_conditions(U, Nσ; eta0_func=eta0)

    mass(u)   = sum(∫( u[1] )*dΩh)                                     # ∫η  (exact invariant)
    etaL2(u)  = sqrt(abs(sum(∫( u[1]*u[1] )*dΩh)))                     # stability indicator
    function energy(u)
        DU = alg_dx(u[2]) + alg_dy(u[3])
        0.5*g*sum(∫( u[1]*u[1] )*dΩh) +
        0.5*sum(∫( (u[2] ⋅ alg_mul(Mt,u[2])) + (u[3] ⋅ alg_mul(Mt,u[3])) )*dΩh) -
        0.5*d0^2*sum(∫( DU ⋅ alg_mul(Bt,DU) )*dΩh)
    end

    op      = build_ode_operator(prob, U, V, trian, dΩh)
    monitor = SolverMonitor()
    solver  = build_ode_solver_distributed(dt; nl_tol=1e-8, monitor=monitor)

    M0 = mass(u0); E0 = energy(u0); L0 = etaL2(u0)
    nsteps = round(Int, Tfinal/dt)
    if r0
        @printf("# cluster_conservation | %d ranks (%d×%d) | mesh %d×%d = %d cells | Nσ=%d\n",
                px*py, px, py, nx, ny, nx*ny, Nσ)
        @printf("# domain %.0f×%.0f | d=%.2f | hump A=%.4g | dt=%.4g | %d steps (T=%.1f)\n",
                Lx, Ly, d0, A0, dt, nsteps, Tfinal)
        @printf("# regime=%s nl_pressure=%s flat_bed=%s | M0=%.6e E0=%.6e\n",
                string(regime_sym()), string(nl_pressure_sym()), string(flat_bed_flag()), M0, E0)
        open(joinpath(outdir, "conservation.csv"), "w") do io
            println(io, "step,t,mass,dmass_rel,energy,denergy_rel,etaL2,nl_iters")
        end
    end

    dmax = 0.0; lmax = L0; step = 0; wall0 = time()
    for (t_n, u_n) in solve(solver, op, 0.0, Tfinal, u0)
        step += 1
        st = take_step_stats!(monitor)
        m  = mass(u_n); e = energy(u_n); l = etaL2(u_n)
        dm = abs(m - M0)/max(abs(M0),1e-30); de = abs(e - E0)/max(abs(E0),1e-30)
        dmax = max(dmax, dm); lmax = max(lmax, l)
        if r0 && (step % prevery == 0 || step == nsteps)
            wall = time() - wall0
            @printf("  step %6d  t=%.3f  Δmass=%.2e  ΔE=%.2e  etaL2=%.4e  nl=%d  (%.1fs)\n",
                    step, t_n, dm, de, l, st.nl_iters, wall)
            open(joinpath(outdir, "conservation.csv"), "a") do io
                @printf(io, "%d,%.6f,%.10e,%.3e,%.10e,%.3e,%.6e,%d\n",
                        step, t_n, m, dm, e, de, l, st.nl_iters)
            end
            flush(stdout)
        end
        (isnan(l) || l > 1e3*L0) && (r0 && @warn("diverged at t=$t_n"); break)
    end

    if r0
        @printf("\n# SUMMARY: steps=%d  max mass drift=%.3e  max etaL2/L0=%.3f\n",
                step, dmax, lmax/L0)
        pass_mass = dmax < 1e-6
        pass_bound = lmax < 5*L0 && !isnan(lmax)
        println(pass_mass  ? "  PASS  mass conserved over the whole run (drift < 1e-6)" :
                             "  FAIL  mass drift $dmax")
        println(pass_bound ? "  PASS  amplitude bounded (no blow-up)" :
                             "  FAIL  amplitude unbounded")
        println("# CSV: $(joinpath(outdir, "conservation.csv"))")
        (pass_mass && pass_bound) || error("cluster_conservation: FAILED")
        println("# cluster_conservation: distributed long-run conservation validated.")
    end
    return nothing
end
