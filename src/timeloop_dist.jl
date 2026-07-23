# ==============================================================
#  timeloop_dist.jl — Distributed (MPI) mesh, solver stack and time loop
#
#  Port of ../../LFE-M_2D_solver/src/timeloop2D_dist.jl to the algebraic
#  package. Because the stacked residual/Jacobians are Gridap-native CellField
#  algebra (Operation is forwarded for DistributedCellField), THE SAME
#  residual/jacobian_* run distributed unchanged — including the FULL
#  nonlinear physics (advection, slope pressure, P_full, nl_pressure68). No
#  owned V⊗H loop is needed, unlike the old solver.
#
#  Load-bearing conventions (old solver, revalidated):
#    * distributed solve = ThetaMethod(NewtonSolver(GMRESSolver(;Pr=Jacobi)))
#      from GridapSolvers — LU/NLSolver are sequential-only at scale;
#    * norm(PVector, Inf) is BROKEN (PartitionedArrays 0.3.5) → own_values +
#      reduce(max, …; init=0.0);
#    * all I/O behind i_am_main(ranks); mkpath on rank 0 + MPI.Barrier;
#    * point gauges omitted (inter-rank point search) — diags = (t, eta_max);
#    * launch with ~/.julia/bin/mpiexecjl (system mpiexec → PMIx mismatch).
# ==============================================================

"""
    build_horizontal_model_distributed(ranks, cpu_grid, domain, partition)

Partitioned 2D Cartesian mesh across MPI ranks. `cpu_grid=(px,py)` is the
process topology (`px·py == mpiexec -n`); `domain=(x0,x1,y0,y1)` (flat 4-tuple);
`partition=(nx,ny)` cells (nx divisible by px, ny by py). Returns `(model, trian)`.
"""
function build_horizontal_model_distributed(ranks, cpu_grid::Tuple,
                                                domain::NTuple{4,Float64},
                                                partition::Tuple)
    model = CartesianDiscreteModel(ranks, cpu_grid, domain, partition)
    trian = Triangulation(model)
    return model, trian
end

"""
    build_ode_solver_distributed(dt; solver_type, theta, tableau, nl_iter,
                                     nl_tol, ls_rtol, ls_maxiter, monitor)

Distributed ODE solver factory (template: GridapSWE.jl). Wraps
NewtonSolver(GMRES + Jacobi preconditioner) — the scalable GridapSWE-pattern
linear stack. Integrators: `:sdirk` (fully-implicit `RungeKutta(nls, ls, dt,
tableau)`, default; `:SDIRK_2_2` = L-stable 2nd-order, more stable than
Crank–Nicolson) and `:theta` (Crank–Nicolson). Pass a `SolverMonitor` as
`monitor` to collect per-step convergence statistics (RK: over the stages).
"""
function build_ode_solver_distributed(dt::Float64;
                                          solver_type :: Symbol  = :sdirk,
                                          theta       :: Float64 = 0.5,
                                          tableau     :: Symbol  = :SDIRK_2_2,
                                          nl_iter     :: Int     = 50,
                                          nl_tol      :: Float64 = 1e-6,
                                          ls_rtol     :: Float64 = 1e-9,
                                          ls_maxiter  :: Int     = 2000,
                                          monitor                = nothing)
    ls  = GMRESSolver(ls_maxiter; Pr=JacobiLinearSolver(), rtol=ls_rtol,
                      atol=1e-14, verbose=false)
    nls = NewtonSolver(ls; maxiter=nl_iter, atol=nl_tol, rtol=1e-10, verbose=false)
    if monitor !== nothing
        monitor.inner = nls
        nls = monitor
    end
    if solver_type == :sdirk
        return RungeKutta(nls, ls, dt, tableau)
    elseif solver_type == :theta
        return ThetaMethod(nls, dt, theta)
    else
        error("Distributed algebraic solver supports :sdirk (RungeKutta) or :theta.")
    end
end

"""
    eta_max_distributed(eta_n)

Global max |η| across ranks: own_values only (no ghost duplication) + MPI max
reduction. (norm(PVector, Inf) is broken in PartitionedArrays 0.3.5.)
"""
function eta_max_distributed(eta_n)
    free_vals = get_free_dof_values(eta_n)
    local_maxes = map(own_values(free_vals)) do lv
        isempty(lv) ? 0.0 : maximum(abs, lv)
    end
    return reduce(max, local_maxes; init=0.0)
end

"""
    run_time_loop_dist(ranks, op, solver, u0, t0, T_final; output_dir,
                           save_every, trian, Nσ, print_every, print_dt, recon,
                           trial_space, dt, nlp, monitor, checker, check_every,
                           check_tol)

Distributed time loop. Returns `[(t, eta_max, nl_iters, res_nl, t_solve)]`
(eta_max by MPI reduction; no point gauges; the last three are −1/NaN without
a `monitor`). VTK: per-σ-node components of the stacked fields under the OLD
field names (`eta, u1x, u1y, …`), plus the reconstructed `w_s<σ>`/`p_s<σ>`
fields when `recon` is given (from `build_field_recon` — pure CellField
algebra, distributed-transparent).

Runtime diagnostics (printed on rank 0 only; the underlying stats/assemblies
are computed collectively on ALL ranks — do not rank-guard the calls):
  * `monitor` (SolverMonitor) — Newton iterations, residuals, convergence
    flag, last GMRES iteration count, solve wall time per step;
  * `print_every` — report every N steps (default 1); a legacy `print_dt`
    (simulation seconds) overrides it when given;
  * `checker` (ResidualChecker) + `check_every` — independent reassembly
    of the governing equations, verified against `check_tol`.
"""
function run_time_loop_dist(ranks, op, solver, u0,
                                t0::Float64, T_final::Float64;
                                output_dir :: String  = "output",
                                save_every :: Int     = 0,
                                trian                 = nothing,
                                Nσ         :: Int     = 1,
                                print_every:: Int     = 1,
                                print_dt              = nothing,
                                recon                 = nothing,
                                trial_space           = nothing,
                                dt         :: Float64 = 0.0,
                                nlp                   = nothing,   # (prob, ctx) for nl_pressure_full
                                monitor               = nothing,   # SolverMonitor
                                checker               = nothing,   # ResidualChecker
                                check_every:: Int     = 0,
                                check_tol  :: Float64 = 1e-8)
    if i_am_main(ranks)
        mkpath(output_dir)
    end
    MPI.Barrier(MPI.COMM_WORLD)

    odesol = solve(solver, op, t0, T_final, u0)

    prev_vals    = nothing     # previous-step DOFs (PVector) → u̇ backward FD
    need_prev    = recon !== nothing || (checker !== nothing && check_every > 0)
    diags        = NamedTuple{(:t,:eta_max,:nl_iters,:res_nl,:t_solve),
                              Tuple{Float64,Float64,Int,Float64,Float64}}[]
    t_last_print = t0
    step         = 0
    do_vtk       = !isnothing(trian) && save_every > 0
    steps_total  = dt > 0 ? round(Int, (T_final - t0)/dt) : 0
    wall0        = time()
    nl_total     = 0
    t_solve_tot  = 0.0
    n_vtk        = 0

    function _loop(pvd)
        for (t_n, u_n) in odesol
            step  += 1
            stats  = monitor === nothing ? nothing : take_step_stats!(monitor)
            eta_n  = u_n[1]
            emax   = eta_max_distributed(eta_n)
            push!(diags, (t=t_n, eta_max=emax,
                          nl_iters=stats === nothing ? -1 : stats.nl_iters,
                          res_nl=stats === nothing ? NaN : stats.res,
                          t_solve=stats === nothing ? NaN : stats.t_solve))
            if stats !== nothing
                nl_total    += stats.nl_iters
                t_solve_tot += stats.t_solve
            end

            do_print = print_dt === nothing ? (step % print_every == 0) :
                                              (t_n - t_last_print >= Float64(print_dt))
            if do_print
                if i_am_main(ranks)
                    wall  = time() - wall0
                    eta_s = steps_total > step ? (wall/step)*(steps_total - step) : NaN
                    println(step_report(step, t_n, emax, stats;
                                            eta_s=eta_s, tag="[dist]"))
                    flush(stdout)
                end
                t_last_print = t_n
            end
            if stats !== nothing && !stats.converged && i_am_main(ranks)
                println("  *** WARNING: nonlinear solver did NOT converge at step $step (t=$t_n) ***")
                flush(stdout)
            end

            # independent verification of the governing equations (collective!)
            if checker !== nothing && check_every > 0 && step % check_every == 0
                cres = check_residuals(checker, t_n, u_n, prev_vals)
                if cres !== nothing && i_am_main(ranks)
                    println(check_report(step, t_n, cres, check_tol))
                    flush(stdout)
                end
            end

            if !isnothing(pvd) && step % save_every == 0
                t_vtk = @elapsed begin
                    tn_str = replace(@sprintf("%.4f", t_n), "." => "_")
                    fname  = joinpath(output_dir, "sol_t_$(tn_str)")
                    fields = Pair{String,Any}["eta" => eta_n]
                    for k in 1:Nσ
                        push!(fields, "u$(k)x" => alg_component(u_n[2], k))
                        push!(fields, "u$(k)y" => alg_component(u_n[3], k))
                    end
                    if recon !== nothing
                        u_prev = prev_vals === nothing ? nothing :
                                 FEFunction(space_at(trial_space, t_n - dt), prev_vals)
                        append!(fields, extra_field_cellfields(u_n, u_prev, dt, recon, trian))
                    end
                    pvd[t_n] = createvtk(trian, fname; cellfields=fields, append=false)
                end
                n_vtk += 1
                if i_am_main(ranks)
                    @printf("  [vtk] snapshot %d written at t=%.4f (%.2f s)\n", n_vtk, t_n, t_vtk)
                    flush(stdout)
                end
            end

            if need_prev
                prev_vals = copy(get_free_dof_values(u_n))
            end

            # nl_pressure_full: refresh the frozen projections π𝖲, π𝖻 (CG+Jacobi
            # distributed mass solve, see nlpressure.jl :: build_nlp_ctx)
            if nlp !== nothing
                update_nlp_state!(nlp[1], nlp[2], u_n)
            end

            if isnan(emax) || emax > 1e4
                if i_am_main(ranks)
                    @warn "Diverged at t=$t_n (eta_max=$emax)"
                end
                break
            end
        end
    end

    if do_vtk
        pvd_path = joinpath(output_dir, "solution")
        createpvd(ranks, pvd_path) do pvd
            _loop(pvd)
        end
    else
        _loop(nothing)
    end

    if i_am_main(ranks)
        print(final_report(step, isempty(diags) ? t0 : diags[end].t,
                               time() - wall0, nl_total, t_solve_tot, n_vtk;
                               tag="[dist]"))
        flush(stdout)
    end
    return diags
end
