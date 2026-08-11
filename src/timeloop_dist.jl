# ==============================================================
#  timeloop_dist.jl — Distributed (MPI) mesh, solver stack and time loop
#
#  The distributed counterpart of timeloop.jl. Because the stacked
#  residual/Jacobians are expressed entirely in Gridap CellField algebra
#  (`Operation` is forwarded for `DistributedCellField`), the very same
#  `global_residual`/`jacobian_*` run across MPI ranks unchanged — the full
#  nonlinear physics (advection, slope pressure, P_full, nl_pressure68, flat_bed) is
#  available distributed with no separate code path. This file provides the
#  partitioned mesh builder, the scalable Krylov solver stack, and a rank-aware
#  time loop.
#
#  Conventions the distributed path relies on:
#    * the linear systems are solved with GMRESSolver + Jacobi wrapped in
#      NewtonSolver (GridapSolvers) — a direct LU factorisation does not scale to
#      the partitioned matrices at cluster size;
#    * the global max|η| is reduced from own_values (the ∞-norm of a PVector is
#      not usable here), via reduce(max, …; init=0.0);
#    * all I/O is guarded by i_am_main(ranks); the output directory is created on
#      rank 0 behind an MPI.Barrier;
#    * point gauges are not evaluated (they would need an inter-rank point
#      search), so diags carry (t, eta_max, …);
#    * launch with ~/.julia/bin/mpiexecjl so the MPI build matches the runtime.
# ==============================================================

"""
    build_horizontal_model_distributed(ranks, cpu_grid, domain, partition; y_periodic=false)

Partitioned 2D Cartesian mesh across MPI ranks. `cpu_grid=(px,py)` is the
process topology (`px·py == mpiexec -n`); `domain=(x0,x1,y0,y1)` (flat 4-tuple);
`partition=(nx,ny)` cells (nx divisible by px, ny by py). `y_periodic=true` glues
the top/bottom edges (the `:periodic` lateral BC). Returns `(model, trian)`.
"""
function build_horizontal_model_distributed(ranks, cpu_grid::Tuple,
                                                domain::NTuple{4,Float64},
                                                partition::Tuple;
                                                y_periodic::Bool=false)
    model = CartesianDiscreteModel(ranks, cpu_grid, domain, partition;
                                       isperiodic=(false, y_periodic))
    trian = Triangulation(model)
    return model, trian
end

"""
    build_preconditioner(kind::Symbol; gs_iters=1) → LinearSolver

Right preconditioner for the distributed GMRES. `:jacobi` (default, the
historical choice) is point-diagonal and **weak for this operator** — measured
iteration counts are ~450–700 per solve on the local 2-D cases and ~760 on the
quasi-1D flume, and they are *rank-independent*, which is the signature of a
preconditioner problem rather than a decomposition one.

  * `:jacobi`  — `JacobiLinearSolver()`. Cheapest per iteration, most iterations.
  * `:schwarz` — additive Schwarz with an exact **LU solve on each rank's own
                 block** (`SchwarzLinearSolver(LUSolver())`). Much stronger per
                 iteration and a natural fit to the existing MPI partition; the
                 per-rank factorisation costs memory, so watch RSS.
  * `:gs`      — `SymGaussSeidelSmoother(gs_iters)`. Between the two.

`:schwarz` becomes *more* effective as ranks get bigger blocks, i.e. it rewards
FEWER, FATTER ranks — the opposite of Jacobi's behaviour.
"""
function build_preconditioner(kind::Symbol; gs_iters::Int=1)
    kind === :jacobi  && return JacobiLinearSolver()
    kind === :schwarz && return SchwarzLinearSolver(LUSolver())
    kind === :gs      && return SymGaussSeidelSmoother(gs_iters)
    error("build_preconditioner: kind must be :jacobi, :schwarz or :gs (got :$kind)")
end

"""
    build_ode_solver_distributed(dt; solver_type, theta, tableau, nl_iter,
                                     nl_tol, ls_rtol, ls_maxiter, krylov_m, monitor)

Distributed ODE solver factory (template: GridapSWE.jl). Wraps
NewtonSolver(GMRES + Jacobi preconditioner) — the scalable GridapSWE-pattern
linear stack. Integrators: `:sdirk` (fully-implicit `RungeKutta(nls, ls, dt,
tableau)`, default; `:SDIRK_2_2` = L-stable 2nd-order, more stable than
Crank–Nicolson) and `:theta` (Crank–Nicolson). Pass a `SolverMonitor` as
`monitor` to collect per-step convergence statistics (RK: over the stages).

# GMRES parameters — `krylov_m` and `ls_maxiter` are NOT the same thing
`GMRESSolver`'s first POSITIONAL argument is `m`, the size of the stored Krylov
basis (a *memory* bound), while `maxiter` is a KEYWORD giving the iteration
budget (a *time* bound, library default **100**). Passing `ls_maxiter`
positionally therefore reserved an `ls_maxiter`-vector basis while silently
leaving the iteration cap at 100 — which is the bug this signature fixes:

  * `get_solver_caches` allocates `V = [allocate_in_domain(A) for i in 1:m+1]`
    **and** a dense local `H = zeros(m+1, m)` up front. With `m = 2000` on a
    200×40 Q2 mesh over 32 ranks that is ≈139 MB/rank (108.5 MB basis + 30.5 MB
    Hessenberg) per numerical setup, of which ~96 % could never be reached.
    Churned across stages/steps it ratcheted RSS past the 2 GB/core cgroup limit
    and the runs were OOM-killed mid-integration (exit 137) after ~140 steps.
  * every linear solve stopped at exactly 100 iterations — the `gmres=100`
    pinned on every line of the run logs — so `ls_rtol` was never reached, Newton
    got poor steps and needed 8–10 iterations, at ~25–31 s/step.

`restart=true` bounds the basis at `krylov_m+1` vectors: GMRES(m) restarts every
`krylov_m` iterations and runs up to `ls_maxiter` in total. The alternative
(`restart=false`, the library default) lets the basis GROW past `m` via
`expand_krylov_caches!`, which is exactly the unbounded-memory behaviour to
avoid here. Restarting discards the accumulated basis and so loses GMRES's
global optimality — it can stagnate — but that is not a regression relative to
the previous behaviour: the first `krylov_m` iterations are identical, and
everything beyond them is progress the truncated-at-100 configuration never
made. If the iteration count now sits at the cap, the fix is a stronger
preconditioner (Jacobi is weak for this coupled non-hydrostatic operator), not a
larger basis.
"""
function build_ode_solver_distributed(dt::Float64;
                                          solver_type :: Symbol  = :sdirk,
                                          theta       :: Float64 = 0.5,
                                          tableau     :: Symbol  = :SDIRK_2_2,
                                          nl_iter     :: Int     = 50,
                                          nl_tol      :: Float64 = 1e-5,
                                          ls_rtol     :: Float64 = 1e-6,
                                          ls_maxiter  :: Int     = 1000,
                                          krylov_m    :: Int     = 100,
                                          precond     :: Symbol  = :jacobi,
                                          gs_iters    :: Int     = 1,
                                          monitor                = nothing)
    ls  = GMRESSolver(krylov_m;                  # basis size (memory bound)
                      Pr      = build_preconditioner(precond; gs_iters=gs_iters),
                      restart = true,            # cap the basis; do NOT let it grow
                      maxiter = ls_maxiter,      # iteration budget (time bound)
                      rtol    = ls_rtol,
                      atol    = 1e-14, verbose=false)
    nls = NewtonSolver(ls; maxiter=nl_iter, atol=nl_tol, rtol=1e-10, verbose=false)
    if monitor !== nothing
        monitor.inner = nls
        # Read the configuration back OUT of the constructed solver objects (B1).
        # The pre-2026-08-05 bug was a kwarg the caller believed it had passed
        # (`ls_maxiter`) landing on GMRESSolver's positional `m`; a banner built
        # from the kwargs cannot show that, one built from `ls` can.
        lt = ls.log.tols
        monitor.ls_desc = @sprintf(
            "GMRES(m=%d, restart=%s) + %s preconditioner | max iters = %d | rtol = %.1e, atol = %.1e",
            ls.m, string(ls.restart), string(nameof(typeof(ls.Pr))), lt.maxiter, lt.rtol, lt.atol)
        nt = nls.log.tols
        monitor.nls_desc = @sprintf(
            "NewtonSolver (GridapSolvers, exact hand Jacobians) | max iters = %d | atol (‖r‖₂) = %.1e, rtol = %.1e",
            nt.maxiter, nt.atol, nt.rtol)
        monitor.lin_cap = lt.maxiter          # L1: the cap the saturation flag tests against
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
a `monitor`). VTK: per-σ-node components of the stacked fields under readable
field names (`eta, u1x, u1y, …`), plus the reconstructed `w_s<σ>`/`p_s<σ>`
fields when `recon` is given (from `build_field_recon` — pure CellField
algebra, distributed-transparent).

Runtime diagnostics (printed on rank 0 only; the underlying stats/assemblies
are computed collectively on ALL ranks — do not rank-guard the calls):
  * `monitor` (SolverMonitor) — Newton iterations, residuals, convergence
    flag, last GMRES iteration count, solve wall time per step;
  * `print_every` — report every N steps (default 1); an optional `print_dt`
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
                                check_tol  :: Float64 = 1e-8,
                                rundiag               = nothing,   # RunDiagnostics
                                diag_every :: Int     = 0)
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
            # COLLECTIVE — the sampling condition depends only on `step`, which is
            # identical on every rank, so this must NOT be rank-guarded.
            fd = (rundiag !== nothing && diag_every > 0 && step % diag_every == 0) ?
                 field_diagnostics(rundiag, u_n) : nothing
            if fd !== nothing && i_am_main(ranks)
                diag_csv_row(rundiag, step, t_n, fd, stats)
            end
            if do_print
                if i_am_main(ranks)
                    wall  = time() - wall0
                    eta_s = steps_total > step ? (wall/step)*(steps_total - step) : NaN
                    println(step_report(step, t_n, emax, stats;
                                            eta_s=eta_s, tag="[dist]", fd=fd))
                    flush(stdout)
                end
                t_last_print = t_n
            end
            if stats !== nothing && !stats.converged && i_am_main(ranks)
                println("  *** WARNING: nonlinear solver did NOT converge at step $step (t=$t_n) ***")
                flush(stdout)
            end
            if stats !== nothing && stats.lin_sat && i_am_main(ranks)
                println("  *** WARNING: GMRES hit its iteration cap ($(stats.lin_cap)) at step $step ",
                        "— the solve was TRUNCATED, ls_rtol was never reached ***")
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

            # D1: RELATIVE divergence guard (see the sequential twin).
            div_limit = rundiag === nothing ? 1.0e4 : rundiag.div_limit
            if isnan(emax) || emax > div_limit
                if i_am_main(ranks)
                    @warn "Diverged at t=$t_n (eta_max=$emax, limit=$div_limit)"
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
                               tag="[dist]", rundiag=rundiag))
        close_diagnostics(rundiag)
        flush(stdout)
    end
    return diags
end
