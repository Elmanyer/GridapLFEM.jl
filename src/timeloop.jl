# ==============================================================
#  timeloop.jl — ODE solver factory + time loop (stacked layout)
#
#  Port of ../../LFE-M_2D_solver/src/timeloop2D.jl. Only VTK output changes:
#  the stacked VectorValue{Nσ} velocity fields are written per σ-node
#  component with the OLD solver's field names (eta, u1x, u1y, u2x, …), so
#  existing ParaView pipelines work unchanged.
# ==============================================================

"""
    build_ode_solver(dt; solver_type, theta, rho_inf, tableau, nl_iter, nl_tol,
                         show_trace, monitor)

Gridap ODE solver factory (template: GridapSWE.jl `RungeKutta(nls, ls, dt,
tableau)`). Options: `:sdirk` (fully-implicit diagonally-implicit Runge–Kutta,
default — more stable than Crank–Nicolson; tableau `:SDIRK_2_2`, L-stable
2nd-order), `:theta` (Crank–Nicolson θ=0.5, non-dissipative), `:gen_alpha`,
`:rk3` (SDIRK_3_4). Nonlinear solve: Newton + LU. Pass a `SolverMonitor` as
`monitor` to collect per-step convergence statistics (the monitor wraps the
Newton solver transparently; for RK it accumulates over the implicit stages).
"""
function build_ode_solver(dt::Float64;
                              solver_type :: Symbol  = :sdirk,
                              theta       :: Float64 = 0.5,
                              rho_inf     :: Float64 = 0.5,
                              tableau     :: Symbol  = :SDIRK_2_2,
                              nl_iter     :: Int     = 50,
                              nl_tol      :: Float64 = 1e-6,
                              show_trace  :: Bool    = false,
                              monitor                = nothing)
    ls  = LUSolver()
    nls = NLSolver(ls; show_trace=show_trace, method=:newton,
                   iterations=nl_iter, ftol=nl_tol, store_trace=true)
    if monitor !== nothing
        monitor.inner = nls
        nls = monitor
    end
    if solver_type == :sdirk
        return RungeKutta(nls, ls, dt, tableau)
    elseif solver_type == :theta
        return ThetaMethod(nls, dt, theta)
    elseif solver_type == :gen_alpha
        return GeneralizedAlpha(nls, dt, rho_inf)
    elseif solver_type == :rk3
        return RungeKutta(nls, ls, dt, :SDIRK_3_4)
    else
        error("Unknown solver_type; choose :sdirk, :theta, :gen_alpha, or :rk3")
    end
end

"""
    make_initial_conditions(U)                      (sequential-only fast path)
    make_initial_conditions(U, Nσ; eta0_func=nothing)

Initial conditions for the stacked MultiFieldFESpace U. The 2-arg form uses
`interpolate_everywhere` — REQUIRED in distributed mode (`FEFunction(U, zeros)`
creates a plain local array and fails on PVector layouts) and equally valid
sequentially. `eta0_func` (x → η₀) enables initial-condition-release problems
(Gaussian hump, soliton); those REQUIRE `x_wall_bc=true` (closed basin).
"""
function make_initial_conditions(U)
    return FEFunction(U, zeros(Float64, num_free_dofs(U)))
end

function make_initial_conditions(U, Nσ::Int; eta0_func = nothing,
                                     ux0_func = nothing, uy0_func = nothing)
    zvv  = VectorValue(ntuple(_ -> 0.0, Nσ)...)
    eta0 = isnothing(eta0_func) ? (x -> 0.0) : eta0_func
    ux0  = isnothing(ux0_func)  ? (x -> zvv) : ux0_func
    uy0  = isnothing(uy0_func)  ? (x -> zvv) : uy0_func
    return interpolate_everywhere([eta0, ux0, uy0], U)
end

"Extract σ-node component j of a stacked VectorValue{Nσ} CellField (scalar CellField)."
alg_component(Uf, j::Int) = Operation(v -> v[j])(Uf)

"""
    space_at(U, t)

Evaluate a (possibly transient-Dirichlet) trial space at time `t`. Static
spaces pass through unchanged (sequential callable syntax / the distributed
`Arrays.evaluate` no-op). Needed wherever `FEFunction`s are rebuilt from DOF
vectors in wave-generation (transient Dirichlet) runs.
"""
space_at(U, t::Real) = Gridap.Arrays.evaluate(U, t)

"""
    run_time_loop(op, solver, u0, t0, T_final; output_dir, save_every,
                      trian, Nσ, print_every, print_dt, gauges, recon,
                      trial_space, dt, nlp, monitor, checker, check_every,
                      check_tol)

Time loop from t0 to T_final. Returns `[(t, eta_max, gauge_vals, nl_iters,
res_nl, t_solve)]` (the last three are −1/NaN when no `monitor` is given).
`save_every > 0` writes VTK snapshots (fields: eta, u1x, u1y, …, uNσx, uNσy)
plus a `solution.pvd` index. If `recon` (from `build_field_recon`) is given,
the reconstructed `w_s<σ>`/`p_s<σ>` fields are appended to each snapshot
(`trial_space`+`dt` required for the u̇ backward FD of the pressure).

Runtime diagnostics:
  * `monitor` (SolverMonitor, also passed to `build_ode_solver`) —
    per-step Newton iterations, residuals, convergence flag, solve wall time;
  * `print_every` — report line every N steps (default 1 = every step);
    a legacy `print_dt` (simulation seconds) overrides it when given;
  * `checker` (ResidualChecker) + `check_every` — every N steps reassemble
    the governing equations independently and verify ‖R‖∞ ≤ `check_tol`.
Stops early on NaN or eta_max > 1e4.
"""
function run_time_loop(op, solver, u0, t0::Float64, T_final::Float64;
                           output_dir :: String  = "output",
                           save_every :: Int     = 0,
                           trian                 = nothing,
                           Nσ         :: Int     = 1,
                           print_every:: Int     = 1,
                           print_dt              = nothing,   # legacy time-based reporting
                           gauges                = [],
                           recon                 = nothing,
                           trial_space           = nothing,
                           dt         :: Float64 = 0.0,
                           nlp                   = nothing,   # (prob, ctx) for nl_pressure_full
                           monitor               = nothing,   # SolverMonitor
                           checker               = nothing,   # ResidualChecker
                           check_every:: Int     = 0,
                           check_tol  :: Float64 = 1e-8)
    mkpath(output_dir)
    odesol = solve(solver, op, t0, T_final, u0)

    # previous-step DOFs → u̇ backward FD (reconstructed pressure + residual check)
    prev_vals = nothing
    need_prev = recon !== nothing || (checker !== nothing && check_every > 0)

    diags        = NamedTuple{(:t,:eta_max,:gauge_vals,:nl_iters,:res_nl,:t_solve),
                              Tuple{Float64,Float64,Vector{Float64},Int,Float64,Float64}}[]
    t_last_print = t0
    step         = 0
    do_vtk       = !isnothing(trian) && save_every > 0
    steps_total  = dt > 0 ? round(Int, (T_final - t0)/dt) : 0
    wall0        = time()
    nl_total     = 0
    t_solve_tot  = 0.0
    n_vtk        = 0

    gauge_pts = map(g -> (g isa VectorValue ? g : VectorValue(Float64(g[1]), Float64(g[2]))),
                    gauges)

    function _loop(pvd)
        for (t_n, u_n) in odesol
            step  += 1
            stats  = monitor === nothing ? nothing : take_step_stats!(monitor)
            eta_n  = u_n[1]
            emax   = maximum(abs.(get_free_dof_values(eta_n)))
            gauge_data = isempty(gauge_pts) ? Float64[] :
                         [eta_n(pt) for pt in gauge_pts]
            push!(diags, (t=t_n, eta_max=emax, gauge_vals=gauge_data,
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
                wall  = time() - wall0
                eta_s = steps_total > step ? (wall/step)*(steps_total - step) : NaN
                println(step_report(step, t_n, emax, stats; eta_s=eta_s))
                flush(stdout)
                t_last_print = t_n
            end
            if stats !== nothing && !stats.converged
                println("  *** WARNING: nonlinear solver did NOT converge at step $step (t=$t_n) ***")
                flush(stdout)
            end

            # independent verification of the governing equations
            if checker !== nothing && check_every > 0 && step % check_every == 0
                cres = check_residuals(checker, t_n, u_n, prev_vals)
                if cres !== nothing
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
                        # trial space evaluated at t_{n-1}: correct Dirichlet values
                        # for transient-BC runs; static spaces return themselves.
                        u_prev = prev_vals === nothing ? nothing :
                                 FEFunction(space_at(trial_space, t_n - dt), prev_vals)
                        append!(fields, extra_field_cellfields(u_n, u_prev, dt, recon, trian))
                    end
                    pvd[t_n] = createvtk(trian, fname; cellfields=fields, append=false)
                end
                n_vtk += 1
                @printf("  [vtk] snapshot %d written at t=%.4f (%.2f s)\n", n_vtk, t_n, t_vtk)
                flush(stdout)
            end

            if need_prev
                prev_vals = copy(get_free_dof_values(u_n))
            end

            # nl_pressure_full: refresh the frozen projections π𝖲, π𝖻 for the next step
            if nlp !== nothing
                update_nlp_state!(nlp[1], nlp[2], u_n)
            end

            if isnan(emax) || emax > 1e4
                @warn "Diverged at t=$t_n (eta_max=$emax)"
                break
            end
        end
    end

    if do_vtk
        pvd_path = joinpath(output_dir, "solution")
        createpvd(pvd_path) do pvd
            _loop(pvd)
        end
    else
        _loop(nothing)
    end

    print(final_report(step, isempty(diags) ? t0 : diags[end].t,
                           time() - wall0, nl_total, t_solve_tot, n_vtk))
    flush(stdout)
    return diags
end
