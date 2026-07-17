# ==============================================================
#  monitor.jl — runtime solver instrumentation & diagnostics
#
#  Three tools, shared by the sequential and distributed time loops:
#
#  1. SolverMonitor — a transparent NonlinearSolver wrapper. The ODE
#     machinery (ThetaMethod/GeneralizedAlpha/RungeKutta) calls solve! on it
#     once per stage; the wrapper delegates to the real solver, wall-times
#     the call and harvests the convergence data:
#       * Gridap.Algebra.NLSolver  (sequential Newton+LU): NLsolve result
#         stored in the cache — iterations, residual_norm, f/x_converged,
#         initial residual from the stored trace (requires store_trace=true);
#       * GridapSolvers NewtonSolver (distributed Newton+GMRES+Jacobi):
#         ConvergenceLog — num_iters, residuals[1]/residuals[end],
#         tolerances; last linear-solve GMRES iteration count from ls.log.
#     Stats accumulate across stage calls (RK) and are read+reset once per
#     time step with take_step_stats!.
#
#  2. ResidualChecker — independent verification that the governing
#     equations are satisfied. ThetaMethod solves exactly
#         residual(t_n + θΔt, θ u_{n+1} + (1−θ) u_n, (u_{n+1}−u_n)/Δt) = 0
#     so reassembling that residual from the accepted states must return a
#     norm at the nonlinear tolerance — an end-to-end check of the solve
#     (independent assembly path, catches false convergence). The checker
#     also assembles the instantaneous PDE residual at (t_n, u_n, u̇_FD),
#     whose magnitude is the local time-discretisation error (O(Δt)) — it
#     tracks how well the continuous governing equations hold.
#     All norms are PVector-safe (distributed ∞-norm via own_values —
#     norm(PVector, Inf) is broken in PartitionedArrays 0.3.5).
#
#  3. Formatting helpers — solver-configuration banner, per-step report
#     line, residual-check report line, ETA formatting — so the sequential
#     and distributed loops print identical layouts.
# ==============================================================

"""
    SolverMonitor()

Transparent nonlinear-solver wrapper collecting per-step convergence
statistics. Create empty, pass to `build_ode_solver(...; monitor=...)`
(the factory attaches the real solver), read each step with
[`take_step_stats!`](@ref).
"""
mutable struct SolverMonitor <: Gridap.Algebra.NonlinearSolver
    inner     :: Any        # the wrapped NonlinearSolver (set by the factory)
    ncalls    :: Int        # solve! calls since last take_step_stats! (RK: stages)
    nl_iters  :: Int        # accumulated nonlinear iterations
    res0      :: Float64    # residual before the first Newton iteration
    res_final :: Float64    # residual after the last Newton iteration
    converged :: Bool       # AND over all calls since last read
    lin_iters :: Int        # last linear-solve Krylov iterations (−1 = n/a, direct LU)
    t_solve   :: Float64    # accumulated wall time inside solve! [s]
end

SolverMonitor() = SolverMonitor(nothing, 0, 0, NaN, NaN, true, -1, 0.0)

function Gridap.Algebra.solve!(x::AbstractVector, m::SolverMonitor,
                               op::Gridap.Algebra.NonlinearOperator, cache)
    t0    = time()
    cache = Gridap.Algebra.solve!(x, m.inner, op, cache)
    m.t_solve += time() - t0
    m.ncalls  += 1
    _monitor_harvest!(m, m.inner, cache)
    return cache
end

# NLsolve-backed sequential solver: the result object lives on the cache.
function _monitor_harvest!(m::SolverMonitor, nls::NLSolver, cache)
    r = cache.result
    r === nothing && return nothing
    m.nl_iters += r.iterations
    m.res_final = r.residual_norm                      # ‖r‖∞ (NLsolve convention)
    m.converged &= (r.f_converged || r.x_converged)
    if isnan(m.res0)
        try   # iteration-0 entry of the NLsolve trace (needs store_trace=true)
            isempty(r.trace.states) || (m.res0 = r.trace.states[1].fnorm)
        catch
        end
    end
    return nothing
end

# GridapSolvers NewtonSolver (distributed stack): stats live on the ConvergenceLog.
function _monitor_harvest!(m::SolverMonitor, nls::NewtonSolver, cache)
    log   = nls.log
    niter = log.num_iters
    r0    = log.residuals[1]
    r     = log.residuals[niter + 1]                    # ‖r‖₂ (GridapSolvers convention)
    m.nl_iters += niter
    isnan(m.res0) && (m.res0 = r0)
    m.res_final = r
    r_rel = r0 > 0 ? r / r0 : 0.0
    m.converged &= (r <= log.tols.atol) || (r_rel <= log.tols.rtol)
    if hasproperty(nls.ls, :log)                        # GMRES: last linear solve
        m.lin_iters = nls.ls.log.num_iters
    end
    return nothing
end

_monitor_harvest!(m::SolverMonitor, nls, cache) = nothing   # unknown solver: timing only

"""
    take_step_stats!(m::SolverMonitor) → NamedTuple | nothing

Read the statistics accumulated since the previous call and reset the
accumulators. Returns `nothing` when no solve happened in between.
Fields: `ncalls, nl_iters, res0, res, converged, lin_iters, t_solve`.
"""
function take_step_stats!(m::SolverMonitor)
    m.ncalls == 0 && return nothing
    s = (ncalls=m.ncalls, nl_iters=m.nl_iters, res0=m.res0, res=m.res_final,
         converged=m.converged, lin_iters=m.lin_iters, t_solve=m.t_solve)
    m.ncalls = 0; m.nl_iters = 0; m.res0 = NaN; m.res_final = NaN
    m.converged = true; m.lin_iters = -1; m.t_solve = 0.0
    return s
end

# --------------------------------------------------------------
#  Governing-equation residual verification
# --------------------------------------------------------------

# PVector-safe norms (∞-norm of a PVector must go through own_values).
_norm2(v)   = norm(v)
_norminf(v::AbstractVector) = isempty(v) ? 0.0 : maximum(abs, v)
function _norminf(v::PVector)
    lm = map(own_values(v)) do lv
        isempty(lv) ? 0.0 : maximum(abs, lv)
    end
    return reduce(max, lm; init=0.0)
end

"""
    ResidualChecker(prob, U, V, trian, dO, dt, theta, is_theta)

Reassembles the residual of the governing equations from the accepted time
steps (independently of the ODE solver's own assembly). See
[`check_residuals`](@ref).
"""
struct ResidualChecker
    prob     :: LFEMProblem
    U        :: Any
    V        :: Any
    trian    :: Any
    dO       :: Any
    dt       :: Float64
    theta    :: Float64
    is_theta :: Bool      # θ-scheme verification only valid for ThetaMethod
end

"""
    check_residuals(chk, t_n, u_n, prev_vals) → NamedTuple | nothing

`prev_vals` = free-DOF values of u_{n-1}. Returns
  * `res_theta`, `res_theta_inf` — ‖R‖₂/‖R‖∞ of the θ-scheme discrete system
    at the converged states (should sit at the nonlinear tolerance; NaN for
    non-θ integrators);
  * `res_pde`, `res_pde_inf` — instantaneous PDE residual at (t_n, u_n) with
    u̇ from backward differencing (local time-discretisation error, O(Δt)).
Collective in distributed runs (assembly + norms) — call on ALL ranks.
"""
function check_residuals(chk::ResidualChecker, t_n::Float64, u_n, prev_vals)
    (prev_vals === nothing || chk.dt <= 0) && return nothing
    dt = chk.dt
    vn = get_free_dof_values(u_n)

    vdot = copy(vn)
    vdot .= (vn .- prev_vals) ./ dt                    # u̇ = (u_n − u_{n-1})/Δt
    udot = FEFunction(chk.U, vdot)

    res_th, res_th_inf = NaN, NaN
    if chk.is_theta
        θ   = chk.theta
        vth = copy(vn)
        vth .= θ .* vn .+ (1.0 - θ) .* prev_vals       # u_θ (the state ThetaMethod solved at)
        uth = FEFunction(chk.U, vth)
        tth = t_n - dt + θ*dt
        tu  = Gridap.ODEs.TransientCellField(uth, (udot,))
        rv  = assemble_vector(v -> global_residual(tth, tu, v, chk.prob, chk.trian, chk.dO),
                              chk.V)
        res_th, res_th_inf = _norm2(rv), _norminf(rv)
    end

    tu2 = Gridap.ODEs.TransientCellField(u_n, (udot,))
    rv2 = assemble_vector(v -> global_residual(t_n, tu2, v, chk.prob, chk.trian, chk.dO),
                          chk.V)
    return (res_theta=res_th, res_theta_inf=res_th_inf,
            res_pde=_norm2(rv2), res_pde_inf=_norminf(rv2))
end

# --------------------------------------------------------------
#  Report formatting (shared sequential/distributed)
# --------------------------------------------------------------

"Format a duration in seconds as mm:ss or h:mm:ss."
function fmt_hms(s::Float64)
    (isnan(s) || !isfinite(s)) && return "--:--"
    sec = round(Int, s)
    h, rem = divrem(sec, 3600)
    mn, ss = divrem(rem, 60)
    return h > 0 ? @sprintf("%d:%02d:%02d", h, mn, ss) : @sprintf("%02d:%02d", mn, ss)
end

"""
    print_solver_banner(nl_desc, ls_desc; solver_type, theta, dt, t0, T_final,
                            print_every, print_dt, check_every, check_tol, io=stdout)

Solver-configuration banner printed by the drivers before the time loop.
"""
function print_solver_banner(nl_desc::String, ls_desc::String;
                                 solver_type::Symbol, theta::Float64,
                                 dt::Float64, t0::Float64, T_final::Float64,
                                 print_every::Int=1, print_dt=nothing,
                                 check_every::Int=0, check_tol::Float64=NaN,
                                 io::IO=stdout)
    n_steps = dt > 0 ? round(Int, (T_final - t0)/dt) : 0
    println(io, "=== Solver configuration ===")
    println(io, "  Nonlinear solver : ", nl_desc)
    println(io, "  Linear solver    : ", ls_desc)
    tdesc = solver_type == :theta     ? @sprintf("ThetaMethod (θ=%.2f)", theta) :
            solver_type == :gen_alpha ? "GeneralizedAlpha" : "RungeKutta SDIRK-3"
    @printf(io, "  Time integrator  : %s | dt=%g s | t ∈ [%g, %g] → %d steps\n",
            tdesc, dt, t0, T_final, n_steps)
    pdesc = print_dt === nothing ? @sprintf("every %d step(s)", print_every) :
                                   @sprintf("every %.3g s of simulation time", Float64(print_dt))
    println(io, "  Step reporting   : ", pdesc)
    if check_every > 0
        @printf(io, "  Residual check   : every %d step(s) — reassembles the governing\n", check_every)
        @printf(io, "                     equations; θ-scheme ‖R‖∞ must stay ≤ %.1e\n", check_tol)
    else
        println(io, "  Residual check   : disabled (check_every=0)")
    end
    flush(io)
    return nothing
end

"""
    step_report(step, t_n, emax, stats; eta_s=NaN, tag="") → String

One-line per-step progress report: step index, simulation time, max |η|,
nonlinear convergence data (iterations, initial→final residual, flag, last
GMRES count), nonlinear-solve wall time, run ETA.
"""
function step_report(step::Int, t_n::Float64, emax::Float64, stats;
                         eta_s::Float64=NaN, tag::String="")
    io = IOBuffer()
    isempty(tag) || print(io, tag, " ")
    @printf(io, "step %6d  t=%10.4f  eta_max=%.4e", step, t_n, emax)
    if stats !== nothing
        @printf(io, "  | NL %2d it", stats.nl_iters)
        isnan(stats.res0) || @printf(io, "  r0=%.2e", stats.res0)
        isnan(stats.res)  || @printf(io, " → r=%.2e", stats.res)
        print(io, stats.converged ? "  [conv]" : "  [*** NOT CONVERGED ***]")
        stats.lin_iters >= 0 && @printf(io, "  gmres=%d", stats.lin_iters)
        @printf(io, "  | %7.3f s", stats.t_solve)
    end
    isnan(eta_s) || print(io, "  | ETA ", fmt_hms(eta_s))
    return String(take!(io))
end

"""
    check_report(step, t_n, cres, tol) → String

Report line for a governing-equation residual check (`cres` from
[`check_residuals`](@ref)). The θ-scheme ‖R‖∞ is compared against `tol`:
`OK` = the accepted step satisfies the discrete governing equations.
"""
function check_report(step::Int, t_n::Float64, cres, tol::Float64)
    io = IOBuffer()
    @printf(io, "  [check] step %d  t=%.4f:", step, t_n)
    if !isnan(cres.res_theta)
        verdict = cres.res_theta_inf <= tol ? "OK" : "*** WARN: EQUATIONS NOT SATISFIED ***"
        @printf(io, "  θ-scheme ‖R‖₂=%.3e ‖R‖∞=%.3e %s", cres.res_theta,
                cres.res_theta_inf, verdict)
    end
    @printf(io, "  |  PDE(t_n) ‖R‖₂=%.3e ‖R‖∞=%.3e (time-disc. error)",
            cres.res_pde, cres.res_pde_inf)
    return String(take!(io))
end

"""
    final_report(step, t_final, wall_s, nl_total, t_solve_total, n_vtk; tag="") → String

End-of-run performance summary.
"""
function final_report(step::Int, t_final::Float64, wall_s::Float64,
                          nl_total::Int, t_solve_total::Float64, n_vtk::Int;
                          tag::String="")
    io = IOBuffer()
    isempty(tag) || print(io, tag, " ")
    @printf(io, "Done. %d steps, final t=%.3f\n", step, t_final)
    if step > 0
        @printf(io, "  wall time   : %s  (%.3f s/step average)\n",
                fmt_hms(wall_s), wall_s/step)
        @printf(io, "  solve time  : %s  (%.1f%% of wall)\n", fmt_hms(t_solve_total),
                wall_s > 0 ? 100.0*t_solve_total/wall_s : 0.0)
        @printf(io, "  Newton iters: %d total  (%.2f/step average)\n",
                nl_total, nl_total/step)
        n_vtk > 0 && @printf(io, "  VTK writes  : %d snapshots\n", n_vtk)
    end
    return String(take!(io))
end
