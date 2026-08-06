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
    # --- linear-solver saturation tracking (L1) ------------------------------
    # One sample per solve! call = per RK stage (the ConvergenceLog holds only
    # the LAST linear solve of a Newton sequence), so these are min/max/mean
    # over the step's STAGES, not over its Newton iterations.
    lin_min   :: Int        # smallest sampled Krylov count this step (−1 = none)
    lin_max   :: Int        # largest  sampled Krylov count this step
    lin_sum   :: Int        # Σ sampled counts (for the mean)
    lin_calls :: Int        # number of samples
    lin_cap   :: Int        # configured iteration budget (0 = unknown/direct)
    lin_sat   :: Bool       # any sample reached the cap → the solve was TRUNCATED
    # --- solver descriptions, read back from the CONSTRUCTED objects (B1) ----
    # Set by build_ode_solver/build_ode_solver_distributed from the real solver
    # structs, so a mis-passed argument shows up in the banner instead of the
    # kwargs the caller believed it passed (the ls_maxiter/krylov_m bug).
    nls_desc  :: String
    ls_desc   :: String
end

SolverMonitor() = SolverMonitor(nothing, 0, 0, NaN, NaN, true, -1, 0.0,
                                -1, -1, 0, 0, 0, false, "", "")

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
        k = nls.ls.log.num_iters
        m.lin_iters  = k
        m.lin_min    = m.lin_calls == 0 ? k : min(m.lin_min, k)
        m.lin_max    = max(m.lin_max, k)
        m.lin_sum   += k
        m.lin_calls += 1
        # cap known from the log's own tolerances — no reliance on the kwarg
        cap = m.lin_cap > 0 ? m.lin_cap :
              (hasproperty(nls.ls.log, :tols) ? nls.ls.log.tols.maxiter : 0)
        cap > 0 && k >= cap && (m.lin_sat = true)
    end
    return nothing
end

_monitor_harvest!(m::SolverMonitor, nls, cache) = nothing   # unknown solver: timing only

"""
    take_step_stats!(m::SolverMonitor) → NamedTuple | nothing

Read the statistics accumulated since the previous call and reset the
accumulators. Returns `nothing` when no solve happened in between.

Fields: `ncalls` (solve! calls = RK stages), `nl_iters` (SUMMED over the
stages — divide by `ncalls` for the per-stage count), `res0`, `res`,
`converged`, `lin_iters` (last Krylov count), `lin_min`/`lin_max`/`lin_mean`
(over the step's stages), `lin_cap`, `lin_sat` (a solve hit the iteration
budget and was truncated), `t_solve`.
"""
function take_step_stats!(m::SolverMonitor)
    m.ncalls == 0 && return nothing
    s = (ncalls=m.ncalls, nl_iters=m.nl_iters, res0=m.res0, res=m.res_final,
         converged=m.converged, lin_iters=m.lin_iters, t_solve=m.t_solve,
         lin_min=m.lin_min, lin_max=m.lin_max,
         lin_mean=m.lin_calls > 0 ? m.lin_sum/m.lin_calls : NaN,
         lin_cap=m.lin_cap, lin_sat=m.lin_sat)
    m.ncalls = 0; m.nl_iters = 0; m.res0 = NaN; m.res_final = NaN
    m.converged = true; m.lin_iters = -1; m.t_solve = 0.0
    m.lin_min = -1; m.lin_max = -1; m.lin_sum = 0; m.lin_calls = 0; m.lin_sat = false
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
    ResidualChecker(prob, U, V, trian, dΩh, dt, theta, is_theta)

Reassembles the residual of the governing equations from the accepted time
steps (independently of the ODE solver's own assembly). See
[`check_residuals`](@ref).
"""
struct ResidualChecker
    prob     :: LFEMProblem
    U        :: Any
    V        :: Any
    trian    :: Any
    dΩh       :: Any
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
# true when the multifield trial space carries transient Dirichlet members
# (wave-generation BCs) — sequential and distributed
function _has_transient_trial(U)
    U isa MultiFieldFESpace &&
        return any(s -> s isa Gridap.ODEs.TransientTrialFESpace, U.spaces)
    U isa GridapDistributed.DistributedMultiFieldFESpace &&
        return Gridap.ODEs.has_transient(U)
    return false
end

function check_residuals(chk::ResidualChecker, t_n::Float64, u_n, prev_vals)
    (prev_vals === nothing || chk.dt <= 0) && return nothing
    dt = chk.dt
    vn = get_free_dof_values(u_n)

    # Transient-Dirichlet trials (wave generation BCs): evaluate the trial space
    # at the reassembly time — the Dirichlet values of u are g(t) and those of u̇
    # the analytic ġ(t) (time_derivative space), matching the ODE machinery.
    transient = _has_transient_trial(chk.U)
    Uat(t)  = transient ? Gridap.Arrays.evaluate(chk.U, t) : chk.U
    Udat(t) = transient ?
              Gridap.Arrays.evaluate(Gridap.ODEs.time_derivative(chk.U), t) : chk.U

    vdot = copy(vn)
    vdot .= (vn .- prev_vals) ./ dt                    # u̇ = (u_n − u_{n-1})/Δt
    udot = FEFunction(Udat(t_n), vdot)

    res_th, res_th_inf = NaN, NaN
    if chk.is_theta
        θ   = chk.theta
        vth = copy(vn)
        vth .= θ .* vn .+ (1.0 - θ) .* prev_vals       # u_θ (the state ThetaMethod solved at)
        tth = t_n - dt + θ*dt
        uth = FEFunction(Uat(tth), vth)
        udot_th = transient ? FEFunction(Udat(tth), vdot) : udot
        tu  = Gridap.ODEs.TransientCellField(uth, (udot_th,))
        rv  = assemble_vector(v -> global_residual(tth, tu, v, chk.prob, chk.trian, chk.dΩh),
                              chk.V)
        res_th, res_th_inf = _norm2(rv), _norminf(rv)
    end

    tu2 = Gridap.ODEs.TransientCellField(u_n, (udot,))
    rv2 = assemble_vector(v -> global_residual(t_n, tu2, v, chk.prob, chk.trian, chk.dΩh),
                          chk.V)
    return (res_theta=res_th, res_theta_inf=res_th_inf,
            res_pde=_norm2(rv2), res_pde_inf=_norminf(rv2))
end

# ==============================================================
#  Field diagnostics (M1/E1/E2/I1/D1) — see
#  building_files/LOCAL_VALIDATION_PLAN.md §3
#
#  Everything here answers a question the archived cluster logs could not:
#    * M1  how much memory is this rank using, and is it growing?
#    * E1  WHERE is max|η| — interior wave, or pinned at a boundary node?
#    * E2  is |u|/|η| the ratio of a genuine wave (≈ ω/kd) or of the
#          η-dominated open-boundary mode (≈ 0.4)?
#    * I1  are mass and energy conserved to the level test_conservation.jl
#          proves they are when the run is healthy?
#    * D1  a RELATIVE divergence guard: the absolute 1e4 threshold let a
#          linear A=1e-3 run grow four orders of magnitude over 8 h.
#
#  Distributed-safety: every reduction goes through own_values + a single
#  Float64 MPI reduce. The coordinate and damping-profile node values are
#  interpolated into the SAME multifield space the solution lives in and
#  extracted through the SAME `u[1]` path, so their partition and DOF
#  ordering are identical to η's by construction (a standalone single-field
#  PVector is NOT guaranteed to match a multifield component block).
# ==============================================================

"Current resident-set size of this process [bytes]; 0.0 if unavailable."
function rss_bytes()
    try
        # /proc/self/statm field 2 = resident pages. Page size is 4096 on every
        # platform this runs on; the value is a diagnostic, not an invariant.
        fields = split(read("/proc/self/statm", String))
        return parse(Float64, fields[2]) * 4096.0
    catch
        return Float64(Sys.maxrss())          # peak, not current — better than nothing
    end
end

# --- PVector-safe reductions ------------------------------------------------
# `damped=true` keeps nodes with μ>0 (sponge ∪ relaxation zone), `false` keeps
# the undamped interior. Returns 0.0 when the selected set is empty.
function _local_masked_max(lv, lmu, damped::Bool)
    r = 0.0
    @inbounds for i in eachindex(lv)
        ((lmu[i] > 0.0) == damped) || continue
        a = abs(lv[i]); a > r && (r = a)
    end
    return r
end

masked_max(v::AbstractVector, mu::AbstractVector, damped::Bool) =
    _local_masked_max(v, mu, damped)

function masked_max(v::PVector, mu::PVector, damped::Bool)
    lm = map(own_values(v), own_values(mu)) do lv, lmu
        _local_masked_max(lv, lmu, damped)
    end
    return reduce(max, lm; init=0.0)
end

# x-coordinate of (one of) the node(s) attaining |v| ≈ target. −Inf if none.
function _local_x_at(lv, lx, target)
    r = -Inf
    @inbounds for i in eachindex(lv)
        abs(lv[i]) >= target * (1.0 - 1e-12) && (r = max(r, lx[i]))
    end
    return r
end

x_at_max(v::AbstractVector, xv::AbstractVector, target::Float64) =
    _local_x_at(v, xv, target)

function x_at_max(v::PVector, xv::PVector, target::Float64)
    lm = map(own_values(v), own_values(xv)) do lv, lx
        _local_x_at(lv, lx, target)
    end
    return reduce(max, lm; init=-Inf)
end

"Global max over ranks of a rank-local scalar (identity when sequential)."
global_max_scalar(x::Float64, ranks::Nothing) = x
global_max_scalar(x::Float64, ranks) = reduce(max, map(_ -> x, ranks); init=x)

"""
    RunDiagnostics

Per-run diagnostic state: the node-aligned coordinate/damping arrays, the
conserved-quantity baselines, the divergence limit and the CSV handle.
Built once by [`build_run_diagnostics`](@ref), sampled by
[`field_diagnostics`](@ref).
"""
mutable struct RunDiagnostics
    ranks      :: Any        # nothing (sequential) or the distributed ranks handle
    dΩ         :: Any        # Measure — for the ∫ invariants
    d_cf       :: Any        # still-water depth CellField
    g          :: Float64
    Mv         :: Any        # vertical mass tensor (kinetic energy weight)
    x_vals     :: Any        # x-coordinate per η free DOF (aligned with η)
    mu_vals    :: Any        # total damping μ per η free DOF (sponge + relaxation)
    eta_ref    :: Float64    # reference wave amplitude scale (0 ⇒ unknown)
    div_limit  :: Float64    # abort when max|η| exceeds this
    mass0      :: Float64
    energy0    :: Float64
    rss_peak   :: Float64
    csv        :: Any        # IOStream or nothing
end

"""
    build_run_diagnostics(prob, U0, trian, dΩ; ranks=nothing, eta_ref=0.0,
                              div_factor=20.0, output_dir="", diag_csv=false,
                              is_main=true) → RunDiagnostics

Interpolate the x-coordinate and the total damping profile onto the SAME
multifield space the solution lives in (so the free-DOF ordering of field 1
matches η's exactly), record the t=0 mass/energy baselines, and open the CSV
step log. `eta_ref` is the incident amplitude scale (`A_wave`, the sea state's
`Hs`, or `max|η₀|`); with `eta_ref>0` the divergence guard becomes
`div_factor·eta_ref` instead of the blind absolute 1e4.
"""
function build_run_diagnostics(prob, U0, trian, dΩ;
                                   ranks       = nothing,
                                   eta_ref     :: Float64 = 0.0,
                                   div_factor  :: Float64 = 20.0,
                                   output_dir  :: String  = "",
                                   diag_csv    :: Bool    = false,
                                   is_main     :: Bool    = true,
                                   u0                     = nothing)
    Nσ   = prob.Nσ
    zvv  = VectorValue(ntuple(_ -> 0.0, Nσ)...)
    xh   = interpolate_everywhere([x -> x[1], x -> zvv, x -> zvv], U0)
    # total damping seen by the state: sponge + (optional) relaxation zone
    mufn = prob.relax_bc ? (x -> prob.mu_sponge(x) + prob.relax_mu(x)) : prob.mu_sponge
    muh  = interpolate_everywhere([mufn, x -> zvv, x -> zvv], U0)

    d_cf = CellField(prob.h_bathy, trian)
    div_limit = eta_ref > 0.0 ? div_factor * eta_ref : 1.0e4

    csv = nothing
    if diag_csv && is_main && !isempty(output_dir)
        mkpath(output_dir)
        csv = open(joinpath(output_dir, "diagnostics.csv"), "w")
        println(csv, "step,t,eta_max,x_at_max,eta_max_int,eta_max_damped,u_max,",
                     "mass,mass_drift,energy,energy_ratio,",
                     "nl_iters,nl_stages,res0,res,converged,",
                     "lin_last,lin_min,lin_max,lin_sat,t_solve,rss_mb,rss_peak_mb")
        flush(csv)
    end

    rd = RunDiagnostics(ranks, dΩ, d_cf, prob.g, prob.Mv,
                        get_free_dof_values(xh[1]), get_free_dof_values(muh[1]),
                        eta_ref, div_limit, NaN, NaN, 0.0, csv)
    # Seed the conserved-quantity baselines from the ACTUAL initial state, not
    # from the first sampled step — a forced run has already gained mass by
    # then, which would hide exactly the drift the diagnostic looks for.
    u0 === nothing || field_diagnostics(rd, u0)
    return rd
end

"""
    field_diagnostics(rd, u_n) → NamedTuple

Sample the physical diagnostics of the current state. **Collective** — call on
every rank. Returns `eta_max, x_at_max, eta_int, eta_damped, u_max, ratio,
mass, mass_drift, energy, energy_ratio, rss, rss_peak`.

`ratio = max|u| / max|η|` is the kinematic discriminator: a genuine wave gives
≈ `ω/(kd)`, the η-dominated open-boundary mode ≈ 0.4 (CLAUDE.md §8). `max|u|`
is taken over all σ-modes and both components — an upper bound, not the
surface value, so compare ratios across time rather than to theory alone.
"""
function field_diagnostics(rd::RunDiagnostics, u_n)
    etav = get_free_dof_values(u_n[1])
    emax = _norminf(etav)
    xmax = emax > 0 ? x_at_max(etav, rd.x_vals, emax) : NaN
    eint = masked_max(etav, rd.mu_vals, false)
    edmp = masked_max(etav, rd.mu_vals, true)
    umax = max(_norminf(get_free_dof_values(u_n[2])),
               _norminf(get_free_dof_values(u_n[3])))

    η  = u_n[1]
    H  = rd.d_cf + η
    Ux = u_n[2]; Uy = u_n[3]
    # Constant tensors must go through the Operation helpers (tensors.jl) —
    # a bare `TensorValue ⋅ CellField` is not a CellField operation.
    mass   = sum(∫( η ) * rd.dΩ)
    energy = sum(∫( 0.5*rd.g*η*η +
                    0.5*H*((Ux ⋅ alg_mul(rd.Mv, Ux)) +
                           (Uy ⋅ alg_mul(rd.Mv, Uy))) ) * rd.dΩ)
    isnan(rd.mass0)   && (rd.mass0   = mass)
    isnan(rd.energy0) && (rd.energy0 = energy)

    rss = global_max_scalar(rss_bytes(), rd.ranks)
    rss > rd.rss_peak && (rd.rss_peak = rss)

    return (eta_max = emax, x_at_max = xmax, eta_int = eint, eta_damped = edmp,
            u_max = umax, ratio = emax > 0 ? umax/emax : NaN,
            mass = mass, mass_drift = mass - rd.mass0, energy = energy,
            energy_ratio = rd.energy0 != 0 ? energy/rd.energy0 : NaN,
            rss = rss, rss_peak = rd.rss_peak)
end

"""
    resolve_eta_ref(eta_ref, A_wave, wi, eta0_func, domain) → Float64

Reference free-surface amplitude for the relative divergence guard. Explicit
`eta_ref` wins; otherwise it is inferred from how the run is forced — the sea
state's `Hs` for a `WaveInput`, the peak of a released initial surface, or
`A_wave` for an interior wavemaker. Returns 0.0 only if nothing is known, in
which case the guard falls back to the absolute threshold.
"""
function resolve_eta_ref(eta_ref, A_wave, wi, eta0_func, domain)
    eta_ref === nothing || return Float64(eta_ref)
    if wi !== nothing
        hs = 4.0 * sqrt(sum(abs2, wi.amps) / 2)     # significant wave height
        return max(hs, maximum(wi.amps))
    elseif eta0_func !== nothing
        (x0,x1),(y0,y1) = domain isa Tuple{Tuple,Tuple} ? domain :
                          ((domain[1],domain[2]), (domain[3],domain[4]))
        vals = [abs(eta0_func(VectorValue(x, y)))
                for x in range(x0, x1; length=41), y in range(y0, y1; length=9)]
        return maximum(vals)
    else
        return Float64(A_wave)
    end
end

"Append one CSV row (no-op when the log is disabled or on a non-main rank)."
function diag_csv_row(rd::RunDiagnostics, step::Int, t::Float64, fd, stats)
    rd.csv === nothing && return nothing
    ni  = stats === nothing ? -1  : stats.nl_iters
    nst = stats === nothing ? -1  : stats.ncalls
    r0  = stats === nothing ? NaN : stats.res0
    rf  = stats === nothing ? NaN : stats.res
    cv  = stats === nothing ? 1   : Int(stats.converged)
    ll  = stats === nothing ? -1  : stats.lin_iters
    lmn = stats === nothing ? -1  : stats.lin_min
    lmx = stats === nothing ? -1  : stats.lin_max
    lst = stats === nothing ? 0   : Int(stats.lin_sat)
    ts  = stats === nothing ? NaN : stats.t_solve
    @printf(rd.csv, "%d,%.6f,%.8e,%.4f,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,",
            step, t, fd.eta_max, fd.x_at_max, fd.eta_int, fd.eta_damped,
            fd.u_max, fd.mass, fd.mass_drift, fd.energy, fd.energy_ratio)
    @printf(rd.csv, "%d,%d,%.6e,%.6e,%d,%d,%d,%d,%d,%.4f,%.1f,%.1f\n",
            ni, nst, r0, rf, cv, ll, lmn, lmx, lst, ts,
            fd.rss/1e6, fd.rss_peak/1e6)
    flush(rd.csv)
    return nothing
end

close_diagnostics(rd::RunDiagnostics) = (rd.csv === nothing || close(rd.csv); nothing)
close_diagnostics(::Nothing) = nothing

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
                                 monitor=nothing, diag_every::Int=0,
                                 eta_ref::Float64=NaN, div_limit::Float64=NaN,
                                 io::IO=stdout)
    n_steps = dt > 0 ? round(Int, (T_final - t0)/dt) : 0
    # Prefer the descriptions read back from the CONSTRUCTED solver objects:
    # the caller's kwargs are what it BELIEVES it passed, which is exactly the
    # discrepancy that hid the ls_maxiter/krylov_m bug for months.
    if monitor !== nothing
        isempty(monitor.nls_desc) || (nl_desc = monitor.nls_desc)
        isempty(monitor.ls_desc)  || (ls_desc = monitor.ls_desc)
    end
    println(io, "=== Solver configuration ===")
    println(io, "  Nonlinear solver : ", nl_desc)
    println(io, "  Linear solver    : ", ls_desc)
    tdesc = solver_type == :sdirk     ? "RungeKutta (SDIRK_2_2, fully implicit)" :
            solver_type == :theta     ? @sprintf("ThetaMethod (θ=%.2f)", theta) :
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
    if diag_every > 0
        @printf(io, "  Field diagnostics: every %d step(s) — max|η| location + interior/damped\n",
                diag_every)
        println(io, "                     split, |u|/|η|, mass & energy invariants, rank RSS")
        if isfinite(div_limit)
            if isfinite(eta_ref) && eta_ref > 0
                @printf(io, "  Divergence guard : abort at |η| > %.3e m (= %.0f × η_ref %.3e)\n",
                        div_limit, div_limit/eta_ref, eta_ref)
            else
                @printf(io, "  Divergence guard : abort at |η| > %.3e m (absolute — no η_ref given)\n",
                        div_limit)
            end
        end
    else
        println(io, "  Field diagnostics: disabled (diag_every=0)")
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
                         eta_s::Float64=NaN, tag::String="", fd=nothing)
    io = IOBuffer()
    isempty(tag) || print(io, tag, " ")
    @printf(io, "step %6d  t=%10.4f  eta_max=%.4e", step, t_n, emax)
    if fd !== nothing
        # WHERE the maximum sits, and how it splits interior vs damped zone —
        # the open-boundary mode shows up here long before it shows up in emax.
        isnan(fd.x_at_max) || @printf(io, " @x=%.2f", fd.x_at_max)
        @printf(io, " (int %.2e / dmp %.2e)", fd.eta_int, fd.eta_damped)
    end
    if stats !== nothing
        # nl_iters is SUMMED over the RK stages — report the stage count too so
        # the number is unambiguous.
        @printf(io, "  | NL %2d it", stats.nl_iters)
        stats.ncalls > 1 && @printf(io, " (%d st)", stats.ncalls)
        isnan(stats.res0) || @printf(io, "  r0=%.2e", stats.res0)
        isnan(stats.res)  || @printf(io, " → r=%.2e", stats.res)
        print(io, stats.converged ? "  [conv]" : "  [*** NOT CONVERGED ***]")
        if stats.lin_iters >= 0
            if stats.lin_max > stats.lin_min && stats.lin_min >= 0
                @printf(io, "  gmres %d/%d", stats.lin_min, stats.lin_max)
            else
                @printf(io, "  gmres=%d", stats.lin_iters)
            end
            stats.lin_sat && print(io, " [*** AT CAP ***]")
        end
        @printf(io, "  | %7.3f s", stats.t_solve)
    end
    if fd !== nothing
        # Energy as a RATIO to t=0 for a closed system (the informative form);
        # as an absolute when the run starts from rest and the ratio is undefined.
        @printf(io, " | m%+.1e", fd.mass_drift)
        isnan(fd.energy_ratio) ? @printf(io, " E=%.2e", fd.energy) :
                                 @printf(io, " E%.3f", fd.energy_ratio)
        fd.rss > 0 && @printf(io, " | rss %.0fM", fd.rss/1e6)
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
                          tag::String="", rundiag=nothing)
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
        if rundiag !== nothing && rundiag.rss_peak > 0
            @printf(io, "  peak RSS    : %.0f MB (max over ranks)\n", rundiag.rss_peak/1e6)
        end
    end
    return String(take!(io))
end
