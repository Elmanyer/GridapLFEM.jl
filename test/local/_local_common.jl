# ==============================================================
#  _local_common.jl — shared helpers for the LOCAL validation suite
#
#  The local suite (test/local/) validates the solver's INTERNAL MACHINERY —
#  sponge, relaxation zone, boundary behaviour — on quasi-1D flumes small
#  enough to run in minutes on this machine, so the diagnosis loop no longer
#  goes through multi-hour cluster jobs. See
#  building_files/LOCAL_VALIDATION_PLAN.md §4.
#
#  Quasi-1D flume: a narrow 2-D domain (Ly=3 m, ny=3 — Gridap's minimum in a
#  periodic direction) with y-periodic laterals. The solver is structurally
#  2-D; this is the established way to
#  pose a 1-D horizontal problem in it (test_bc_spectrum.jl does the same) and
#  needs NO solver change. It is NOT a 1-D model: the 𝖴y DOFs exist and are
#  solved, and y-periodic modes are admissible. See the plan §5.1.
#
#  Sequential on purpose: the distributed driver has no point gauges
#  (timeloop_dist.jl:21-22) and every measurement here is gauge-based. Cores are
#  used for process-level concurrency ACROSS test files (run_local_tests.sh),
#  not for MPI within one — measured, a direct LU beats a 2-12 rank GMRES split
#  of this mesh by 2-3x (see building_files/SOLVER_ASSESSMENT_2026-08.md §3).
# ==============================================================

using GridapLFEM
using Printf, LinearAlgebra, DelimitedFiles

# --------------------------------------------------------------
#  PASS/FAIL bookkeeping
# --------------------------------------------------------------
mutable struct CheckCounter
    pass :: Int
    fail :: Int
end
CheckCounter() = CheckCounter(0, 0)

function check!(cc::CheckCounter, name, cond, extra="")
    if cond
        println("  PASS  $name $extra"); cc.pass += 1
    else
        println("  FAIL  $name $extra"); cc.fail += 1
    end
    return cond
end

function report!(cc::CheckCounter, title)
    println()
    println("=" ^ 66)
    @printf("  %s — Results: %d PASS,  %d FAIL\n", title, cc.pass, cc.fail)
    println("=" ^ 66)
    cc.fail > 0 && error("$title: $(cc.fail) failed!")
    return nothing
end

# --------------------------------------------------------------
#  Spectral gauge analysis
#  (window_dft / goda_suzuki are the helpers proven in test_bc_spectrum.jl;
#   that file keeps its own frozen copies — it is a passing reference and is
#   deliberately not refactored onto this module.)
# --------------------------------------------------------------

"Complex DFT coefficient (2/N · Σ v e^{−iωt}) of gauge `idx` over the LAST `Twin` seconds."
function window_dft(diags, idx::Int, omega::Float64, Twin::Float64)
    ts = [d.t for d in diags]
    gv = [d.gauge_vals[idx] for d in diags]
    sel = ts .> (ts[end] - Twin) + 1e-9
    tv = ts[sel]; gw = gv[sel]
    return 2.0 * dot(gw, exp.(-im .* omega .* tv)) / length(gw)
end

window_amplitude(diags, idx::Int, omega::Float64, Twin::Float64) =
    abs(window_dft(diags, idx, omega, Twin))

"""
    goda_suzuki(diags, omega, k, i1, i2, x1, x2, Twin) → (a_I, a_R)

Two-gauge incident/reflected decomposition at frequency ω and wavenumber k,
from gauges `i1`,`i2` at `x1`,`x2`. Invalid when k(x₂−x₁) ≈ nπ — keep the
separation in (0.3, 2.8) rad.
"""
function goda_suzuki(diags, omega::Float64, k::Float64, i1::Int, i2::Int,
                     x1::Float64, x2::Float64, Twin::Float64)
    C1 = window_dft(diags, i1, omega, Twin)
    C2 = window_dft(diags, i2, omega, Twin)
    s  = 2.0 * abs(sin(k * (x2 - x1)))
    aI = abs(exp(im*k*x2)*C1 - exp(im*k*x1)*C2) / s
    aR = abs(exp(-im*k*x2)*C1 - exp(-im*k*x1)*C2) / s
    return aI, aR
end

"Phase (rad) of gauge `idx` at ω over the last `Twin` s."
gauge_phase(diags, idx::Int, omega::Float64, Twin::Float64) =
    angle(window_dft(diags, idx, omega, Twin))

# --------------------------------------------------------------
#  Wave kinematics
# --------------------------------------------------------------

"Airy group velocity at wavenumber k and depth d."
function group_velocity(k::Float64, d::Float64, g::Float64=9.81)
    kd = k * d
    c  = sqrt(g * tanh(kd) / k)
    return 0.5 * c * (1.0 + 2kd / sinh(2kd))
end

"""
    fit_loglinear(x, y) → (slope, intercept, r2)

Least-squares fit of `y = slope·x + intercept` with the coefficient of
determination. Used on `ln a` against the theoretical damping exponent, so
`slope` is the ratio measured/theory and `r2` measures how well the *shape*
of the damping law is reproduced.
"""
function fit_loglinear(x::Vector{Float64}, y::Vector{Float64})
    n  = length(x)
    mx = sum(x)/n; my = sum(y)/n
    sxx = sum((x .- mx).^2); sxy = sum((x .- mx) .* (y .- my))
    slope = sxy / sxx
    inter = my - slope*mx
    yhat  = slope .* x .+ inter
    ssres = sum((y .- yhat).^2); sstot = sum((y .- my).^2)
    r2    = sstot > 0 ? 1.0 - ssres/sstot : 1.0
    return slope, inter, r2
end

"Median of a vector (no Statistics dependency in the local suite)."
median_of(v) = (s = sort(collect(v)); n = length(s);
                n == 0 ? NaN : (isodd(n) ? s[(n+1)÷2] : 0.5*(s[n÷2] + s[n÷2+1])))

"""
    growth_rate(ts, vals) → per-unit-time exponential growth rate of |vals|

Slope of a least-squares fit of ln|vals| against t. Positive ⇒ the field is
growing exponentially; the boundary-mode gate uses this.
"""
function growth_rate(ts::Vector{Float64}, vals::Vector{Float64})
    sel = vals .> 0
    sum(sel) < 3 && return NaN
    s, _, _ = fit_loglinear(ts[sel], log.(vals[sel]))
    return s
end

# --------------------------------------------------------------
#  Diagnostics CSV (written by the solver, monitor.jl :: diag_csv_row)
# --------------------------------------------------------------

"""
    read_diagnostics(dir) → NamedTuple of column vectors

Read `<dir>/diagnostics.csv`. Columns: step, t, eta_max, x_at_max,
eta_max_int, eta_max_damped, u_max, mass, mass_drift, energy, energy_ratio,
nl_iters, nl_stages, res0, res, converged, lin_last, lin_min, lin_max,
lin_sat, t_solve, rss_mb, rss_peak_mb.
"""
function read_diagnostics(dir::String)
    path = joinpath(dir, "diagnostics.csv")
    isfile(path) || error("read_diagnostics: no diagnostics.csv in $dir")
    data, header = readdlm(path, ','; header=true)
    isempty(data) && error("read_diagnostics: $path has no rows — the run's " *
                           "diag_every never fired (it defaults to print_every; " *
                           "pass diag_every explicitly in a test that reads the CSV)")
    names = Symbol.(strip.(vec(header)))
    return NamedTuple{Tuple(names)}(Tuple(Float64.(data[:, j]) for j in 1:length(names)))
end

"""
    tail_window(dg, t_min) → NamedTuple restricted to `t ≥ t_min`

Errors loudly if the window is empty. That happens when a run ABORTED EARLY —
the divergence guard fired and there are no samples in the requested late-time
window — and the failure mode to avoid is a bare `maximum` on an empty vector
several lines later, which reports a `reduce over an empty collection` error
instead of the actual problem (the run diverged).
"""
function tail_window(dg, t_min::Float64)
    sel = dg.t .>= t_min
    count(sel) == 0 && error("tail_window: no samples with t ≥ $t_min " *
        "(the run ended at t = $(isempty(dg.t) ? "—" : dg.t[end]); it almost " *
        "certainly aborted early — check for a divergence-guard warning)")
    return NamedTuple{keys(dg)}(Tuple(v[sel] for v in values(dg)))
end

# --------------------------------------------------------------
#  The shared quasi-1D flume
# --------------------------------------------------------------

# FLUME — reference quasi-1D configuration for the whole local suite, chosen to
# match this machine's measured cost point (≈0.35 s/step at ~4.7k free DOFs, so
# one 8-period run ≈ 112 s). kd = 5.5 is the deep-water regime CLAUDE.md §8
# identifies as the hardest for the sponge.
# Guarded so the file can be re-included in one session (test runner / REPL)
# without a "redefinition of constant" warning.
@isdefined(FLUME) || const FLUME = (
    M         = 2,
    Ly        = 3.0,
    ny        = 3,        # MINIMUM for y_wall_bc=:periodic — Gridap's
                          # CartesianDescriptor asserts "a minimum of 3 elements
                          # is required in any periodic direction"
                          # (Gridap/Geometry/CartesianGrids.jl:39). ny=1 and
                          # ny=2 are rejected outright; measured 2026-08-06.
    dx        = 0.5,      # 8 cells per wavelength with Q2 at kd=5.5
    p         = 2,
    d         = 3.5,
    T         = 1.6,      # ⇒ λ = 4.0 m, kd = 5.5
    dt        = 0.04,     # 40 steps per period
    A         = 0.001,
)

"Number of x-cells for a flume of length Lx at the reference resolution."
flume_nx(Lx::Float64) = round(Int, Lx / FLUME.dx)

"Airy wavenumber of the reference flume wave."
flume_k(; d=FLUME.d, T=FLUME.T, g=9.81) = find_wavenumber(2π/T, d, g)

"Scratch output directory for a local test case (never inside the repo tree by default)."
function local_outdir(name::String)
    root = get(ENV, "LFEM_LOCAL_OUT",
               joinpath(@__DIR__, "..", "..", "output", "local"))
    dir = joinpath(root, name)
    mkpath(dir)
    return dir
end
