# ==============================================================
#  inspect_run.jl — judge a finished (or running) simulation in seconds
#
#  Reads `<output_dir>/diagnostics.csv` — the machine-readable step log the
#  solver writes alongside the human log (monitor.jl :: diag_csv_row) — and
#  answers the questions the archived cluster logs could not:
#
#    * did the run stay bounded, or is |η| growing exponentially?
#    * is the maximum in the INTERIOR (a wave) or pinned in the damped zone
#      at a boundary (the spurious open-boundary mode)?
#    * is the linear solver converging, or silently truncated at its cap?
#    * is memory flat, or ratcheting up toward the cgroup limit?
#    * are mass and energy behaving?
#
#  Deliberately dependency-free (stdlib only) so it runs against a cluster
#  output directory without the postprocessing environment. For fields,
#  spectra and plots use postprocessing/GridapBALFEMPost instead.
#
#  USAGE
#    julia --project=. examples/inspect_run.jl <output_dir> [<output_dir> ...]
#    julia --project=. examples/inspect_run.jl output/local_2d/*        # compare cases
# ==============================================================

using Printf, DelimitedFiles, Statistics

function read_diag(dir::AbstractString)
    path = joinpath(dir, "diagnostics.csv")
    isfile(path) || return nothing
    data, header = readdlm(path, ','; header=true)
    isempty(data) && return nothing
    names = Symbol.(strip.(vec(header)))
    return NamedTuple{Tuple(names)}(Tuple(Float64.(data[:, j]) for j in 1:length(names)))
end

"Least-squares exponential growth rate of a positive series (1/time units)."
function growth_rate(t, v)
    sel = v .> 0
    count(sel) < 3 && return NaN
    x = t[sel]; y = log.(v[sel])
    mx, my = mean(x), mean(y)
    return sum((x .- mx) .* (y .- my)) / sum((x .- mx).^2)
end

verdict(ok) = ok ? "OK  " : "WARN"

function inspect(dir::AbstractString)
    println("\n" * "=" ^ 72)
    println("  ", dir)
    println("=" ^ 72)
    dg = read_diag(dir)
    if dg === nothing
        println("  no usable diagnostics.csv (run with diag_every > 0 and diag_csv=true)")
        return false
    end

    n     = length(dg.t)
    t0, t1 = dg.t[1], dg.t[end]
    late  = dg.t .>= t0 + 0.66*(t1 - t0)          # last third of the run
    ok    = true

    @printf("  %d samples, t ∈ [%.2f, %.2f] s\n\n", n, t0, t1)

    # --- amplitude and where it sits -------------------------------------
    emax, imax = findmax(dg.eta_max)
    if emax == 0
        # An identically-zero surface is a rest-state run: every ratio below is
        # 0/0, so say so once and skip the amplitude gates rather than emit a
        # string of NaN "warnings" for a run that is behaving perfectly.
        println("  max|η|            : 0 — the surface is exactly at rest for the whole run")
    else
        @printf("  max|η|            : %.4e m  at t=%.2f s, x=%.2f m\n",
                emax, dg.t[imax], dg.x_at_max[imax])
    end
    rate = growth_rate(dg.t[late], dg.eta_max[late])
    # NaN = too few positive samples to fit (e.g. a rest state): not a growth signal.
    grow_ok = isnan(rate) || rate <= 0.02
    ok &= grow_ok
    @printf("  [%s] growth rate  : %s over the last third%s\n",
            verdict(grow_ok), isnan(rate) ? "n/a (no signal)" : @sprintf("%+.4f /s", rate),
            grow_ok ? "" : "   ← GROWING")

    r_dmp = maximum(dg.eta_max_damped[late]) / max(maximum(dg.eta_max_int[late]), 1e-300)
    loc_ok = r_dmp < 1.0
    ok &= loc_ok
    @printf("  [%s] localisation : max|η| damped/interior = %.3f%s\n",
            verdict(loc_ok), r_dmp,
            loc_ok ? "" : "   ← the maximum is IN the damped zone (boundary mode?)")

    ratio = median(dg.u_max[late] ./ max.(dg.eta_max[late], 1e-30))
    @printf("        kinematics  : median |u|/|η| = %.3f 1/s (a wave gives ω/tanh(kd);\n", ratio)
    println("                      an η-dominated boundary mode gives much less)")

    # --- invariants -------------------------------------------------------
    md = maximum(abs, dg.mass_drift)
    @printf("        mass drift  : %.3e (max |∫η − ∫η₀|)\n", md)
    if all(isfinite, dg.energy_ratio)
        @printf("        energy      : E/E₀ ∈ [%.4f, %.4f]\n",
                minimum(dg.energy_ratio), maximum(dg.energy_ratio))
    else
        @printf("        energy      : %.3e → %.3e (started from rest, no ratio)\n",
                dg.energy[1], dg.energy[end])
    end

    # --- solver health ----------------------------------------------------
    if any(dg.lin_last .>= 0)
        nsat = count(dg.lin_sat .> 0)
        sat_ok = nsat == 0
        ok &= sat_ok
        @printf("  [%s] GMRES       : %.0f–%.0f iters (mean %.0f), %d/%d samples AT THE CAP%s\n",
                verdict(sat_ok), minimum(dg.lin_min[dg.lin_min .>= 0]),
                maximum(dg.lin_max), mean(dg.lin_last[dg.lin_last .>= 0]), nsat, n,
                sat_ok ? "" : "   ← truncated solves, ls_rtol never reached")
    else
        println("        linear solve: direct (sequential LU) — no iteration count")
    end

    nconv = count(dg.converged .< 1)
    conv_ok = nconv == 0
    ok &= conv_ok
    @printf("  [%s] Newton       : %.1f it/step over %.0f stage(s), %d non-converged step(s)\n",
            verdict(conv_ok), mean(dg.nl_iters), median(dg.nl_stages), nconv)

    # --- memory -----------------------------------------------------------
    if any(dg.rss_mb .> 0)
        drift = dg.rss_mb[end] - dg.rss_mb[1]
        mem_ok = drift < 0.25 * dg.rss_mb[1]
        ok &= mem_ok
        @printf("  [%s] memory      : %.0f → %.0f MB (peak %.0f), drift %+.0f MB%s\n",
                verdict(mem_ok), dg.rss_mb[1], dg.rss_mb[end], maximum(dg.rss_peak_mb),
                drift, mem_ok ? "" : "   ← RATCHETING (a leak, or caches churning)")
    end

    println()
    println(ok ? "  VERDICT: healthy" : "  VERDICT: needs attention (see the WARN lines)")
    return ok
end

dirs = isempty(ARGS) ? String[] : ARGS
if isempty(dirs)
    println("usage: julia --project=. examples/inspect_run.jl <output_dir> [...]")
    exit(2)
end
all_ok = true
for d in dirs
    isdir(d) || (println("\nskipping (not a directory): $d"); continue)
    global all_ok &= inspect(d)
end
exit(all_ok ? 0 : 1)
