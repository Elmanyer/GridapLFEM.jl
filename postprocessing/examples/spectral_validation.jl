# ==============================================================
#  spectral_validation.jl — measured vs target sea-state spectrum
#
#  Postprocessing of a Dirichlet-BC irregular-sea run (produced by
#  examples/bc_irregular_sea.jl): reads the gauge CSV (fast path) and/or the
#  VTK output, estimates the PSD at each gauge with Welch's method, overlays
#  the JONSWAP target density, reports measured vs target Hs, and checks the
#  wave-height statistics against the Rayleigh law of a linear sea.
#
#  RUN (from the repo root, postprocessing env):
#    julia --project=GridapLFEM.jl/postprocessing \
#          GridapLFEM.jl/postprocessing/examples/spectral_validation.jl
# ==============================================================

if !isdefined(Main, :GridapLFEMPost)
    include(joinpath(@__DIR__, "..", "src", "GridapLFEMPost.jl"))
end
using .GridapLFEMPost
using Printf

# target sea state (must match examples/bc_irregular_sea.jl)
Hs_t, Tp_t = 0.002, 1.6

rundir = joinpath(@__DIR__, "..", "..", "output", "bc_irregular_sea")
outdir = joinpath(rundir, "post")
mkpath(outdir)

println("=" ^ 60)
println("  spectral_validation.jl — irregular-sea run analysis")
println("=" ^ 60)

# ── gauge CSV (dense time series — the primary spectral record) ──
csv = joinpath(rundir, "gauges.csv")
isfile(csv) || error("gauge CSV not found: $csv — run examples/bc_irregular_sea.jl first")
tbl = load_csv(csv)
t   = tbl["t"]
gnames = [n for n in tbl.names if n != "t"]
@printf("  gauges: %s   (%d samples, T=%.1f s)\n",
        join(gnames, ", "), length(t), t[end]-t[1])

# steady window: drop the ramp/fill-in transient (first third)
i0 = length(t) ÷ 3
tw = t[i0:end]

for name in gnames
    local v = tbl[name][i0:end]
    local f, S = psd_welch(tw, v; nseg=6)
    local Hs_m = significant_height(f, S)
    @printf("  %s: Hs measured = %.5f m  (target %.5f m, ratio %.3f)\n",
            name, Hs_m, Hs_t, Hs_m/Hs_t)
    local p = plot_sea_spectrum(tw, v; target=(Hs=Hs_t, Tp=Tp_t), nseg=6)
    savefig(p, joinpath(outdir, "spectrum_$(name).png"))
end

# wave-height statistics at the mid-domain gauge (g2 if present)
gmid = "g2" in gnames ? "g2" : gnames[end]
p = plot_exceedance(tw, tbl[gmid][i0:end]; Hs=Hs_t)
savefig(p, joinpath(outdir, "exceedance_$(gmid).png"))

# ── optional: spatial field from the VTK output ────────────────
pvd = joinpath(rundir, "solution.pvd")
if isfile(pvd)
    sim = load_simulation(rundir)
    p = plot_field(sim, "eta"; it=nsnapshots(sim))
    savefig(p, joinpath(outdir, "eta_final.png"))
    p = plot_hovmoller(sim, "eta", (0.0, 5.0), (100.0, 5.0))
    savefig(p, joinpath(outdir, "eta_hovmoller.png"))
    println("  VTK field plots written (eta_final, eta_hovmoller)")
end

println("\n  plots in: $outdir")
