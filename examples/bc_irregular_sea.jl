# ==============================================================
#  bc_irregular_sea.jl — long-crested JONSWAP sea from Dirichlet BCs
#
#  THE WaveSpec.jl coupling showcase: a stochastic sea state (JONSWAP
#  spectrum, energy-domain sampling, unidirectional) is synthesised by
#  WaveSpec, converted to a WaveInput component table (seeded phases —
#  reproducible), prescribed as Dirichlet data on the left boundary, and
#  propagated through the flume by the LFE model. Gauge records are dumped to
#  CSV for spectral validation with the postprocessing library
#  (postprocessing/examples/spectral_validation.jl).
#
#  Linear regime: Hs = 2 mm (component amplitudes ≲ the A ≤ 0.001 m rule).
#
#  Interpreting the QUICK default (30 Tp): only the near-inflow gauge sees a
#  fully developed sea (measured Hs ratio 0.979 on 2026-07-23); the mid/far
#  gauges are still filling in (group-speed arrival ~32 s / >48 s) — that is
#  a short-run artifact, not attenuation. Production runs (200+ Tp) develop
#  the whole domain and sharpen the Welch peak (Δf ~ 1/window).
#
#  RUN:  julia --project=. GridapBALFEM.jl/examples/bc_irregular_sea.jl
# ==============================================================

using GridapBALFEM
using Printf
using DelimitedFiles

println("=" ^ 60)
println("  bc_irregular_sea.jl — JONSWAP sea via Dirichlet generation")
println("=" ^ 60)

# ── Sea state ─────────────────────────────────────────────────
h_val = 3.5
Hs    = 0.002                       # 2 mm — linear regime
Tp    = 1.6
g     = 9.81
nfreq = 21                          # 20 spectral bins
seed  = 20260723                    # reproducible realisation

spec  = WaveSpec.ContinuousSpectrums.JONSWAP(Hs, Tp)
dspec = WaveSpec.SpectralSpreading.DiscreteSpectralSpreading(
            spec, WaveSpec.SpectralSampling.UniformSampling(),
            1.0/(2.5*Tp), 1.0/(0.55*Tp), nfreq;
            domain=WaveSpec.SpectralSampling.Energy)   # equal-energy bins
spread = WaveSpec.AngularSpreading.DiscreteAngularSpreading(0.0; seed=seed)
state  = WaveSpec.AiryWaves.AiryState(dspec, spread, h_val)
state  = WaveSpec.AiryWaves.change_seed!(state, seed)   # reproducible phases

# ── Domain / numerics (quick default; production: 200+ Tp, 200×10 mesh) ──
Lx, Ly  = 100.0, 6.0
nx, ny  = 200, 4
T_final = 30.0 * Tp
dt      = Tp / 30
gauges  = [(8.0, Ly/2), (40.0, Ly/2), (64.0, Ly/2)]

diags, vert, prob = setup_and_run(
    M           = 2,
    flat_bed    = true,   # flat sea bed (∇h≡0)
    c_bdy       = [0.0, 0.728, 1.0],
    domain      = ((0.0, Lx), (0.0, Ly)),
    partition   = (nx, ny),
    p_horizontal    = 2,
    h_val       = h_val,
    g           = g,
    T_wave      = Tp,               # reporting only (kd banner)
    A_wave      = Hs/2,
    wave_gen    = :bc_gen,     # WaveSpec AiryState at the boundary
    wave_bc     = state,            # ← WaveSpec AiryState → Dirichlet data
    bc_side     = :left,
    bc_profile  = :model,
    sponge_wL   = 0.0,
    sponge_wR   = 20.0,
    sponge_wB   = 0.0,
    sponge_wT   = 0.0,
    mu_max      = 5.0,
    T_final     = T_final,
    dt          = dt,
    regime      = :linear,          # :linear | :nonlinear  (replaces the retired
                                    #   linearised=/advection= kwarg pair)
    nl_pressure = :none,            # 𝓝 blocks off — meaningless in :linear
    save_every  = 60,
    output_dir  = joinpath(@__DIR__, "..", "output", "bc_irregular_sea"),
    gauges      = gauges,
)

# ── Gauge CSV dump (postprocessing input) ─────────────────────
outdir = joinpath(@__DIR__, "..", "output", "bc_irregular_sea")
mkpath(outdir)
csv = joinpath(outdir, "gauges.csv")
open(csv, "w") do io
    println(io, "t," * join(["g$(i)" for i in 1:length(gauges)], ","))
    for d in diags
        isempty(d.gauge_vals) && continue
        println(io, string(d.t) * "," * join(string.(d.gauge_vals), ","))
    end
end

emax  = maximum(d.eta_max for d in diags)
g2    = [d.gauge_vals[2] for d in diags if !isempty(d.gauge_vals)]
n2    = length(g2) ÷ 2
Hs_g2 = 4.0 * sqrt(sum(abs2, g2[n2:end] .- sum(g2[n2:end])/length(g2[n2:end])) /
                   length(g2[n2:end]))
@printf("\n  max η over run     = %.5f m   (target Hs = %.4f m)\n", emax, Hs)
@printf("  Hs (4σ) at gauge 2 = %.5f m   (steady half-window)\n", Hs_g2)
println("  gauge CSV : $csv")
println("  VTK output: GridapBALFEM.jl/output/bc_irregular_sea/solution.pvd")
println("  → spectral validation: postprocessing/examples/spectral_validation.jl")
