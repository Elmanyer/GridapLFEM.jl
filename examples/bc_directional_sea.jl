# ==============================================================
#  bc_directional_sea.jl — short-crested (directional) JONSWAP sea
#
#  JONSWAP spectrum + cosine-power angular spreading: every (ωᵢ, θⱼ) bin is a
#  plane-wave component with its own direction; the Dirichlet data on the left
#  boundary prescribes η, 𝖴x AND 𝖴y (directional seas require
#  y_wall_bc=:open — the lateral boundaries are sponge-absorbed instead of
#  solid walls). The result is a short-crested 2D sea surface.
#
#  RUN:  julia --project=. GridapLFEM.jl/examples/bc_directional_sea.jl
# ==============================================================

using GridapLFEM
using Printf

println("=" ^ 60)
println("  bc_directional_sea.jl — short-crested sea via Dirichlet BCs")
println("=" ^ 60)

# ── Sea state: JONSWAP × cosine-power spreading ───────────────
h_val = 3.5
Hs    = 0.002
Tp    = 1.6
g     = 9.81
seed  = 20260723

spec  = WaveSpec.ContinuousSpectrums.JONSWAP(Hs, Tp)
dspec = WaveSpec.SpectralSpreading.DiscreteSpectralSpreading(
            spec, WaveSpec.SpectralSampling.UniformSampling(),
            1.0/(2.5*Tp), 1.0/(0.55*Tp), 13;
            domain=WaveSpec.SpectralSampling.Energy)
# cosine-power spreading, mean direction 0 (+x), σ_θ = 20°, cut at ±60°
spread = WaveSpec.AngularSpreading.DiscreteAngularSpreading(
            :cosinepow, 0.0, 20.0*pi/180, -pi/3, pi/3, 7)
state  = WaveSpec.AiryWaves.AiryState(dspec, spread, h_val)
state  = WaveSpec.AiryWaves.change_seed!(state, seed)   # reproducible phases

# ── Domain / numerics (quick default) ─────────────────────────
Lx, Ly  = 80.0, 40.0
nx, ny  = 160, 80
T_final = 25.0 * Tp
dt      = Tp / 30
gauges  = [(20.0, Ly/2), (40.0, Ly/2), (40.0, 3*Ly/4)]

diags, vert, prob = setup_and_run(
    M           = 2,
    flat_bed    = true,   # flat sea bed (∇h≡0)
    c_bdy       = [0.0, 0.728, 1.0],
    domain      = ((0.0, Lx), (0.0, Ly)),
    partition   = (nx, ny),
    p_horizontal    = 2,
    h_val       = h_val,
    g           = g,
    T_wave      = Tp,
    A_wave      = Hs/2,
    wave_gen    = :bc_gen,     # WaveSpec AiryState at the boundary
    wave_bc     = state,            # directional AiryState → η, 𝖴x AND 𝖴y BCs
    bc_side     = :left,
    bc_profile  = :model,
    y_wall_bc   = :open,            # REQUIRED for directional inflow
    sponge_wL   = 0.0,
    sponge_wR   = 16.0,
    sponge_wB   = 10.0,             # lateral absorption replaces the walls
    sponge_wT   = 10.0,
    mu_max      = 5.0,
    T_final     = T_final,
    dt          = dt,
    regime      = :linear,          # :linear | :nonlinear  (replaces the retired
                                    #   linearised=/advection= kwarg pair)
    nl_pressure = :none,            # 𝓝 blocks off — meaningless in :linear
    save_every  = 30,
    output_dir  = joinpath(@__DIR__, "..", "output", "bc_directional_sea"),
    gauges      = gauges,
)

emax = maximum(d.eta_max for d in diags)
@printf("\n  max η over run = %.5f m  (target Hs = %.4f m)\n", emax, Hs)
println("  VTK output: GridapLFEM.jl/output/bc_directional_sea/solution.pvd")
println("  (animate with postprocessing: animate_field(sim, \"eta\"))")
