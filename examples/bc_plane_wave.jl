# ==============================================================
#  bc_plane_wave.jl — plane wave from Dirichlet boundary generation
#
#  The `plane_wave.jl` twin with NO interior wavemaker: a regular wave enters
#  through time-varying Dirichlet data (η, 𝖴x) on the LEFT boundary (the
#  discrete-eigenmode :model polarization), propagates down the flume and is
#  absorbed by the right sponge. Hann ramp over 2 periods; cold start.
#
#  RUN:  julia --project=. GridapLFEM.jl/examples/bc_plane_wave.jl
# ==============================================================

if !isdefined(Main, :GridapLFEM)
    include(joinpath(@__DIR__, "..", "src", "GridapLFEM.jl"))
end
using .GridapLFEM
using Printf

println("=" ^ 60)
println("  bc_plane_wave.jl — Dirichlet-generated plane wave (LFE-2)")
println("=" ^ 60)

# ── Physical parameters (Reg06-like linear regime) ────────────
h_val  = 3.5
T_wave = 1.6
A_wave = 0.001
g      = 9.81

# ── Domain (quick default) ────────────────────────────────────
Lx, Ly = 100.0, 10.0
nx, ny = 200, 10

# ── Time stepping (quick default; production: 50+ periods) ────
T_final = 10.0 * T_wave
dt      = T_wave / 30

# ── Wave gauges (y = Ly/2) ────────────────────────────────────
gauges = [(2*4.0, Ly/2), (4*4.0, Ly/2), (8*4.0, Ly/2)]

diags, vert, prob = setup_and_run(
    M           = 2,
    flat_bed    = true,   # flat sea bed (∇h≡0)
    c_bdy       = [0.0, 0.728, 1.0],
    domain      = ((0.0, Lx), (0.0, Ly)),
    partition   = (nx, ny),
    p_horizontal    = 2,
    h_val       = h_val,
    g           = g,
    T_wave      = T_wave,
    A_wave      = A_wave,
    wave_gen    = :bc_gen,  # parametrised boundary wave
    wave_bc     = :regular,         # ← Dirichlet generation (no wavemaker)
    bc_side     = :left,
    bc_profile  = :model,           # discrete LFE-M eigenmode polarization
    sponge_wL   = 0.0,              # nothing may damp the generation boundary
    sponge_wR   = 20.0,
    sponge_wB   = 0.0,
    sponge_wT   = 0.0,
    mu_max      = 5.0,
    T_final     = T_final,
    dt          = dt,
    linearised  = true,             # linear regime benchmark (A = 0.001)
    advection   = false,
    save_every  = 30,               # one VTK snapshot per period
    output_dir  = joinpath(@__DIR__, "..", "output", "bc_plane_wave"),
    gauges      = gauges,
)

# ── Post ──────────────────────────────────────────────────────
emax = maximum(d.eta_max for d in diags)
@printf("\n  max η over run = %.5f m  (A = %.4f m)\n", emax, A_wave)
for (i, gxy) in enumerate(gauges)
    gv = [d.gauge_vals[i] for d in diags if !isempty(d.gauge_vals)]
    n2 = length(gv) ÷ 2
    amp = isempty(gv) ? 0.0 : maximum(abs.(gv[n2:end]))
    @printf("  gauge %d at x=%.1f m:  steady amplitude ≈ %.5f m\n", i, gxy[1], amp)
end
println("\n  VTK output: GridapLFEM.jl/output/bc_plane_wave/solution.pvd")
