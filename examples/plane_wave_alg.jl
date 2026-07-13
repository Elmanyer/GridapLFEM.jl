# ==============================================================
#  plane_wave_alg.jl — Long-Crested Plane Wave (2D flume, algebraic solver)
#
#  Port of ../../LFE-M_2D_solver/examples/plane_wave2D.jl to the stacked
#  algebraic package. Monochromatic plane wave from a Gaussian line source.
#
#  Physical parameters (Reg06 / Yang & Liu 2024 §4.1):
#    d = 3.5 m,  T = 1.6 s,  A = 0.001 m (linear regime, kd = 5.5)
#
#  DEFAULT = QUICK validation size (8 periods, 200×10 cells, ~previously
#  validated configuration). Production run (old solver values):
#    Lx=200, Ly=30, nx=400, ny=15, T_final=50*T_wave  → multi-hour.
#
#  RUN:  julia --project=. LFEM_2D/examples/plane_wave_alg.jl
# ==============================================================

if !isdefined(Main, :LFEModelAlg)
    include(joinpath(@__DIR__, "..", "src", "LFEModelAlg.jl"))
end
using .LFEModelAlg
using Printf

println("=" ^ 60)
println("  plane_wave_alg.jl — long-crested plane wave (LFE-2, algebraic)")
println("=" ^ 60)

# ── Physical parameters ──────────────────────────────────────
d_val  = 3.5
T_wave = 1.6
A_wave = 0.001
g      = 9.81

# ── Domain (quick default; production: 200×30 m, 400×15 cells) ─
Lx, Ly = 100.0, 10.0
nx, ny = 200, 10

# ── Time stepping (quick default; production: 50 periods) ─────
T_final = 8.0 * T_wave
dt      = T_wave / 30

# ── Sponge and wavemaker ──────────────────────────────────────
x_wm      = 15.0
sponge_wL = 15.0
sponge_wR = 20.0
mu_max    = 5.0

# ── Wave gauges (y = Ly/2) ────────────────────────────────────
gauges = [(x_wm + 2*4.0, Ly/2), (x_wm + 4*4.0, Ly/2), (x_wm + 8*4.0, Ly/2)]

diags, vert, prob = setup_and_run_alg(
    M           = 2,
    c_bdy       = [0.0, 0.728, 1.0],
    domain      = ((0.0, Lx), (0.0, Ly)),
    partition   = (nx, ny),
    fe_order    = 2,
    d_val       = d_val,
    g           = g,
    T_wave      = T_wave,
    A_wave      = A_wave,
    x_wm        = x_wm,
    y_wm        = nothing,          # line source → plane wave
    sponge_wL   = sponge_wL,
    sponge_wR   = sponge_wR,
    sponge_wB   = 0.0,
    sponge_wT   = 0.0,
    mu_max      = mu_max,
    T_final     = T_final,
    dt          = dt,
    linearised  = true,             # linear regime benchmark (A = 0.001)
    advection   = false,
    save_every  = 30,               # one VTK snapshot per period
    output_dir  = joinpath(@__DIR__, "..", "output", "plane_wave_alg"),
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
println("\n  VTK output: LFEM_2D/output/plane_wave_alg/solution.pvd")
