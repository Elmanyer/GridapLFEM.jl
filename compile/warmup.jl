
# ==============================================================
#  warmup.jl — Small long-crested plane wave problem to compile julia module
#
#  Gaussian LINE-source wavemaker → plane waves down a long flume, absorbed by
#  sponge layers at both x-ends. Writes η, per-node u/v components AND the
#  reconstructed vertical velocity (w_s*) and total pressure (p_s*) fields.
#  Runs the FULL nonlinear physics distributed (one Gridap path).
#
# ==============================================================

const ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(ROOT, "src", "GridapLFEM.jl"))

using .GridapLFEM
using Printf

println("=" ^ 60)
println("  warmup.jl — small long-crested plane wave problem (LFE-2)")
println("=" ^ 60)

# ── Physical parameters ──────────────────────────────────────
d_val  = 3.5
T_wave = 1.6
A_wave = 0.001
g      = 9.81

# ── Domain (quick default; production: 200×30 m, 400×15 cells) ─
Lx, Ly = 1.0, 1.0
nx, ny = 2, 2

# ── Time stepping (quick default; production: 50 periods) ─────
T_final = 0.1 * T_wave
dt      = T_wave / 30

# ── Sponge and wavemaker ──────────────────────────────────────
x_wm      = 0.2
sponge_wL = 0.2
sponge_wR = 0.2
mu_max    = 5.0

# ── Wave gauges (y = Ly/2) ────────────────────────────────────
gauges = [(x_wm + 2*4.0, Ly/2), (x_wm + 4*4.0, Ly/2), (x_wm + 8*4.0, Ly/2)]

diags, vert, prob = setup_and_run(
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
    linearised  = false,             # linear regime benchmark (A = 0.001)
    advection   = true,
    save_every  = 0,                # one VTK snapshot per period
    output_dir  = joinpath(@__DIR__, "..", "output", "plane_wave"),
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
println("\n  Warmup done!")