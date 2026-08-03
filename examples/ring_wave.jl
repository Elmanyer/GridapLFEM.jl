# ==============================================================
#  ring_wave.jl — Ring Waves from a Point Source (algebraic solver)
#
#  Port of ../../LFE-M_2D_solver/examples/ring_wave2D.jl to the stacked
#  algebraic package. Gaussian point-source wavemaker at the centre of a
#  square basin; sponge on all four boundaries.
#
#  Expected: circular wavefronts (radial symmetry), amplitude ∝ 1/√r
#  (cylindrical spreading, far field), no significant boundary reflection.
#
#  DEFAULT = QUICK validation size (8 periods, 80×80 cells). Production:
#    L=200, n_side=200, T_final=20*T_wave  → multi-hour.
#
#  RUN:  julia --project=. GridapLFEM.jl/examples/ring_wave.jl
# ==============================================================

if !isdefined(Main, :GridapLFEM)
    include(joinpath(@__DIR__, "..", "src", "GridapLFEM.jl"))
end
using .GridapLFEM
using Printf

println("=" ^ 60)
println("  ring_wave.jl — point-source ring waves (LFE-2, algebraic)")
println("=" ^ 60)

# ── Physical parameters ──────────────────────────────────────
h_val  = 3.5
T_wave = 1.6
A_wave = 0.001
g      = 9.81

# ── Domain (quick default; production: 200 m / 200 cells per side) ─
L      = 80.0
n_side = 80
Lx, Ly = L, L
nx, ny = n_side, n_side

# ── Time stepping (quick default; production: 20 periods) ─────
T_final = 8.0 * T_wave
dt      = T_wave / 30

# ── Point source at centre; 4-side sponge ─────────────────────
x_wm, y_wm = Lx/2, Ly/2
sponge_w   = 10.0
mu_max     = 10.0

# ── Radial gauges along +x from the centre ────────────────────
r_gauges = [5.0, 10.0, 15.0, 20.0]
gauges   = [(x_wm + r, y_wm) for r in r_gauges]
# symmetry checks: same radius along +y and diagonal
push!(gauges, (x_wm, y_wm + 10.0))
push!(gauges, (x_wm + 10.0/sqrt(2.0), y_wm + 10.0/sqrt(2.0)))

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
    wave_gen    = :inner_res,       # interior Gaussian source
    x_wm        = x_wm,
    y_wm        = y_wm,             # point source → ring waves
    sponge_wL   = sponge_w,
    sponge_wR   = sponge_w,
    sponge_wB   = sponge_w,
    sponge_wT   = sponge_w,
    mu_max      = mu_max,
    T_final     = T_final,
    dt          = dt,
    linearised  = true,
    advection   = false,
    save_every  = 30,
    output_dir  = joinpath(@__DIR__, "..", "output", "ring_wave"),
    gauges      = gauges,
)

# ── Post: radial decay + symmetry ─────────────────────────────
emax = maximum(d.eta_max for d in diags)
@printf("\n  max η over run = %.5f m\n", emax)
amps = Float64[]
for (i, r) in enumerate(r_gauges)
    gv = [d.gauge_vals[i] for d in diags if !isempty(d.gauge_vals)]
    n2 = length(gv) ÷ 2
    amp = isempty(gv) ? 0.0 : maximum(abs.(gv[n2:end]))
    push!(amps, amp)
    @printf("  r=%5.1f m:  amplitude ≈ %.6f m\n", r, amp)
end
gv_y  = [d.gauge_vals[end-1] for d in diags if !isempty(d.gauge_vals)]
gv_d  = [d.gauge_vals[end]   for d in diags if !isempty(d.gauge_vals)]
n2 = length(gv_y) ÷ 2
amp_y = maximum(abs.(gv_y[n2:end])); amp_d = maximum(abs.(gv_d[n2:end]))
@printf("  symmetry at r=10:  +x %.6f  +y %.6f  diag %.6f\n", amps[2], amp_y, amp_d)
println("\n  VTK output: GridapLFEM.jl/output/ring_wave/solution.pvd")
