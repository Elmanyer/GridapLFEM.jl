# ==============================================================
#  test_relaxation_1d.jl — the inflow relaxation zone generates the prescribed
#                          wave and absorbs what comes back
#
#  The relaxation zone (utilities.jl :: `relax_bc`, problem.jl :: `relax_mu`
#  / `relax_tg`) blends the state toward the incident field over a strip
#  adjacent to the Dirichlet inflow. It has TWO jobs, and this file gates both:
#
#    GENERATION — inside the zone the solution must BE the prescribed incident
#      wave (right amplitude, right phase gradient). If it is not, every
#      downstream amplitude measurement in the domain is off by that error.
#
#    ABSORPTION — waves travelling back toward the inflow must be absorbed
#      there instead of re-reflecting off the Dirichlet boundary. Without this
#      a reflective domain accumulates energy indefinitely. The test is
#      differential: the SAME closed-end flume is run with and without the
#      zone, and the run without it must accumulate measurably more.
#
#  The incident wave is built here as an explicit `WaveInput` (rather than
#  `wave_bc=:regular`) so the test owns the exact component table it is
#  gating against — amplitude, model wavenumber and ramp all come from `wi`.
#
#  RUN:  julia --project=. test/local/test_relaxation_1d.jl      (~6-9 min)
# ==============================================================

include(joinpath(@__DIR__, "_local_common.jl"))

println("=" ^ 66)
println("  test_relaxation_1d.jl — inflow relaxation zone")
println("=" ^ 66)

cc = CheckCounter()

# ---- the prescribed incident wave --------------------------------------
vert = assemble_vertical_tensors(FLUME.M, 1, [0.0, 0.728, 1.0])
A_in = FLUME.A
wi   = WaveInput(vert; A=A_in, T=FLUME.T, d=FLUME.d, T_ramp=2*FLUME.T,
                 profile=:model)
k    = wi.ks[1]                      # MODEL wavenumber (not Airy) — what the BC imposes
omega = wi.omegas[1]
lam  = 2π / k

@printf("\n  incident wave: A=%.4f m, T=%.2f s, kd=%.2f, λ=%.2f m (model dispersion)\n",
        A_in, FLUME.T, k*FLUME.d, lam)

# =====================================================================
#  PART 1 — GENERATION: inside the zone the state must be the incident wave
# =====================================================================
println("\n--- Part 1: generation inside the relaxation zone ---")

Lx1   = 40.0
nx1   = flume_nx(Lx1)
w_rel = lam                                    # one wavelength, the driver's own default
n_per = 12
Tf1   = n_per * FLUME.T
Twin  = 4 * FLUME.T

# 3 stations inside the zone [0, λ]; a Goda–Suzuki pair just outside it.
x_in   = [0.25*lam, 0.50*lam, 0.75*lam]
x_o1, x_o2 = 2.0*lam, 2.0*lam + 1.0
y_c    = FLUME.Ly / 2
gauges = vcat([(x, y_c) for x in x_in], [(x_o1, y_c), (x_o2, y_c)])
i_o1, i_o2 = 4, 5

out1 = local_outdir("relax_generation")
diags1, _, _ = setup_and_run(
    M=FLUME.M, domain=((0.0,Lx1),(0.0,FLUME.Ly)), partition=(nx1,FLUME.ny),
    p_horizontal=FLUME.p, h_val=FLUME.d, flat_bed=true,
    T_wave=FLUME.T, A_wave=A_in,
    wave_bc=wi, bc_side=:left, relax_bc=true, relax_width=w_rel,
    sponge_wL=0.0, sponge_wR=12.0, mu_max=40.0,
    T_final=Tf1, dt=FLUME.dt, regime=:linear, nl_pressure=:none,
    y_wall_bc=:periodic, x_wall_bc=false,
    save_every=0, gauges=gauges, print_every=100, check_every=0, diag_every=5,
    output_dir=out1)

emax1 = maximum(d.eta_max for d in diags1)
check!(cc, "[gen] run stable", !isnan(emax1) && emax1 < 20*A_in,
       @sprintf("(max|η| = %.2e m)", emax1))

# --- amplitude inside the zone = prescribed incident amplitude -----------
for (j, x) in enumerate(x_in)
    a   = window_amplitude(diags1, j, omega, Twin)
    err = abs(a - A_in) / A_in
    check!(cc, @sprintf("[gen] amplitude at x=%.2f m within 8%% of A_in", x),
           err < 0.08, @sprintf("(a=%.3e vs A=%.3e, err=%.1f%%)", a, A_in, 100err))
end

# --- phase gradient inside the zone = the model wavenumber ---------------
# Absolute phase carries the ramp/start-time offset; the DIFFERENCE between two
# in-zone stations is k·Δx and is convention-free.
ph  = [gauge_phase(diags1, j, omega, Twin) for j in 1:length(x_in)]
dph = mod(ph[3] - ph[1] + π, 2π) - π                 # wrapped to (−π, π]
dph_expect = mod(k*(x_in[3] - x_in[1]) + π, 2π) - π
err_deg = abs(rad2deg(mod(dph - dph_expect + π, 2π) - π))
check!(cc, "[gen] phase gradient inside the zone matches k", err_deg < 10.0,
       @sprintf("(Δφ=%.1f° vs k·Δx=%.1f°, err=%.1f°)",
                rad2deg(dph), rad2deg(dph_expect), err_deg))

# --- the zone does not itself reflect ------------------------------------
aI, aR = goda_suzuki(diags1, omega, k, i_o1, i_o2, x_o1, x_o2, Twin)
check!(cc, "[gen] incident amplitude downstream of the zone within 10% of A_in",
       abs(aI - A_in)/A_in < 0.10, @sprintf("(a_I=%.3e vs %.3e)", aI, A_in))
check!(cc, "[gen] reflection outside the zone < 8%", aR/aI < 0.08,
       @sprintf("(a_R/a_I = %.2f%%)", 100*aR/aI))

# =====================================================================
#  PART 2 — ABSORPTION: the zone must swallow an outgoing wave
# =====================================================================
println("\n--- Part 2: absorption at the inflow (hump release, zero incident wave) ---")

#  DESIGN NOTE — why this is a hump release and not a forced closed-end flume.
#  The first design forced a wave into a flume with a solid far wall and compared
#  the accumulated response with and without the zone, expecting the reflected
#  train to re-reflect off the Dirichlet inflow and fill the basin. MEASURED: it
#  does not. `max|η|` came out 2.19e-3 (zone on) vs 2.46e-3 (off) — 1.12×, against
#  a 1.5× gate — and the late-time energy was within 6 % (off/on = 0.94). The
#  premise was wrong: a Dirichlet boundary PINS the state to the prescribed
#  incident field, which is itself strongly absorbing, so the basin is bounded
#  with or without the zone and the experiment cannot discriminate.
#
#  This design removes the competing mechanism instead. The incident amplitude is
#  set to ~0, so the relaxation target is the still-water state and the zone is a
#  pure absorber; a Gaussian hump is released in the interior, travels to the
#  inflow, and either is swallowed (zone on) or bounces off the Dirichlet wall and
#  keeps ringing (zone off). The discriminator is the ENERGY left in the basin.
Lx2   = 6*lam
nx2   = flume_nx(Lx2)
Tf2   = 20 * FLUME.T
A_eps = 1e-9                          # ≈ no incident wave: the zone's target is rest
wi0   = WaveInput(vert; A=A_eps, T=FLUME.T, d=FLUME.d, T_ramp=0.0, profile=:model)
hump  = x -> 0.001 * exp(-((x[1] - 0.6*Lx2)/1.5)^2)
gauges2 = [(0.35*Lx2, y_c), (0.35*Lx2 + 1.0, y_c)]

energy_case = Dict{Bool,Float64}()
for use_relax in (true, false)
    tag  = use_relax ? "relax_on" : "relax_off"
    out2 = local_outdir("relax_absorption_" * tag)
    local diags2, _, _ = setup_and_run(
        M=FLUME.M, domain=((0.0,Lx2),(0.0,FLUME.Ly)), partition=(nx2,FLUME.ny),
        p_horizontal=FLUME.p, h_val=FLUME.d, flat_bed=true,
        T_wave=FLUME.T, A_wave=A_eps,
        wave_bc=wi0, bc_side=:left, relax_bc=use_relax, relax_width=w_rel,
        sponge_wL=0.0, sponge_wR=0.0, mu_max=40.0,   # NO sponge anywhere: the zone is the only absorber
        T_final=Tf2, dt=FLUME.dt, regime=:linear, nl_pressure=:none,
        y_wall_bc=:periodic, x_wall_bc=true,          # solid right wall
        eta0_func=hump,                               # the energy under test
        save_every=0, gauges=gauges2, print_every=200, check_every=0, diag_every=5,
        output_dir=out2, eta_ref=0.001, div_factor=200.0)

    dg  = read_diagnostics(out2)
    E0  = maximum(dg.energy[dg.t .<= 2.0])            # the released energy
    Ef  = maximum(tail_window(dg, Tf2 - 4*FLUME.T).energy)
    energy_case[use_relax] = Ef / E0
    @printf("      %-9s : E_released = %.3e → E_remaining/E₀ = %.3f\n", tag, E0, Ef/E0)
end

check!(cc, "[abs] the zone absorbs most of an outgoing wave",
       energy_case[true] < 0.25,
       @sprintf("(E_remaining/E₀ = %.3f with the zone)", energy_case[true]))
check!(cc, "[abs] switching the zone OFF leaves measurably more energy in the basin",
       energy_case[false] / energy_case[true] > 2.0,
       @sprintf("(off/on = %.2f×: %.3f vs %.3f)", energy_case[false]/energy_case[true],
                energy_case[false], energy_case[true]))

report!(cc, "test_relaxation_1d")
