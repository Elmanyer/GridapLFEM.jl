# ==============================================================
#  test_reststate_1d.jl — the rest state must stay at rest
#
#  The cheapest possible detector of a broken gravity/pressure balance, and
#  the first thing to run after touching the residual or the diagnostics.
#
#  The gravity term is assembled in the integrated-by-parts ENERGY form
#  −(g/2)(H²−d²)(𝚽·DW) (CLAUDE.md §5): subtracting the still-water baseline
#  makes the discrete rest state exactly force-free, so an undisturbed surface
#  must not drift — over a FLAT bed and, the sharper case, over a SLOPING one,
#  where the bed-slope pressure packages are all active and must cancel.
#
#  Also asserts that the diagnostics themselves are sane at rest: mass drift,
#  energy and the max|η| split must all be at round-off.
#
#  RUN:  julia --project=. test/local/test_reststate_1d.jl        (~1 min)
# ==============================================================

include(joinpath(@__DIR__, "_local_common.jl"))

println("=" ^ 66)
println("  test_reststate_1d.jl — rest state, flat and sloping bed")
println("=" ^ 66)

cc = CheckCounter()

Lx  = 20.0
nx  = flume_nx(Lx)
Tf  = 40 * FLUME.dt                       # 40 steps: long enough for a drift to show

# A submerged bar: the sloping-bed case exercises every ∇h term at once.
d0   = FLUME.d
bar  = x -> d0 - 0.5*1.0*(tanh((x[1]-8.0)/2.0) - tanh((x[1]-14.0)/2.0))

for (name, h_bathy, flat_bed) in (("flat bed",    nothing, true),
                                  ("sloping bed", bar,     false))
    println("\n--- $name ---")
    outdir = local_outdir("reststate_" * replace(name, " " => "_"))
    local diags, _, _ = setup_and_run(
        M=FLUME.M, domain=((0.0,Lx),(0.0,FLUME.Ly)), partition=(nx,FLUME.ny),
        p_horizontal=FLUME.p, h_val=d0, h_bathy=h_bathy, flat_bed=flat_bed,
        T_wave=FLUME.T, A_wave=0.0,                 # NO forcing at all
        x_wm=Lx/2, sponge_wL=0.0, sponge_wR=0.0, mu_max=0.0,
        T_final=Tf, dt=FLUME.dt, regime=:nonlinear, nl_pressure=:none,
        y_wall_bc=:periodic, x_wall_bc=true,        # closed basin: no outflow at all
        save_every=0, gauges=[(Lx/2, FLUME.Ly/2)], print_every=1000, diag_every=1,
        check_every=0, output_dir=outdir,
        eta_ref=1.0e-3,                             # a nominal scale for the guard
    )

    emax = maximum(d.eta_max for d in diags)
    check!(cc, "[$name] surface stays at rest", emax < 1e-12,
           @sprintf("(max|η| = %.2e m over %d steps)", emax, length(diags)))

    dg = read_diagnostics(outdir)
    check!(cc, "[$name] mass drift at round-off",
           maximum(abs, dg.mass_drift) < 1e-12,
           @sprintf("(max |Δmass| = %.2e)", maximum(abs, dg.mass_drift)))
    check!(cc, "[$name] no energy generated",
           maximum(dg.energy) < 1e-18,
           @sprintf("(max E = %.2e J/ρ)", maximum(dg.energy)))
    check!(cc, "[$name] diagnostics CSV complete",
           length(dg.t) == length(diags) && all(dg.converged .== 1),
           @sprintf("(%d rows, all Newton-converged)", length(dg.t)))
end

report!(cc, "test_reststate_1d")
