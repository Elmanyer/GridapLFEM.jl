# ==============================================================
#  test_boundary_modes_1d.jl — no spurious growth at the domain boundaries
#
#  WHAT THIS PROTECTS. At a free (`x_wall_bc=false`) outflow the model supports
#  a boundary-localised, η-dominated mode. A velocity-only sponge is
#  structurally blind to it, so it grew unchecked in the archived cluster runs:
#  a linear A=1e-3 plane wave reached 44 m over 8 h, and an irregular sea NaN'd
#  at t=13.8 after 44 h. The `+∫ μ q η` surface damping (`95f5ec6`) is the fix;
#  this file is the gate that keeps it fixed, and it is the primary consumer of
#  the field diagnostics added for it (monitor.jl: E1/E2/I1/D1).
#
#  THE SIGNATURE (CLAUDE.md §8) and the gate each clause becomes:
#    * grows with an e-folding time comparable to T      → G1 growth rate
#    * peaks AT the boundary and decays inward           → G2 damped/interior
#    * the location of max|η| sits on the boundary       → G3 x_at_max
#    * carries large η with little u (|u|/|η| well below → G4 kinematic ratio
#      a genuine wave's)                                    band
#    * mass/energy stop behaving                         → G5 invariants
#
#  A NEGATIVE CONTROL is mandatory here: an under-damped configuration must
#  FAIL the gates. Without it a future regression that silences the detector
#  would leave every gate passing on a broken solver.
#
#  RUN:  julia --project=. test/local/test_boundary_modes_1d.jl   (~15-25 min)
# ==============================================================

include(joinpath(@__DIR__, "_local_common.jl"))

println("=" ^ 66)
println("  test_boundary_modes_1d.jl — boundary stability")
println("=" ^ 66)

cc = CheckCounter()

Lx    = 40.0
nx    = flume_nx(Lx)
n_per = parse(Int, get(ENV, "LFEM_TEST_PERIODS", "18"))
Tf    = n_per * FLUME.T
Twin  = 8 * FLUME.T                 # analysis window: the last 8 periods
k     = flume_k()
omega = 2π / FLUME.T
y_c   = FLUME.Ly / 2

# Kinematic reference for G4. `u_max` from the diagnostics is the largest
# nodal velocity over ALL σ-modes and both components, so the comparison is
# against the SURFACE velocity ratio ω/tanh(kd) — not the depth-averaged
# ω/(kd) quoted in CLAUDE.md §8 for a depth-averaged probe.
ratio_wave = omega / tanh(k*FLUME.d)

@printf("\n  flume %.0f×%.0f m, %d×%d cells | kd=%.2f | %d periods (T_final=%.1f s)\n",
        Lx, FLUME.Ly, nx, FLUME.ny, k*FLUME.d, n_per, Tf)
@printf("  expected |u|/|η| for a genuine wave ≈ ω/tanh(kd) = %.2f 1/s\n", ratio_wave)

"""
Run one configuration and return its late-time diagnostics plus whether the
run survived to T_final (the divergence guard aborts early on blow-up).
"""
function run_case(name; mu_max, x_wall_bc, y_wall_bc, eta0_func=nothing,
                        A_wave=FLUME.A, div_factor=200.0)
    outdir = local_outdir("boundary_" * name)
    diags, _, _ = setup_and_run(
        M=FLUME.M, domain=((0.0,Lx),(0.0,FLUME.Ly)), partition=(nx,FLUME.ny),
        p_horizontal=FLUME.p, h_val=FLUME.d, flat_bed=true,
        T_wave=FLUME.T, A_wave=A_wave, x_wm=Lx/2, y_wm=nothing,
        sponge_wL=10.0, sponge_wR=10.0, mu_max=mu_max,
        T_final=Tf, dt=FLUME.dt, regime=:linear, nl_pressure=:none,
        y_wall_bc=y_wall_bc, x_wall_bc=x_wall_bc, eta0_func=eta0_func,
        save_every=0, gauges=[(Lx/2, y_c)], print_every=200, check_every=0, diag_every=5,
        output_dir=outdir,
        eta_ref=A_wave > 0 ? A_wave : nothing, div_factor=div_factor)
    completed = !isempty(diags) && diags[end].t >= Tf - 1e-9
    dg  = read_diagnostics(outdir)
    return (diags=diags, dg=dg, late=tail_window(dg, Tf - Twin), completed=completed)
end

"""
Evaluate the signature gates; returns the number of VIOLATIONS detected.

`open_x=false` for a CLOSED basin: G3 ("max|η| never sits on the domain edge")
encodes *"the maximum must not be pinned against an absorbing/open boundary"*,
and that has no meaning at a solid wall — a standing wave in a closed basin
puts an antinode exactly ON the wall, so the gate would fail correct physics.
(Measured: the IC-release case reports `x_edge = 0.00 m` while every other
signature clause is healthy.) The remaining four gates still apply.
"""
function gate_case(cc, label, r; open_x::Bool=true)
    n_fail_before = cc.fail
    late = r.late

    if !r.completed
        # The run never reached T_final: the relative divergence guard fired.
        # For a production configuration that is a failure; for the negative
        # control it is the STRONGEST possible detection, so it must count as a
        # violation — returning 0 here would make the control look undetected.
        check!(cc, "[$label] run reached T_final (divergence guard did not fire)",
               false, "(*** aborted early — the divergence guard fired ***)")
        return 1
    end

    rate  = growth_rate(late.t, late.eta_max)
    tol_r = 0.05 / FLUME.T                       # 5 % growth per wave period
    check!(cc, "[$label] G1 no exponential growth", abs(rate) < tol_r,
           @sprintf("(rate = %+.4f /s, limit %.4f = 5%%/T)", rate, tol_r))

    r_dmp = maximum(late.eta_max_damped) / maximum(late.eta_max_int)
    check!(cc, "[$label] G2 maximum is NOT localised in the damped zone", r_dmp < 1.0,
           @sprintf("(damped/interior = %.3f)", r_dmp))

    if open_x
        x_edge = minimum(min.(late.x_at_max, Lx .- late.x_at_max))
        check!(cc, "[$label] G3 max|η| never sits on the domain edge",
               x_edge > 2*FLUME.dx,
               @sprintf("(closest approach to a boundary = %.2f m)", x_edge))
    else
        println("  SKIP  [$label] G3 not applicable to a closed basin " *
                "(a standing wave has an antinode ON the wall)")
    end

    rr = median_of(late.u_max ./ max.(late.eta_max, 1e-30))
    check!(cc, "[$label] G4 |u|/|η| is that of a wave, not of an η-dominated mode",
           0.3*ratio_wave < rr < 3.0*ratio_wave,
           @sprintf("(measured %.2f vs %.2f 1/s expected)", rr, ratio_wave))

    check!(cc, "[$label] G5 mass drift bounded",
           maximum(abs, late.mass_drift) < 1.0,
           @sprintf("(max |Δmass| = %.2e)", maximum(abs, late.mass_drift)))

    return cc.fail - n_fail_before
end

median_of(v) = (s = sort(v); n = length(s);
                isodd(n) ? s[(n+1)÷2] : 0.5*(s[n÷2] + s[n÷2+1]))

# =====================================================================
#  (i) open x-ends with a strong sponge — the configuration that failed
# =====================================================================
println("\n--- (i) open x-ends, μ_max = 40 (the production configuration) ---")
r_open = run_case("open_strong"; mu_max=40.0, x_wall_bc=false, y_wall_bc=:periodic)
gate_case(cc, "open/μ=40", r_open)

# =====================================================================
#  (ii) closed basin, initial-condition release
# =====================================================================
println("\n--- (ii) closed basin, IC release (x_wall_bc=true) ---")
hump = x -> 0.001 * exp(-((x[1] - Lx/2)/2.0)^2)
r_ic = run_case("closed_ic"; mu_max=0.0, x_wall_bc=true, y_wall_bc=:periodic,
                eta0_func=hump, A_wave=0.0)
gate_case(cc, "closed/IC", r_ic; open_x=false)
# A closed, undamped basin is conservative: energy must not grow.
let late = r_ic.late
    check!(cc, "[closed/IC] energy does not grow in a closed undamped basin",
           maximum(late.energy_ratio) < 1.05,
           @sprintf("(max E/E₀ = %.4f)", maximum(late.energy_ratio)))
end

# =====================================================================
#  (iii) lateral BC: solid wall vs periodic (streamwise config fixed)
# =====================================================================
println("\n--- (iii) lateral BC: :wall ---")
r_wall = run_case("open_ywall"; mu_max=40.0, x_wall_bc=false, y_wall_bc=:wall)
gate_case(cc, "open/y-wall", r_wall)

# =====================================================================
#  NEGATIVE CONTROL — an under-damped open boundary MUST be detected
# =====================================================================
println("\n--- (N) negative control: open x-ends, μ_max = 0.05 ---")
println("    (this run is EXPECTED to violate the gates; its PASS/FAIL lines are")
println("     scored on a separate counter and only the verdict is credited)")
r_bad  = run_case("open_underdamped"; mu_max=0.05, x_wall_bc=false,
                  y_wall_bc=:periodic, div_factor=1000.0)
cc_neg = CheckCounter()                       # deliberate failures live here, not in cc
n_bad  = gate_case(cc_neg, "NEGATIVE", r_bad)
check!(cc, "negative control IS detected (≥1 signature gate fires)", n_bad >= 1,
       @sprintf("(%d of the %d gates fired)", n_bad, cc_neg.pass + cc_neg.fail))

report!(cc, "test_boundary_modes_1d")
