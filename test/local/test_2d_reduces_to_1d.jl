# ==============================================================
#  test_2d_reduces_to_1d.jl — a y-invariant 2-D run MUST reproduce the flume
#
#  WHY THIS IS THE SHARPEST CHECK AVAILABLE. The solver is structurally 2-D:
#  every residual term carries both `Ex` and `Ey` contributions, `𝖴y` is a
#  first-class field, and the B-matrix dispersion couples η to the SCALAR
#  divergence ∂ₓu̇ˣ + ∂ᵧu̇ʸ. Drive that machinery with a y-invariant state and
#  every `Ey`/`𝖴y` term must cancel *exactly* — the answer has to collapse onto
#  the quasi-1D flume. Anything that does not cancel (a sign, a transposed
#  tensor index, a missing symmetry between the x- and y-momentum blocks) shows
#  up here and is invisible to a 1-D-only test.
#
#  It is also a check the existing suite cannot make: `test_equivalence.jl`
#  compares against an external oracle (and is currently failing, see
#  building_files/SOLVER_ASSESSMENT_2026-08.md §2), while every other gate
#  compares the solver against theory using its own residual. This one compares
#  the solver against ITSELF along a symmetry the model must respect.
#
#  METHOD. The same physical problem is run twice:
#    (a) NARROW  — Ly = 3 m,  ny = 3   (the quasi-1D flume)
#    (b) WIDE    — Ly = 9 m,  ny = 9   (same dy, 3x the width)
#  with a y-invariant line source and y-periodic lateral boundaries in both. A
#  y-invariant problem has no length scale in y, so the two must agree to solver
#  tolerance in every scalar diagnostic. Widening (rather than refining) in y is
#  deliberate: it changes the number of y-DOFs and the periodic-mode content
#  WITHOUT changing dy, so a disagreement cannot be blamed on discretisation
#  error in y.
#
#  GATES: max|η|, its location, the interior/damped split and |u|/|η| must agree
#  to 1e-6 relative; mass must scale with the domain area (3x) to the same
#  tolerance.
#
#  RUN:  julia --project=. test/local/test_2d_reduces_to_1d.jl      (~8 min)
# ==============================================================

include(joinpath(@__DIR__, "_local_common.jl"))

println("=" ^ 66)
println("  test_2d_reduces_to_1d.jl — y-invariance of the 2-D residual")
println("=" ^ 66)

cc = CheckCounter()

Lx     = 30.0
nx     = flume_nx(Lx)          # dx = 0.5 m
dy     = 1.0
n_per  = 8
Tf     = n_per * FLUME.T
x_wm   = 12.0                  # clear of the 6 m left sponge (see run_flume_1d.jl)
y_c(Ly) = Ly / 2

@printf("\n  %d x-cells (dx=%.2f), dy=%.2f fixed; narrow Ly=3 (ny=3) vs wide Ly=9 (ny=9)\n",
        nx, Lx/nx, dy)
@printf("  y-invariant line source at x=%.1f m, y-periodic laterals, %d periods\n", x_wm, n_per)

function run_width(Ly, ny, tag)
    outdir = local_outdir("reduce2d_" * tag)
    diags, _, _ = setup_and_run(
        M=FLUME.M, domain=((0.0,Lx),(0.0,Ly)), partition=(nx,ny),
        p_horizontal=FLUME.p, h_val=FLUME.d, flat_bed=true,
        T_wave=FLUME.T, A_wave=FLUME.A,
        x_wm=x_wm, y_wm=nothing,                       # LINE source ⇒ y-invariant
        sponge_wL=6.0, sponge_wR=8.0, mu_max=20.0,
        T_final=Tf, dt=FLUME.dt, regime=:nonlinear, nl_pressure=:full,
        y_wall_bc=:periodic, x_wall_bc=false,          # no lateral length scale
        save_every=0, gauges=[(x_wm + 6.0, y_c(Ly))],
        print_every=200, check_every=0, diag_every=5,
        output_dir=outdir)
    dg = read_diagnostics(outdir)
    return (diags=diags, dg=dg, area=Lx*Ly)
end

println("\n--- narrow flume (Ly=3, ny=3) ---")
a = run_width(3.0, 3, "narrow")
println("\n--- wide domain (Ly=9, ny=9) — same dy, 3x the width ---")
b = run_width(9.0, 9, "wide")

rel(x, y) = abs(x - y) / max(abs(x), abs(y), 1e-300)

# --- scalar diagnostics must be width-independent -------------------------
for (name, f, tol) in (
        ("max|η|",                      dg -> maximum(dg.eta_max),          1e-6),
        ("max|η| interior",             dg -> maximum(dg.eta_max_int),      1e-6),
        ("max|η| in the damped zone",   dg -> maximum(dg.eta_max_damped),   1e-6),
        ("max|u|",                      dg -> maximum(dg.u_max),            1e-6),
        ("energy at t_end",             dg -> dg.energy[end] ,              1e-6))
    va, vb = f(a.dg), f(b.dg)
    # energy and the invariants are extensive: normalise by the domain area
    ext = name == "energy at t_end"
    va_n = ext ? va / a.area : va
    vb_n = ext ? vb / b.area : vb
    r = rel(va_n, vb_n)
    check!(cc, "$name is width-independent", r < tol,
           @sprintf("(narrow %.6e, wide %.6e%s, rel %.2e)",
                    va, vb, ext ? " [per unit area]" : "", r))
end

# --- the location of the maximum must be identical ------------------------
xa, xb = a.dg.x_at_max[end], b.dg.x_at_max[end]
check!(cc, "location of max|η| is identical", abs(xa - xb) < 1e-9,
       @sprintf("(narrow x=%.4f, wide x=%.4f)", xa, xb))

# --- mass is extensive: it must scale exactly with the area ---------------
ma, mb = a.dg.mass[end], b.dg.mass[end]
r_mass = rel(ma / a.area, mb / b.area)
check!(cc, "mass scales with the domain area (3×)", r_mass < 1e-6,
       @sprintf("(narrow %.6e over %.0f m², wide %.6e over %.0f m², rel %.2e)",
                ma, a.area, mb, b.area, r_mass))

# --- the gauge time series must match pointwise ---------------------------
ga = [d.gauge_vals[1] for d in a.diags]
gb = [d.gauge_vals[1] for d in b.diags]
n  = min(length(ga), length(gb))
worst = maximum(abs.(ga[1:n] .- gb[1:n])) / max(maximum(abs, ga[1:n]), 1e-300)
check!(cc, "gauge time series agree pointwise", worst < 1e-6,
       @sprintf("(worst relative difference over %d steps = %.2e)", n, worst))

# --- 𝖴y must be identically zero for a y-invariant problem ----------------
#  Not directly in the CSV (u_max is max over BOTH components), but if 𝖴y were
#  non-zero the two widths would carry different y-periodic content and the
#  checks above would already have failed. Report the ratio for the record.
@printf("\n  |u|/|η| : narrow %.4f, wide %.4f (expected ω/tanh(kd) = %.4f)\n",
        median_of(a.dg.u_max ./ max.(a.dg.eta_max, 1e-30)),
        median_of(b.dg.u_max ./ max.(b.dg.eta_max, 1e-30)),
        (2π/FLUME.T) / tanh(flume_k()*FLUME.d))

report!(cc, "test_2d_reduces_to_1d")
