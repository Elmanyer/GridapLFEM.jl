# ==============================================================
#  test_mms_convergence.jl — THE VERIFICATION TEST
#
#  Solves the forced problem R − F = 0 with F derived analytically from the
#  governing equations (src/mms.jl, gated by test_mms_forcing.jl) and measures
#  the ORDER OF ACCURACY. This is the only test in the repository that can
#  detect a residual which is self-consistently wrong.
#
#  READ THE RATE, NOT THE ERROR. A wrong term almost never destroys convergence
#  outright — it REDUCES THE ORDER (ValidationTests.tex):
#      slope 3  -> correct for Q2 (p+1)
#      slope 2  -> a term evaluated one derivative too coarsely
#      slope 1  -> a first-order-in-h inconsistency
#      slope 0  -> A GENUINELY WRONG COEFFICIENT (the error stops decreasing
#                  because it is not a discretisation error at all)
#
#  SCOPE: Stage 1 only — linear core on a flat bed. Verifies mass,
#  M-acceleration, gΦ∇η and the d²B dispersion term. It says NOTHING about the
#  nonlinear core, the curved-bed packages or the 𝓝 blocks; those need their own
#  stages (see building_files/MMS_ANALYTIC_PLAN.md §1).
#
#  RUN:  julia --project=. test/test_mms_convergence.jl
#  Cost: a few minutes sequential (small meshes, direct LU).
# ==============================================================

using GridapLFEM
using Printf

println("="^70)
println("  test_mms_convergence.jl — analytic MMS, order of accuracy")
println("="^70)

n_pass = 0; n_fail = 0
function check(name, cond)
    global n_pass, n_fail
    if cond; println("  PASS  $name"); n_pass += 1
    else;    println("  FAIL  $name"); n_fail += 1; end
end

const LX, LY = 1.7, 1.1
const DEPTH  = 1.0
const CASE   = (Lx=LX, Ly=LY, d=DEPTH, M=2, p_horizontal=2)

# ---------------------------------------------------------------------------
#  G6 — SPATIAL order. Q2 ⇒ expect 3 in L².
#  dt is deliberately tiny so the O(Δt²) contribution sits below the smallest
#  spatial error in the sequence; G8 verifies that this actually holds rather
#  than assuming it.
# ---------------------------------------------------------------------------
println("\n--- G6: spatial refinement (expect p ≈ 3) ---")
sp = run_mms_refinement(:space; levels=3, nx0=6, ny0=4, dt0=2e-4,
                        T_final=0.02, CASE...)
check(@sprintf("G6 η spatial order 3 (got %.3f)", sp.p_eta), abs(sp.p_eta - 3.0) < 0.3)
check(@sprintf("G6 u spatial order 3 (got %.3f)", sp.p_u),   abs(sp.p_u   - 3.0) < 0.3)

# ---------------------------------------------------------------------------
#  G7 — TEMPORAL order. SDIRK_2_2 ⇒ expect 2.
#  Mesh fixed and fine so the spatial error is below the temporal one.
# ---------------------------------------------------------------------------
println("\n--- G7: temporal refinement (expect q ≈ 2) ---")
tp = run_mms_refinement(:time; levels=3, nx_fine=24, ny_fine=16, dt0=8e-3,
                        T_final=0.08, CASE...)
check(@sprintf("G7 η temporal order 2 (got %.3f)", tp.p_eta), abs(tp.p_eta - 2.0) < 0.3)
check(@sprintf("G7 u temporal order 2 (got %.3f)", tp.p_u),   abs(tp.p_u   - 2.0) < 0.3)

# ---------------------------------------------------------------------------
#  G8 — the spatial study must not be measuring the temporal error.
#  Halving dt at the FINEST spatial level must barely move the error.
# ---------------------------------------------------------------------------
println("\n--- G8: dt-independence of the spatial study ---")
_, _, rel = mms_dt_independence(; nx=24, ny=16, dt=2e-4, T_final=0.02, CASE...)
check(@sprintf("G8 halving dt changes e_eta by <5%% (got %.2f%%)", 100rel), rel < 0.05)

# ---------------------------------------------------------------------------
#  G9 — with the forcing OFF and a zero initial state the rest state is exact.
#  Confirms the mms_src hook is genuinely inert when unused, so every physical
#  run is bit-identical to before this feature existed.
# ---------------------------------------------------------------------------
println("\n--- G9: forcing off ⇒ exact rest state ---")
vert = assemble_vertical_tensors(2, 1, [0.0, 0.728, 1.0])
diags, _, _ = setup_and_run(M=2, h_val=DEPTH, T_wave=1.6, A_wave=0.0,
                            domain=((0.0,LX),(0.0,LY)), partition=(6,4),
                            p_horizontal=2, x_wm=0.5*LX, y_wm=nothing,
                            sponge_wL=0.0, sponge_wR=0.0, mu_max=0.0,
                            T_final=0.05, dt=1e-2, regime=:linear,
                            save_every=0, print_every=typemax(Int))
emax = maximum(d.eta_max for d in diags)
@printf("    max|η| over the run = %.3e\n", emax)
check("G9 rest state exact with no forcing (max|η| = 0)", emax == 0.0)

println()
println("="^70)
@printf("  Results: %d PASS,  %d FAIL\n", n_pass, n_fail)
@printf("  Measured orders:  space p_eta=%.3f p_u=%.3f  |  time q_eta=%.3f q_u=%.3f\n",
        sp.p_eta, sp.p_u, tp.p_eta, tp.p_u)
println("="^70)
n_fail > 0 ? error("test_mms_convergence: $n_fail failed!") :
             println("  VERIFIED: the Stage-1 linear flat-bed operator converges at the theoretical order.")
