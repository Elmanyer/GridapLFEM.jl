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
#      optimal  -> correct
#      one less -> a term evaluated one derivative too coarsely
#      slope 1  -> a first-order-in-h inconsistency
#      slope 0  -> A GENUINELY WRONG COEFFICIENT (the error stops decreasing
#                  because it is not a discretisation error at all)
#
#  ⚠ THAT LAST READING IS ONLY VALID IF THE STUDY IS PROPERLY ISOLATED — check
#  that FIRST, because the two causes look identical. A refinement study measures
#  the rate of whichever error DOMINATES; if the other discretisation is the
#  larger one, the error stops decreasing and you read slope 0 while the operator
#  is perfectly correct. This is not hypothetical: G7 read 0.024 (η) and 0.195
#  (u) purely because it integrated over 1.7 % of the solution's period, so the
#  temporal error was 25x SMALLER than the spatial floor it was sitting on (see
#  the note on G7). G8 and G10 exist to rule this out in both directions, and
#  they should be believed before any conclusion is drawn from a slope.
#
#  ⚠ THE FE PAIRING IS PART OF THE GATE, NOT AN INCIDENTAL CHOICE.
#  η enters momentum undifferentiated (via ∇·v after IBP), so it plays the role
#  pressure plays in Stokes, and EQUAL-ORDER continuous spaces are inf-sup
#  deficient: Q_p/Q_p converges at p, not p+1, in BOTH fields. This test used to
#  run Q2/Q2 while asserting the theoretical 3 — a gate no correct solver could
#  pass. It now runs the Taylor-Hood-like Q3/Q2 (velocity one order above the
#  surface), the only pairing measured optimal in both fields across the 12-study
#  campaign (building_files/MMS_CONVERGENCE_CAMPAIGN.md):
#      η ∈ Q2 ⇒ optimal 3        u ∈ Q3 ⇒ optimal 4        (DIFFERENT optima)
#  If you lower the pairing back to equal order, lower these expectations to p,
#  or the test measures the FE spaces rather than the residual.
#
#  SCOPE: Model 1 only — linear core on a flat bed. Verifies mass,
#  M-acceleration, gΦ∇η and the d²B dispersion term. It says NOTHING about the
#  nonlinear core, the curved-bed packages or the 𝓝 blocks. Models 2 (linear /
#  variable bed) and 3 (nonlinear / flat bed) are verified by
#  test_mms_convergence_nonlinear.jl and examples/local_mms/run_varbed_study.jl.
#
#  RUN:  julia --project=. test/test_mms_convergence.jl
#  Cost: ~40–60 min sequential (direct LU). It is NOT "a few minutes" as this
#  header used to claim: the Q3/Q2 pairing needed to reach the theoretical order
#  costs roughly 4x the retired Q2/Q2 setup, and G10 adds two more solves.
#  That is the price of a gate that can actually be passed; a cheap gate
#  asserting an unreachable number is worth nothing.
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
#  Q3/Q2 — see the pairing note in the header. p_eta < p_horizontal is what makes
#  the theoretical p+1 reachable in both fields.
const P_U    = 3                     # velocity order  ⇒ optimal u rate P_U+1 = 4
const P_ETA  = 2                     # surface  order  ⇒ optimal η rate P_ETA+1 = 3
const CASE   = (Lx=LX, Ly=LY, d=DEPTH, M=2, p_horizontal=P_U, p_eta=P_ETA)

# ---------------------------------------------------------------------------
#  G6 — SPATIAL order. Q3/Q2 ⇒ expect 3 for η and 4 for u in L².
#  dt is deliberately tiny so the O(Δt²) contribution sits below the smallest
#  spatial error in the sequence; G8 verifies that this actually holds rather
#  than assuming it.
# ---------------------------------------------------------------------------
println("\n--- G6: spatial refinement (expect p_η ≈ 3, p_u ≈ 4) ---")
sp = run_mms_refinement(:space; levels=3, nx0=6, ny0=4, dt0=2e-4,
                        T_final=0.02, CASE...)
check(@sprintf("G6 η spatial order %d (got %.3f)", P_ETA+1, sp.p_eta),
      abs(sp.p_eta - (P_ETA+1)) < 0.3)
check(@sprintf("G6 u spatial order %d (got %.3f)", P_U+1, sp.p_u),
      abs(sp.p_u - (P_U+1)) < 0.3)

# ---------------------------------------------------------------------------
#  G7 — TEMPORAL order. SDIRK_2_2 ⇒ expect 2.
#  Mesh fixed and fine so the spatial error is below the temporal one — and G10
#  below VERIFIES that rather than assuming it.
#
#  ⚠ T_final MUST BE A REAL FRACTION OF THE SOLUTION'S PERIOD.
#  The manufactured field oscillates at omega = 1.3, i.e. a period of 2π/1.3 ≈
#  4.83 s. This gate used to integrate to T_final = 0.08 — **1.7 % of one
#  period** — over which the solution barely changes, so the O(Δt²) error had
#  almost nothing to act on: it came out ~7e-7 against a spatial floor of
#  2.5e-5, and the measured "temporal rate" was really the spatial error sitting
#  still. (The giveaway: e_eta moved 2.543e-5 → 2.468e-5 on halving Δt, and
#  2.48e-5 is EXACTLY G6's finest spatial error.)
#  T_final ≈ 1.2 s covers about a quarter period, which is what makes the
#  temporal error dominant. Refining Δt is pointless if nothing is evolving.
# ---------------------------------------------------------------------------
#  🔴 G7 IS MIS-SPECIFIED AND CURRENTLY FAILS — DIAGNOSED 2026-08-19, NOT YET
#  RE-SPECIFIED. IT IS NOT A SOLVER DEFECT: G6 above reproduces its documented
#  values (2.995 / 3.770) BIT-IDENTICALLY, and the verified-scope table is
#  entirely spatial, so nothing about the model's verification status is affected.
#
#  Measured (all :theta unless stated), fitted slope over the window:
#      T=1.2  nx=24  :sdirk (as gated)   p_eta 1.382   p_u 1.332   <- the failure
#      T=1.2  nx=24  :sdirk  4 levels    p_eta 1.762   p_u 1.252
#      T=1.2  nx=24  :theta              p_eta 2.406   p_u 1.272
#      T=1.2  nx=36  :theta              p_eta 2.424   p_u 1.573
#      T=2.4  nx=24  :theta              p_eta 2.893   p_u 1.068
#      T=2.4  nx=36  dt0=0.15            p_eta 2.653   p_u 2.025
#      T=2.4  nx=36  dt0=0.075           p_eta 1.974   p_u 2.019
#
#  WHAT IS ESTABLISHED. **u is second order in time**, beyond doubt: pairwise
#  rates 1.992 / 2.056 / 2.016 and 2.056 / 2.016 / 1.986 in two independent
#  windows. **eta is second order too**, but its asymptotic window is only ONE
#  refinement wide: it OVER-converges above dt~0.0375 (rate 2.631) and SATURATES
#  on its spatial floor below dt~0.01875 (rate 1.157).
#
#  ⚠ SO THE p_eta=1.974 IN THE LAST ROW IS NOT A CLEAN MEASUREMENT. It hits the
#  target only because the coarse-end over-convergence and the fine-end
#  saturation cancel in the least-squares fit. Do not adopt that window and
#  declare victory — read the RATE SEQUENCE, not the fitted slope.
#
#  WHY eta IS HARDER THAN u HERE, AND IT IS STRUCTURAL: under the Taylor-Hood
#  pairing Q3/Q2 that G6 needs for spatial optimality, eta lives in the LOWER
#  order space, so its spatial error floor is relatively higher and squeezes its
#  temporal window from below. The pairing that optimises the SPATIAL study
#  pessimises the TEMPORAL one for eta. Any re-specification must face this.
#
#  OPTIONS (a decision with a runtime cost — deliberately left open):
#    (1) T=2.4 + nx=36 and gate on the RATE SEQUENCE per field rather than the
#        fitted slope. Correct, but takes G7 from ~20 min to ~1 h.
#    (2) Gate u on temporal order (wide clean window) and drop/relax eta, with
#        the reason recorded.
#    (3) Push eta's spatial floor down with nx>=48. Most expensive.
#  Do NOT "fix" this by widening the +-0.3 tolerance: the window is what is
#  wrong, not the expectation.
#
#  ⚠ AND FIX G10 REGARDLESS. It is the guard that should have caught this and
#  did not: it measured 3.4 % contamination for u against 0.5 % for eta -- a
#  SEVENFOLD asymmetry -- and passed both on one shared 10 % threshold. It
#  already computes the two numbers separately; it just does not act on them
#  separately.
const T7   = 1.2      # ≈ ¼ of the 4.83 s period — enough evolution to measure
const DT7  = 0.15     # ⇒ 8 / 16 / 32 steps over the three levels
println("\n--- G7: temporal refinement (expect q ≈ 2) ---")
tp = run_mms_refinement(:time; levels=3, nx_fine=24, ny_fine=16, dt0=DT7,
                        T_final=T7, CASE...)
check(@sprintf("G7 η temporal order 2 (got %.3f)", tp.p_eta), abs(tp.p_eta - 2.0) < 0.3)
check(@sprintf("G7 u temporal order 2 (got %.3f)", tp.p_u),   abs(tp.p_u   - 2.0) < 0.3)

# ---------------------------------------------------------------------------
#  G10 — the temporal study must not be measuring the SPATIAL error.
#  This is the exact mirror of G8, and its ABSENCE is what let G7 be
#  mis-specified for so long: G8 guarded the spatial study against temporal
#  contamination, but nothing guarded the temporal study against spatial
#  contamination. At the finest Δt, DOUBLING THE MESH must barely move the error.
#  Whenever you fix one of a pair of refinement studies, check that the mirror
#  guard exists — a one-sided guard is a blind spot by construction.
# ---------------------------------------------------------------------------
#  1.5x rather than 2x: at Q3 that already cuts the spatial error by 1.5³ ≈ 3.4,
#  so a spatially-contaminated error would move by tens of percent — a perfectly
#  sharp discriminator — while a 2x refinement would mean a ~98k-DOF direct solve
#  per step purely to sharpen a guard that is already decisive.
println("\n--- G10: mesh-independence of the temporal study ---")
let dt_fine = DT7 / 4
    coarse = run_mms_case(; nx=24, ny=16, dt=dt_fine, T_final=T7, CASE...)
    fine   = run_mms_case(; nx=36, ny=24, dt=dt_fine, T_final=T7, CASE...)
    rel_e  = abs(fine.e_eta - coarse.e_eta) / coarse.e_eta
    rel_u  = abs(fine.e_u   - coarse.e_u)   / coarse.e_u
    @printf("    refining the mesh 1.5x at dt=%.4g:  e_eta %.4e → %.4e (%.1f%%)   e_u %.4e → %.4e (%.1f%%)\n",
            dt_fine, coarse.e_eta, fine.e_eta, 100rel_e, coarse.e_u, fine.e_u, 100rel_u)
    check(@sprintf("G10 η: mesh refinement changes e_eta by <10%% (got %.1f%%)", 100rel_e),
          rel_e < 0.10)
    check(@sprintf("G10 u: mesh refinement changes e_u by <10%% (got %.1f%%)", 100rel_u),
          rel_u < 0.10)
end

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
