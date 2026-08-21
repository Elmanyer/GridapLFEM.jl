# ==============================================================
#  test_mms_convergence_nonlinear.jl — ORDER OF ACCURACY for the nonlinear
#  analytic-MMS models (the verification proper)
#
#  Specification: ValidationTests.tex §subsec: mms model3 / §subsec: mms model4
#                 and §subsubsec: mms measure.
#  Plan:          building_files/MMS_NONLINEAR_PLAN.md §2.3.
#
#  WHAT THIS CERTIFIES, AND WHY IT IS DIFFERENT FROM test_selfconsistency.jl.
#  The forcing is derived from the GOVERNING EQUATIONS (src/mms.jl), never from
#  the residual, so a wrong term in problem.jl does NOT cancel — it shows up as a
#  reduced order of accuracy. Reaching the theoretical order is therefore code
#  verification in the sense of Roache, not self-consistency.
#
#  PAIRING. Q3/Q2, i.e. velocity one order above the surface. Equal order costs
#  one order in u (measured); on Q_p/Q_{p−1} both fields reach their optimum:
#      p_η → p_e+1 = 3      p_u → p_u+1 = 4
#
#  TOLERANCE. nl_tol = 1e-9, NOT 1e-12. The nonlinear Jacobians are deliberately
#  quasi-Newton (the pressure blocks' η-dependence is frozen — see the coverage
#  note in problem.jl), so Newton converges linearly and stalls around 1e-10.
#  1e-9 sits two to three orders below the smallest discretisation error measured
#  here, so it cannot affect the rate.
#
#  NEVER loosen nl_tol to make a study pass: the algebraic error would then sit
#  INSIDE the discretisation error being measured and the rate would be
#  meaningless.
#
#  ⚠ AND DO NOT REACH FOR ITERATION BUDGET EITHER, WITHOUT FIRST ASKING WHICH
#  KIND OF NON-CONVERGENCE YOU HAVE. A Newton solve can fail two ways that look
#  identical in the log: converging SLOWLY (a higher-order term missing from the
#  Jacobian — more iterations fix it) or converging to the WRONG FIXED POINT (an
#  O(1) term missing — no budget ever fixes it). This file briefly carried
#  nl_iter=400 for Model 4 on the first reading; the truth was the second, and
#  the real fix was to assemble the 𝓐/𝓚 package in ∂R/∂u̇ (2026-08-17).
#  test_jacobians_ad.jl distinguishes the two directly, by measuring how the
#  hand↔AD gap scales with state amplitude: vanishing ⇒ slow, flat ⇒ wrong.
#
#  RUNTIME. ~25–30 min per :none study (3 levels). The :native study is
#  SUBSTANTIALLY slower — the 𝓝 forcing costs ~4x a :none evaluation (measured
#  ~25 vs ~100-150 µs/point), because Ψ carries an Nσ²x8 component sum that the
#  outer gradient then differentiates. Budget ~1 h for it. Set MMS_NL_LEVELS to
#  shorten.
#
#  THE VERTICAL BASIS IS A PARAMETER: MMS_M (elements) and MMS_PVERT (order),
#  defaulting to the P1LFE-2 every study in this repository has ever run. The
#  rates asserted below are properties of the HORIZONTAL discretisation, so they
#  must hold for ANY vertical basis — overriding these is how the basis-agnosticism
#  the model is named for gets tested (building_files/PENDING_TASKS.md §1).
#  ⚠ The forcing cost scales as Nσ²; P1LFE-4 or P2LFE-3 is several times the
#  default. Prune with MMS_NL_LEVELS before sweeping.
#
#  ENV
#    MMS_NL_LEVELS  refinement levels                       3
#    MMS_NL_NX0     coarsest nx                             8
#    MMS_M          vertical elements M                     2
#    MMS_PVERT      vertical FE order p  (Nσ = M·p+1)       1
#    MMS_NL_FULL    1 ⇒ also run the two :full models       0  (see below)
#
#  RUN:  julia --project=. test/test_mms_convergence_nonlinear.jl
#        MMS_M=3 julia --project=. test/test_mms_convergence_nonlinear.jl
# ==============================================================

using GridapBALFEM
using Printf

println("=" ^ 76)
println("  test_mms_convergence_nonlinear.jl — order of accuracy, nonlinear models")
println("=" ^ 76)

const LEVELS = parse(Int, get(ENV, "MMS_NL_LEVELS", "3"))
const NX0    = parse(Int, get(ENV, "MMS_NL_NX0",    "8"))
const TOLP   = 0.3          # |p_obs − p_opt| gate, as in §subsubsec: mms measure
#  VERTICAL basis — see the header. c_bdy is left to resolve_cbdy, which is what
#  makes M ≠ 2 legal at all: run_conv_study used to hard-wire the M=2 node set and
#  threw the length(c_bdy)==M+1 assertion for anything else (fixed 2026-08-21).
const M_VERT = parse(Int, get(ENV, "MMS_M",     "2"))
const P_VERT = parse(Int, get(ENV, "MMS_PVERT", "1"))
const RUN_FULL = get(ENV, "MMS_NL_FULL", "0") != "0"
@printf("  vertical basis: P%dLFE-%d  (Nσ = %d)   levels=%d  nx0=%d\n",
        P_VERT, M_VERT, M_VERT*P_VERT + 1, LEVELS, NX0)

n_pass = 0; n_fail = 0
function check(name, cond, extra = "")
    global n_pass, n_fail
    cond ? (println("  PASS  $name $extra"); n_pass += 1) :
           (println("  FAIL  $name $extra"); n_fail += 1)
end

#  Studies in the order of MMS_NONLINEAR_PLAN.md §3 — simplest first, so a broken
#  rate is attributable to the block just added.
studies = [
    (name = "Model 3  nonlinear / flat bed      / :none",
     regime = :nonlinear, flat_bed = true,  nl_pressure = :none, nl_iter = 50,
     rate_gate_u = true),
    #  Model 4 runs at the DEFAULT budget. It briefly carried nl_iter=400, added
    #  when the stall at ‖r‖=4.8e-8 was read as "quasi-Newton convergence is just
    #  slow, give it more iterations". That diagnosis was WRONG and the extra
    #  budget would never have helped: ∂R/∂u̇ was missing the 𝓐/𝓚 slope package,
    #  whose prefactor H·∇h does NOT scale with the solution, so Newton was
    #  converging to a fixed point of the wrong map — an O(1) error, not a slow
    #  one. With that block assembled (2026-08-17) Model 4 converges in the
    #  ordinary number of iterations. Lesson: distinguish "converging slowly"
    #  from "converging to the wrong thing" BEFORE spending iterations on it;
    #  test_jacobians_ad.jl tells them apart by amplitude scaling.
    (name = "Model 4  nonlinear / variable bed  / :none",
     regime = :nonlinear, flat_bed = false, nl_pressure = :none, nl_iter = 50,
     rate_gate_u = true),
    #  ⛔ DO NOT SIMPLY UNCOMMENT ALL FOUR. The 𝓝 forcing became available on
    #  2026-08-18 (B1 was a closure variable-capture bug, not the tag-precedence
    #  limit recorded here before), but only `:native` is RATE-TESTABLE:
    #
    #    :native — components {3,6,7,8} are assembled NATIVELY, so the forcing and
    #              the solver encode the same operator. Model 3 measured
    #              p_η=2.996, p_u=3.997 and Model 4 p_η=2.996, p_u=3.998 —
#              theoretical order in both. BOTH are enabled below.
    #
    #    :full   — adds {1,2,4,5}, whose irreducible ∂²η the SOLVER evaluates from
    #              FROZEN L² PROJECTIONS lagged one step (src/nlpressure.jl) while
    #              the MMS forcing computes them EXACTLY. Different operators, so
    #              the study measures the surrogate and `e_u` STALLS AT A CONSTANT
    #              (5.988e-03; p_u = −0.00 over a 16× refinement) while `e_η` keeps
    #              its rate. ⚠ That is the IDENTICAL fingerprint to the 2026-08-17
    #              gravity DEFECT and here it means the opposite — the MMS cannot
    #              tell a wrong operator from a deliberately approximated one.
    #              Enabling it as a RATE gate would assert something false.
    #              If you want :full in the suite, pin the FLOOR (e_u ≈ 5.99e-3)
    #              as a regression value instead — that detects a change in the
    #              approximation, which a rate gate never could.
    (name = "Model 3 / :native", regime=:nonlinear, flat_bed=true,  nl_pressure=:native,
     nl_iter=50, rate_gate_u = true),
    (name = "Model 4 / :native", regime=:nonlinear, flat_bed=false, nl_pressure=:native,
     nl_iter=50, rate_gate_u = true),
]

#  ---- the :full pair — OPT-IN (MMS_NL_FULL=1), AND NOT A RATE GATE ON u ------
#  These complete the eight-model grid, so a vertical-basis sweep can cover every
#  configuration the solver offers. They are OFF by default for two reasons, one
#  of cost and one of meaning:
#
#    * cost — each is the most expensive tier (all eight 𝓝 components in the
#      forcing), and enabling them by default would multiply this file's runtime
#      and silently change its documented 8/8 score;
#    * meaning — `rate_gate_u = false`. `e_u` STALLS AT A CONSTANT by
#      construction: the solver evaluates the irreducible ∂²η of components
#      {1,2,4,5} from frozen L² projections lagged one step while the forcing
#      computes them exactly, so the two encode DIFFERENT operators and no mesh
#      refinement closes the gap (VERIFICATION.md §4). `e_η` keeps its rate and IS
#      gated. The `e_u` floor is REPORTED, labelled as a floor.
#
#  ⚠ DO NOT "fix" this by gating p_u — that would assert something false — and do
#  NOT pin a floor CONSTANT here either: the floor is a property of (M, p_vert,
#  domain, depth, levels), and PENDING_TASKS.md §1 tier 3 exists precisely to ask
#  whether it depends on Nσ. Pinning the P1LFE-2 number would break every other
#  vertical basis the moment someone swept one.
if RUN_FULL
    append!(studies, [
        (name = "Model 7  nonlinear / flat bed      / :full", regime=:nonlinear,
         flat_bed=true,  nl_pressure=:full, nl_iter=50, rate_gate_u = false),
        (name = "Model 8  nonlinear / variable bed  / :full", regime=:nonlinear,
         flat_bed=false, nl_pressure=:full, nl_iter=50, rate_gate_u = false),
    ])
else
    println("\n  [skip] the two :full models — set MMS_NL_FULL=1 to include them.")
    println("         They are NOT rate-gated on u (frozen-projection floor); see the note above.")
end

for s in studies
    println("\n" * "-"^76)
    println("  $(s.name)")
    println("-"^76); flush(stdout)
    #  A study that cannot COMPLETE must be reported as a failed gate, not allowed
    #  to abort the run. `run_conv_study` throws when Newton exhausts its budget,
    #  and Model 4 is exactly the study expected to do that — letting it propagate
    #  would destroy the report of every study that DID work, including Model 3's
    #  verification result. Report it, keep going, fail at the end.
    r = try
        run_conv_study(; p_u = 3, domain = :d1, mode = :static,
                         levels = LEVELS, nx0 = NX0, ny_1d = 3,
                         Lx = 1.7, Ly = 1.1, d = 2.5,
                         M = M_VERT, p_vert = P_VERT,   # c_bdy ⇒ resolve_cbdy(M)
                         dt = 1e-5, nsteps = 100, nl_tol = 1e-9,
                         nl_iter = s.nl_iter,
                         regime = s.regime, nl_pressure = s.nl_pressure,
                         flat_bed = s.flat_bed, a_b = s.flat_bed ? 0.0 : 0.2,
                         kbx = 1.3, kby = 0.0, verbose = true)
    catch e
        msg = first(split(sprint(showerror, e), '\n'))
        check("$(s.name): study completed", false, "\n        $msg")
        if occursin("did not converge", msg)
            println("        ⇒ the SOLVE is under-converged, which is NOT evidence about the")
            println("          forcing or the operator: a stalled Newton reads exactly like a")
            println("          wrong residual. The nonlinear Jacobians are quasi-Newton by")
            println("          design, so convergence is linear. Raise nl_iter (currently")
            println("          $(s.nl_iter)) and re-run. If the residual PLATEAUS rather than")
            println("          decreasing slowly, budget will not help and the real fix is to")
            println("          complete the nonlinear Jacobians — see MMS_NONLINEAR_PLAN.md §5b.")
            println("        ⇒ do NOT loosen nl_tol to make this pass: the algebraic error would")
            println("          then sit inside the discretisation error being measured.")
        end
        flush(stdout)
        continue
    end
    check("$(s.name): p_eta → $(Int(r.opt_eta))", abs(r.fit_eta - r.opt_eta) < TOLP,
          @sprintf("(got %.3f)", r.fit_eta))
    if s.rate_gate_u
        check("$(s.name): p_u   → $(Int(r.opt_u))",   abs(r.fit_u   - r.opt_u)   < TOLP,
              @sprintf("(got %.3f)", r.fit_u))
    else
        #  REPORTED, NOT GATED. See the :full note above: a stalled p_u is the
        #  designed behaviour of the frozen-projection surrogate, so asserting a
        #  rate would manufacture a defect, and asserting a floor CONSTANT would
        #  break under any other vertical basis. What the sweep wants is the number.
        @printf("  REPORT  %s: e_u FLOOR = %.6e  (p_u = %.3f over %d levels, %s, Nσ=%d)\n",
                s.name, r.e_u[end], r.fit_u, length(r.h), r.tag, r.Nsigma)
        println("          not gated: :full's u-error is capped by the frozen L² projections,")
        println("          not by the mesh — VERIFICATION.md §4. p_η above IS gated.")
    end
    flush(stdout)
end

println()
println("=" ^ 76)
@printf("  Results: %d PASS,  %d FAIL\n", n_pass, n_fail)
println("=" ^ 76)
n_fail > 0 ? error("test_mms_convergence_nonlinear: $n_fail failed!") :
             println("  Nonlinear MMS models reach their theoretical order of accuracy.")
