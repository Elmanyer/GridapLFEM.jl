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
#  RUNTIME. ~25–30 min per study (3 levels). Set MMS_NL_LEVELS to shorten.
#
#  RUN:  julia --project=. test/test_mms_convergence_nonlinear.jl
# ==============================================================

using GridapLFEM
using Printf

println("=" ^ 76)
println("  test_mms_convergence_nonlinear.jl — order of accuracy, nonlinear models")
println("=" ^ 76)

const LEVELS = parse(Int, get(ENV, "MMS_NL_LEVELS", "3"))
const NX0    = parse(Int, get(ENV, "MMS_NL_NX0",    "8"))
const TOLP   = 0.3          # |p_obs − p_opt| gate, as in §subsubsec: mms measure

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
     regime = :nonlinear, flat_bed = true,  nl_pressure = :none),
    (name = "Model 4  nonlinear / variable bed  / :none",
     regime = :nonlinear, flat_bed = false, nl_pressure = :none),
    #  :native and :full are specified in ValidationTests.tex but not yet
    #  implementable — see the tag-precedence limitation recorded in src/mms.jl
    #  and MMS_NONLINEAR_PLAN.md §6. They are listed here, commented, so the gap
    #  is visible in the test file rather than only in a plan document.
    # (name = "Model 3 / :native", regime=:nonlinear, flat_bed=true,  nl_pressure=:native),
    # (name = "Model 3 / :full",   regime=:nonlinear, flat_bed=true,  nl_pressure=:full),
    # (name = "Model 4 / :native", regime=:nonlinear, flat_bed=false, nl_pressure=:native),
    # (name = "Model 4 / :full",   regime=:nonlinear, flat_bed=false, nl_pressure=:full),
]

for s in studies
    println("\n" * "-"^76)
    println("  $(s.name)")
    println("-"^76); flush(stdout)
    r = run_conv_study(; p_u = 3, domain = :d1, mode = :static,
                         levels = LEVELS, nx0 = NX0, ny_1d = 3,
                         Lx = 1.7, Ly = 1.1, d = 2.5, M = 2,
                         dt = 1e-5, nsteps = 100, nl_tol = 1e-9,
                         regime = s.regime, nl_pressure = s.nl_pressure,
                         flat_bed = s.flat_bed, a_b = s.flat_bed ? 0.0 : 0.2,
                         kbx = 1.3, kby = 0.0, verbose = true)
    check("$(s.name): p_eta → $(Int(r.opt_eta))", abs(r.fit_eta - r.opt_eta) < TOLP,
          @sprintf("(got %.3f)", r.fit_eta))
    check("$(s.name): p_u   → $(Int(r.opt_u))",   abs(r.fit_u   - r.opt_u)   < TOLP,
          @sprintf("(got %.3f)", r.fit_u))
    flush(stdout)
end

println()
println("=" ^ 76)
@printf("  Results: %d PASS,  %d FAIL\n", n_pass, n_fail)
println("=" ^ 76)
n_fail > 0 ? error("test_mms_convergence_nonlinear: $n_fail failed!") :
             println("  Nonlinear MMS models reach their theoretical order of accuracy.")
