# ==============================================================
#  run_varbed_study.jl — ONE variable-bathymetry MMS convergence study
#
#  Verifies the VARIABLE-BED operator — the ∇h packages (𝓐/𝓚, L¹, the bed-slope
#  half of the IBP) that no flat-bed study can reach, because every one of those
#  terms vanishes identically when ∇h ≡ 0. That structural blind spot is what let
#  a double-counted 𝓐/𝓚 package survive the entire suite for weeks; a sloping bed
#  is the only configuration that tests this code at all.
#
#  Specification: LinearModel.tex eq: linearised system momentum (linear) and
#  the term classification in GridapImplementation.tex (all four models).
#  Plan: building_files/MMS_VARBED_PLAN.md, building_files/MMS_NONLINEAR_PLAN.md
#
#  MODELS REACHABLE FROM HERE (REGIME × the fixed variable bed):
#    REGIME=linear     -> Model 2  (VERIFIED 2026-08-15: p_eta 3.000 / p_u 4.000)
#    REGIME=nonlinear  -> Model 4  (advection + full leading pressure over a slope)
#  Model 1/3 (flat bed) are the business of run_mms_convergence.jl.
#
#  CONVENTIONS ADOPTED FROM THE FLAT-BED CAMPAIGN'S BLIND SPOTS:
#    * d != 1  -- d=1 makes multiplication by h the IDENTITY, so the whole
#      h-weighting of the momentum equation becomes unobservable.
#    * a_b > 0 -- an actually varying bed, else this degenerates to the flat case.
#    * Q_p/Q_{p-1} -- equal order costs one convergence order (measured); this is
#      the only pairing measured optimal in BOTH fields.
#
#  ⚠ NONLINEAR: BUDGET, NEVER TOLERANCE. The nonlinear Jacobians are quasi-Newton
#  by design (the pressure blocks contribute no η-derivative and the 𝓐/𝓚 package
#  is absent from ∂R/∂u̇), so Newton converges LINEARLY — and over a variable bed
#  strictly more is missing than on a flat one. Model 4 stalled at ‖r‖=4.8e-8 on
#  the default 50 iterations; the answer is NL_ITER, not a looser NL_TOL. A
#  loosened tolerance would put the algebraic error inside the discretisation
#  error being measured, and the rate would mean nothing.
#
#  ENV: PU, DOMAIN(d1|d2), MODE(static|transient), LEVELS, NX0, AB, D0,
#       REGIME(linear|nonlinear), NLP(none|native|full), NL_ITER, NL_TOL,
#       M (vertical elements), PVERT (vertical FE order)  ⇒ P{PVERT}LFE-{M}
#
#  RUN
#    julia --project=. examples/local_mms/run_varbed_study.jl              # Model 2
#    REGIME=nonlinear julia --project=. examples/local_mms/run_varbed_study.jl  # Model 4
# ==============================================================
using GridapBALFEM, Printf
gi(k,d)=parse(Int,get(ENV,k,string(d))); gf(k,d)=parse(Float64,get(ENV,k,string(d)))
p_u=gi("PU",3); dom=Symbol(get(ENV,"DOMAIN","d1")); mode=Symbol(get(ENV,"MODE","static"))
lv=gi("LEVELS",4); nx0=gi("NX0",16); ab=gf("AB",0.2); d0=gf("D0",2.5)
regime = Symbol(get(ENV,"REGIME","linear"))
nlp    = Symbol(get(ENV,"NLP","none"))
#  VERTICAL basis — a parameter, not a constant. c_bdy resolves from M.
Mv=gi("M",2); pv=gi("PVERT",1)
#  A linear problem converges in ONE Newton iteration per stage (the linear
#  Jacobian branch is exact), so 50 is generous there and 400 is the measured
#  need over a slope in the nonlinear branch.
nl_iter = gi("NL_ITER", regime === :linear ? 50 : 400)
nl_tol  = gf("NL_TOL",  regime === :linear ? 1e-14 : 1e-9)
model   = regime === :linear ? 2 : 4

println("#"^72)
println("#  VARIABLE-BED MMS — Model $model  ($regime / :$nlp)  P$(pv)LFE-$(Mv)  Q$(p_u)/Q$(p_u-1)  $dom  $mode")
println("#    h = $d0 (1 + $ab sin(1.3x))   d0!=1 and a_b>0 are deliberate")
println("#    gates:  u -> $(p_u+1)   eta -> $(p_u)   (different optima, by design)")
println("#    Newton: nl_iter=$nl_iter  nl_tol=$nl_tol")
println("#"^72); flush(stdout)
r = run_conv_study(; p_u=p_u, domain=dom, mode=mode, levels=lv, nx0=nx0, ny_1d=3,
                     Lx=1.7, Ly=1.1, d=d0, M=Mv, p_vert=pv, dt=1e-5, nsteps=100,
                     regime=regime, nl_pressure=nlp,
                     nl_iter=nl_iter, nl_tol=nl_tol,
                     flat_bed=false, a_b=ab, kbx=1.3, kby=0.0, verbose=true)
@printf("\n>>> %s :  p_eta=%.3f (opt %.0f)   p_u=%.3f (opt %.0f)\n",
        r.tag, r.fit_eta, r.opt_eta, r.fit_u, r.opt_u)
ok = abs(r.fit_eta - r.opt_eta) < 0.3 && abs(r.fit_u - r.opt_u) < 0.3
println(ok ? ">>> Model $model reaches its theoretical order." :
             ">>> CHECK: off-optimal. Before blaming the residual, rule out (a) the FE\n" *
             ">>>        pairing, (b) an under-converged Newton solve (raise NL_ITER and\n" *
             ">>>        re-run — a stalled solve reads exactly like a wrong operator).")
