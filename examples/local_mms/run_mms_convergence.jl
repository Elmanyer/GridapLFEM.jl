# ==============================================================
#  run_mms_convergence.jl — PARAMETRIC analytic-MMS convergence study
#
#  The production form of test_mms_convergence.jl: longer refinement sequences,
#  a CSV of the results, and env configuration, for when you want the measured
#  order to a tighter tolerance than a test gate needs — or want to see WHERE a
#  rate breaks rather than just that it did.
#
#  SEQUENTIAL BY DESIGN. These are small problems, and a direct LU beats every
#  MPI split 2-3x at this size (measured, building_files/CONFIGURATION.md §6); the
#  driver also needs the final FE solution to form the L² error.
#
#  CONFIG (env)
#    BALFEM_MMS_MODE     space | time | both        both
#    BALFEM_MMS_LEVELS   refinement levels           4
#    BALFEM_MMS_NX0/NY0  coarsest mesh (space)       6 / 4
#    BALFEM_MMS_DT0      time step                   2e-4 (space) / 8e-3 (time)
#    BALFEM_MMS_NXF/NYF  fixed fine mesh (time)      24 / 16
#    BALFEM_MMS_TFINAL   final time                  0.02 (space) / 0.08 (time)
#    BALFEM_MMS_M        vertical elements M         2
#    BALFEM_MMS_PVERT    vertical FE order p          1   (Nσ = M·p+1 ⇒ P{p}LFE-{M})
#    BALFEM_MMS_ORDER    VELOCITY FE order p_u       3   (optimal u rate = p_u+1)
#    BALFEM_MMS_PETA     SURFACE  FE order p_eta     p_u−1 (optimal eta rate = p_eta+1)
#    BALFEM_MMS_LX/LY    domain                      1.7 / 1.1
#    BALFEM_MMS_D        still-water depth           1.0
#    BALFEM_MMS_SOLVER   sdirk | theta               sdirk
#    BALFEM_MMS_OUT      output directory            output/local/mms
#    -- model selection (the four MMS models) --
#    BALFEM_MMS_REGIME   linear | nonlinear          linear
#    BALFEM_MMS_NLP      none | native | full        none   (≠none available since 2026-08-18)
#    BALFEM_MMS_FLATBED  1 flat | 0 variable bed     1
#    BALFEM_MMS_AB       bed amplitude if FLATBED=0  0.2
#    BALFEM_MMS_NLITER   Newton budget               50 linear / 400 nonlinear
#
#  ⚠ THE FE PAIRING IS PART OF THE MEASUREMENT. η enters momentum
#  undifferentiated (via ∇·v after IBP), so it plays the pressure role of a
#  Stokes system and EQUAL-ORDER spaces are inf-sup deficient: Q_p/Q_p converges
#  at p in both fields, not p+1. The default here is the Taylor-Hood-like
#  Q_p/Q_{p−1}, the only pairing measured optimal in both fields
#  (building_files/MMS_CONVERGENCE_CAMPAIGN.md). Reading a rate without knowing
#  the pairing is how a healthy solver gets mistaken for a broken one.
#
#  ⚠ NONLINEAR MODELS NEED ITERATION BUDGET, NOT LOOSER TOLERANCES. The
#  nonlinear Jacobians are quasi-Newton by design, so Newton converges linearly;
#  raise BALFEM_MMS_NLITER. Loosening nl_tol would put the algebraic error inside
#  the discretisation error being measured.
#
#  RUN
#    julia --project=. examples/local_mms/run_mms_convergence.jl
#    BALFEM_MMS_MODE=space BALFEM_MMS_LEVELS=5 julia --project=. examples/local_mms/run_mms_convergence.jl
#    # Model 3 (nonlinear, flat bed):
#    BALFEM_MMS_REGIME=nonlinear julia --project=. examples/local_mms/run_mms_convergence.jl
#    # Model 2 (linear, variable bed):
#    BALFEM_MMS_FLATBED=0 BALFEM_MMS_D=2.5 julia --project=. examples/local_mms/run_mms_convergence.jl
# ==============================================================

using GridapBALFEM
using Printf

genv(k, d)   = get(ENV, k, d)
genv_i(k, d) = parse(Int,     get(ENV, k, string(d)))
genv_f(k, d) = parse(Float64, get(ENV, k, string(d)))

mode    = Symbol(genv("BALFEM_MMS_MODE", "both"))
levels  = genv_i("BALFEM_MMS_LEVELS", 4)
nx0     = genv_i("BALFEM_MMS_NX0", 6);   ny0  = genv_i("BALFEM_MMS_NY0", 4)
nxf     = genv_i("BALFEM_MMS_NXF", 24);  nyf  = genv_i("BALFEM_MMS_NYF", 16)
#  VERTICAL basis: (M, p_vert). c_bdy is left to resolve_cbdy — the optimised
#  Yang & Liu boundaries where they exist for this M, a uniform split otherwise.
#  The node positions change the error CONSTANT, never the ORDER, so a rate study
#  may use any reasonable set; the optimised ones are a DISPERSION question.
Mvert   = genv_i("BALFEM_MMS_M", 2)
p_vert  = genv_i("BALFEM_MMS_PVERT", 1)
order   = genv_i("BALFEM_MMS_ORDER", 3)          # velocity order p_u (Q3 by default)
#  Surface order. Default = order−1 (Taylor-Hood-like), the ONLY pairing measured
#  optimal in BOTH fields. Set equal to `order` to reproduce the old equal-order
#  behaviour — but then expect p, not p+1, in both fields.
p_eta   = genv_i("BALFEM_MMS_PETA", order - 1)
Lx      = genv_f("BALFEM_MMS_LX", 1.7);  Ly   = genv_f("BALFEM_MMS_LY", 1.1)
dval    = genv_f("BALFEM_MMS_D", 1.0)
solver  = Symbol(genv("BALFEM_MMS_SOLVER", "sdirk"))
outdir  = genv("BALFEM_MMS_OUT", "output/local/mms")
mkpath(outdir)

#  Model selection — the SAME three symbols drive the forcing and the solver
#  (run_mms_case passes both from one variable each), so they cannot drift apart.
regime      = Symbol(genv("BALFEM_MMS_REGIME", "linear"))
nlp         = Symbol(genv("BALFEM_MMS_NLP", "none"))
flat_bed    = genv_i("BALFEM_MMS_FLATBED", 1) != 0
a_b         = genv_f("BALFEM_MMS_AB", 0.2)       # bed amplitude when flat_bed=0
nl_iter     = genv_i("BALFEM_MMS_NLITER", regime === :linear ? 50 : 400)
#  Variable bed ⇒ hand run_mms_case the same bathymetry object it gives the forcing.
hfun    = flat_bed ? nothing : bathymetry_field(; d0=dval, a_b=a_b, kbx=1.3, kby=0.0)

CASE = (Lx=Lx, Ly=Ly, d=dval, M=Mvert, p_vert=p_vert, p_horizontal=order, p_eta=p_eta,
        solver_type=solver, regime=regime, nl_pressure=nlp, flat_bed=flat_bed,
        hfun=hfun, nl_iter=nl_iter)

#  The two fields have DIFFERENT optimal rates under a mixed pairing.
opt_eta, opt_u = p_eta + 1, order + 1
model_no = regime === :linear ? (flat_bed ? 1 : 2) : (flat_bed ? 3 : 4)

println("#"^70)
println("#  ANALYTIC MMS — Model $model_no  ($(regime) / $(flat_bed ? "flat" : "variable") bed / :$(nlp))")
println("#    domain $(Lx)×$(Ly) m | d=$(dval) | P$(p_vert)LFE-$(Mvert) (Nσ=$(Mvert*p_vert+1)) | Q$(order)/Q$(p_eta) | $(solver)")
println("#    expected: space  eta→$(opt_eta)  u→$(opt_u)   |   time 2")
if order == p_eta
    println("#    ⚠ EQUAL ORDER: eta enters momentum undifferentiated (∇·v after IBP), so")
    println("#      this pairing is inf-sup deficient and converges at p, NOT p+1. The")
    println("#      expectations printed above are unreachable — use BALFEM_MMS_PETA=$(order-1).")
end
println("#"^70)

rows = []
if mode in (:space, :both)
    dt0 = genv_f("BALFEM_MMS_DT0", 2e-4)
    tF  = genv_f("BALFEM_MMS_TFINAL", 0.02)
    sp  = run_mms_refinement(:space; levels=levels, nx0=nx0, ny0=ny0,
                             dt0=dt0, T_final=tF, CASE...)
    for i in eachindex(sp.param)
        push!(rows, ("space", sp.param[i], dt0, sp.e_eta[i], sp.e_u[i],
                     sp.p_eta, sp.p_u, Float64(opt_eta), Float64(opt_u)))
    end
    @printf("\n  >>> SPATIAL  p_eta = %.3f (expected %d)   p_u = %.3f (expected %d)\n",
            sp.p_eta, opt_eta, sp.p_u, opt_u)
end
if mode in (:time, :both)
    dt0 = genv_f("BALFEM_MMS_DT0_T", 8e-3)
    tF  = genv_f("BALFEM_MMS_TFINAL_T", 0.08)
    tp  = run_mms_refinement(:time; levels=levels, nx_fine=nxf, ny_fine=nyf,
                             dt0=dt0, T_final=tF, CASE...)
    for i in eachindex(tp.param)
        push!(rows, ("time", NaN, tp.param[i], tp.e_eta[i], tp.e_u[i],
                     tp.p_eta, tp.p_u, 2.0, 2.0))
    end
    @printf("\n  >>> TEMPORAL q_eta = %.3f   q_u = %.3f   (expected 2)\n",
            tp.p_eta, tp.p_u)
end

csv = joinpath(outdir, "mms_convergence.csv")
open(csv, "w") do io
    println(io, "study,h,dt,e_eta,e_u,p_eta,p_u,expected_eta,expected_u")
    for r in rows
        @printf(io, "%s,%.8g,%.8g,%.10e,%.10e,%.4f,%.4f,%.1f,%.1f\n", r...)
    end
end
println("\n  wrote $csv")
println("\n  Reading a failure (ValidationTests.tex):")
println("    optimal ($(opt_eta) for eta, $(opt_u) for u) → correct")
println("    one order less → a term evaluated one derivative too coarsely")
println("    slope 1 → first-order inconsistency   slope 0 → A WRONG COEFFICIENT")
println("    but check the PAIRING first: equal order costs one order by itself.")
