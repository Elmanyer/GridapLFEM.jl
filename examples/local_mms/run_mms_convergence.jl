# ==============================================================
#  run_mms_convergence.jl — PARAMETRIC analytic-MMS convergence study
#
#  The production form of test_mms_convergence.jl: longer refinement sequences,
#  a CSV of the results, and env configuration, for when you want the measured
#  order to a tighter tolerance than a test gate needs — or want to see WHERE a
#  rate breaks rather than just that it did.
#
#  SEQUENTIAL BY DESIGN. These are small problems, and a direct LU beats every
#  MPI split 2-3x at this size (measured, LOCAL_TESTS_RESULTS.md §5.3); the
#  driver also needs the final FE solution to form the L² error.
#
#  CONFIG (env)
#    LFEM_MMS_MODE     space | time | both        both
#    LFEM_MMS_LEVELS   refinement levels           4
#    LFEM_MMS_NX0/NY0  coarsest mesh (space)       6 / 4
#    LFEM_MMS_DT0      time step                   2e-4 (space) / 8e-3 (time)
#    LFEM_MMS_NXF/NYF  fixed fine mesh (time)      24 / 16
#    LFEM_MMS_TFINAL   final time                  0.02 (space) / 0.08 (time)
#    LFEM_MMS_M        vertical elements           2
#    LFEM_MMS_ORDER    horizontal FE order         2   (expected space rate = ORDER+1)
#    LFEM_MMS_LX/LY    domain                      1.7 / 1.1
#    LFEM_MMS_D        still-water depth           1.0
#    LFEM_MMS_SOLVER   sdirk | theta               sdirk
#    LFEM_MMS_OUT      output directory            output/local/mms
#
#  RUN
#    julia --project=. examples/local_mms/run_mms_convergence.jl
#    LFEM_MMS_MODE=space LFEM_MMS_LEVELS=5 julia --project=. examples/local_mms/run_mms_convergence.jl
# ==============================================================

using GridapLFEM
using Printf

genv(k, d)   = get(ENV, k, d)
genv_i(k, d) = parse(Int,     get(ENV, k, string(d)))
genv_f(k, d) = parse(Float64, get(ENV, k, string(d)))

mode    = Symbol(genv("LFEM_MMS_MODE", "both"))
levels  = genv_i("LFEM_MMS_LEVELS", 4)
nx0     = genv_i("LFEM_MMS_NX0", 6);   ny0  = genv_i("LFEM_MMS_NY0", 4)
nxf     = genv_i("LFEM_MMS_NXF", 24);  nyf  = genv_i("LFEM_MMS_NYF", 16)
Mvert   = genv_i("LFEM_MMS_M", 2)
order   = genv_i("LFEM_MMS_ORDER", 2)
Lx      = genv_f("LFEM_MMS_LX", 1.7);  Ly   = genv_f("LFEM_MMS_LY", 1.1)
dval    = genv_f("LFEM_MMS_D", 1.0)
solver  = Symbol(genv("LFEM_MMS_SOLVER", "sdirk"))
outdir  = genv("LFEM_MMS_OUT", "output/local/mms")
mkpath(outdir)

CASE = (Lx=Lx, Ly=Ly, d=dval, M=Mvert, p_horizontal=order, solver_type=solver)

println("#"^70)
println("#  ANALYTIC MMS — Stage 1 (linear core, flat bed)")
println("#    domain $(Lx)×$(Ly) m | d=$(dval) | M=$(Mvert) | Q$(order) | $(solver)")
println("#    expected: space $(order+1)   time 2")
println("#    NOTE: this verifies the LINEAR FLAT-BED operator only.")
println("#"^70)

rows = []
if mode in (:space, :both)
    dt0 = genv_f("LFEM_MMS_DT0", 2e-4)
    tF  = genv_f("LFEM_MMS_TFINAL", 0.02)
    sp  = run_mms_refinement(:space; levels=levels, nx0=nx0, ny0=ny0,
                             dt0=dt0, T_final=tF, CASE...)
    for i in eachindex(sp.param)
        push!(rows, ("space", sp.param[i], dt0, sp.e_eta[i], sp.e_u[i],
                     sp.p_eta, sp.p_u, Float64(order+1)))
    end
    @printf("\n  >>> SPATIAL  p_eta = %.3f   p_u = %.3f   (expected %d)\n",
            sp.p_eta, sp.p_u, order+1)
end
if mode in (:time, :both)
    dt0 = genv_f("LFEM_MMS_DT0_T", 8e-3)
    tF  = genv_f("LFEM_MMS_TFINAL_T", 0.08)
    tp  = run_mms_refinement(:time; levels=levels, nx_fine=nxf, ny_fine=nyf,
                             dt0=dt0, T_final=tF, CASE...)
    for i in eachindex(tp.param)
        push!(rows, ("time", NaN, tp.param[i], tp.e_eta[i], tp.e_u[i],
                     tp.p_eta, tp.p_u, 2.0))
    end
    @printf("\n  >>> TEMPORAL q_eta = %.3f   q_u = %.3f   (expected 2)\n",
            tp.p_eta, tp.p_u)
end

csv = joinpath(outdir, "mms_convergence.csv")
open(csv, "w") do io
    println(io, "study,h,dt,e_eta,e_u,p_eta,p_u,expected")
    for r in rows
        @printf(io, "%s,%.8g,%.8g,%.10e,%.10e,%.4f,%.4f,%.1f\n", r...)
    end
end
println("\n  wrote $csv")
println("\n  Reading a failure (ValidationTests.tex):")
println("    slope $(order+1) → correct        slope 2 → a term one derivative too coarse")
println("    slope 1 → first-order inconsistency   slope 0 → A WRONG COEFFICIENT")
