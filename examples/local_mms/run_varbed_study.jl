# ==============================================================
#  run_varbed_study.jl — ONE variable-bathymetry MMS convergence study
#
#  Verifies the LINEARISED VARIABLE-BED operator (LinearModel.tex
#  eq: linearised system momentum), whose grad-h terms Stage 1 never touched.
#  Plan: building_files/MMS_VARBED_PLAN.md
#
#  CONVENTIONS ADOPTED FROM THE FLAT-BED CAMPAIGN'S BLIND SPOTS:
#    * d != 1  -- d=1 makes multiplication by h the IDENTITY, so the whole
#      h-weighting of the momentum equation becomes unobservable.
#    * a_b > 0 -- an actually varying bed, else this degenerates to Stage 1.
#    * Q_p/Q_{p-1} -- equal order costs one convergence order (measured).
#
#  ENV: PU, DOMAIN(d1|d2), MODE(static|transient), LEVELS, NX0, AB, D0
# ==============================================================
using GridapLFEM, Printf
gi(k,d)=parse(Int,get(ENV,k,string(d))); gf(k,d)=parse(Float64,get(ENV,k,string(d)))
p_u=gi("PU",3); dom=Symbol(get(ENV,"DOMAIN","d1")); mode=Symbol(get(ENV,"MODE","static"))
lv=gi("LEVELS",4); nx0=gi("NX0",16); ab=gf("AB",0.2); d0=gf("D0",2.5)
println("#"^72)
println("#  VARIABLE-BED MMS  Q$(p_u)/Q$(p_u-1)  $dom  $mode")
println("#    h = $d0 (1 + $ab sin(1.3x))   d0!=1 and a_b>0 are deliberate")
println("#    gates:  u -> $(p_u+1)   eta -> $(p_u)   (different optima, by design)")
println("#"^72); flush(stdout)
r = run_conv_study(; p_u=p_u, domain=dom, mode=mode, levels=lv, nx0=nx0, ny_1d=3,
                     Lx=1.7, Ly=1.1, d=d0, M=2, dt=1e-5, nsteps=100,
                     flat_bed=false, a_b=ab, kbx=1.3, kby=0.0, verbose=true)
@printf("\n>>> %s :  p_eta=%.3f (opt %.0f)   p_u=%.3f (opt %.0f)\n",
        r.tag, r.fit_eta, r.opt_eta, r.fit_u, r.opt_u)
