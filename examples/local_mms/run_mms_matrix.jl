# ==============================================================
#  run_mms_matrix.jl — the MMS convergence CAMPAIGN runner
#
#  Runs a set of convergence studies (pairing × domain × exec × time-mode) and
#  writes one CSV row per mesh level plus a summary table.
#  Plan: building_files/MMS_CONVERGENCE_CAMPAIGN.md
#
#  GATES (note the two fields have DIFFERENT optima):
#      u  in Q_p      ⇒ optimal L² rate p+1
#      eta in Q_{p-1} ⇒ optimal L² rate p        (NOT p+1)
#
#  ENV
#    LFEM_CONV_PU      comma list of velocity orders   2,3,4   (⇒ Q2/Q1,Q3/Q2,Q4/Q3)
#    LFEM_CONV_DOMAIN  d1 | d2                          d1
#    LFEM_CONV_MODE    static | transient | both        both
#    LFEM_CONV_LEVELS  refinement levels                4
#    LFEM_CONV_NX0     coarsest nx                      8
#    LFEM_CONV_NY0     coarsest ny (2-D only)           8
#    LFEM_CONV_DT      time step                        1e-4
#    LFEM_CONV_NSTEPS  steps                            100
#    LFEM_CONV_DIST    0 | 1 distributed                0
#    LFEM_CONV_PX/PY   MPI grid (distributed)           2 / 2
#    LFEM_CONV_LSRTOL  GMRES rtol (distributed)         1e-13
#    LFEM_CONV_NLTOL   Newton tol                       1e-14
#    LFEM_CONV_OUT     output dir                       output/local/mms_conv
#    LFEM_CONV_REGIME  linear | nonlinear                linear
#    LFEM_CONV_NLP     none | native | full              none  (≠none available since 2026-08-18)
#    LFEM_CONV_FLATBED 1 flat bed | 0 variable bed       1
#    LFEM_CONV_AB      bed amplitude when FLATBED=0      0.2
#    LFEM_CONV_NLITER  Newton budget                     50 linear / 400 nonlinear
# ==============================================================
using GridapLFEM, Printf
genv(k,d)=get(ENV,k,d); genv_i(k,d)=parse(Int,get(ENV,k,string(d)))
genv_f(k,d)=parse(Float64,get(ENV,k,string(d))); genv_b(k,d)=genv_i(k,d)!=0

pus    = parse.(Int, split(genv("LFEM_CONV_PU","2,3,4"), ","))
domain = Symbol("d"*string(genv("LFEM_CONV_DOMAIN","d1")[end]))
modes  = let m=genv("LFEM_CONV_MODE","both")
    m=="both" ? [:static,:transient] : [Symbol(m)] end
levels = genv_i("LFEM_CONV_LEVELS",4)
nx0    = genv_i("LFEM_CONV_NX0",8); ny0 = genv_i("LFEM_CONV_NY0",8)
dt     = genv_f("LFEM_CONV_DT",1e-4); nsteps = genv_i("LFEM_CONV_NSTEPS",100)
dist   = genv_b("LFEM_CONV_DIST",0)
px,py  = genv_i("LFEM_CONV_PX",2), genv_i("LFEM_CONV_PY",2)
lsrtol = genv_f("LFEM_CONV_LSRTOL",1e-13); nltol = genv_f("LFEM_CONV_NLTOL",1e-14)
outdir = genv("LFEM_CONV_OUT","output/local/mms_conv"); mkpath(outdir)
#  MODEL selection. The campaign's original question was the FE PAIRING, which is
#  orthogonal to the model — so the same sweep can now be run for any of the four
#  MMS models instead of only the linear flat-bed one. The three symbols drive the
#  forcing and the solver from one variable each, so they cannot drift apart.
regime  = Symbol(genv("LFEM_CONV_REGIME","linear"))
nlp     = Symbol(genv("LFEM_CONV_NLP","none"))
flatbed = genv_b("LFEM_CONV_FLATBED",1)
a_b     = genv_f("LFEM_CONV_AB",0.2)          # bed amplitude when FLATBED=0
#  Nonlinear ⇒ quasi-Newton ⇒ LINEAR convergence ⇒ needs budget, not a looser
#  tolerance. Loosening nl_tol would bury the algebraic error inside the
#  discretisation error the rate is measuring.
nliter  = genv_i("LFEM_CONV_NLITER", regime === :linear ? 50 : 400)
model   = regime === :linear ? (flatbed ? 1 : 2) : (flatbed ? 3 : 4)

println("#"^76)
println("#  MMS CONVERGENCE CAMPAIGN — Model $model ($(regime) / $(flatbed ? "flat" : "variable") bed / :$(nlp))")
println("#    pairings Q_p/Q_{p-1} for p = $(pus) | domain=$domain | modes=$modes")
println("#    levels=$levels nx0=$nx0 dt=$dt nsteps=$nsteps | $(dist ? "MPI $(px)x$(py)" : "sequential (direct LU)")")
println("#    gates:  u → p+1     eta → p      (different optima, by design)")
println("#"^76)

rows=[]; summ=[]
for p_u in pus, mode in modes
    println("\n>>> Q$(p_u)/Q$(p_u-1)  $(domain)  $(mode)")
    r = run_conv_study(; p_u=p_u, domain=domain, mode=mode, levels=levels,
                         nx0=nx0, ny0=ny0, dt=dt, nsteps=nsteps,
                         distributed=dist, cpu_grid=(px,py),
                         regime=regime, nl_pressure=nlp,
                         flat_bed=flatbed, a_b=flatbed ? 0.0 : a_b,
                         nl_iter=nliter,
                         ls_rtol=lsrtol, nl_tol=nltol)
    for i in eachindex(r.h)
        push!(rows,(r.tag,r.h[i],r.ndofs[i],r.e_eta[i],r.e_u[i]))
    end
    okη = abs(r.pw_eta[end]-r.opt_eta)<0.3; oku = abs(r.pw_u[end]-r.opt_u)<0.3
    push!(summ,(r.tag,r.fit_eta,r.opt_eta,r.fit_u,r.opt_u,
                r.pw_eta[end],r.pw_u[end], okη&&oku))
end

csv=joinpath(outdir,"mms_conv_$(domain)_$(dist ? "dist" : "seq").csv")
open(csv,"w") do io
    println(io,"study,h,ndofs,e_eta,e_u")
    for r in rows; @printf(io,"%s,%.8g,%d,%.10e,%.10e\n",r...); end
end
println("\n","="^76)
println("  CAMPAIGN SUMMARY   (last pairwise rate is the asymptotic one)")
println("="^76)
@printf("  %-30s %-14s %-14s %s\n","study","eta fit/opt","u fit/opt","verdict")
for s in summ
    @printf("  %-30s %5.2f/%-8.0f %5.2f/%-8.0f %s\n",
            s[1],s[2],s[3],s[4],s[5], s[8] ? "PASS" : "CHECK")
end
println("\n  wrote $csv")
