# ==============================================================
#  run_vertical_basis_study.jl — MMS convergence ACROSS VERTICAL BASES
#
#  THE CLAIM THIS MEASURES. BALFE-M asserts BASIS-AGNOSTICISM: the model family
#  works for an ARBITRARY vertical FE basis. Every convergence study, every model
#  and every gate in this repository had, until 2026-08-21, run exactly ONE
#  member — P1LFE-2 (M=2, p=1, Nσ=3) — because `run_conv_study` hard-wired
#  `assemble_vertical_tensors(M, 1, [0.0, 0.728, 1.0])`: `p_vert` was not a
#  parameter at all and `c_bdy` was pinned to the M=2 node set, so any M ≠ 2 threw
#  the `length(c_bdy) == M+1` assertion. The evidence for the property the project
#  is named for was one data point. This script produces the rest.
#
#  Design: building_files/PENDING_TASKS.md §1 (tiers 1–3).
#
#  ⚠ THIS BELONGS LOCAL AND SEQUENTIAL. MMS measures the order of accuracy under
#  MESH REFINEMENT; the domain size does not enter a rate at all (h = Lx/nx, so a
#  bigger domain at fixed nx is simply a coarser mesh). Cluster resources buy more
#  refinement levels, not new information about the operator — and the expensive
#  axis here is the VERTICAL resolution, whose forcing cost scales as Nσ² at each
#  quadrature point and does not parallelise away.
#
#  ⚠ WHAT A RATE HERE DOES AND DOES NOT SHOW. `c_bdy` changes the error CONSTANT,
#  never the ORDER. So this study tests the generality of the DISCRETISATION, and
#  is silent about whether the σ-node positions are well chosen — that is a
#  DISPERSION-accuracy question, measured by applicable kd (test_dispersion_curve's
#  territory). Reporting one as evidence for the other is a category error. For
#  p ≥ 2 no published optimum exists and `resolve_cbdy` falls back to the Yang &
#  Liu boundaries for that M (a uniform split when M > 4), which is fine for a rate.
#
#  ⚠ TIER 3 (:full) IS NOT A RATE STUDY. `:full` pins p_u at −0.00 by
#  construction: the solver evaluates components {1,2,4,5} from frozen L²
#  projections lagged one step while the forcing computes them exactly, so the two
#  encode different operators and refinement never closes the gap. The FLOOR value
#  is what tier 3 wants — does it depend on Nσ? — and this script reports it,
#  labelled as a floor, with no rate gate on u. Enabling a rate gate there would
#  manufacture a defect that does not exist.
#
#  COST. Nσ = M·p+1 and the forcing scales as Nσ², multiplied by the tier factor
#  (:native forcing is ~4–6× :none). Relative to P1LFE-2: P1LFE-3 1.8×, P1LFE-4
#  2.8×, P2LFE-2 2.8×, P2LFE-3 5.4×, P2LFE-4 9.0×. PRUNE THE GRID BEFORE RUNNING.
#
#  TOLERANCES must stay ~8 orders tighter than production (nl_tol 1e-12…1e-14,
#  ls_rtol 1e-13) or the study measures the SOLVER, not the discretisation. The
#  defaults below are already right; do not relax them to make a study finish.
#  nl_tol=1e-14 is unreachable for mode=:transient — keep the study static, or use
#  1e-12 there.
#
#  ENV
#    VB_BASES     "M:p,M:p,…"  vertical bases       2:1,3:1,4:1,2:2
#    VB_MODELS    comma list of model numbers 1-8   1,2,3,4,5,6   (7,8 = :full)
#    VB_PU        velocity FE order p_u             3      (⇒ Q3/Q2)
#    VB_DOMAIN    d1 | d2                           d1
#    VB_MODE      static | transient                static
#    VB_LEVELS    refinement levels                 4
#    VB_NX0       coarsest nx                       8
#    VB_D         still-water depth                 2.5    (d ≠ 1 deliberately)
#    VB_AB        bed amplitude (variable-bed)      0.2
#    VB_NLTOL     Newton tolerance                  1e-12
#    VB_NLITER    Newton budget                     50 linear / 400 nonlinear
#    VB_OUT       output directory                  output/local/mms_vbasis
#
#  RUN
#    julia --project=. examples/local_mms/run_vertical_basis_study.jl
#    # tier 1, the paper claim (30 studies — hours):
#    VB_BASES=2:1,3:1,4:1,2:2,3:2 VB_MODELS=1,2,3,4,5,6 \
#        julia --project=. examples/local_mms/run_vertical_basis_study.jl
#    # tier 3, the :full floor vs Nσ:
#    VB_MODELS=7,8 VB_BASES=2:1,3:1 \
#        julia --project=. examples/local_mms/run_vertical_basis_study.jl
# ==============================================================
using GridapBALFEM, Printf

gs(k,d)=get(ENV,k,d); gi(k,d)=parse(Int,get(ENV,k,string(d)))
gf(k,d)=parse(Float64,get(ENV,k,string(d)))

#  The eight models of §4 of CLAUDE.md, as (regime, flat_bed, nl_pressure).
#  `rate_u` is false exactly where the MMS cannot reach the operator — see the
#  tier-3 note above. Numbering matches the verified-scope table.
const MODELS = Dict(
    1 => (regime=:linear,    flat_bed=true,  nlp=:none,   rate_u=true),
    2 => (regime=:linear,    flat_bed=false, nlp=:none,   rate_u=true),
    3 => (regime=:nonlinear, flat_bed=true,  nlp=:none,   rate_u=true),
    4 => (regime=:nonlinear, flat_bed=false, nlp=:none,   rate_u=true),
    5 => (regime=:nonlinear, flat_bed=true,  nlp=:native, rate_u=true),
    6 => (regime=:nonlinear, flat_bed=false, nlp=:native, rate_u=true),
    7 => (regime=:nonlinear, flat_bed=true,  nlp=:full,   rate_u=false),
    8 => (regime=:nonlinear, flat_bed=false, nlp=:full,   rate_u=false),
)

bases = map(split(gs("VB_BASES","2:1,3:1,4:1,2:2"), ",")) do tok
    mp = split(strip(tok), ":")
    length(mp) == 2 || error("VB_BASES: expected \"M:p\" entries, got \"$tok\"")
    (M=parse(Int,mp[1]), p=parse(Int,mp[2]))
end
mods   = parse.(Int, split(gs("VB_MODELS","1,2,3,4,5,6"), ","))
p_u    = gi("VB_PU",3); dom = Symbol(gs("VB_DOMAIN","d1"))
mode   = Symbol(gs("VB_MODE","static")); levels = gi("VB_LEVELS",4)
nx0    = gi("VB_NX0",8); dval = gf("VB_D",2.5); ab = gf("VB_AB",0.2)
nltol  = gf("VB_NLTOL",1e-12)
outdir = gs("VB_OUT","output/local/mms_vbasis"); mkpath(outdir)
const TOLP = 0.3

println("#"^84)
println("#  VERTICAL-BASIS CONVERGENCE STUDY — is the order INDEPENDENT of the σ-basis?")
println("#    bases  : ", join(["P$(b.p)LFE-$(b.M) (Nσ=$(b.M*b.p+1))" for b in bases], "  "))
println("#    models : $mods    pairing Q$(p_u)/Q$(p_u-1)   $dom  $mode  levels=$levels nx0=$nx0")
println("#    gates  : u → $(p_u+1)   eta → $(p_u)   (different optima, by design)")
println("#    ⚠ models 7/8 (:full) are NOT rate-gated on u — the floor is reported instead")
println("#"^84); flush(stdout)

rows = []; summ = []
for b in bases, mno in mods
    haskey(MODELS, mno) || error("VB_MODELS: unknown model $mno (valid 1–8)")
    m = MODELS[mno]
    nliter = gi("VB_NLITER", m.regime === :linear ? 50 : 400)
    println("\n" * "-"^84)
    println("  P$(b.p)LFE-$(b.M)  ×  Model $mno  ($(m.regime) / $(m.flat_bed ? "flat" : "varbed") / :$(m.nlp))")
    println("-"^84); flush(stdout)
    #  A study that cannot COMPLETE is a failed entry, not an aborted campaign:
    #  letting it propagate would destroy the report of every study that did work.
    r = try
        run_conv_study(; p_u=p_u, domain=dom, mode=mode, levels=levels,
                         nx0=nx0, ny_1d=3, Lx=1.7, Ly=1.1, d=dval,
                         M=b.M, p_vert=b.p,          # c_bdy ⇒ resolve_cbdy(M)
                         dt=1e-5, nsteps=100, nl_tol=nltol, nl_iter=nliter,
                         regime=m.regime, nl_pressure=m.nlp,
                         flat_bed=m.flat_bed, a_b=m.flat_bed ? 0.0 : ab,
                         kbx=1.3, kby=0.0, verbose=true)
    catch e
        msg = first(split(sprint(showerror, e), '\n'))
        println("  ERROR  $msg")
        occursin("did not converge", msg) && println(
            "         ⇒ under-converged Newton reads exactly like a wrong operator.\n" *
            "           Raise VB_NLITER (now $nliter). Do NOT loosen VB_NLTOL — that\n" *
            "           would put the algebraic error inside the rate being measured.")
        push!(summ, (tag="P$(b.p)LFE-$(b.M) Model $mno", Nsigma=b.M*b.p+1,
                     fit_eta=NaN, opt_eta=NaN, fit_u=NaN, opt_u=NaN,
                     e_u_floor=NaN, rate_u=m.rate_u, verdict="ERROR"))
        flush(stdout); continue
    end
    for i in eachindex(r.h)
        push!(rows, (r.tag, r.Nsigma, r.h[i], r.ndofs[i], r.e_eta[i], r.e_u[i]))
    end
    #  Read the LAST PAIRWISE rate, not the fitted slope: the fit averages over
    #  pre-asymptotic levels, and a saturated slope and a wrong coefficient produce
    #  the same fitted number.
    okη = abs(r.pw_eta[end] - r.opt_eta) < TOLP
    oku = m.rate_u ? abs(r.pw_u[end] - r.opt_u) < TOLP : true
    push!(summ, (tag=r.tag, Nsigma=r.Nsigma,
                 fit_eta=r.fit_eta, opt_eta=r.opt_eta,
                 fit_u=r.fit_u, opt_u=r.opt_u, e_u_floor=r.e_u[end],
                 rate_u=m.rate_u, verdict=(okη && oku) ? "PASS" : "CHECK"))
    m.rate_u || @printf("    :full ⇒ e_u FLOOR = %.6e at Nσ=%d (p_u=%.3f, NOT a rate)\n",
                        r.e_u[end], r.Nsigma, r.fit_u)
    flush(stdout)
end

csv = joinpath(outdir, "mms_vbasis_$(dom)_$(mode).csv")
open(csv,"w") do io
    println(io, "study,Nsigma,h,ndofs,e_eta,e_u")
    for r in rows; @printf(io, "%s,%d,%.8g,%d,%.10e,%.10e\n", r...); end
end

println("\n" * "="^100)
println("  VERTICAL-BASIS SUMMARY   (u is not rate-gated for :full — floor reported instead)")
println("="^100)
@printf("  %-46s %4s  %-13s %-13s %-12s %s\n",
        "study","Nσ","eta fit/opt","u fit/opt","e_u(finest)","verdict")
for s in summ
    @printf("  %-46s %4d  %5.2f/%-7.0f %5.2f/%-7s %.4e  %s\n",
            s.tag, s.Nsigma, s.fit_eta, s.opt_eta, s.fit_u,
            s.rate_u ? string(Int(s.opt_u)) : "floor", s.e_u_floor, s.verdict)
end
println("\n  wrote $csv")
println("\n  Interpreting this table:")
println("    every rate optimal across bases ⇒ the order is INDEPENDENT of the σ-basis,")
println("      which is the direct quantitative support for the basis-agnosticism claim;")
println("    one basis off-optimal ⇒ check the ERROR MAGNITUDE and the pairwise SEQUENCE")
println("      before concluding — a saturated study and a wrong coefficient look identical;")
println("    :full rows ⇒ compare e_u(finest) ACROSS Nσ. That, not a rate, is tier 3's answer.")
