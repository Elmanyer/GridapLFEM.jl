# ==============================================================
#  test_nlpressure_distributed.jl — distributed nl_pressure_full smoke
#                                       (4 ranks, 2×2)
#
#  Validates the distributed frozen-projection mass solve
#  (nlpressure.jl :: build_nlp_ctx(...; distributed=true) →
#  CGSolver(JacobiLinearSolver())) by running the SAME tanh-bar case as gate
#  G3 of test_nlpressure.jl (ALL pressure flags on) both distributed and
#  sequentially, and comparing the eta_max trajectories.
#
#  RUN:
#    ~/.julia/bin/mpiexecjl --project=. -n 4 julia --project=. \
#        GridapLFEM.jl/test/test_nlpressure_distributed.jl
# ==============================================================

include(joinpath(@__DIR__, "..", "src", "GridapLFEM.jl"))
using .GridapLFEM
using Printf

# sequential reference (same config, run once via setup_and_run): see
# test_nlpressure.jl gate G3 — bar run, 40 steps.
const REF_EMAX = 0.0068979802 # sequential max η over the bar run (LU-based projections)
const REF_RTOL = 5e-3         # CG(1e-10)+Jacobi projections vs LU ⇒ loose agreement

bar(x) = 3.5 - 0.5*1.5*(tanh((x[1]-20.0)/4.0) - tanh((x[1]-32.0)/4.0))

diags, vert, prob = setup_and_run_distributed(
    cpu_grid=(2,2),
    M=2, h_val=3.5, T_wave=2.0, A_wave=0.001,
    domain=((0.0,60.0),(0.0,2.0)), partition=(60,2), p_horizontal=2,
    x_wm=8.0, sponge_wL=8.0, sponge_wR=8.0, mu_max=30.0,
    T_final=2.0, dt=0.05, h_bathy=bar,
    regime=:nonlinear, nl_pressure=:full, flat_bed=false,   # tanh bar → variable bathymetry (∇h ON)
    save_every=0, print_dt=0.4)

is_rank0 = get(ENV, "OMPI_COMM_WORLD_RANK", get(ENV, "PMI_RANK", "0")) == "0"
if is_rank0
    emax = maximum(d.eta_max for d in diags)
    nan_free = all(!isnan(d.eta_max) for d in diags)
    rel = abs(emax - REF_EMAX) / REF_EMAX
    println("=" ^ 60)
    println("  test_nlpressure_distributed.jl — results (4 ranks, 2×2)")
    println("=" ^ 60)
    @printf("  steps=%d  max η=%.7f  (ref %.7f, rel=%.2e)\n", length(diags), emax, REF_EMAX, rel)
    results = Dict(
        "no NaN"                                => nan_free,
        "bounded (max η < 0.02)"                => emax < 0.02,
        "matches sequential (rel < $(REF_RTOL))" => rel < REF_RTOL,
    )
    for (name, ok) in sort(collect(results); by=first)
        println(ok ? "  PASS  $name" : "  FAIL  $name")
    end
    n_pass = count(values(results)); n_fail = length(results) - n_pass
    @printf("  Results: %d PASS,  %d FAIL\n", n_pass, n_fail)
    flush(stdout)
    n_fail > 0 && error("test_nlpressure_distributed: $n_fail failed!")
    println("  Distributed nl_pressure_full (CG+Jacobi mass solve) validated.")
    flush(stdout)
end
