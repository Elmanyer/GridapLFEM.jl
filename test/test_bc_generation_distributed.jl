# ==============================================================
#  test_bc_generation_distributed.jl — Dirichlet generation, 4 MPI ranks
#
#  The distributed twin of test_bc_generation.jl case A (small): a regular
#  wave (kd=3, :model polarization) generated from left-boundary Dirichlet
#  data on a 2×2 process grid (transient Dirichlet through GridapDistributed's
#  DistributedTransientTrialFESpace + GMRES+Jacobi+Newton). The WaveInput is
#  deterministic (plain arrays) so every rank builds identical boundary data.
#  Checks: no NaN, bounded, wave generated, agreement of max-η with the
#  sequential reference (GMRES vs LU ⇒ loose 2e-3 rel).
#
#  RUN (from the project root; system mpiexec fails with PMIx mismatch):
#    ~/.julia/bin/mpiexecjl --project=. -n 4 julia --project=. \
#        GridapLFEM.jl/test/test_bc_generation_distributed.jl
# ==============================================================

include(joinpath(@__DIR__, "..", "src", "GridapLFEM.jl"))
using .GridapLFEM
using Printf

# sequential reference: same config via setup_and_run (LU+Newton) — see the
# generation block at the bottom of this file (measured 2026-07-23)
const REF_EMAX = 5.5027223e-04
const REF_RTOL = 2e-3

# kd = 3 at T = 1.6 s (matches test_bc_generation.jl)
g = 9.81; T_wave = 1.6; A_wave = 0.0005
d_val = 1.899

diags, vert, prob = setup_and_run_distributed(
    cpu_grid=(2,2),
    M=2, c_bdy=[0.0,0.728,1.0], d_val=d_val, T_wave=T_wave, A_wave=A_wave,
    domain=((0.0,24.0),(0.0,2.0)), partition=(24,2), fe_order=2,
    wave_bc=:regular, bc_side=:left, bc_profile=:model,
    sponge_wL=0.0, sponge_wR=6.0, sponge_wB=0.0, sponge_wT=0.0, mu_max=30.0,
    T_final=8*T_wave, dt=T_wave/24, linearised=true, advection=false,
    save_every=0, print_every=50, check_every=0,
)

emax     = maximum(d.eta_max for d in diags)
nan_free = all(!isnan(d.eta_max) for d in diags)
rel      = abs(emax - REF_EMAX) / REF_EMAX

results = Dict{String,Bool}(
    "no NaN"                          => nan_free,
    "bounded (max η < 10A)"           => emax < 10*A_wave,
    "wave generated (max η > 0.5A)"   => emax > 0.5*A_wave,
    "matches sequential (rel<$(REF_RTOL))" => rel < REF_RTOL,
)

is_rank0 = get(ENV, "OMPI_COMM_WORLD_RANK", get(ENV, "PMI_RANK", "0")) == "0"
if is_rank0
    println("=" ^ 60)
    println("  test_bc_generation_distributed.jl — results (4 ranks, 2×2)")
    println("=" ^ 60)
    @printf("  steps=%d  max η=%.7e  (ref %.7e, rel=%.2e)\n",
            length(diags), emax, REF_EMAX, rel)
    n_pass = count(values(results)); n_fail = length(results) - n_pass
    for (name, ok) in sort(collect(results); by=first)
        println(ok ? "  PASS  $name" : "  FAIL  $name")
    end
    @printf("  Results: %d PASS,  %d FAIL\n", n_pass, n_fail)
    flush(stdout)
    n_fail > 0 && error("test_bc_generation_distributed: $n_fail failed!")
    println("  Distributed Dirichlet wave generation validated.")
    flush(stdout)
end

# ---- sequential reference generation (run once, paste REF_EMAX above) --------
# diags, _, _ = setup_and_run(
#     M=2, c_bdy=[0.0,0.728,1.0], d_val=1.899, T_wave=1.6, A_wave=0.0005,
#     domain=((0.0,24.0),(0.0,2.0)), partition=(24,2), fe_order=2,
#     wave_bc=:regular, bc_side=:left, bc_profile=:model,
#     sponge_wL=0.0, sponge_wR=6.0, sponge_wB=0.0, sponge_wT=0.0, mu_max=30.0,
#     T_final=8*1.6, dt=1.6/24, linearised=true, advection=false,
#     save_every=0, print_every=50, check_every=0)
# @printf("REF_EMAX = %.7e\n", maximum(d.eta_max for d in diags))
