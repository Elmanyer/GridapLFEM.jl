# ==============================================================
#  warmup.jl — trace-compile the LFE-M solver into the sysimage
#
#  Run as PackageCompiler's precompile_execution_file (see compile.jl). It runs a
#  handful of TINY solves so the heavy Gridap FEM kernels — the residual, both hand
#  Jacobians, the pressure machinery and the time loop — get specialized and baked
#  into the sysimage. Both regimes (:linear and :nonlinear + :full pressure) and
#  BOTH the sequential (CellField) and distributed (DistributedCellField) code
#  paths are exercised, because they specialize to different types.
#
#  Meshes are deliberately minimal (4x4, ~2 steps); the FIRST solve triggers the
#  full ~30-45 min cold compile — that is the whole point, we pay it once here so
#  every cluster rank later just LOADS the sysimage instead of recompiling (which
#  is what was OOM-killing the 32-rank runs).
# ==============================================================

const ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(ROOT, "src", "GridapLFEM.jl"))
using .GridapLFEM

# ── tiny SEQUENTIAL solve — specializes residual/Jacobians for CellField ──────
seq(reg, nlp) = setup_and_run(
    M=2, c_bdy=[0.0, 0.728, 1.0], domain=((0.0, 4.0), (0.0, 4.0)), partition=(4, 4),
    p_horizontal=2, h_val=3.5, T_wave=1.6, A_wave=0.001,
    x_wm=1.0, y_wm=nothing, sponge_wL=0.5, sponge_wR=0.5, mu_max=5.0,
    T_final=0.1, dt=0.05, regime=reg, nl_pressure=nlp, flat_bed=true,
    save_every=0, print_every=10^6,
    output_dir=joinpath(ROOT, "output", "warmup_seq"))

# ── tiny DISTRIBUTED solve (1 rank) — specializes for DistributedCellField+GMRES
dist(reg, nlp) = setup_and_run_distributed(
    cpu_grid=(1, 1), M=2, c_bdy=[0.0, 0.728, 1.0],
    domain=(0.0, 4.0, 0.0, 4.0), partition=(4, 4),
    p_horizontal=2, h_val=3.5, T_wave=1.6, A_wave=0.001,
    x_wm=1.0, y_wm=nothing, sponge_wL=0.5, sponge_wR=0.5, mu_max=5.0,
    T_final=0.1, dt=0.05, regime=reg, nl_pressure=nlp, flat_bed=true,
    save_every=0, print_every=10^6,
    output_dir=joinpath(ROOT, "output", "warmup_dist"))

println("warmup: sequential traces (linear + nonlinear/full) …")
seq(:linear, :none)
seq(:nonlinear, :full)

println("warmup: distributed traces (cpu_grid=(1,1)) …")
try
    dist(:linear, :none)
    dist(:nonlinear, :full)
catch e
    @warn "distributed warmup skipped (MPI unavailable in this build subprocess); " *
          "the sequential trace is still baked, the distributed glue will JIT on " *
          "the first cluster run." exception = e
end

println("warmup done.")
