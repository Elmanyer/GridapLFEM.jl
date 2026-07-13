# ==============================================================
#  GridapLFEM.jl — Algebraic (loop-free) 2D LFE-M Wave Solver
#
#  Stacked-layout reimplementation of the LFE-M depth-integrated wave model
#  (Yang & Liu 2024, JFM 999 A32), following main.tex §8 (corrected: includes
#  the leading-pressure/dispersion term R_P) and the operator simplifications
#  of algebraic_residual_math.md.
#
#  Layout:  MultiField = [η, 𝖴x, 𝖴y]  with 𝖴x,𝖴y ∈ VectorValue{Nσ}
#  (3 fields total — NOT 1+2Nσ scalar fields). All vertical (layer) sums are
#  native constant-tensor contractions; the residual contains no per-layer
#  loops and no MultiField decomposition beyond u[1],u[2],u[3].
#
#  Usage:
#    include("GridapLFEM.jl/src/GridapLFEM.jl"); using .GridapLFEM
#    diags, vert, prob = setup_and_run_alg(M=2, T_wave=1.6, A_wave=0.001)
#
#  Validated against the per-layer oracle solver (../LFE-M_2D_solver/):
#  see test/test_equivalence_alg.jl (virtual-work match ~1e-15) and
#  algebraic_solver_plan.md.
# ==============================================================

module GridapLFEM

using Gridap
using Gridap.Algebra
using Gridap.FESpaces
using Gridap.ReferenceFEs
using Gridap.Geometry
using Gridap.CellData
using Gridap.ODEs
using Gridap.TensorValues
using LinearAlgebra
using SparseArrays
using Printf
# Distributed (MPI) stack — same versions/pattern as the old solver:
#   GridapDistributed 0.4.x, PartitionedArrays 0.3.x, MPI 0.20.x,
#   GridapSolvers 0.6.x (GMRES + Jacobi + NewtonSolver for the distributed solve).
using GridapDistributed
using PartitionedArrays
using MPI
using GridapSolvers
using GridapSolvers.LinearSolvers
using GridapSolvers.NonlinearSolvers

# 2D unit vectors + 1D sigma-mesh gradient extractor
const Ex     = VectorValue(1.0, 0.0)
const Ey     = VectorValue(0.0, 1.0)
const E1_sig = VectorValue(1.0)

include("vertical_alg.jl")       # Stage 1: σ-mesh + full vertical tensor set (incl. P, Pcal)
include("tensors_alg.jl")        # constant-tensor constructors + pointwise Operation helpers
include("horizontal_alg.jl")     # Stage 2: 2D mesh + stacked [η,𝖴x,𝖴y] FE spaces + wall BCs
include("problem_alg.jl")        # AlgebraicLFEM struct + loop-free residual + hand Jacobians
include("nlpressure_alg.jl")     # FULL nonlinear pressure (native / ∇h exact-IBP / frozen proj.)
include("reconstruct_alg.jl")    # w / total-pressure σ-level VTK fields (serial + distributed)
include("timeloop_alg.jl")       # ODE solver factory + sequential time loop (VTK + recon)
include("utilities_alg.jl")      # dispersion analysis, sponge, wavemakers, sequential driver
include("timeloop_alg_dist.jl")  # DISTRIBUTED mesh builder + GMRES solver + time loop
include("utilities_alg_dist.jl") # DISTRIBUTED driver setup_and_run_alg_distributed

# Vertical stage
export build_vertical_model_alg
export assemble_vertical_tensors_alg

# Tensor/algebra helpers
export alg_to_vec, alg_to_tensor2, alg_to_tensor3
export alg_dx, alg_dy, alg_mul, alg_dot, alg_dc3, alg_outer, alg_vec2

# Horizontal stage
export build_horizontal_model_alg
export build_fe_spaces_alg

# Problem
export AlgebraicLFEM, build_problem_alg
export residual_alg, jacobian_u_alg, jacobian_u_t_alg
export build_ode_operator_alg, build_ode_operator_alg_ad

# Nonlinear pressure (full physics)
export nlp_native_contrib, nlp_gradh_contrib, nlp_frozen_N
export nlp_gradH_frozen_contrib, nlp_P_frozen_contrib
export build_nlp_ctx, update_nlp_state!

# Time integration
export build_ode_solver_alg
export make_initial_conditions_alg
export run_time_loop_alg

# Utilities
export find_wavenumber_alg
export dispersion_ratio_alg, applicable_kd_alg
export make_sponge_alg, make_wavemaker_line_alg, make_wavemaker_point_alg
export setup_and_run_alg

# Field reconstruction (w / total pressure at σ-levels, VTK)
export build_field_recon_alg, extra_field_cellfields_alg, recon_level_str_alg

# Distributed (MPI)
export build_horizontal_model_alg_distributed
export build_ode_solver_alg_distributed
export run_time_loop_alg_dist
export setup_and_run_alg_distributed

end # module GridapLFEM
