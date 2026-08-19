# ==============================================================
#  GridapBALFEM.jl — Algebraic (loop-free) 2D LFE-M Wave Solver
#
#  Depth-integrated, non-hydrostatic free-surface wave solver for the LFE-M
#  model (Yang & Liu 2024, JFM 999 A32). The water column σ∈[0,1] is resolved
#  by Nσ vertical finite-element modes; the vertical velocity and the
#  non-hydrostatic pressure are eliminated analytically into small, precomputed
#  vertical tensors, leaving a system of 2D PDEs in (H, 𝖴) solved on a
#  horizontal FE mesh by Gridap. The global residual assembled here is the
#  virtual-work form of §8 of the accompanying derivation (main.tex), whose
#  leading-pressure term R_P carries the frequency dispersion.
#
#  Layout:  MultiField = [η, 𝖴x, 𝖴y]  with 𝖴x,𝖴y ∈ VectorValue{Nσ}
#  (3 fields total). Stacking the vertical/layer index into the FE value type
#  turns every layer sum into a native constant-tensor contraction, so the
#  residual contains no per-layer loops and touches the MultiField only through
#  u[1]=η, u[2]=𝖴x, u[3]=𝖴y. This keeps the assembly well-typed and compact,
#  and lets the identical code run sequentially and distributed.
#
#  Usage (GridapBALFEM is a Julia package; activate this directory as the project):
#    julia --project=/path/to/GridapBALFEM.jl
#    using GridapBALFEM
#    diags, vert, prob = setup_and_run(M=2, T_wave=1.6, A_wave=0.001)
#
#  It is loaded with `using GridapBALFEM`, never `include`d: being a real package is
#  what lets PackageCompiler bake the solver AND its Gridap specialisations into
#  the cluster system image (see compile/), and what gives every sequential run a
#  cached precompile. An `include()`d module is rebuilt in every process, which is
#  what used to OOM-kill the 32-64 rank runs.
# ==============================================================

module GridapBALFEM

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
using ForwardDiff          # analytic-MMS generic strong-form evaluator (src/mms.jl) only
# Distributed (MPI) stack: GridapDistributed 0.4.x, PartitionedArrays 0.3.x,
#   MPI 0.20.x, GridapSolvers 0.6.x (GMRES + Jacobi + NewtonSolver — the scalable
#   Krylov solve used for the partitioned linear systems on the cluster).
using GridapDistributed
using PartitionedArrays
using MPI
using GridapSolvers
using GridapSolvers.LinearSolvers
using GridapSolvers.NonlinearSolvers
# Stochastic sea-state synthesis (Dirichlet boundary wave generation)
using WaveSpec

# 2D unit vectors + 1D sigma-mesh gradient extractor
const Ex     = VectorValue(1.0, 0.0)
const Ey     = VectorValue(0.0, 1.0)
const E1_sig = VectorValue(1.0)

# Global physical constants. `g` and `rho` are declared once here and used as the
# default for every `g`/`rho` keyword argument in the module (a single source of
# truth); the kwargs remain overridable per call.
const g   = 9.81      # gravitational acceleration [m/s²]
const rho = 1025.0    # water density [kg/m³]

include("vertical.jl")       # Stage 1: σ-mesh + full vertical tensor set (incl. P, Pcal)
include("tensors.jl")        # constant-tensor constructors + pointwise Operation helpers
include("horizontal.jl")     # Stage 2: 2D mesh + stacked [η,𝖴x,𝖴y] FE spaces + wall BCs
include("problem.jl")        # BALFEMProblem struct + loop-free residual + hand Jacobians
include("nlpressure.jl")     # FULL nonlinear pressure (native / ∇h exact-IBP / frozen proj.)
include("reconstruct.jl")    # w / total-pressure σ-level VTK fields (serial + distributed)
include("monitor.jl")        # solver monitor + governing-eq residual checker + reports
include("timeloop.jl")       # ODE solver factory + sequential time loop (VTK + recon)
include("utilities.jl")      # dispersion analysis, sponge, wavemakers, sequential driver
include("waveinput.jl")      # Dirichlet boundary wave generation + WaveSpec coupling
include("timeloop_dist.jl")  # DISTRIBUTED mesh builder + GMRES solver + time loop
include("utilities_dist.jl") # DISTRIBUTED driver setup_and_run_distributed
include("errors.jl")         # L² error norms + convergence-rate fitting
include("mms.jl")            # ANALYTIC (verification) MMS forcing — independent of problem.jl
include("mms_driver.jl")     # MMS solve + spatial/temporal refinement studies

# Vertical stage
export build_vertical_model
export assemble_vertical_tensors

# Tensor/algebra helpers
export alg_to_vec, alg_to_tensor2, alg_to_tensor3
export alg_dx, alg_dy, alg_mul, alg_dot, alg_dc3, alg_outer, alg_vec2

# Horizontal stage
export build_horizontal_model
export build_fe_spaces

# Problem
export BALFEMProblem, build_problem, build_problem_raw, resolve_physics
export global_residual, jacobian_u, jacobian_u_t
export build_ode_operator, build_ode_operator_ad

# Nonlinear pressure (full physics)
export nlp_native_contrib, nlp_gradh_contrib, nlp_frozen_N
export nlp_gradH_frozen_contrib, nlp_P_frozen_contrib
export build_nlp_ctx, update_nlp_state!

# Time integration
export build_ode_solver, build_preconditioner
export make_initial_conditions
export run_time_loop

# Runtime diagnostics (solver monitoring + governing-eq verification)
export SolverMonitor, take_step_stats!
export ResidualChecker, check_residuals
export print_solver_banner, step_report, check_report, final_report
# Field diagnostics: max|η| location + interior/damped split, |u|/|η|, mass and
# energy invariants, per-rank RSS, relative divergence guard, CSV step log.
export RunDiagnostics, build_run_diagnostics, field_diagnostics
export resolve_eta_ref, rss_bytes, masked_max, x_at_max

# --- L² errors / convergence (errors.jl) ------------------------------------
export l2, l2_error, l2_norm_exact, error_measure
export convergence_rate, refinement_table

# --- analytic (verification) MMS (mms.jl, mms_driver.jl) --------------------
export MMSField, field_callables
export mms_eta, mms_ux, mms_uy
export mms_exact_eta, mms_exact_ux, mms_exact_uy
export mms_forcing_stage1, strong_residual_stage1
export model_celerity, eigenmode_callables, standing_mode, exact_cfs
export bathymetry_field, strong_residual_linear, strong_residual_model, mms_forcing
export run_mms_case, run_mms_refinement, mms_dt_independence
export run_model_case, run_model_refinement
export run_conv_study, run_mms_case_distributed
export diag_csv_row, close_diagnostics

# Utilities
export find_wavenumber
export dispersion_ratio, applicable_kd
export make_sponge, make_wavemaker_line, make_wavemaker_point
export setup_and_run

# Dirichlet boundary wave generation (waveinput.jl) + WaveSpec re-export
export WaveInput, eta_bc, ux_bc, uy_bc, incident_fields
export sigma_dof_nodes, model_wavenumber, ramp_value, waveinput_summary
export WaveSpec

# Field reconstruction (w / total pressure at σ-levels, VTK)
export build_field_recon, extra_field_cellfields, recon_level_str

# Distributed (MPI)
export build_horizontal_model_distributed
export build_ode_solver_distributed
export run_time_loop_dist
export setup_and_run_distributed

end # module GridapBALFEM
