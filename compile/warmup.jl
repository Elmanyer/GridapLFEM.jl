
# Script executed in compile.jl to precompile the GridapSWE module and create a system image for faster loading in future runs. 
# After precompiling the module, the created system image can be used to execute module functions with command:
# julia --project=. -J TS_SWE_sysimage.so script_to_run.jl
# or in parallel
# mpiexecjl -n 6 julia --project=. -J TS_SWE_sysimage.so script_to_run.jl

using MPI
if !MPI.Initialized()
    MPI.Init()
end

using GridapSWE

println("GridapSWE --| Running warmup for the GridapSWE module...")

dom_params = Dict(
        :Lx => 0.001,           # Domain x-axis length
        :Ly => 0.001,           # Domain y-axis length
        :Nx => 4,               # Domain x-axis number of cells
        :Ny => 4)               # Domain y-axis number of cells 
        
# Solver parameters
solver_params_seq = Dict(
        :dt => 0.001,             # Time step size
        :t0 => 0.0,               # Initial time
        :tF => 0.001,             # Final time
        :tableau => :SDIRK_3_3)   # Butcher tableau for the time-stepping scheme (e.g., SDIRK_3_3 for a 3-stage, 3rd-order SDIRK method)

solver_params_dis = Dict(
        :dt => 0.001,             # Time step size
        :t0 => 0.0,               # Initial time
        :tF => 0.001,             # Final time
        :tableau => :SDIRK_3_3,   # Butcher tableau for the time-stepping scheme (e.g., SDIRK_3_3 for a 3-stage, 3rd-order SDIRK method)
        :ls_atol => 1e-8,         # Absolute tolerance for the linear solver
        :ls_rtol => 1e-6,         # Relative tolerance for the linear solver
        :ls_maxiter => 1000,      # Maximum number of iterations for the linear solver
        :nls_atol => 1e-6,        # Absolute tolerance for the nonlinear solver
        :nls_rtol => 1e-4,        # Relative tolerance for the nonlinear solver
        :nls_maxiter => 1000)     # Maximum number of iterations for the nonlinear solver

println("GridapSWE --|   -> Running sequential benchmark...")
GridapSWE.run_sequential_benchmark(:sinusoidal, :CG, dom_params, solver_params_seq)
GridapSWE.run_sequential_benchmark(:sinusoidal, :CG_SUPG, dom_params, solver_params_seq)
GridapSWE.run_sequential_benchmark(:sinusoidal, :DG, dom_params, solver_params_seq)
println("GridapSWE --|   Done!")

println("GridapSWE --|   -> Running distributed benchmark...")
GridapSWE.run_distributed_benchmark(:sinusoidal, :CG, dom_params, solver_params_dis, (1,1))
GridapSWE.run_distributed_benchmark(:sinusoidal, :CG_SUPG, dom_params, solver_params_dis, (1,1))
GridapSWE.run_distributed_benchmark(:sinusoidal, :DG, dom_params, solver_params_dis, (1,1))
println("GridapSWE --|   Done!")

println("GridapSWE --| Warmup completed for the GridapSWE module. The system image will now be created.")