# ==============================================================
#  timeloop_alg_dist.jl — Distributed (MPI) mesh, solver stack and time loop
#
#  Port of ../../LFE-M_2D_solver/src/timeloop2D_dist.jl to the algebraic
#  package. Because the stacked residual/Jacobians are Gridap-native CellField
#  algebra (Operation is forwarded for DistributedCellField), THE SAME
#  residual_alg/jacobian_*_alg run distributed unchanged — including the FULL
#  nonlinear physics (advection, slope pressure, P_full, nl_pressure68). No
#  owned V⊗H loop is needed, unlike the old solver.
#
#  Load-bearing conventions (old solver, revalidated):
#    * distributed solve = ThetaMethod(NewtonSolver(GMRESSolver(;Pr=Jacobi)))
#      from GridapSolvers — LU/NLSolver are sequential-only at scale;
#    * norm(PVector, Inf) is BROKEN (PartitionedArrays 0.3.5) → own_values +
#      reduce(max, …; init=0.0);
#    * all I/O behind i_am_main(ranks); mkpath on rank 0 + MPI.Barrier;
#    * point gauges omitted (inter-rank point search) — diags = (t, eta_max);
#    * launch with ~/.julia/bin/mpiexecjl (system mpiexec → PMIx mismatch).
# ==============================================================

"""
    build_horizontal_model_alg_distributed(ranks, cpu_grid, domain, partition)

Partitioned 2D Cartesian mesh across MPI ranks. `cpu_grid=(px,py)` is the
process topology (`px·py == mpiexec -n`); `domain=(x0,x1,y0,y1)` (flat 4-tuple);
`partition=(nx,ny)` cells (nx divisible by px, ny by py). Returns `(model, trian)`.
"""
function build_horizontal_model_alg_distributed(ranks, cpu_grid::Tuple,
                                                domain::NTuple{4,Float64},
                                                partition::Tuple)
    model = CartesianDiscreteModel(ranks, cpu_grid, domain, partition)
    trian = Triangulation(model)
    return model, trian
end

"""
    build_ode_solver_alg_distributed(dt; theta, nl_iter, nl_tol, ls_rtol, ls_maxiter)

Distributed ODE solver factory: ThetaMethod wrapping NewtonSolver(GMRES with
Jacobi preconditioner) — the scalable GridapSWE-pattern stack.
"""
function build_ode_solver_alg_distributed(dt::Float64;
                                          solver_type :: Symbol  = :theta,
                                          theta       :: Float64 = 0.5,
                                          nl_iter     :: Int     = 20,
                                          nl_tol      :: Float64 = 1e-10,
                                          ls_rtol     :: Float64 = 1e-9,
                                          ls_maxiter  :: Int     = 800)
    ls  = GMRESSolver(ls_maxiter; Pr=JacobiLinearSolver(), rtol=ls_rtol,
                      atol=1e-14, verbose=false)
    nls = NewtonSolver(ls; maxiter=nl_iter, atol=nl_tol, rtol=1e-10, verbose=false)
    if solver_type == :theta
        return ThetaMethod(nls, dt, theta)
    else
        error("Distributed algebraic solver supports :theta (GMRES+Jacobi+Newton).")
    end
end

"""
    eta_max_alg_distributed(eta_n)

Global max |η| across ranks: own_values only (no ghost duplication) + MPI max
reduction. (norm(PVector, Inf) is broken in PartitionedArrays 0.3.5.)
"""
function eta_max_alg_distributed(eta_n)
    free_vals = get_free_dof_values(eta_n)
    local_maxes = map(own_values(free_vals)) do lv
        isempty(lv) ? 0.0 : maximum(abs, lv)
    end
    return reduce(max, local_maxes; init=0.0)
end

"""
    run_time_loop_alg_dist(ranks, op, solver, u0, t0, T_final; output_dir,
                           save_every, trian, Nσ, print_dt, recon, trial_space, dt)

Distributed time loop. Returns `[(t, eta_max)]` (eta_max by MPI reduction; no
point gauges). VTK: per-σ-node components of the stacked fields under the OLD
field names (`eta, u1x, u1y, …`), plus the reconstructed `w_s<σ>`/`p_s<σ>`
fields when `recon` is given (from `build_field_recon_alg` — pure CellField
algebra, distributed-transparent).
"""
function run_time_loop_alg_dist(ranks, op, solver, u0,
                                t0::Float64, T_final::Float64;
                                output_dir :: String  = "output",
                                save_every :: Int     = 0,
                                trian                 = nothing,
                                Nσ         :: Int     = 1,
                                print_dt   :: Float64 = 10.0,
                                recon                 = nothing,
                                trial_space           = nothing,
                                dt         :: Float64 = 0.0,
                                nlp                   = nothing)   # (prob, ctx) for nl_pressure_full
    if i_am_main(ranks)
        mkpath(output_dir)
    end
    MPI.Barrier(MPI.COMM_WORLD)

    odesol = solve(solver, op, t0, T_final, u0)

    prev_vals    = nothing     # previous-step DOFs (PVector) → u̇ backward FD
    diags        = NamedTuple{(:t,:eta_max),Tuple{Float64,Float64}}[]
    t_last_print = t0
    step         = 0
    do_vtk       = !isnothing(trian) && save_every > 0

    function _loop(pvd)
        for (t_n, u_n) in odesol
            step  += 1
            eta_n  = u_n[1]
            emax   = eta_max_alg_distributed(eta_n)
            push!(diags, (t=t_n, eta_max=emax))

            if i_am_main(ranks) && t_n - t_last_print >= print_dt
                @printf("  [alg-dist] t=%8.3f   eta_max=%10.6f   step %d\n",
                        t_n, emax, step)
                flush(stdout)
                t_last_print = t_n
            end

            if !isnothing(pvd) && step % save_every == 0
                tn_str = replace(@sprintf("%.4f", t_n), "." => "_")
                fname  = joinpath(output_dir, "sol_t_$(tn_str)")
                fields = Pair{String,Any}["eta" => eta_n]
                for k in 1:Nσ
                    push!(fields, "u$(k)x" => alg_component(u_n[2], k))
                    push!(fields, "u$(k)y" => alg_component(u_n[3], k))
                end
                if recon !== nothing
                    u_prev = prev_vals === nothing ? nothing :
                             FEFunction(trial_space, prev_vals)
                    append!(fields, extra_field_cellfields_alg(u_n, u_prev, dt, recon, trian))
                end
                pvd[t_n] = createvtk(trian, fname; cellfields=fields)
            end

            if recon !== nothing
                prev_vals = copy(get_free_dof_values(u_n))
            end

            # nl_pressure_full: refresh the frozen projections π𝖲, π𝖻 (CG+Jacobi
            # distributed mass solve, see nlpressure_alg.jl :: build_nlp_ctx)
            if nlp !== nothing
                update_nlp_state!(nlp[1], nlp[2], u_n)
            end

            if isnan(emax) || emax > 1e4
                if i_am_main(ranks)
                    @warn "Diverged at t=$t_n (eta_max=$emax)"
                end
                break
            end
        end
    end

    if do_vtk
        pvd_path = joinpath(output_dir, "solution")
        createpvd(ranks, pvd_path) do pvd
            _loop(pvd)
        end
    else
        _loop(nothing)
    end

    if i_am_main(ranks)
        @printf("[alg-dist] Done. %d steps, final t=%.3f\n",
                step, isempty(diags) ? t0 : diags[end].t)
        flush(stdout)
    end
    return diags
end
