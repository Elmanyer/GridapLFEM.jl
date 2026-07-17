# ==============================================================
#  timeloop_alg.jl — ODE solver factory + time loop (stacked layout)
#
#  Port of ../../LFE-M_2D_solver/src/timeloop2D.jl. Only VTK output changes:
#  the stacked VectorValue{Nσ} velocity fields are written per σ-node
#  component with the OLD solver's field names (eta, u1x, u1y, u2x, …), so
#  existing ParaView pipelines work unchanged.
# ==============================================================

"""
    build_ode_solver_alg(dt; solver_type, theta, rho_inf, nl_iter, nl_tol, show_trace)

Gridap ODE solver factory: `:theta` (Crank–Nicolson θ=0.5, recommended),
`:gen_alpha`, `:rk3` (SDIRK). Nonlinear solve: Newton + LU.
"""
function build_ode_solver_alg(dt::Float64;
                              solver_type :: Symbol  = :theta,
                              theta       :: Float64 = 0.5,
                              rho_inf     :: Float64 = 0.5,
                              nl_iter     :: Int     = 20,
                              nl_tol      :: Float64 = 1e-10,
                              show_trace  :: Bool    = false)
    ls  = LUSolver()
    nls = NLSolver(ls; show_trace=show_trace, method=:newton,
                   iterations=nl_iter, ftol=nl_tol)
    if solver_type == :theta
        return ThetaMethod(nls, dt, theta)
    elseif solver_type == :gen_alpha
        return GeneralizedAlpha(nls, dt, rho_inf)
    elseif solver_type == :rk3
        return RungeKutta(nls, ls, dt, :SDIRK_3_4)
    else
        error("Unknown solver_type; choose :theta, :gen_alpha, or :rk3")
    end
end

"""
    make_initial_conditions_alg(U)                      (sequential-only fast path)
    make_initial_conditions_alg(U, Nσ; eta0_func=nothing)

Initial conditions for the stacked MultiFieldFESpace U. The 2-arg form uses
`interpolate_everywhere` — REQUIRED in distributed mode (`FEFunction(U, zeros)`
creates a plain local array and fails on PVector layouts) and equally valid
sequentially. `eta0_func` (x → η₀) enables initial-condition-release problems
(Gaussian hump, soliton); those REQUIRE `x_wall_bc=true` (closed basin).
"""
function make_initial_conditions_alg(U)
    return FEFunction(U, zeros(Float64, num_free_dofs(U)))
end

function make_initial_conditions_alg(U, Nσ::Int; eta0_func = nothing)
    zvv  = VectorValue(ntuple(_ -> 0.0, Nσ)...)
    eta0 = isnothing(eta0_func) ? (x -> 0.0) : eta0_func
    return interpolate_everywhere([eta0, x -> zvv, x -> zvv], U)
end

"Extract σ-node component j of a stacked VectorValue{Nσ} CellField (scalar CellField)."
alg_component(Uf, j::Int) = Operation(v -> v[j])(Uf)

"""
    run_time_loop_alg(op, solver, u0, t0, T_final; output_dir, save_every,
                      trian, Nσ, print_dt, gauges, recon, trial_space, dt)

Time loop from t0 to T_final. Returns `[(t, eta_max, gauge_vals)]`.
`save_every > 0` writes VTK snapshots (fields: eta, u1x, u1y, …, uNσx, uNσy)
plus a `solution.pvd` index. If `recon` (from `build_field_recon_alg`) is given,
the reconstructed `w_s<σ>`/`p_s<σ>` fields are appended to each snapshot
(`trial_space`+`dt` required for the u̇ backward FD of the pressure).
Stops early on NaN or eta_max > 1e4.
"""
function run_time_loop_alg(op, solver, u0, t0::Float64, T_final::Float64;
                           output_dir :: String  = "output",
                           save_every :: Int     = 0,
                           trian                 = nothing,
                           Nσ         :: Int     = 1,
                           print_dt   :: Float64 = 10.0,
                           gauges                = [],
                           recon                 = nothing,
                           trial_space           = nothing,
                           dt         :: Float64 = 0.0,
                           nlp                   = nothing)   # (prob, ctx) for nl_pressure_full
    mkpath(output_dir)
    odesol = solve(solver, op, t0, T_final, u0)

    # previous-step DOFs → u̇ backward FD for the reconstructed pressure
    prev_vals = nothing

    diags        = NamedTuple{(:t,:eta_max,:gauge_vals),
                              Tuple{Float64,Float64,Vector{Float64}}}[]
    t_last_print = t0
    step         = 0
    do_vtk       = !isnothing(trian) && save_every > 0

    gauge_pts = map(g -> (g isa VectorValue ? g : VectorValue(Float64(g[1]), Float64(g[2]))),
                    gauges)

    function _loop(pvd)
        for (t_n, u_n) in odesol
            step  += 1
            eta_n  = u_n[1]
            emax   = maximum(abs.(get_free_dof_values(eta_n)))
            gauge_data = isempty(gauge_pts) ? Float64[] :
                         [eta_n(pt) for pt in gauge_pts]
            push!(diags, (t=t_n, eta_max=emax, gauge_vals=gauge_data))

            if t_n - t_last_print >= print_dt
                @printf("  t=%8.3f   eta_max=%10.6f   step %d\n", t_n, emax, step)
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
                pvd[t_n] = createvtk(trian, fname; cellfields=fields; append=false)
            end

            if recon !== nothing
                prev_vals = copy(get_free_dof_values(u_n))
            end

            # nl_pressure_full: refresh the frozen projections π𝖲, π𝖻 for the next step
            if nlp !== nothing
                update_nlp_state!(nlp[1], nlp[2], u_n)
            end

            if isnan(emax) || emax > 1e4
                @warn "Diverged at t=$t_n (eta_max=$emax)"
                break
            end
        end
    end

    if do_vtk
        pvd_path = joinpath(output_dir, "solution")
        createpvd(pvd_path) do pvd
            _loop(pvd)
        end
    else
        _loop(nothing)
    end

    @printf("Done. %d steps, final t=%.3f\n", step,
            isempty(diags) ? t0 : diags[end].t)
    return diags
end
