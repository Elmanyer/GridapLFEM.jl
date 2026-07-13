# ==============================================================
#  utilities_alg_dist.jl — Distributed driver for the algebraic LFE-M solver
#
#  setup_and_run_alg_distributed: parallel counterpart of setup_and_run_alg,
#  built on the Gridap-native distributed pattern (as the old solver's
#  setup_and_run_lfem_distributed / GridapSWE run_distributed):
#      with_mpi() do distribute
#          ranks = distribute(LinearIndices((prod(cpu_grid),)))
#          model = CartesianDiscreteModel(ranks, cpu_grid, domain, partition)
#          … TransientFEOperator + GMRES+Jacobi+Newton + solve(…) iterator …
#      end
#
#  ── FULL-PHYSICS SCOPE (the improvement over the old solver) ──
#  The stacked algebraic residual is cheap and Gridap-native, so this ONE
#  distributed path carries every physics flag — advection, lin_pressure,
#  P_full, nl_pressure68 — with the same hand Jacobians as the sequential
#  driver. (The old solver had to restrict its Gridap distributed path to the
#  linear core and route the nonlinear advection through an owned V⊗H loop.)
#
#  Launch with Julia's own launcher (system mpiexec has a PMIx mismatch here):
#    ~/.julia/bin/mpiexecjl --project=. -n 4 julia --project=. my_script.jl
#  with the script calling setup_and_run_alg_distributed(cpu_grid=(2,2), …)
#  and n == prod(cpu_grid).
# ==============================================================

"""
    setup_and_run_alg_distributed(; cpu_grid, M, p_vert, c_bdy, domain, partition,
                                    fe_order, d_val, g, T_wave, A_wave, x_wm, y_wm,
                                    sponge_*, mu_max, T_final, dt, theta, output_dir,
                                    save_every, y_wall_bc, x_wall_bc, linearised,
                                    advection, lin_pressure, P_full, nl_pressure68,
                                    d_func, eta0_func, write_w, write_pressure, rho,
                                    nl_iter, nl_tol, ls_rtol, ls_maxiter, print_dt)

Distributed (MPI) driver for the stacked algebraic LFE-M solver. Mirrors
`setup_and_run_alg` with the horizontal mesh/assembly distributed across
`cpu_grid = (px, py)` ranks (`px·py == mpiexecjl -n N`); the vertical
pre-computation is replicated per rank (tiny). ALL physics flags are supported
in parallel. Returns `(diags, vert, prob)` with `diags = [(t, eta_max)]`
(eta_max via MPI reduction; point gauges are sequential-only).
"""
function setup_and_run_alg_distributed(;
    cpu_grid     :: Tuple   = (2, 2),
    # Vertical
    M            :: Int     = 2,
    p_vert       :: Int     = 1,
    c_bdy                   = nothing,
    # Horizontal
    domain                  = ((0.0, 60.0), (0.0, 20.0)),
    partition    :: Tuple   = (120, 40),
    fe_order     :: Int     = 2,
    # Physics
    d_val        :: Float64 = 3.5,
    g            :: Float64 = 9.81,
    T_wave       :: Float64 = 1.6,
    A_wave       :: Float64 = 0.001,
    x_wm         :: Float64 = 12.0,
    y_wm                    = nothing,     # nothing → line source (plane wave)
    sponge_wL    :: Float64 = 12.0,
    sponge_wR    :: Float64 = 12.0,
    sponge_wB    :: Float64 = 0.0,
    sponge_wT    :: Float64 = 0.0,
    mu_max       :: Float64 = 5.0,
    # Time integration
    T_final      :: Float64 = 12.8,
    dt           :: Float64 = 0.02,
    theta        :: Float64 = 0.5,
    # Output
    output_dir   :: String  = joinpath(@__DIR__, "..", "output", "alg_dist_out"),
    save_every   :: Int     = 0,
    print_dt                = nothing,
    # BCs
    y_wall_bc    :: Bool    = true,
    x_wall_bc    :: Bool    = false,
    # Physics flags — ALL supported distributed
    linearised   :: Bool    = false,
    advection    :: Bool    = true,
    lin_pressure :: Bool    = false,
    P_full       :: Bool    = false,
    nl_pressure68:: Bool    = false,      # native NL-pressure set c∈{3,6,7,8} — distributed
    nl_pressure_full :: Bool = false,     # + c∈{1,2,4,5}: frozen L²-projections, distributed
                                          #   mass solve = CGSolver(JacobiLinearSolver())
    nlp_cg_rtol  :: Float64 = 1e-10,      # projection-solve tolerance (nl_pressure_full)
    nlp_cg_maxiter :: Int   = 500,
    d_func                  = nothing,
    eta0_func               = nothing,     # η₀(x) initial release (REQUIRES x_wall_bc=true)
    # Field output
    write_w        :: Bool    = false,
    write_pressure :: Bool    = false,
    rho            :: Float64 = 1025.0,
    # Solver tolerances
    nl_iter      :: Int     = 20,
    nl_tol       :: Float64 = 1e-10,
    ls_rtol      :: Float64 = 1e-9,
    ls_maxiter   :: Int     = 800,
)
    n_procs = prod(cpu_grid)

    result = with_mpi() do distribute
        ranks = distribute(LinearIndices((n_procs,)))

        if i_am_main(ranks)
            println("=== 2D LFE-M ALGEBRAIC Distributed Solver (stacked [η,𝖴x,𝖴y]) ===")
            println("  cpu_grid: $cpu_grid  ($(n_procs) ranks total)")
            flush(stdout)
        end

        # ----- Stage 1: Vertical pre-computation (all ranks, identical) -----
        c_bdy_used = isnothing(c_bdy) ?
            get(ALG_DEFAULT_CBDY, M, collect(LinRange(0.0, 1.0, M+1))) : c_bdy
        vert = assemble_vertical_tensors_alg(M, p_vert, c_bdy_used)
        if i_am_main(ranks)
            @printf("  Nσ=%d   ΣΦ=%.6f\n", vert.N_dof, sum(vert.Phi))
        end

        # ----- Stage 2: Distributed 2D horizontal mesh + stacked spaces -----
        if domain isa Tuple{Tuple,Tuple}
            (x0,x1), (y0,y1) = domain
        else
            x0,x1,y0,y1 = domain
        end
        dom_flat = (Float64(x0), Float64(x1), Float64(y0), Float64(y1))
        nx, ny   = partition

        model, trian = build_horizontal_model_alg_distributed(ranks, cpu_grid,
                                                              dom_flat, (nx, ny))
        dO   = Measure(trian, 2*fe_order + 2)
        U, V = build_fe_spaces_alg(model, fe_order, vert.N_dof;
                                   y_wall_bc=y_wall_bc, x_wall_bc=x_wall_bc)
        if i_am_main(ranks)
            @printf("  domain [%.1f,%.1f]×[%.1f,%.1f]  partition %d×%d\n", x0,x1,y0,y1,nx,ny)
            @printf("  Fields: 3 (η + 2 stacked VectorValue{%d}) | lin=%s adv=%s linP=%s Pfull=%s nlP68=%s nlPfull=%s\n",
                    vert.N_dof, string(linearised), string(advection),
                    string(lin_pressure), string(P_full), string(nl_pressure68),
                    string(nl_pressure_full))
        end

        # ----- Stage 3: Physics setup (all ranks) -----
        omega  = 2.0 * pi / T_wave
        k_wave = find_wavenumber_alg(omega, d_val, g)
        if i_am_main(ranks)
            @printf("  Wave: λ=%.2f m, kd=%.2f   CFL_x ~ %.3f\n",
                    2pi/k_wave, k_wave*d_val, sqrt(g*d_val)*dt/((x1-x0)/nx))
        end
        sponge = make_sponge_alg(dom_flat, sponge_wL, sponge_wR, sponge_wB, sponge_wT, mu_max)
        wm = isnothing(y_wm) ? make_wavemaker_line_alg(x_wm, A_wave, T_wave, k_wave) :
                               make_wavemaker_point_alg(x_wm, Float64(y_wm), A_wave, T_wave)
        dfn = isnothing(d_func) ? (x -> d_val) : d_func

        prob = build_problem_alg(vert; g=g, d_func=dfn,
            linearised=linearised, advection=advection, lin_pressure=lin_pressure,
            P_full=P_full, nl_pressure68=nl_pressure68, nl_pressure_full=nl_pressure_full,
            mu_sponge=sponge, wm_src=wm)

        # ----- Stage 4: Distributed ODE operator + solver + IC -----
        op     = build_ode_operator_alg(prob, U, V, trian, dO)
        solver = build_ode_solver_alg_distributed(dt; theta=theta,
                     nl_iter=nl_iter, nl_tol=nl_tol,
                     ls_rtol=ls_rtol, ls_maxiter=ls_maxiter)
        # interpolate_everywhere is REQUIRED distributed (FEFunction(U, zeros) fails)
        u0 = make_initial_conditions_alg(U, vert.N_dof; eta0_func=eta0_func)

        # nl_pressure_full: frozen-projection context (CG+Jacobi mass solve, distributed=true —
        # base `lu` has no method for a partitioned PSparseMatrix, see nlpressure_alg.jl)
        nlp = nl_pressure_full ?
              (prob, build_nlp_ctx(model, fe_order, vert.N_dof, trian, dO;
                                   distributed=true, cg_rtol=nlp_cg_rtol,
                                   cg_maxiter=nlp_cg_maxiter)) : nothing

        # ----- Stage 5: Time loop -----
        recon = build_field_recon_alg(vert, dfn, g; rho=rho,
                                      write_w=write_w, write_pressure=write_pressure)
        if recon !== nothing && i_am_main(ranks)
            @printf("  Field output: write_w=%s write_pressure=%s at σ-levels %s\n",
                    string(write_w), string(write_pressure),
                    string(round.(recon.levels; digits=3)))
        end
        if i_am_main(ranks)
            println("\n=== Time loop (algebraic, distributed) ===")
            flush(stdout)
        end
        pdt = isnothing(print_dt) ? max(dt, T_final/50.0) : Float64(print_dt)
        diags = run_time_loop_alg_dist(ranks, op, solver, u0, 0.0, T_final;
                    output_dir  = output_dir,
                    save_every  = save_every,
                    trian       = (save_every > 0 ? trian : nothing),
                    Nσ          = vert.N_dof,
                    print_dt    = pdt,
                    recon       = recon, trial_space = U, dt = dt, nlp = nlp)

        return (diags, vert, prob)
    end
    return result
end
