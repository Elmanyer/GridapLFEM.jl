# ==============================================================
#  utilities_dist.jl — Distributed (MPI) driver for the LFE-M solver
#
#  `setup_and_run_distributed` is the parallel counterpart of `setup_and_run`.
#  The horizontal mesh, FE spaces, assembly and solve are partitioned across MPI
#  ranks; the whole run is wrapped in `with_mpi() do distribute … end`, which
#  gives each rank its slice of the process grid:
#      with_mpi() do distribute
#          ranks = distribute(LinearIndices((prod(cpu_grid),)))
#          model = CartesianDiscreteModel(ranks, cpu_grid, domain, partition)
#          … TransientFEOperator + GMRES+Jacobi+Newton + solve(…) iterator …
#      end
#
#  Because the stacked residual and its hand Jacobians are expressed purely in
#  Gridap CellField algebra, this single distributed path carries every physics
#  flag — advection, lin_pressure, P_full, nl_pressure68, nl_pressure_full —
#  using the very same code as the sequential driver; only the linear solver
#  (GMRES + Jacobi + Newton) and the reductions differ.
#
#  The vertical pre-computation is tiny and identical on every rank, so it is
#  simply replicated rather than distributed.
#
#  Launch with Julia's own MPI launcher so the runtime matches the MPI build:
#    ~/.julia/bin/mpiexecjl --project=. -n 4 julia --project=. my_script.jl
#  with the script calling setup_and_run_distributed(cpu_grid=(2,2), …) and
#  n == prod(cpu_grid).
# ==============================================================

"""
    setup_and_run_distributed(; cpu_grid, kwargs...) → (diags, vert, prob)

Distributed (MPI) driver for the stacked LFE-M solver. Same physics and workflow
as [`setup_and_run`](@ref), with the horizontal mesh/assembly/solve partitioned
across `cpu_grid = (px, py)` ranks (`px·py` must equal `mpiexecjl -n N`). Every
physics flag is available in parallel. Returns `(diags, vert, prob)` with
`diags = [(t, eta_max, …)]`; the global `eta_max` is obtained by MPI reduction,
and point gauges are not evaluated in parallel. The keyword arguments are
documented inline on the signature below.
"""
function setup_and_run_distributed(;
    cpu_grid     :: Tuple   = (2, 2),      # (px,py) MPI process grid; px·py == mpiexecjl -n N
    # ---- Vertical discretisation (Stage 1, replicated on every rank) ---------
    M            :: Int     = 2,           # number of vertical σ-elements (LFE-M order)
    p_vertical       :: Int     = 1,           # σ-element polynomial order (Nσ = M·p_vertical+1)
    c_bdy                   = nothing,     # σ-node positions in [0,1]; nothing → optimised set
    # ---- Horizontal discretisation (partitioned over cpu_grid) ---------------
    domain                  = ((0.0, 60.0), (0.0, 20.0)),  # ((x0,x1),(y0,y1)) extent [m]
    partition    :: Tuple   = (120, 40),   # (nx,ny) cells (ideally nx%px==0, ny%py==0)
    p_horizontal     :: Int     = 2,           # horizontal FE order (≥2 required)
    # ---- Physical parameters -------------------------------------------------
    h_val        :: Float64 = 3.5,         # still-water depth [m]
    g            :: Float64 = g,        # gravitational acceleration [m/s²]
    T_wave       :: Float64 = 1.6,         # forcing wave period [s]
    A_wave       :: Float64 = 0.001,       # forcing wave amplitude [m]
    # ---- Internal wavemaker (used only when wave_bc is nothing) ---------------
    x_wm         :: Float64 = 12.0,        # wavemaker x-position [m]
    y_wm                    = nothing,     # nothing → line source (plane wave); number → point source
    # ---- Sponge layers -------------------------------------------------------
    sponge_wL    :: Float64 = 12.0,        # left-edge sponge width [m]
    sponge_wR    :: Float64 = 12.0,        # right-edge sponge width [m]
    sponge_wB    :: Float64 = 0.0,         # bottom-edge sponge width [m]
    sponge_wT    :: Float64 = 0.0,         # top-edge sponge width [m]
    mu_max       :: Float64 = 5.0,         # peak sponge / relaxation strength
    # ---- Time integration ----------------------------------------------------
    T_final      :: Float64 = 12.8,        # final simulated time [s]
    dt           :: Float64 = 0.02,        # time step [s]
    solver_type  :: Symbol  = :sdirk,      # integrator: :sdirk (default) | :theta
    tableau      :: Symbol  = :SDIRK_2_2,  # Runge–Kutta tableau when solver_type == :sdirk
    theta        :: Float64 = 0.5,         # θ for :theta (0.5 = Crank–Nicolson)
    # ---- Output --------------------------------------------------------------
    output_dir   :: String  = joinpath(@__DIR__, "..", "output", "dist_out"),  # VTK/pvd destination
    save_every   :: Int     = 0,           # write a VTK snapshot every N steps (0 = no VTK)
    print_dt                = nothing,     # if set, report every this many simulated seconds instead
    # ---- Boundary conditions -------------------------------------------------
    y_wall_bc    :: Symbol  = :wall,       # y-edge BC: :wall (𝖴y=0) | :open (natural) | :periodic
    x_wall_bc    :: Bool    = false,       # solid walls on the x-edges (true for closed-basin IC)
    # ---- Physics flags (all supported in parallel) ---------------------------
    linearised   :: Bool    = false,       # linear regime
    advection    :: Bool    = true,        # nonlinear advection block
    lin_pressure :: Bool    = false,       # A/K linear slope-pressure package
    P_full       :: Bool    = false,       # keep all three slope components of R_P
    nl_pressure68:: Bool    = false,       # nonlinear-pressure native set c∈{3,6,7,8}
    nl_pressure_full :: Bool = false,      # + c∈{1,2,4,5}; distributed mass solve = CG + Jacobi
    nlp_cg_rtol  :: Float64 = 1e-10,       # CG tolerance for the frozen-projection solve
    nlp_cg_maxiter :: Int   = 500,         # CG iteration cap for that solve
    h_bathy                  = nothing,     # x → d(x): variable bathymetry (overrides h_val)
    eta0_func               = nothing,     # x → η₀(x): initial release (REQUIRES x_wall_bc=true)
    # ---- Dirichlet boundary wave generation (deterministic per rank) ---------
    wave_bc                 = nothing,     # nothing | :regular | WaveInput | WaveSpec AiryState
    bc_side      :: Symbol  = :left,       # generation boundary (:left/:right)
    bc_profile   :: Symbol  = :model,      # vertical polarization (:model/:airy)
    T_ramp                  = nothing,     # Hann ramp-up time [s]; nothing → 2 peak periods
    ic_from_bc   :: Bool    = false,       # hot-start from the incident wave (needs T_ramp=0)
    relax_bc     :: Bool    = false,       # relaxation zone at the inflow
    relax_width  :: Float64 = 0.0,         # relaxation-zone width [m]; 0 → one peak wavelength
    # ---- Reconstructed field output ------------------------------------------
    write_w        :: Bool    = false,     # also write vertical-velocity fields w_s<σ>
    write_pressure :: Bool    = false,     # also write total-pressure fields p_s<σ>
    rho            :: Float64 = rho,    # water density [kg/m³] (pressure output)
    # ---- Solver tolerances / diagnostics -------------------------------------
    nl_iter      :: Int     = 50,          # max Newton iterations per stage
    nl_tol       :: Float64 = 1e-6,        # Newton residual tolerance (‖r‖₂) — production default
    ls_rtol      :: Float64 = 1e-9,        # GMRES relative tolerance (kept tight for good Newton steps)
    ls_maxiter   :: Int     = 2000,        # GMRES iteration cap per Newton step
    print_every  :: Int     = 1,           # print a step report every N steps (print_dt overrides)
    check_every  :: Int     = 50,          # re-verify the governing equations every N steps (0 = off)
    check_tol    :: Float64 = 1e-8,        # tolerance for that verification (‖R‖∞)
)
    n_procs = prod(cpu_grid)                 # total MPI ranks implied by the process grid

    # Everything runs inside with_mpi: it initialises MPI and hands back a
    # `distribute` that turns a global index set into this rank's local share.
    result = with_mpi() do distribute
        ranks = distribute(LinearIndices((n_procs,)))   # this rank's handle in the grid

        # All console output is guarded by i_am_main so only rank 0 prints.
        if i_am_main(ranks)
            println("=== 2D LFE-M ALGEBRAIC Distributed Solver (stacked [η,𝖴x,𝖴y]) ===")
            println("  cpu_grid: $cpu_grid  ($(n_procs) ranks total)")
            flush(stdout)
        end

        # ----- Stage 1: Vertical pre-computation (identical work on every rank) -
        c_bdy_used = isnothing(c_bdy) ?
            get(DEFAULT_CBDY, M, collect(LinRange(0.0, 1.0, M+1))) : c_bdy
        vert = assemble_vertical_tensors(M, p_vertical, c_bdy_used)
        if i_am_main(ranks)
            @printf("  Nσ=%d   ΣΦ=%.6f\n", vert.N_dof, sum(vert.Phi))
        end

        # ----- Stage 2: distributed horizontal mesh + stacked spaces -----------
        # Unpack the domain corners (nested or flat tuple) and cell counts.
        if domain isa Tuple{Tuple,Tuple}
            (x0,x1), (y0,y1) = domain
        else
            x0,x1,y0,y1 = domain
        end
        dom_flat = (Float64(x0), Float64(x1), Float64(y0), Float64(y1))
        nx, ny   = partition

        # ----- Dirichlet boundary wave generation -----------------------------
        # Built identically on every rank: the WaveInput snapshots the seeded
        # phases/amplitudes into plain arrays, so no communication is needed and
        # every rank prescribes the same boundary data.
        wi = nothing
        if wave_bc !== nothing
            bc_side in (:left, :right) ||
                error("setup_and_run_distributed: bc_side must be :left or :right")
            Tr = T_ramp === nothing ? nothing : Float64(T_ramp)
            wi = wave_bc isa WaveInput ? wave_bc :
                 wave_bc === :regular ?
                     WaveInput(vert; A=A_wave, T=T_wave, d=h_val, g=g,
                               T_ramp=(Tr === nothing ? 2.0*T_wave : Tr),
                               profile=bc_profile) :
                     WaveInput(vert, wave_bc; d=h_val, g=g, T_ramp=Tr,
                               profile=bc_profile)
            i_am_main(ranks) && waveinput_summary(wi)
            wi.directional && y_wall_bc == :wall &&
                error("setup_and_run_distributed: directional sea requires " *
                      "y_wall_bc=:open (lateral sponges) or :periodic")
            ic_from_bc && wi.T_ramp > 0.0 &&
                error("setup_and_run_distributed: ic_from_bc=true requires T_ramp=0.0")
        end
        # Time-varying Dirichlet data for the inflow (η, 𝖴x, and 𝖴y if directional).
        inflow = wi === nothing ? nothing :
                 (side=bc_side, eta=eta_bc(wi), ux=ux_bc(wi),
                  uy=wi.directional ? uy_bc(wi) : nothing)

        # Lateral (y) boundary condition: :wall / :open / :periodic.
        y_wall_bc in (:wall, :open, :periodic) ||
            error("setup_and_run_distributed: y_wall_bc must be :wall, :open or :periodic (got :$y_wall_bc)")
        y_periodic = y_wall_bc == :periodic
        if y_periodic && i_am_main(ranks)
            (sponge_wB > 0 || sponge_wT > 0) &&
                @warn "y_wall_bc=:periodic — lateral sponges (sponge_wB/wT) are redundant; set them to 0"
            !isnothing(y_wm) &&
                @warn "y_wall_bc=:periodic with a point source is not y-periodic (image array); use a line source"
        end

        # Partition the Cartesian mesh over the process grid (periodic in y when
        # requested), build the quadrature measure, and create the stacked FE spaces.
        model, trian = build_horizontal_model_distributed(ranks, cpu_grid,
                                                              dom_flat, (nx, ny);
                                                              y_periodic=y_periodic)
        dΩh   = Measure(trian, 2*p_horizontal + 2)
        U, V = build_fe_spaces(model, p_horizontal, vert.N_dof;
                                   y_wall_bc=y_wall_bc, x_wall_bc=x_wall_bc,
                                   inflow=inflow)
        if i_am_main(ranks)
            @printf("  domain [%.1f,%.1f]×[%.1f,%.1f]  partition %d×%d\n", x0,x1,y0,y1,nx,ny)
            @printf("  Fields: 3 (η + 2 stacked VectorValue{%d}) | lin=%s adv=%s linP=%s Pfull=%s nlP68=%s nlPfull=%s\n",
                    vert.N_dof, string(linearised), string(advection),
                    string(lin_pressure), string(P_full), string(nl_pressure68),
                    string(nl_pressure_full))
        end

        # ----- Stage 3: forcing setup (identical on every rank) ----------------
        # Forcing frequency + wavenumber (also used for the reported CFL number).
        omega  = 2.0 * pi / T_wave
        k_wave = find_wavenumber(omega, h_val, g)
        if i_am_main(ranks)
            @printf("  Wave: λ=%.2f m, kd=%.2f   CFL_x ~ %.3f\n",
                    2pi/k_wave, k_wave*h_val, sqrt(g*h_val)*dt/((x1-x0)/nx))
        end
        # Sponge profile; internal source (none/line/point) as in the sequential driver.
        sponge = make_sponge(dom_flat, sponge_wL, sponge_wR, sponge_wB, sponge_wT, mu_max)
        wm = wi !== nothing ? ((x, t) -> 0.0) :
             isnothing(y_wm) ? make_wavemaker_line(x_wm, A_wave, T_wave, k_wave) :
                               make_wavemaker_point(x_wm, Float64(y_wm), A_wave, T_wave)
        dfn = isnothing(h_bathy) ? (x -> h_val) : h_bathy   # bathymetry

        # Optional relaxation zone next to the inflow (generation + absorption).
        relax_mu_fn = x -> 0.0
        relax_tg    = nothing
        use_relax   = wi !== nothing && relax_bc
        if use_relax
            wrx = relax_width > 0.0 ? relax_width : 2.0*pi/wi.ks[argmax(wi.amps)]
            relax_mu_fn = bc_side == :left ?
                (x -> x[1] < x0 + wrx ? mu_max*((x0 + wrx - x[1])/wrx)^2 : 0.0) :
                (x -> x[1] > x1 - wrx ? mu_max*((x[1] - (x1 - wrx))/wrx)^2 : 0.0)
            relax_tg = incident_fields(wi)
        end

        # Same problem bundle as the sequential driver — identical residual.
        prob = build_problem(vert; g=g, h_bathy=dfn,
            linearised=linearised, advection=advection, lin_pressure=lin_pressure,
            P_full=P_full, nl_pressure68=nl_pressure68, nl_pressure_full=nl_pressure_full,
            mu_sponge=sponge, wm_src=wm,
            relax_bc=use_relax, relax_mu=relax_mu_fn, relax_tg=relax_tg)

        # ----- Stage 4: distributed operator + solver stack + initial state ----
        op      = build_ode_operator(prob, U, V, trian, dΩh)   # same hand Jacobians
        monitor = SolverMonitor()
        # Distributed integrator: Newton + GMRES/Jacobi (the scalable linear solve).
        solver  = build_ode_solver_distributed(dt; solver_type=solver_type,
                      theta=theta, tableau=tableau,
                      nl_iter=nl_iter, nl_tol=nl_tol,
                      ls_rtol=ls_rtol, ls_maxiter=ls_maxiter, monitor=monitor)
        checker = check_every > 0 ?
                  ResidualChecker(prob, U, V, trian, dΩh, dt, theta,
                                     solver_type == :theta) : nothing
        # Initial state via interpolate_everywhere (required for distributed spaces);
        # space_at evaluates a transient trial at t=0 (a no-op for static spaces).
        u0 = if wi !== nothing && ic_from_bc
            inc = incident_fields(wi)
            make_initial_conditions(space_at(U, 0.0), vert.N_dof;
                eta0_func = x -> inc.eta(x, 0.0),
                ux0_func  = x -> inc.ux(x, 0.0),
                uy0_func  = x -> inc.uy(x, 0.0))
        else
            make_initial_conditions(space_at(U, 0.0), vert.N_dof; eta0_func=eta0_func)
        end

        # For nl_pressure_full, the frozen-projection mass solve uses CG + Jacobi
        # here (a partitioned matrix has no direct-factorisation method).
        nlp = nl_pressure_full ?
              (prob, build_nlp_ctx(model, p_horizontal, vert.N_dof, trian, dΩh;
                                   distributed=true, cg_rtol=nlp_cg_rtol,
                                   cg_maxiter=nlp_cg_maxiter)) : nothing

        # ----- Stage 5: time loop ---------------------------------------------
        # Reconstruction context for optional w/p VTK output.
        recon = build_field_recon(vert, dfn, g; rho=rho,
                                      write_w=write_w, write_pressure=write_pressure)
        if recon !== nothing && i_am_main(ranks)
            @printf("  Field output: write_w=%s write_pressure=%s at σ-levels %s\n",
                    string(write_w), string(write_pressure),
                    string(round.(recon.levels; digits=3)))
        end
        if i_am_main(ranks)
            println()
            print_solver_banner(
                @sprintf("NewtonSolver (GridapSolvers, exact hand Jacobians) | max iters = %d | atol (‖r‖₂) = %.1e, rtol = 1.0e-10",
                         nl_iter, nl_tol),
                @sprintf("GMRES + Jacobi preconditioner | max iters = %d | rtol = %.1e, atol = 1.0e-14",
                         ls_maxiter, ls_rtol);
                solver_type=solver_type, theta=theta, dt=dt, t0=0.0, T_final=T_final,
                print_every=print_every, print_dt=print_dt,
                check_every=check_every, check_tol=check_tol)
            println("\n=== Time loop (algebraic, distributed) ===")
            flush(stdout)
        end
        # March the transient problem; run_time_loop_dist reduces eta_max across
        # ranks and writes distributed VTK. Returns the per-step diagnostics.
        diags = run_time_loop_dist(ranks, op, solver, u0, 0.0, T_final;
                    output_dir  = output_dir,
                    save_every  = save_every,
                    trian       = (save_every > 0 ? trian : nothing),
                    Nσ          = vert.N_dof,
                    print_every = print_every,
                    print_dt    = print_dt,
                    recon       = recon, trial_space = U, dt = dt, nlp = nlp,
                    monitor     = monitor, checker = checker,
                    check_every = check_every, check_tol = check_tol)

        return (diags, vert, prob)
    end
    return result
end
