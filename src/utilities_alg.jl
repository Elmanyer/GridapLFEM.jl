# ==============================================================
#  utilities_alg.jl — dispersion analysis, sponge, wavemakers, driver
#
#  Port of ../../LFE-M_2D_solver/src/utilities2D.jl + utilities_lfem2D.jl
#  adapted to the algebraic package (v2 vertical tensor names Mmat/B/Phi;
#  stacked FE spaces; residual/Jacobian from problem_alg.jl).
# ==============================================================

"""
    find_wavenumber_alg(omega, d, g)

Newton iteration for the linear dispersion relation ω² = g k tanh(kd).
"""
function find_wavenumber_alg(omega::Float64, d::Float64, g::Float64)
    k = omega^2 / g
    for _ in 1:100
        f  = g * k * tanh(k*d) - omega^2
        df = g * (tanh(k*d) + k*d*(1.0 - tanh(k*d)^2))
        dk = f / df
        k -= dk
        abs(dk) < 1e-13 * abs(k) && break
    end
    return k
end

"""
    dispersion_ratio_alg(vert, g, d, kd_vals)

Model-to-exact phase speed ratio Cm/Ce per kd:
  Ce = √(g tanh(kd)/k),   Cm² = g·d · Φᵀ (Mmat − B·kd²)⁻¹ Φ
(`B ≤ 0`, so Mmat − B·kd² = Mmat + |B|·kd² > 0.)
"""
function dispersion_ratio_alg(vert, g::Float64, d::Float64,
                              kd_vals::AbstractVector{Float64})
    A, B, Phi = vert.Mmat, vert.B, vert.Phi
    ratios = zeros(length(kd_vals))
    for (idx, kd) in enumerate(kd_vals)
        k     = kd / d
        Ce    = sqrt(g * tanh(kd) / k)
        M_eff = A .- B .* kd^2
        try
            Cm_sq = g * d * dot(Phi, M_eff \ Phi)
            ratios[idx] = Cm_sq > 0 ? sqrt(Cm_sq) / Ce : NaN
        catch
            ratios[idx] = NaN
        end
    end
    return ratios
end

"""
    applicable_kd_alg(vert, g, d; err=0.02, kd_max=200.0, n=2000)

Maximum kd with |Cm/Ce − 1| ≤ err.
"""
function applicable_kd_alg(vert, g::Float64, d::Float64;
                           err::Float64=0.02, kd_max::Float64=200.0, n::Int=2000)
    kd_vals = LinRange(0.01, kd_max, n)
    ratios  = dispersion_ratio_alg(vert, g, d, kd_vals)
    idx     = findlast(abs.(ratios .- 1.0) .<= err)
    return isnothing(idx) ? NaN : Float64(kd_vals[idx])
end

"""
    make_sponge_alg(domain, wL, wR, wB, wT, mu_max)

Quadratic sponge μ(x) on up to four boundaries; corner regions clamp at
mu_max (max of the contributions, not the sum).
"""
function make_sponge_alg(domain::Tuple, wL::Float64, wR::Float64,
                         wB::Float64, wT::Float64, mu_max::Float64)
    if domain isa Tuple{Tuple,Tuple}
        (x0,x1), (y0,y1) = domain
    else
        x0,x1,y0,y1 = domain
    end
    function mu_fn(x)
        xv = Float64(x[1]); yv = Float64(x[2])
        mu = 0.0
        wL > 0 && xv < x0 + wL && (mu = max(mu, mu_max * ((x0+wL-xv)/wL)^2))
        wR > 0 && xv > x1 - wR && (mu = max(mu, mu_max * ((xv-(x1-wR))/wR)^2))
        wB > 0 && yv < y0 + wB && (mu = max(mu, mu_max * ((y0+wB-yv)/wB)^2))
        wT > 0 && yv > y1 - wT && (mu = max(mu, mu_max * ((yv-(y1-wT))/wT)^2))
        return mu
    end
    return mu_fn
end

"""
    make_wavemaker_line_alg(x_wm, A, T, k_wave; sigma_wm=1.5)

Gaussian line mass source at x = x_wm (long-crested plane waves):
  S(x,t) = 2Aω exp(−((x−x_wm)/σ)²) cos(ωt)
(factor 2: waves radiate in ±x). Enters continuity as −∫ q·S.
"""
function make_wavemaker_line_alg(x_wm::Float64, A::Float64, T::Float64,
                                 k_wave::Float64; sigma_wm::Float64=1.5)
    omega = 2.0 * pi / T
    function wm_fn(x, t)
        xv = Float64(x[1])
        return 2.0 * A * omega * exp(-((xv - x_wm)/sigma_wm)^2) * cos(omega * t)
    end
    return wm_fn
end

"""
    make_wavemaker_point_alg(x_wm, y_wm, A, T; sigma_wm=1.5)

Gaussian point mass source at (x_wm, y_wm) (ring waves).
"""
function make_wavemaker_point_alg(x_wm::Float64, y_wm::Float64,
                                  A::Float64, T::Float64; sigma_wm::Float64=1.5)
    omega = 2.0 * pi / T
    function wm_fn(x, t)
        xv = Float64(x[1]); yv = Float64(x[2])
        r2 = ((xv - x_wm)^2 + (yv - y_wm)^2) / sigma_wm^2
        return 2.0 * A * omega * exp(-r2) * cos(omega * t)
    end
    return wm_fn
end

# Optimised vertical node positions (Yang & Liu 2024, Table 1)
const ALG_DEFAULT_CBDY = Dict(
    1 => [0.0, 1.0],
    2 => [0.0, 0.728, 1.0],
    3 => [0.0, 0.726, 0.925, 1.0],
    4 => [0.0, 0.745, 0.923, 0.977, 1.0],
)

"""
    setup_and_run_alg(; kwargs...) → (diags, vert, prob)

Full driver for the algebraic solver (mirror of the old `setup_and_run_lfem`).
`y_wm = nothing` → line source (plane wave); a number → point source (ring).
Flags: `linearised, advection, lin_pressure, P_full, nl_pressure68, use_ad`.
"""
function setup_and_run_alg(;
    M            :: Int     = 2,
    p_vert       :: Int     = 1,
    c_bdy                   = nothing,
    domain                  = ((0.0, 60.0), (0.0, 10.0)),
    partition    :: Tuple   = (120, 20),
    fe_order     :: Int     = 2,
    d_val        :: Float64 = 3.5,
    g            :: Float64 = 9.81,
    T_wave       :: Float64 = 1.6,
    A_wave       :: Float64 = 0.001,
    x_wm         :: Float64 = 12.0,
    y_wm                    = nothing,
    sponge_wL    :: Float64 = 12.0,
    sponge_wR    :: Float64 = 12.0,
    sponge_wB    :: Float64 = 0.0,
    sponge_wT    :: Float64 = 0.0,
    mu_max       :: Float64 = 5.0,
    T_final      :: Float64 = 12.8,
    dt           :: Float64 = 0.02,
    solver_type  :: Symbol  = :theta,
    theta        :: Float64 = 0.5,
    rho_inf      :: Float64 = 0.5,
    output_dir   :: String  = joinpath(@__DIR__, "..", "output", "alg_out"),
    save_every   :: Int     = 0,
    gauges                  = [],
    y_wall_bc    :: Bool    = true,
    x_wall_bc    :: Bool    = false,
    linearised   :: Bool    = false,
    advection    :: Bool    = true,
    lin_pressure :: Bool    = false,
    P_full       :: Bool    = false,
    nl_pressure68:: Bool    = false,      # native NL-pressure set c∈{3,6,7,8} (all blocks)
    nl_pressure_full :: Bool = false,     # + c∈{1,2,4,5} (∇h exact-IBP; ∇H/𝓟 frozen projections)
    d_func                  = nothing,
    eta0_func               = nothing,    # initial free surface η₀(x) (IC problems: set x_wall_bc=true!)
    use_ad       :: Bool    = false,
    show_trace   :: Bool    = false,
    # Nonlinear solver controls
    nl_iter      :: Int     = 20,         # max Newton iterations per step
    nl_tol       :: Float64 = 1e-10,      # Newton ftol (‖r‖∞, NLsolve)
    # Runtime diagnostics
    print_every  :: Int     = 1,          # step report every N steps (1 = every step)
    check_every  :: Int     = 50,         # governing-eq residual check every N steps (0 = off)
    check_tol    :: Float64 = 1e-8,       # ‖R_θ‖∞ verification threshold
    # Field output (reconstructed at the Nσ vertical σ-nodes)
    write_w        :: Bool    = false,    # vertical velocity fields w_s<σ>
    write_pressure :: Bool    = false,    # total pressure fields p_s<σ>
    rho            :: Float64 = 1025.0,   # water density [kg/m³] (pressure output)
)
    if isnothing(c_bdy)
        c_bdy = get(ALG_DEFAULT_CBDY, M, collect(LinRange(0.0, 1.0, M + 1)))
    end

    println("=== Vertical FE problem (algebraic LFE-M) ===")
    vert = assemble_vertical_tensors_alg(M, p_vert, c_bdy)
    @printf("  Nσ=%d   ΣΦ=%.6f\n", vert.N_dof, sum(vert.Phi))

    println("\n=== 2D Horizontal FE problem (stacked [η,𝖴x,𝖴y]) ===")
    model, trian = build_horizontal_model_alg(domain, partition)
    dO = Measure(trian, 2*fe_order + 2)
    U, V = build_fe_spaces_alg(model, fe_order, vert.N_dof;
                               y_wall_bc=y_wall_bc, x_wall_bc=x_wall_bc)
    @printf("  Fields: 3 (η + 2 stacked VectorValue{%d})   free DOFs: %d\n",
            vert.N_dof, num_free_dofs(U))

    omega  = 2.0*pi/T_wave
    k_wave = find_wavenumber_alg(omega, d_val, g)
    @printf("  Wave: λ=%.2f m, kd=%.2f\n", 2pi/k_wave, k_wave*d_val)

    sponge = make_sponge_alg(domain, sponge_wL, sponge_wR, sponge_wB, sponge_wT, mu_max)
    wm = isnothing(y_wm) ? make_wavemaker_line_alg(x_wm, A_wave, T_wave, k_wave) :
                           make_wavemaker_point_alg(x_wm, Float64(y_wm), A_wave, T_wave)
    dfn = isnothing(d_func) ? (x -> d_val) : d_func

    prob = build_problem_alg(vert; g=g, d_func=dfn,
        linearised=linearised, advection=advection, lin_pressure=lin_pressure,
        P_full=P_full, nl_pressure68=nl_pressure68, nl_pressure_full=nl_pressure_full,
        mu_sponge=sponge, wm_src=wm)

    op = use_ad ? build_ode_operator_alg_ad(prob, U, V, trian, dO) :
                  build_ode_operator_alg(prob, U, V, trian, dO)
    monitor = SolverMonitorAlg()
    solver  = build_ode_solver_alg(dt; solver_type=solver_type, theta=theta,
                                   rho_inf=rho_inf, nl_iter=nl_iter, nl_tol=nl_tol,
                                   show_trace=show_trace, monitor=monitor)
    checker = check_every > 0 ?
              ResidualCheckerAlg(prob, U, V, trian, dO, dt, theta,
                                 solver_type == :theta) : nothing
    u0 = isnothing(eta0_func) ? make_initial_conditions_alg(U) :
         make_initial_conditions_alg(U, vert.N_dof; eta0_func=eta0_func)

    # nl_pressure_full: frozen-projection context (mass matrix factorised once)
    nlp = nl_pressure_full ?
          (prob, build_nlp_ctx(model, fe_order, vert.N_dof, trian, dO)) : nothing

    recon = build_field_recon_alg(vert, dfn, g; rho=rho,
                                  write_w=write_w, write_pressure=write_pressure)
    if recon !== nothing
        @printf("  Field output: write_w=%s write_pressure=%s at σ-levels %s\n",
                string(write_w), string(write_pressure),
                string(round.(recon.levels; digits=3)))
    end

    println()
    print_solver_banner_alg(
        @sprintf("Newton (NLsolve, exact hand Jacobians) | max iters = %d | ftol (‖r‖∞) = %.1e",
                 nl_iter, nl_tol),
        "LU direct factorisation (sequential)";
        solver_type=solver_type, theta=theta, dt=dt, t0=0.0, T_final=T_final,
        print_every=print_every, check_every=check_every, check_tol=check_tol)

    println("\n=== Time loop (algebraic) ===")
    diags = run_time_loop_alg(op, solver, u0, 0.0, T_final;
                              output_dir=output_dir, save_every=save_every,
                              trian=trian, Nσ=vert.N_dof,
                              print_every=print_every, gauges=gauges,
                              recon=recon, trial_space=U, dt=dt, nlp=nlp,
                              monitor=monitor, checker=checker,
                              check_every=check_every, check_tol=check_tol)
    return diags, vert, prob
end
