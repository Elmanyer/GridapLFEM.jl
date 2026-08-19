# ==============================================================
#  utilities.jl — physical setup helpers + the sequential driver
#
#  Everything needed to turn physical inputs into a running simulation on one
#  process: the dispersion-relation solver (`find_wavenumber`), the quadratic
#  sponge-layer profile (`make_sponge`), the internal Gaussian wavemakers
#  (`make_wavemaker_line/point`), and the top-level driver `setup_and_run`,
#  which wires the two stages together and marches the time loop.
# ==============================================================

"""
    find_wavenumber(omega, d, g)

Newton iteration for the linear (Airy) dispersion relation ω² = g k tanh(kd).
"""
function find_wavenumber(omega::Float64, d::Float64, g::Float64)
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
    dispersion_ratio(vert, g, d, kd_vals)

- vert:    the vertical tensor bundle from `assemble_vertical_tensors`.
- g:       gravitational acceleration [m/s²].
- d:       still-water depth [m].
- kd_vals: vector of kd values to evaluate.

Model-to-exact phase speed ratio Cm/Ce per kd:
  Ce = √(g tanh(kd)/k),   Cm² = g·d · Φᵀ (Mmat − B·kd²)⁻¹ Φ
(`B ≤ 0`, so Mmat − B·kd² = Mmat + |B|·kd² > 0.)
"""
function dispersion_ratio(vert, g::Float64, d::Float64,
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
    applicable_kd(vert, g, d; err=0.02, kd_max=200.0, n=2000)

Maximum kd with |Cm/Ce − 1| ≤ err.
"""
function applicable_kd(vert, g::Float64, d::Float64;
                           err::Float64=0.02, kd_max::Float64=200.0, n::Int=2000)
    kd_vals = LinRange(0.01, kd_max, n)
    ratios  = dispersion_ratio(vert, g, d, kd_vals)
    idx     = findlast(abs.(ratios .- 1.0) .<= err)
    return isnothing(idx) ? NaN : Float64(kd_vals[idx])
end

"""
    make_sponge(domain, wL, wR, wB, wT, mu_max)

- domain: ((x0,x1),(y0,y1)) or (x0,x1,y0,y1) rectangle.
- wL:     width of the LEFT sponge layer.
- wR:     width of the RIGHT sponge layer.
- wB:     width of the BOTTOM sponge layer.
- wT:     width of the TOP sponge layer.
- mu_max: maximum sponge strength.

Quadratic sponge μ(x) on up to four boundaries; corner regions clamp at
mu_max (max of the contributions, not the sum).
"""
function make_sponge(domain::Tuple, wL::Float64, wR::Float64,
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
    make_wavemaker_line(x_wm, A, T, k_wave; sigma_wm=1.5)

Gaussian line mass source at x = x_wm (long-crested plane waves):
  S(x,t) = 2Aω exp(−((x−x_wm)/σ)²) cos(ωt)
(factor 2: waves radiate in ±x). Enters continuity as −∫ q·S.
"""
function make_wavemaker_line(x_wm::Float64, A::Float64, T::Float64,
                                 k_wave::Float64; sigma_wm::Float64=1.5)
    omega = 2.0 * pi / T
    function wm_fn(x, t)
        xv = Float64(x[1])
        return 2.0 * A * omega * exp(-((xv - x_wm)/sigma_wm)^2) * cos(omega * t)
    end
    return wm_fn
end

"""
    make_wavemaker_point(x_wm, y_wm, A, T; sigma_wm=1.5)

Gaussian point mass source at (x_wm, y_wm) (ring waves).
"""
function make_wavemaker_point(x_wm::Float64, y_wm::Float64,
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
const DEFAULT_CBDY = Dict(
    1 => [0.0, 1.0],
    2 => [0.0, 0.728, 1.0],
    3 => [0.0, 0.726, 0.925, 1.0],
    4 => [0.0, 0.745, 0.923, 0.977, 1.0],
)

"""
    setup_and_run(; kwargs...) → (diags, vert, prob)

Top-level sequential driver. It executes the whole workflow in order —
Stage 1 (vertical tensors), the horizontal mesh and FE spaces, the forcing
(wavemaker/sponge or Dirichlet wave generation), the residual/operator, the
time integrator, the initial state, and finally the time loop — and returns the
per-step diagnostics `diags`, the vertical tensor bundle `vert`, and the problem
bundle `prob`. Every physical/numerical choice is a keyword argument (documented
inline on the signature below); the defaults describe a small plane-wave case.

Wave forcing comes from EITHER an internal wavemaker OR Dirichlet boundary
generation, chosen by the arguments:
  * internal wavemaker: `y_wm = nothing` → line source (long-crested plane
    wave); a number → point source at (x_wm, y_wm) (radial/ring waves);
  * Dirichlet boundary generation (waveinput.jl): pass `wave_bc` =
  * `:regular`             — monochromatic wave built from `A_wave`/`T_wave`;
  * a `WaveInput`          — any prebuilt component table;
  * a `WaveSpec.AiryWaves.AiryState` — stochastic sea state (auto-converted).
The interior wavemaker is then disabled and η/𝖴x (and 𝖴y for directional
seas) are prescribed on `bc_side` (`:left`/`:right`). Related kwargs:
`bc_profile` (`:model`/`:airy` vertical polarization), `T_ramp` (Hann ramp,
`nothing` → 2 peak periods), `ic_from_bc` (hot start from the incident field;
requires `T_ramp=0.0`), `relax_bc`+`relax_width` (generation/absorption
relaxation zone adjacent to the inflow, strength `mu_max`).
"""

"""
    check_flat_bed_consistency(h_bathy, domain, flat_bed; ns=7, rtol=1e-8) → Bool

Warn if the `flat_bed` switch disagrees with the prescribed bathymetry `h_bathy`
(`x → d(x)`) sampled on a small grid over `domain`: `flat_bed=true` over a varying
bed silently drops the ∇h (sloping-bed) physics, while `flat_bed=false` over a
constant bed assembles ∇h-terms that vanish anyway. Returns whether the bed is
constant. Sampling a pure function is cheap and rank-independent.
"""
function check_flat_bed_consistency(h_bathy, domain, flat_bed::Bool;
                                    ns::Int = 7, rtol::Float64 = 1e-8)
    (x0, x1), (y0, y1) = domain isa Tuple{Tuple,Tuple} ? domain :
                         ((domain[1], domain[2]), (domain[3], domain[4]))
    hs = [h_bathy(VectorValue(x, y)) for x in range(x0, x1; length=ns),
                                         y in range(y0, y1; length=ns)]
    hmin, hmax = extrema(hs); hmean = sum(hs) / length(hs)
    is_const = (hmax - hmin) <= rtol * max(abs(hmean), 1.0)
    if flat_bed && !is_const
        @warn "flat_bed=true but the bathymetry varies over the domain " *
              "(Δh=$(round(hmax - hmin; sigdigits=3)) m): the ∇h (sloping-bed) terms are being " *
              "dropped and the bed-slope physics is NOT represented — use flat_bed=false for variable bathymetry."
    elseif !flat_bed && is_const
        @info "flat_bed=false but h_bathy is constant (d=$(round(hmean; sigdigits=4)) m): " *
              "the ∇h terms vanish anyway; set flat_bed=true to skip assembling them."
    end
    return is_const
end

"""
    resolve_wave_gen(wave_gen, wave_bc) → Symbol

Map the user's `wave_gen` selector to one of the two mechanisms `:inner_res |
:bc_gen`, validating it against `wave_bc`:

  - `:inner_res` — interior Gaussian source (line ⇒ plane wave, point ⇒ ring wave);
                   requires `wave_bc === nothing`.
  - `:bc_gen`    — boundary Dirichlet generation. The boundary source — a
                   parametrised regular wave (from `A_wave`/`T_wave`/`wave_dir`),
                   a prebuilt `WaveInput`, or a WaveSpec `AiryState` — is dispatched
                   on the TYPE of `wave_bc` inside the driver; all feed the SAME
                   Dirichlet machinery (they differ only in how the `WaveInput`
                   component table is populated, not in the boundary mechanism).

`:auto` (the default) infers it: `wave_bc === nothing` → `:inner_res`, else `:bc_gen`.
"""
function resolve_wave_gen(wave_gen::Symbol, wave_bc)
    if wave_gen === :auto
        return wave_bc === nothing ? :inner_res : :bc_gen
    elseif wave_gen === :inner_res
        wave_bc === nothing ||
            error("wave_gen=:inner_res uses the interior source; leave wave_bc=nothing")
        return :inner_res
    elseif wave_gen === :bc_gen
        return :bc_gen
    else
        error("wave_gen must be :auto, :inner_res or :bc_gen (got :$wave_gen)")
    end
end

function setup_and_run(;
    # ---- Vertical discretisation ---------------------------------------------
    M            :: Int     = 2,          # number of vertical σ-elements (BALFE-M order: 2/3/4)
    p_vertical   :: Int     = 1,          # polynomial order of each σ-element (Nσ = M·p_vertical+1)
    c_bdy                   = nothing,    # σ-node boundary positions in [0,1]; nothing → optimised set
    # ---- Horizontal discretisation -------------------------------------------
    domain                  = ((0.0, 60.0), (0.0, 10.0)),  # ((x0,x1),(y0,y1)) extent [m]
    partition    :: Tuple   = (120, 20),  # (nx,ny) number of horizontal cells
    p_horizontal :: Int     = 2,          # horizontal FE order (must be ≥2: Q1 zeroes the dispersion)
    p_eta        :: Int     = 0,          # surface FE order. 0 ⇒ EQUAL ORDER (= p_horizontal),
                                          #   which is the historical default and is UNCHANGED.
                                          #   Set p_eta = p_horizontal−1 for the Taylor-Hood-like
                                          #   pairing: η enters momentum undifferentiated (via ∇·v
                                          #   after IBP), so it plays the pressure role of a Stokes
                                          #   system and equal-order continuous spaces are inf-sup
                                          #   deficient — the analytic MMS measures order p there
                                          #   rather than p+1, in BOTH fields.
                                          #   ⚠ A BETTER RATE IS NOT A BETTER ANSWER AT A GIVEN MESH:
                                          #   at nx=24 the equal-order Q3/Q3 was 40x MORE ACCURATE
                                          #   than Q3/Q2, because η sits in a richer space. Compare
                                          #   error-vs-DOF at your production resolution before
                                          #   switching. See MMS_CONVERGENCE_CAMPAIGN.md.
    # ---- Physical parameters -------------------------------------------------
    h_val        :: Float64 = 3.5,        # still-water depth [m] (flat bed unless h_bathy given)
    g            :: Float64 = g,          # gravitational acceleration [m/s²]
    T_wave       :: Float64 = 1.6,        # forcing wave period [s]
    A_wave       :: Float64 = 0.001,      # forcing wave amplitude [m] (keep small for stability)
    # ---- Internal wavemaker (used only when wave_bc is nothing) ---------------
    x_wm         :: Float64 = 12.0,       # wavemaker x-position [m]
    y_wm                    = nothing,    # nothing → line source (plane wave); number → point source
    # ---- Sponge layers (quadratic damping toward each boundary) ---------------
    sponge_wL    :: Float64 = 12.0,       # left-edge sponge width [m] (0 = none)
    sponge_wR    :: Float64 = 12.0,       # right-edge sponge width [m]
    sponge_wB    :: Float64 = 0.0,        # bottom-edge (y0) sponge width [m]
    sponge_wT    :: Float64 = 0.0,        # top-edge (y1) sponge width [m]
    mu_max       :: Float64 = 5.0,        # peak sponge strength (also the relaxation-zone strength)
    # ---- Time integration ----------------------------------------------------
    T_final      :: Float64 = 12.8,       # final simulated time [s]
    dt           :: Float64 = 0.02,       # time step [s]
    solver_type  :: Symbol  = :sdirk,     # integrator: :sdirk (default) | :theta | :gen_alpha | :rk3
    tableau      :: Symbol  = :SDIRK_2_2, # Runge–Kutta tableau when solver_type == :sdirk
    theta        :: Float64 = 0.5,        # θ for :theta (0.5 = Crank–Nicolson)
    rho_inf      :: Float64 = 0.5,        # high-frequency damping for :gen_alpha (ρ∞∈[0,1])
    # ---- Output --------------------------------------------------------------
    output_dir   :: String  = joinpath(@__DIR__, "..", "output", "seq_out"),  # VTK/pvd destination
    save_every   :: Int     = 0,          # write a VTK snapshot every N steps (0 = no VTK)
    gauges                  = [],         # list of (x,y) probe points; η is sampled there each step
    # ---- Boundary conditions -------------------------------------------------
    y_wall_bc    :: Symbol  = :wall,      # y-edge BC: :wall (𝖴y=0) | :open (natural) | :periodic (y-periodic)
    x_wall_bc    :: Bool    = false,      # solid walls on the x-edges (𝖴x=0); true for closed-basin IC
    # ---- Physics flags (switch individual residual terms on/off) --------------
    regime       :: Symbol  = :nonlinear, # :linear (linearised, no advection) | :nonlinear
    nl_pressure  :: Symbol  = :none,      # nonlinear pressure: :none | :native {3,6,7,8} | :full {+1,2,4,5}
    flat_bed     :: Bool    = false,      # sea-bed geometry: false = variable bathymetry (∇h≠0),
                                          #   true = flat bed (∇h≡0; every ∇h-term dropped, ∇η-terms kept)
    h_bathy                 = nothing,    # x → d(x): variable bathymetry (overrides h_val)
    eta0_func               = nothing,    # x → η₀(x): initial free surface (IC release: set x_wall_bc=true)
    # ---- Dirichlet boundary wave generation (waveinput.jl) --------------------
    wave_gen     :: Symbol  = :auto,      # :inner_res | :bc_gen  (:auto infers from wave_bc)
    wave_bc                 = nothing,    # nothing | :regular | WaveInput | WaveSpec AiryState
    wave_dir     :: Float64 = 0.0,        # propagation angle vs +x for a :bc_gen boundary wave [rad]
    bc_side      :: Symbol  = :left,      # generation boundary (:left/:right)
    bc_profile   :: Symbol  = :model,     # vertical polarization of the inflow (:model/:airy)
    T_ramp                  = nothing,    # Hann ramp-up time [s]; nothing → 2 peak periods
    ic_from_bc   :: Bool    = false,      # hot-start the field from the incident wave (needs T_ramp=0)
    relax_bc     :: Bool    = false,      # relaxation (generation/absorption) zone at the inflow
    relax_width  :: Float64 = 0.0,        # relaxation-zone width [m]; 0 → one peak wavelength
    # ---- Solver / diagnostics ------------------------------------------------
    use_ad       :: Bool    = false,      # build Jacobians by AD instead of the hand Jacobians
    show_trace   :: Bool    = false,      # print the Newton iteration trace
    nl_iter      :: Int     = 50,         # max Newton iterations per stage
    nl_tol       :: Float64 = 1e-5,       # Newton residual tolerance (‖r‖∞). Production default.
                                          #   MEASURED 2026-08-12, and it is NOT a null change:
                                          #   Newton takes an INTEGER number of iterations, so this
                                          #   tolerance acts as a step function. At 1e-6 Newton needs
                                          #   3 iterations/step; at 1e-5 (and at 1e-4 — bit-identical)
                                          #   it needs 2, and max|eta| shifts by 3.8e-5 relative.
                                          #   Accepted on the error budget, NOT on a null result: the
                                          #   dropped 3rd iteration polishes a residual (~3e-6) some
                                          #   600x smaller than the O(dt^2) time-discretisation error
                                          #   the answer already carries (‖R‖∞ ~ 1.8e-3, measured by
                                          #   the run's own residual check). Tests that need a sharper
                                          #   answer pin 1e-8 explicitly.
    print_every  :: Int     = 1,          # print a step report every N steps (1 = every step)
    check_every  :: Int     = 50,         # re-verify the governing equations every N steps (0 = off)
    check_tol    :: Float64 = 1e-8,       # tolerance for that verification (‖R‖∞)
    # ---- Field diagnostics (monitor.jl: max|η| location, invariants, RSS) -----
    diag_every   :: Int     = 0,          # sample every N steps (0 → = print_every; −1 = disabled)
    diag_csv     :: Bool    = true,       # write output_dir/diagnostics.csv (needs save_every≥0 dir)
    eta_ref                 = nothing,    # reference amplitude for the divergence guard (auto)
    div_factor   :: Float64 = 20.0,       # abort when max|η| > div_factor · eta_ref
    # ---- Reconstructed field output (at the Nσ vertical σ-nodes) --------------
    write_w        :: Bool    = false,    # also write vertical-velocity fields w_s<σ> to VTK
    write_pressure :: Bool    = false,    # also write total-pressure fields p_s<σ> to VTK
    rho            :: Float64 = rho,      # water density [kg/m³] (used for the pressure output)
)
    # Choose the σ-node positions: the paper's optimised set for this M when
    # available, otherwise a uniform split of [0,1] into M+1 nodes.
    if isnothing(c_bdy)
        c_bdy = get(DEFAULT_CBDY, M, collect(LinRange(0.0, 1.0, M + 1)))
    end

    # --- STAGE 1: VERTICAL PRE-COMPUTATION (MESH INDEPENDENT, DONE ONCE) -------
    # Build the σ-basis and integrate every vertical tensor the residual needs.
    println("=== Vertical FE problem (algebraic BALFE-M) ===")
    vert = assemble_vertical_tensors(M, p_vertical, c_bdy)
    @printf("  Nσ=%d   ΣΦ=%.6f\n", vert.N_dof, sum(vert.Phi))   # ΣΦ=1 sanity check

    # Lateral (y) boundary condition: :wall / :open / :periodic.
    y_wall_bc in (:wall, :open, :periodic) ||
        error("setup_and_run: y_wall_bc must be :wall, :open or :periodic (got :$y_wall_bc)")
    y_periodic = y_wall_bc == :periodic  # set y_periodic = true   when  y_wall_bc == :periodic
    if y_periodic
        # Check for bottom/top sponge layers, incompatible with y-periodic BCs (they are redundant).
        (sponge_wB > 0 || sponge_wT > 0) &&
            @warn "y_wall_bc=:periodic — lateral sponges (sponge_wB/wT) are redundant on a " *
                  "periodic domain; set them to 0"
        # Check for a point source wavemaker, which is not y-periodic (the periodic image array is not a point source).
        !isnothing(y_wm) &&
            @warn "y_wall_bc=:periodic with a point source is not y-periodic (periodic image " *
                  "array); use a line source (y_wm=nothing)"
    end

    # --- STAGE 2 SETUP: HORIZONTAL MESH + INTEGRATION MEASURE ----------------- 
    # `y_periodic` glues the top/bottom edges when y_wall_bc == :periodic.
    model, trian = build_horizontal_model(domain, partition; y_periodic=y_periodic)
    # quadrature degree = 2·p_horizontal+2 integrates the nonlinear (product) terms exactly enough.
    dΩh = Measure(trian, 2*max(p_horizontal, p_eta == 0 ? p_horizontal : p_eta) + 2)

    # Forcing frequency and the matching wavenumber from the Airy relation
    # ω² = g k tanh(kd) (used to size the wavemaker and report kd).
    omega  = 2.0*pi/T_wave                      # compute wave frequency from the period
    k_wave = find_wavenumber(omega, h_val, g)   # compute corresponding wavenumber from the Airy dispersion relation

    # dfn: the bathymetry function: either a constant h_val or a user-supplied h_bathy(x).
    dfn    = isnothing(h_bathy) ? (x -> h_val) : h_bathy    # bathymetry: constant h_val or user d(x)

    # Unpack the domain corners (accept either nested or flat tuple form).
    if domain isa Tuple{Tuple,Tuple}
        (x0d, x1d), (y0d, y1d) = domain
    else
        x0d, x1d, y0d, y1d = domain
    end

    # ---- DIRICHLET BOUNDARY WAVE GENERATION (WAVEINPUT.JL) --------------------
    # If wave_bc is set, build the component table `wi` that drives the inflow
    # boundary; the interior wavemaker is disabled below. `wi` stays nothing for
    # the internal-wavemaker path.
    wg = resolve_wave_gen(wave_gen, wave_bc)
    @printf("  Wave generation: %s\n", string(wg))
    wi = nothing
    if wg === :bc_gen
        bc_side in (:left, :right) ||
            error("setup_and_run: bc_side must be :left or :right (got :$bc_side)")
        Tr = T_ramp === nothing ? nothing : Float64(T_ramp)

        # The boundary source is dispatched on the TYPE of wave_bc.
        if wave_bc isa WaveInput  # a prebuilt WaveInput passes through
            wi = wave_bc
        elseif wave_bc isa WaveSpec.AiryWaves.AiryState  # a WaveSpec AiryState is converted
            wi = WaveInput(vert, wave_bc; d=h_val, g=g, T_ramp=Tr, profile=bc_profile)
        else     # otherwise (nothing / :regular) a parametrised regular plane wave is built from A_wave/T_wave/wave_dir. 
            tr_val = Tr === nothing ? 2.0 * T_wave : Tr
            wi = WaveInput(vert; A=A_wave, T=T_wave, d=h_val, g=g, theta=wave_dir,
                        T_ramp=tr_val, profile=bc_profile)
        end
        # All three feed the same Dirichlet machinery, which needs to be feed the component table `wi`,
        # a WaveInput structure (the only difference is how the table is populated).

        println()
        waveinput_summary(wi)      # print Hs/Tp/components of the generated sea
        # Sanity: the generation boundary must sit over a constant depth equal to
        # the depth the boundary data was built for.
         
        # Identify the x-coordinate of the generation boundary (x0d or x1d) 
        xg = bc_side == :left ? x0d : x1d

        # Sample the bathymetry along the y-direction at that x coordinate (xg = generation BC position). The depth must be constant
        dsamp = [dfn(VectorValue(xg, y0d + s*(y1d - y0d))) for s in 0.0:0.25:1.0]

        # Check if all sampled depths are approximately equal to the WaveInput depth (wi.d) within a relative tolerance of 1e-8. If not, issue a warning.
        all(v -> isapprox(v, wi.d; rtol=1e-8), dsamp) ||
            @warn "wave_bc: depth along the generation boundary is not constant " *
                  "(or differs from the WaveInput depth $(wi.d) m)"

        # A directional (θ≠0) sea needs open/periodic lateral boundaries, not walls.
        wi.directional && y_wall_bc == :wall &&
            error("setup_and_run: a directional sea (θ≠0 components) requires " *
                  "y_wall_bc=:open (lateral sponges) or :periodic")
        wi.directional && y_periodic &&
            @warn "y_wall_bc=:periodic with a directional sea requires each component's " *
                  "transverse wavenumber k·sinθ to be a box harmonic 2π/Ly (not enforced)"

        # A hot start supplies the field at t=0, so it is incompatible with a ramp.
        ic_from_bc && wi.T_ramp > 0.0 &&
            error("setup_and_run: ic_from_bc=true requires T_ramp=0.0 " *
                  "(the hot start replaces the ramp)")

        # Warn if a sponge sits on the inflow (it would eat the incoming wave)…
        gen_w = bc_side == :left ? sponge_wL : sponge_wR
        gen_w > 0.0 && !relax_bc &&
            @warn "wave_bc: a plain sponge overlaps the generation boundary and " *
                  "damps the incident wave; set its width to 0 or use relax_bc=true"
        # …and if there is no sponge on the far side to absorb the outgoing wave.
        opp_w = bc_side == :left ? sponge_wR : sponge_wL
        opp_w > 0.0 ||
            @warn "wave_bc: no sponge opposite the inflow — expect reflections"
    end

    # --- STACKED FE SPACES [η,𝖴x,𝖴y] + BOUNDARY CONDITIONS wi --------------------
    println("\n=== 2D Horizontal FE problem (stacked [η,𝖴x,𝖴y]) ===")
    # For a generated sea, pass the time-varying Dirichlet data (η, 𝖴x, and 𝖴y
    # for directional seas) so build_fe_spaces makes the matching transient trials.

    # Build inflow BC data for build_fe_spaces. 
    # If wi is nothing, the interior wavemaker is used and no Dirichlet BCs are applied. 
    if wi === nothing
        inflow = nothing
    # Otherwise, the inflow BCs are built from the WaveInput structure wi.
    else
        if wi.directional
            uy_val = uy_bc(wi)
        else
            uy_val = nothing
        end
        inflow = (side = bc_side, eta = eta_bc(wi), ux = ux_bc(wi), uy = uy_val)
    end

    # Build the stacked FE spaces for the horizontal problem, applying the inflow BCs if provided.
    pe = p_eta == 0 ? p_horizontal : p_eta   # 0 ⇒ equal order (unchanged default)
    U, V = build_fe_spaces(model, 
                                p_horizontal,           # horizontal (velocity) FE order
                                vert.N_dof;             # number of vertical DOFs = number of stacked fields
                                y_wall_bc=y_wall_bc,    # lateral BC type
                                x_wall_bc=x_wall_bc,    # solid wall BC on x-edges
                                inflow=inflow,          # inflow BC data (η, 𝖴x, 𝖴y) if provided
                                p_eta=pe)               # surface FE order (see the kwarg note)

    @printf("  Fields: 3 (η + 2 stacked VectorValue{%d})   free DOFs: %d\n",
            vert.N_dof, num_free_dofs(U(0.0)))
    @printf("  Wave: λ=%.2f m, kd=%.2f\n", 2pi/k_wave, k_wave*h_val)

    # --- Forcing: sponge profile + internal wavemaker source ------------------
    # Sponge damping μ(x,y) grows quadratically toward the flagged boundaries.
    sponge = make_sponge(domain, sponge_wL, sponge_wR, sponge_wB, sponge_wT, mu_max)

    # Internal source S(x,t): none when generating at a boundary; a line source
    # (plane waves) when y_wm is nothing; a point source (ring waves) otherwise.
    if wi !== nothing
        wm = (x, t) -> 0.0  # Dirichlet generation disables the interior wavemaker
    elseif isnothing(y_wm)
        wm = make_wavemaker_line(x_wm, A_wave, T_wave, k_wave)  # line source (plane waves)
    else
        wm = make_wavemaker_point(x_wm, Float64(y_wm), A_wave, T_wave)  # point source (ring waves)
    end

    # --- Optional relaxation zone next to the inflow (generation + absorption) -
    # Blends the state toward the incident wave over a strip at the boundary.
    relax_mu_fn = x -> 0.0
    relax_tg    = nothing
    use_relax   = wi !== nothing && relax_bc
    if use_relax
        wrx = relax_width > 0.0 ? relax_width : 2.0*pi/wi.ks[argmax(wi.amps)]
        relax_mu_fn = bc_side == :left ?
            (x -> x[1] < x0d + wrx ? mu_max*((x0d + wrx - x[1])/wrx)^2 : 0.0) :
            (x -> x[1] > x1d - wrx ? mu_max*((x[1] - (x1d - wrx))/wrx)^2 : 0.0)
        relax_tg = incident_fields(wi)
        @printf("  Relaxation zone: width=%.2f m, μ_max=%.2f, %s boundary\n",
                wrx, mu_max, string(bc_side))
    end

    # Warn if the user has set flat_bed=true but the bathymetry varies, or vice versa.
    check_flat_bed_consistency(dfn, domain, flat_bed)   # warn on bed ↔ flat_bed mismatch

    # --- Assemble the problem bundle: vertical tensors → Gridap constants +
    #     the depth, forcing, and physics flags that define the residual. -------
    prob = build_problem(vert; g=g, 
                        h_bathy=dfn,                # bathymetry function (x → d(x))
                        regime=regime,              # linear/nonlinear physics
                        nl_pressure=nl_pressure,    # nonlinear pressure treatment
                        flat_bed=flat_bed,          # whether to drop ∇h terms (flat bed)
                        mu_sponge=sponge,           # sponge damping profile μ(x,y)
                        wm_src=wm,                  # internal wavemaker source S(x,t)
                        relax_bc=use_relax,         # whether to use a relaxation zone at the inflow
                        relax_mu=relax_mu_fn,       # relaxation-zone damping profile μ(x,y)
                        relax_tg=relax_tg)          # incident wave target for the relaxation zone

    # Build problem TransientFEOperator ->  Wrap the residual (+ Jacobians) into a Gridap operator
    # `use_ad` swaps the hand Jacobians for AD-generated ones (cross-checking only).
    op = use_ad ? build_ode_operator_ad(prob, U, V, trian, dΩh) :
                  build_ode_operator(prob, U, V, trian, dΩh)

    # The monitor transparently wraps the Newton solver to harvest per-step stats.
    monitor = SolverMonitor()

    # Build the time integrator (SDIRK by default) around a Newton+LU nonlinear solve.
    solver  = build_ode_solver(dt;  solver_type=solver_type,    # :sdirk | :theta | :gen_alpha | :rk3
                                    theta=theta,                # θ for :theta (0.5 = Crank–Nicolson)
                                    rho_inf=rho_inf,            # high-frequency damping for :gen_alpha (ρ∞∈[0,1])
                                    tableau=tableau,            # SDIRK tableau for :sdirk
                                    nl_iter=nl_iter,            # max Newton iterations per stage
                                    nl_tol=nl_tol,              # Newton residual tolerance (‖r‖∞)
                                    show_trace=show_trace,      # print the Newton iteration trace
                                    monitor=monitor)            # wrap the Newton solver to harvest per-step stats

    # Optional independent re-assembly of the governing equations for verification
    # -> check if the governing equations are satisfied at the current solution (‖R‖∞ < check_tol).
    # (its θ-scheme self-check is meaningful only for the :theta integrator).
    checker = check_every > 0 ?
              ResidualChecker(prob, U, V, trian, dΩh, dt, theta,
                                 solver_type == :theta) : nothing

    # Initial condition. Four cases: hot-start from the incident wave; rest state
    # for a generated sea; rest state (default); or a prescribed η₀(x) release.
    u0 = if wi !== nothing && ic_from_bc
        inc = incident_fields(wi)
        make_initial_conditions(U(0.0), vert.N_dof;
            eta0_func = x -> inc.eta(x, 0.0),
            ux0_func  = x -> inc.ux(x, 0.0),
            uy0_func  = x -> inc.uy(x, 0.0))
    elseif wi !== nothing
        make_initial_conditions(U(0.0), vert.N_dof; eta0_func=eta0_func)
    elseif isnothing(eta0_func)
        make_initial_conditions(U)
    else
        make_initial_conditions(U, vert.N_dof; eta0_func=eta0_func)
    end

    # For nl_pressure=:full, build the frozen-projection context (mass matrix
    # factorised once) used to evaluate the irreducible ∇H/𝓟 pressure halves.
    nlp = nl_pressure == :full ?
          (prob, build_nlp_ctx(model, p_horizontal, vert.N_dof, trian, dΩh)) : nothing

    # Reconstruction context for optional w/p VTK output (nothing if both off).
    recon = build_field_recon(vert, dfn, g; rho=rho,
                                  write_w=write_w, write_pressure=write_pressure)
    if recon !== nothing
        @printf("  Field output: write_w=%s write_pressure=%s at σ-levels %s\n",
                string(write_w), string(write_pressure),
                string(round.(recon.levels; digits=3)))
    end

    # Field diagnostics: max|η| location + interior/damped split, |u|/|η|, mass
    # and energy invariants, process RSS, and the relative divergence guard.
    # `diag_every=0` follows `print_every` so every reported line is complete;
    # a negative value switches the whole block off.
    diag_n   = diag_every == 0 ? max(print_every, 1) : diag_every
    eta_ref_v = resolve_eta_ref(eta_ref, A_wave, wi, eta0_func, domain)
    rundiag  = diag_n > 0 ?
               build_run_diagnostics(prob, wi !== nothing ? U(0.0) : U, trian, dΩh;
                                     eta_ref=eta_ref_v, div_factor=div_factor,
                                     output_dir=output_dir, diag_csv=diag_csv,
                                     u0=u0) : nothing

    # Print the solver-configuration banner (integrator, tolerances, dt/steps).
    println()
    print_solver_banner(
        @sprintf("Newton (NLsolve, exact hand Jacobians) | max iters = %d | ftol (‖r‖∞) = %.1e",
                 nl_iter, nl_tol),
        "LU direct factorisation (sequential)";
        solver_type=solver_type, theta=theta, dt=dt, t0=0.0, T_final=T_final,
        print_every=print_every, check_every=check_every, check_tol=check_tol,
        monitor=monitor, diag_every=max(diag_n, 0), eta_ref=eta_ref_v,
        div_limit=rundiag === nothing ? NaN : rundiag.div_limit)

    # --- March the transient problem from 0 to T_final, collecting per-step
    #     diagnostics and writing VTK/gauge output as requested. ---------------
    println("\n=== Time loop (algebraic) ===")
    diags = run_time_loop(op, solver,                   # Gridap operator + time integrator, constructed at build_ode_solver
                            u0,                         # initial condition (Gridap FE function)
                            0.0,                        # initial time
                            T_final;                    # final time
                            output_dir=output_dir,      # VTK/pvd destination (output directory)
                            save_every=save_every,      # write a VTK snapshot every N steps (0 = no VTK)
                            trian=trian,                # horizontal mesh (for VTK output)
                            Nσ=vert.N_dof,              # number of vertical σ-levels = number of stacked horizontal fields (+ 1 for η) 
                            print_every=print_every,    # print a step report every N steps (1 = every step)
                            gauges=gauges,              # list of (x,y) probe points; η is sampled there each step
                            recon=recon,                # reconstruction context for optional w/p VTK output (nothing if both off)
                            trial_space=U,              # trial FE space (for VTK output)
                            dt=dt,                      # time step [s]
                            nlp=nlp,                    # frozen-projection context for nl_pressure=:full (nothing if not used)
                            monitor=monitor,            # wrap the Newton solver to harvest per-step stats
                            checker=checker,            # optional independent re-assembly of the governing equations for verification
                            check_every=check_every,    # re-verify the governing equations every N steps (0 = off)
                            check_tol=check_tol,        # tolerance for that verification (‖R‖∞)
                            rundiag=rundiag,            # field-diagnostics state (max|η| location, invariants, RSS)
                            diag_every=max(diag_n, 0))  # sample the field diagnostics every N steps (0 = off)

    return diags, vert, prob
end
