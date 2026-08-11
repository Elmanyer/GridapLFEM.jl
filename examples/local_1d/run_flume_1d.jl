# ==============================================================
#  run_flume_1d.jl — PARAMETRIC quasi-1D flume (local + cluster)
#
#  ONE script for every 1-D-horizontal case. The physics, the wave-generation
#  mechanism and the geometry are all selected by environment variables (set in
#  the launcher); nothing is hard-coded per case. See
#  building_files/LOCAL_VALIDATION_PLAN.md §5.
#
#  WHAT "1-D horizontal" MEANS HERE. The solver is structurally 2-D
#  (CartesianDiscreteModel on a rectangle, Ex/Ey throughout the residual), so a
#  1-D horizontal problem is posed as a NARROW FLUME: a long, thin domain with
#  y-periodic (or y-wall) lateral boundaries and the minimum number of cells
#  across. This needs no solver change and is the repository's established way
#  of doing it (test_bc_spectrum.jl does the same).
#
#  It is NOT a 1-D model, and the difference matters when interpreting results:
#    * the 𝖴y DOFs exist and are solved. For a y-invariant state they are ~0,
#      but they still enter the Jacobian, the GMRES spectrum and the cost;
#    * y_wall_bc=:periodic admits y-periodic modes that a true 1-D model has
#      none of — good for isolating streamwise behaviour from lateral-wall
#      artefacts, but it is an extra mode family, not fewer;
#    * ny CANNOT go below 3 with :periodic — Gridap asserts "a minimum of 3
#      elements is required in any periodic direction"
#      (Gridap/src/Geometry/CartesianGrids.jl:39). ny=1 and ny=2 are rejected
#      at mesh construction. :wall accepts ny=2, but ny=3 is used for both so
#      the wall/periodic comparison is like-for-like.
#
#  CONFIG (env; defaults = the linear flat-bed reference case)
#    LFEM_WAVE_GEN     inner | bc | sea            inner
#                        inner = interior Gaussian line source (plane wave)
#                        bc    = Dirichlet boundary generation, regular wave
#                        sea   = Dirichlet boundary generation, WaveSpec JONSWAP
#    LFEM_REGIME       linear | nonlinear          linear
#    LFEM_NL_PRESSURE  none | native | full        none
#    LFEM_FLAT_BED     1 flat | 0 submerged bar    1
#    LFEM_MPI          0 sequential | 1 MPI        1   (sequential keeps GAUGES;
#                                                       the distributed driver
#                                                       has none — see the plan §4.2)
#    LFEM_PX           MPI ranks in x (LFEM_MPI=1) 12  (px·1 must equal mpiexec -n)
#    LFEM_LX/LY        domain [m]                  60 / 3
#    LFEM_NX/NY        cells                       240 / 3  (dx=0.25 = 16 cells/λ)
#    LFEM_D            still-water depth [m]       3.5
#    LFEM_TWAVE        period [s]                  1.6   (⇒ kd=5.5, λ=4.0 m)
#    LFEM_AWAVE        amplitude [m]               0.001
#    LFEM_PERIODS      duration in wave periods    16 inner / 26 bc,sea (transit-based)
#    LFEM_RELAX        inflow relaxation zone      1 for bc/sea, 0 for inner
#    LFEM_XWM          interior source position    sponge_wL + 6 m (must clear the sponge)
#    LFEM_HBAR/XBAR/WBAR   bar shape (FLAT_BED=0)  1.0 / 30 / 5
#  plus every knob of examples/distributed/_dist_common.jl (solver, tolerances,
#  sea state, output).
#
#  RUN
#    local (12-rank)   : run/local/run_1d_<case>.sh
#    by hand           : LFEM_MPI=1 LFEM_PX=12 ~/.julia/bin/mpiexecjl --project=. -n 12 \
#                          julia --project=. examples/local_1d/run_flume_1d.jl
#    sequential+gauges : LFEM_MPI=0 julia --project=. examples/local_1d/run_flume_1d.jl
#    cluster           : sbatch run/dist_small/run_1d_<case>.sh
# ==============================================================

include(joinpath(@__DIR__, "..", "distributed", "_dist_common.jl"))

# base-case physics (launchers override; get! ⇒ the banner and the solver can
# never disagree, because both read the same resolved ENV)
get!(ENV, "LFEM_REGIME",      "linear")
get!(ENV, "LFEM_NL_PRESSURE", "none")
get!(ENV, "LFEM_FLAT_BED",    "1")

wave_gen_kind = lowercase(genv("LFEM_WAVE_GEN", "inner"))
wave_gen_kind in ("inner", "bc", "sea") ||
    error("LFEM_WAVE_GEN must be inner, bc or sea (got $wave_gen_kind)")
use_mpi = genv_b("LFEM_MPI", 1)

# ---- geometry / numerics -------------------------------------------------
M       = genv_i("LFEM_M", 2)
Lx, Ly  = genv_f("LFEM_LX", 60.0), genv_f("LFEM_LY", 3.0)
nx, ny  = genv_i("LFEM_NX", 240), genv_i("LFEM_NY", 3)
feord   = genv_i("LFEM_FE_ORDER", 2)
d       = genv_f("LFEM_D", 3.5)
Twave   = genv_f("LFEM_TWAVE", 1.6)
Awave   = genv_f("LFEM_AWAVE", 0.001)
dt      = genv_f("LFEM_DT", 0.04)
#  DEFAULT DURATION IS SET FROM THE TRANSIT TIME, and it differs by generation
#  type. At kd=5.5 the group velocity is only c_g≈1.25 m/s, so filling the flume
#  from the source to the far sponge takes:
#      interior source at x≈18 → sponge at 45 : 27 m / 1.25 = 21.6 s = 13.5 T
#      boundary source at x=0  → sponge at 45 : 45 m / 1.25 = 36.0 s = 22.5 T
#  Running a BC case for 16 T therefore leaves the far half of the domain EMPTY,
#  and `max|η|` keeps creeping up as the front advances — which reads as a
#  spurious "growth rate" (+0.028/s measured) even though the amplitude is
#  rock-steady at A. Measured that way once; defaults now cover the transit.
periods = genv_f("LFEM_PERIODS", wave_gen_kind == "inner" ? 16.0 : 26.0)
Tfinal  = haskey(ENV, "LFEM_TFINAL") ? genv_f("LFEM_TFINAL", 0.0) : periods*Twave
save_ev = genv_i("LFEM_SAVE_EVERY", 10)
mumax   = genv_f("LFEM_MUMAX", 40.0)

ny >= 3 || error("LFEM_NY must be ≥ 3 (Gridap's periodic-direction minimum)")

#  SIZING FOR A 12-RANK PARTITION (2026-08-06). The flume was 50 m at dx=0.5
#  (8 cells/λ) on 1 core; it is now 60 m at dx=0.25 (16 cells/λ) decomposed
#  12×1. The extra cores went into RESOLUTION rather than length on purpose:
#  at kd=5.5 the group velocity is only c_g=1.25 m/s, so every metre of extra
#  flume costs 0.8 s of simulated transit before the far field is usable —
#  lengthening the domain would have spent the new cores on waiting.

# ---- bathymetry: flat, or a y-invariant submerged bar --------------------
usebar = !flat_bed_flag(1)
hbar   = genv_f("LFEM_HBAR", 1.0)
xbar   = genv_f("LFEM_XBAR", 30.0)
wbar   = genv_f("LFEM_WBAR", 5.0)
sramp  = wbar / 3.0
h_bathy = usebar ?
    (x -> d - 0.5*hbar*(tanh((x[1]-(xbar-wbar))/sramp) - tanh((x[1]-(xbar+wbar))/sramp))) :
    nothing
bedtag = usebar ? "bar" : "flat"

# ---- wave generation -----------------------------------------------------
# `wave_bc` decides the mechanism by TYPE (resolve_wave_gen): nothing ⇒ interior
# source; a WaveInput ⇒ Dirichlet boundary generation. The sea state is built
# from a seeded spectrum, so every MPI rank gets an identical component table.
vert0 = assemble_vertical_tensors(M, 1, cbdy_override() === nothing ?
                                  [0.0, 0.728, 1.0] : cbdy_override())
if wave_gen_kind == "inner"
    wave_bc  = nothing
    spL      = genv_f("LFEM_SPONGE_L", 12.0)
    #  THE SOURCE MUST CLEAR THE SPONGE. The old default `0.2*Lx` happened to
    #  equal `sponge_wL` at both the previous (50 m / 10 m) and the re-sized
    #  (60 m / 12 m) geometry, putting the Gaussian line source exactly on the
    #  sponge edge: the run then reports max|η| pinned at x = x_wm with
    #  eta_max_damped/eta_max_int ≈ 0.95, i.e. the wave is being absorbed as
    #  fast as it is made and there is no clean plane wave to measure.
    #  Default is now the sponge edge + 1.5 wavelengths (λ = 4.0 m at kd=5.5).
    x_wm     = genv_f("LFEM_XWM", spL + 6.0)
    use_relax = genv_b("LFEM_RELAX", 0)
elseif wave_gen_kind == "bc"
    wave_bc  = WaveInput(vert0; A=Awave, T=Twave, d=d,
                         T_ramp=genv_f("LFEM_TRAMP", 2*Twave), profile=bc_profile_sym())
    x_wm     = 0.0
    spL      = 0.0                       # the inflow boundary must not be sponged
    use_relax = genv_b("LFEM_RELAX", 1)
else                                      # sea
    wave_bc  = WaveInput(vert0, build_airy_state(d); d=d,
                         T_ramp=tramp_val(), profile=bc_profile_sym())
    x_wm     = 0.0
    spL      = 0.0
    use_relax = genv_b("LFEM_RELAX", 1)
end
spR = genv_f("LFEM_SPONGE_R", 15.0)

# Refuse a geometry where the interior source sits inside (or within half a
# wavelength of) the left sponge — it silently destroys the case.
if wave_gen_kind == "inner" && x_wm < spL + 2.0
    error("LFEM_XWM ($(x_wm) m) is inside or too close to the left sponge " *
          "(width $(spL) m). The source would be absorbed as fast as it radiates. " *
          "Move it to at least $(spL + 2.0) m, or shrink LFEM_SPONGE_L.")
end

# Gauge rake (SEQUENTIAL ONLY — the distributed driver evaluates no points).
# Stations at 20/40/60/80 % of the flume plus a Goda-Suzuki pair at mid-length.
y_c    = Ly/2
gauges = use_mpi ? Tuple{Float64,Float64}[] :
         [(0.2Lx, y_c), (0.4Lx, y_c), (0.5Lx, y_c), (0.5Lx + 1.0, y_c),
          (0.6Lx, y_c), (0.8Lx, y_c)]

tag    = @sprintf("%s_%s_%s_%s_A%g_T%g", wave_gen_kind, regime_sym(),
                  nl_pressure_sym(), bedtag, Awave, Twave)
outdir = genv("LFEM_OUTDIR", joinpath(ROOT, "output", "local_1d", "flume_$(tag)_M$(M)"))

if is_rank0()
    @printf("############################################################\n")
    @printf("# QUASI-1D FLUME | gen=%s | %s %s %s | A=%g T=%g\n",
            wave_gen_kind, regime_sym(), nl_pressure_sym(), bedtag, Awave, Twave)
    @printf("#   M=%d | domain %.0f×%.0f m | mesh %d×%d (dx=%.3f) | %s\n",
            M, Lx, Ly, nx, ny, Lx/nx,
            use_mpi ? "MPI $(genv_i("LFEM_PX",12))×1 ranks" : "sequential (+gauges)")
    @printf("#   dt=%g s | %g periods → T_final=%.1f s | sponge L/R = %.0f/%.0f, μ=%.0f\n",
            dt, periods, Tfinal, spL, spR, mumax)
    @printf("#   out=%s\n", outdir)
    @printf("############################################################\n")
    flush(stdout)
end

common = (M=M, c_bdy=cbdy_override(), p_horizontal=feord,
          h_val=d, h_bathy=h_bathy, T_wave=Twave, A_wave=Awave,
          x_wm=x_wm, y_wm=nothing,
          sponge_wL=spL, sponge_wR=spR, sponge_wB=0.0, sponge_wT=0.0, mu_max=mumax,
          T_final=Tfinal, dt=dt,
          regime=regime_sym(), nl_pressure=nl_pressure_sym(),
          flat_bed=flat_bed_flag(1),
          y_wall_bc=Symbol(genv("LFEM_YBC", "periodic")), x_wall_bc=false,
          wave_bc=wave_bc, bc_side=bc_side_sym(), bc_profile=bc_profile_sym(),
          relax_bc=use_relax, relax_width=relax_w_val(),
          output_dir=outdir, save_every=save_ev,
          write_w=write_w_flag(), write_pressure=write_p_flag(), rho=rho_val(),
          solver_type=solver_sym(), tableau=tableau_sym(),
          nl_iter=nl_iter_val(), nl_tol=nl_tol_val(),
          print_every=genv_i("LFEM_PRINT_EVERY", 10),
          check_every=genv_i("LFEM_CHECK_EVERY", 0),
          diag_every=genv_i("LFEM_DIAG_EVERY", 0),
          div_factor=genv_f("LFEM_DIV_FACTOR", 20.0))

if use_mpi
    px = genv_i("LFEM_PX", 12)
    nx % px == 0 || error("LFEM_NX ($nx) must be divisible by LFEM_PX ($px)")
    diags, vert, prob = setup_and_run_distributed(;
        cpu_grid=(px, 1), domain=(0.0, Lx, 0.0, Ly), partition=(nx, ny),
        ls_rtol=ls_rtol_val(), ls_maxiter=ls_maxiter_val(), krylov_m=krylov_m_val(), precond=precond_sym(),
        common...)
else
    diags, vert, prob = setup_and_run(;
        domain=((0.0, Lx), (0.0, Ly)), partition=(nx, ny),
        gauges=gauges, common...)
end

is_rank0() && @printf("flume_1d [%s] done: %d steps → %s\n", tag, length(diags), outdir)
