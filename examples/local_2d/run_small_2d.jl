# ==============================================================
#  run_small_2d.jl — PARAMETRIC small 2-D case (local, 12-rank partition)
#
#  Scaled-down siblings of the cluster suite in run/dist_small/, sized so a
#  case finishes in minutes on this workstation instead of hours on 32 ranks.
#  The point is a fast feedback loop: run it, read diagnostics.csv, decide.
#  See building_files/LOCAL_VALIDATION_PLAN.md §6.
#
#  Each case maps onto a cluster sibling so a local result is predictive:
#    line/linear/flat      ↔ run_lin_periodic_plane_small.sh
#    line/nonlinear/none   ↔ run_nl_none_periodic_plane_small.sh
#    line/nonlinear/full   ↔ run_nl_periodic_plane_flat_small.sh
#    line/nl/full/bar      ↔ run_nl_periodic_plane_varbed_small.sh
#    point/nonlinear       ↔ run_nl_ring_flat_small.sh
#    bc/linear/flat        ↔ run_lin_bc_plane_flat_small.sh
#    bc/nl/full/bar        ↔ run_nl_bc_plane_varbed_small.sh
#    sea/nl/full/flat      ↔ run_nl_irregular_sea_small.sh
#
#  DISTRIBUTED BY DEFAULT, and therefore WITHOUT POINT GAUGES: the distributed
#  driver evaluates none (timeloop_dist.jl:21-22). Every assertion about these
#  runs comes from `<output_dir>/diagnostics.csv` — max|η| and WHERE it sits,
#  the interior/damped split, |u|/|η|, mass and energy, GMRES saturation and
#  per-rank RSS. That is why Task 2 (the instrumentation) precedes this file.
#  Set LFEM_MPI=0 for a sequential run if you specifically want gauges; it is
#  slower here because the sequential path factorises the whole matrix.
#
#  CONFIG (env; defaults = the linear flat-bed plane-wave reference)
#    LFEM_WAVE_GEN     line | point | bc | sea       line
#                        line  = interior Gaussian LINE source  ⇒ plane wave
#                        point = interior Gaussian POINT source ⇒ ring wave
#                        bc    = Dirichlet boundary, regular wave
#                        sea   = Dirichlet boundary, WaveSpec JONSWAP sea
#    LFEM_REGIME       linear | nonlinear             linear
#    LFEM_NL_PRESSURE  none | native | full           none
#    LFEM_FLAT_BED     1 flat | 0 submerged bar       1
#    LFEM_PX/LFEM_PY   MPI grid (px·py = mpiexec -n)  4 / 3      (= 12 ranks)
#    LFEM_LX/LY        domain [m]                     40 / 15
#    LFEM_NX/NY        cells                          96 / 36    (dx=dy=0.417 m)
#    LFEM_AWAVE        amplitude [m]                  0.001
#    LFEM_PERIODS      duration in wave periods       10
#  plus every knob of examples/distributed/_dist_common.jl.
#
#  RUN
#    run/local/run_2d_<case>.sh                        (recommended)
#    LFEM_MPI=1 ~/.julia/bin/mpiexecjl --project=. -n 12 \
#        julia --project=. examples/local_2d/run_small_2d.jl
# ==============================================================

include(joinpath(@__DIR__, "..", "distributed", "_dist_common.jl"))

get!(ENV, "LFEM_REGIME",      "linear")
get!(ENV, "LFEM_NL_PRESSURE", "none")
get!(ENV, "LFEM_FLAT_BED",    "1")

kind = lowercase(genv("LFEM_WAVE_GEN", "line"))
kind in ("line", "point", "bc", "sea") ||
    error("LFEM_WAVE_GEN must be line, point, bc or sea (got $kind)")
use_mpi = genv_b("LFEM_MPI", 1)

# ---- geometry / numerics -------------------------------------------------
M       = genv_i("LFEM_M", 2)
px, py  = genv_i("LFEM_PX", 4), genv_i("LFEM_PY", 3)     # 4·3 = 12-rank partition
Lx, Ly  = genv_f("LFEM_LX", 40.0), genv_f("LFEM_LY", 15.0)
nx, ny  = genv_i("LFEM_NX", 96), genv_i("LFEM_NY", 36)   # dx = dy = 0.417 m
#  SIZING FOR A 12-RANK PARTITION (2026-08-06): the domain grew 25×10 → 40×15 m
#  (2.4× the area, so the wave train has fetch before the sponge) AND the cells
#  shrank 0.52/0.50 → 0.417 m (≈9.6 per wavelength at kd=5.5). ≈98.6k free DOFs
#  against ≈29k before — the cores bought a bigger, better-resolved case, not a
#  faster one.
feord   = genv_i("LFEM_FE_ORDER", 2)
d       = genv_f("LFEM_D", 3.5)
Twave   = genv_f("LFEM_TWAVE", 1.6)                      # kd = 5.5, λ = 4.0 m
Awave   = genv_f("LFEM_AWAVE", 0.001)
dt      = genv_f("LFEM_DT", 0.04)
periods = genv_f("LFEM_PERIODS", 10.0)
Tfinal  = haskey(ENV, "LFEM_TFINAL") ? genv_f("LFEM_TFINAL", 0.0) : periods*Twave
save_ev = genv_i("LFEM_SAVE_EVERY", 10)
mumax   = genv_f("LFEM_MUMAX", 40.0)

if use_mpi
    nx % px == 0 || error("LFEM_NX ($nx) must be divisible by LFEM_PX ($px)")
    ny % py == 0 || error("LFEM_NY ($ny) must be divisible by LFEM_PY ($py)")
    px*py <= 12 || @warn "px·py = $(px*py) exceeds this machine's 12-rank partition — expect oversubscription"
end

# ---- bathymetry ----------------------------------------------------------
usebar = !flat_bed_flag(1)
hbar   = genv_f("LFEM_HBAR", 1.0)
xbar   = genv_f("LFEM_XBAR", 0.5*Lx)
wbar   = genv_f("LFEM_WBAR", 3.0)
sramp  = wbar / 3.0
h_bathy = usebar ?
    (x -> d - 0.5*hbar*(tanh((x[1]-(xbar-wbar))/sramp) - tanh((x[1]-(xbar+wbar))/sramp))) :
    nothing
bedtag = usebar ? "bar" : "flat"

# ---- wave generation + the lateral BC that goes with it ------------------
# A ring wave needs solid lateral walls (the cluster ring case runs that way);
# a long-crested plane wave wants y-periodic so no lateral wall reflects it.
vert0 = assemble_vertical_tensors(M, 1, cbdy_override() === nothing ?
                                  [0.0, 0.728, 1.0] : cbdy_override())
if kind == "line"
    wave_bc = nothing; y_wm = nothing
    x_wm  = genv_f("LFEM_XWM", 0.25*Lx); spL = genv_f("LFEM_SPONGE_L", 8.0)
    ybc   = Symbol(genv("LFEM_YBC", "periodic")); spB = spT = 0.0
    use_relax = genv_b("LFEM_RELAX", 0)
elseif kind == "point"
    wave_bc = nothing; y_wm = genv_f("LFEM_YWM", 0.5*Ly)
    x_wm  = genv_f("LFEM_XWM", 0.5*Lx); spL = genv_f("LFEM_SPONGE_L", 8.0)
    ybc   = Symbol(genv("LFEM_YBC", "wall")); spB = spT = genv_f("LFEM_SPONGE_BT", 3.0)
    use_relax = genv_b("LFEM_RELAX", 0)
elseif kind == "bc"
    wave_bc = WaveInput(vert0; A=Awave, T=Twave, d=d,
                        T_ramp=genv_f("LFEM_TRAMP", 2*Twave), profile=bc_profile_sym())
    y_wm = nothing; x_wm = 0.0; spL = 0.0
    ybc  = Symbol(genv("LFEM_YBC", "periodic")); spB = spT = 0.0
    use_relax = genv_b("LFEM_RELAX", 1)
else                                        # sea
    wave_bc = WaveInput(vert0, build_airy_state(d); d=d,
                        T_ramp=tramp_val(), profile=bc_profile_sym())
    y_wm = nothing; x_wm = 0.0; spL = 0.0
    # a directional sea prescribes 𝖴y at the inflow, which is incompatible with
    # the y-wall corner tags (horizontal.jl:128) — periodic is the safe default
    ybc  = Symbol(genv("LFEM_YBC", "periodic")); spB = spT = 0.0
    use_relax = genv_b("LFEM_RELAX", 1)
end
spR = genv_f("LFEM_SPONGE_R", 10.0)

tag    = @sprintf("%s_%s_%s_%s_A%g", kind, regime_sym(), nl_pressure_sym(), bedtag, Awave)
outdir = genv("LFEM_OUTDIR", joinpath(ROOT, "output", "local_2d", "small2d_$(tag)_M$(M)"))

if is_rank0()
    @printf("############################################################\n")
    @printf("# SMALL 2-D (local) | gen=%s | %s %s %s | A=%g T=%g\n",
            kind, regime_sym(), nl_pressure_sym(), bedtag, Awave, Twave)
    @printf("#   M=%d | domain %.0f×%.0f m | mesh %d×%d (dx=%.2f, dy=%.2f) | %s\n",
            M, Lx, Ly, nx, ny, Lx/nx, Ly/ny,
            use_mpi ? "MPI $(px)×$(py) = $(px*py) ranks (NO gauges)" : "sequential")
    @printf("#   dt=%g | %g periods → T=%.1f s | y-BC=%s | sponge L/R/B/T = %.0f/%.0f/%.0f/%.0f μ=%.0f\n",
            dt, periods, Tfinal, ybc, spL, spR, spB, spT, mumax)
    @printf("#   out=%s\n", outdir)
    @printf("#   judge this run from %s/diagnostics.csv\n", outdir)
    @printf("############################################################\n")
    flush(stdout)
end

common = (M=M, c_bdy=cbdy_override(), p_horizontal=feord,
          h_val=d, h_bathy=h_bathy, T_wave=Twave, A_wave=Awave,
          x_wm=x_wm, y_wm=y_wm,
          sponge_wL=spL, sponge_wR=spR, sponge_wB=spB, sponge_wT=spT, mu_max=mumax,
          T_final=Tfinal, dt=dt,
          regime=regime_sym(), nl_pressure=nl_pressure_sym(), flat_bed=flat_bed_flag(1),
          y_wall_bc=ybc, x_wall_bc=false,
          wave_bc=wave_bc, bc_side=bc_side_sym(), bc_profile=bc_profile_sym(),
          relax_bc=use_relax, relax_width=relax_w_val(),
          output_dir=outdir, save_every=save_ev,
          write_w=write_w_flag(), write_pressure=write_p_flag(), rho=rho_val(),
          solver_type=solver_sym(), tableau=tableau_sym(),
          nl_iter=nl_iter_val(), nl_tol=nl_tol_val(),
          print_every=genv_i("LFEM_PRINT_EVERY", 10),
          check_every=genv_i("LFEM_CHECK_EVERY", 0),
          diag_every=genv_i("LFEM_DIAG_EVERY", 10),
          div_factor=genv_f("LFEM_DIV_FACTOR", 20.0))

if use_mpi
    diags, vert, prob = setup_and_run_distributed(;
        cpu_grid=(px, py), domain=(0.0, Lx, 0.0, Ly), partition=(nx, ny),
        ls_rtol=ls_rtol_val(), ls_maxiter=ls_maxiter_val(), krylov_m=krylov_m_val(),
        common...)
else
    diags, vert, prob = setup_and_run(;
        domain=((0.0, Lx), (0.0, Ly)), partition=(nx, ny),
        gauges=[(0.5Lx, 0.5Ly), (0.5Lx + 1.0, 0.5Ly)], common...)
end

is_rank0() && @printf("small_2d [%s] done: %d steps → %s\n", tag, length(diags), outdir)
