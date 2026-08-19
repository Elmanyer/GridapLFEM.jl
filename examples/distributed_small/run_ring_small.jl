# ==============================================================
#  run_ring_small.jl — PARAMETRIC small-domain ring wave (interior point source)
#
#  Gaussian POINT-source wavemaker at the domain centre (25,10) => circular
#  wavefronts, 4-side sponge absorption, on the 50x20 m small domain. Covers the
#  ring observation case (case 4). Physics via environment variables.
#
#  Config via env (base = nonlinear / full pressure / flat bed / A=0.1):
#    BALFEM_REGIME       linear | nonlinear                    (default nonlinear)
#    BALFEM_NL_PRESSURE  none | native | full                  (default full)
#    BALFEM_AWAVE        wave amplitude [m]                    (default 0.1)
#    BALFEM_TWAVE        wave period [s]                       (default 2.0)
#
#  Fixed defaults: domain 40x40, mesh 160x160 (square 0.25 m cells — isotropic is
#  required here, see below), partition 8x8 (64 ranks), p_horizontal 2, dt 0.02,
#  d 3.5, 4-side sponges X/Y 7 m (= 1.12 lambda), mu_max 40, periods 12,
#  save_every 10.
#
#  ISOTROPIC CELLS ARE A SOLVER REQUIREMENT, NOT AN AESTHETIC ONE: GMRES iteration
#  count grows with mesh anisotropy (a 4:1 mesh measured ~760 iters/solve against
#  ~480 isotropic). This case also carries solid lateral walls, which cost a further
#  ~40%% — it is the most linear-solver-expensive case in the suite.
#
#  Duration: point source at the centre (20,20), sponge from r=16 m, c_g=1.58 m/s
#  => transit 10.1 s = 5.1 T, t_settle = 8.1 T. 12 periods is ample.
#  (Flat bed only: a ring has no meaningful shore-parallel bar variant.)
#
#  LAUNCH (px*py MUST equal -n): see run/dist_small/run_nl_ring_flat_small.sh
# ==============================================================

include(joinpath(@__DIR__, "..", "distributed", "_dist_common.jl"))

get!(ENV, "BALFEM_REGIME", "nonlinear")
get!(ENV, "BALFEM_NL_PRESSURE", "full")
get!(ENV, "BALFEM_FLAT_BED", "1")

M       = genv_i("BALFEM_M", 2)

#  Vertical BASIS ORDER. The model is named P{p_vert}LFE-{M}: `Pp` is the

#  vertical Lagrange order and `M` the number of vertical elements, so the run

#  says which member of the BALFE-M family it actually exercises. Default p=1

#  reproduces the piecewise-linear models of Yang & Liu.

p_vert  = genv_i("BALFEM_P_VERT", 1)

model_name = "P$(p_vert)LFE-$(M)"
px, py  = genv_i("BALFEM_PX", 8), genv_i("BALFEM_PY", 8)      # 8*8 = 64 ranks
nx, ny  = genv_i("BALFEM_NX", 160), genv_i("BALFEM_NY", 160)   # square cells 0.25x0.25
feord   = genv_i("BALFEM_FE_ORDER", 2)
#  p_eta = 0 keeps the historical EQUAL-ORDER spaces (unchanged default).
#  Set BALFEM_P_ETA = BALFEM_FE_ORDER-1 for the Taylor-Hood-like pairing, which is
#  the only one measured to reach the theoretical order in BOTH fields. It is
#  NOT automatically the better production choice: at a GIVEN mesh the
#  equal-order spaces were 40x more accurate, because eta sits in a richer
#  space. Compare error-vs-DOF before switching.
p_eta   = genv_i("BALFEM_P_ETA", 0)
Lx, Ly  = genv_f("BALFEM_LX", 40.0), genv_f("BALFEM_LY", 40.0)
d       = genv_f("BALFEM_D", 3.5)
Twave   = genv_f("BALFEM_TWAVE", 2.0)
Awave   = genv_f("BALFEM_AWAVE", 0.1)
x_wm    = genv_f("BALFEM_XWM", Lx/2)
y_wm    = genv_f("BALFEM_YWM", Ly/2)
#  Sponge width is set by the WAVELENGTH, not by the domain: at T=2.0/d=3.5,
#  lambda = 6.23 m, and a sponge under 1 lambda reflects. 4.0 m was 0.64 lambda.
#  7 m = 1.12 lambda leaves a clean radius of 13 m = 2.1 lambda, ample for the
#  1/sqrt(r) radial-decay check. Strength cannot substitute: mu_max/omega is
#  already ~13, far past the mu ~ 5*omega saturation point where WIDTH is the
#  only remaining lever.
spX     = genv_f("BALFEM_SPONGE_X", 7.0)
spY     = genv_f("BALFEM_SPONGE_Y", 7.0)
mumax   = genv_f("BALFEM_MUMAX", 40.0)   # strong: kill the reflected/boundary mode fast
dt      = genv_f("BALFEM_DT", 0.02)
periods = genv_f("BALFEM_PERIODS", 12.0)
Tfinal  = haskey(ENV, "BALFEM_TFINAL") ? genv_f("BALFEM_TFINAL", 0.0) : periods * Twave
save_ev = genv_i("BALFEM_SAVE_EVERY", 10)

tag     = "$(regime_sym())_$(nl_pressure_sym())_A$(Awave)_T$(Twave)"
outdir  = genv("BALFEM_OUTDIR", joinpath(ROOT, "output", "small_ring_$(tag)_$(model_name)"))

banner("SMALL | ring wave (point source, flat bed) | $(regime_sym()) $(nl_pressure_sym()) A=$Awave",
       M, (px,py), (nx,ny), nx*ny, outdir)

diags, vert, prob = setup_and_run_distributed(
    cpu_grid=(px,py), M=M, p_vertical=p_vert, c_bdy=cbdy_override(), p_horizontal=feord, p_eta=p_eta,
    domain=(0.0,Lx,0.0,Ly), partition=(nx,ny),
    wave_gen=:inner_res,                                           # interior point source
    h_val=d, T_wave=Twave, A_wave=Awave, x_wm=x_wm, y_wm=y_wm,      # point source => ring
    sponge_wL=spX, sponge_wR=spX, sponge_wB=spY, sponge_wT=spY, mu_max=mumax,
    T_final=Tfinal, dt=dt,
    regime=regime_sym(), nl_pressure=nl_pressure_sym(), flat_bed=flat_bed_flag(1),
    y_wall_bc=:wall, x_wall_bc=false,
    output_dir=outdir, save_every=save_ev,
    write_w=write_w_flag(), write_pressure=write_p_flag(), rho=rho_val(),
    solver_type=solver_sym(), tableau=tableau_sym(),
    nl_iter=nl_iter_val(), nl_tol=nl_tol_val(),
    ls_rtol=ls_rtol_val(), ls_maxiter=ls_maxiter_val(), krylov_m=krylov_m_val(), precond=precond_sym(),
    diag_every=diag_every_val(), diag_csv=diag_csv_flag(),
    div_factor=div_factor_val(), eta_ref=eta_ref_val(),
    print_every=genv_i("BALFEM_PRINT_EVERY", 10))

is_rank0() && @printf("ring [%s] done: %d steps, %d snapshots to %s\n",
                      tag, length(diags), length(diags) ÷ max(save_ev,1), outdir)
