# ==============================================================
#  run_ring_small.jl — PARAMETRIC small-domain ring wave (interior point source)
#
#  Gaussian POINT-source wavemaker at the domain centre (25,10) => circular
#  wavefronts, 4-side sponge absorption, on the 50x20 m small domain. Covers the
#  ring observation case (case 4). Physics via environment variables.
#
#  Config via env (base = nonlinear / full pressure / flat bed / A=0.1):
#    LFEM_REGIME       linear | nonlinear                    (default nonlinear)
#    LFEM_NL_PRESSURE  none | native | full                  (default full)
#    LFEM_AWAVE        wave amplitude [m]                    (default 0.1)
#    LFEM_TWAVE        wave period [s]                       (default 2.0)
#
#  Fixed defaults: domain 50x20, mesh 200x80 (square 0.25 m cells, isotropic for
#  the ring), partition 8x8 (64 ranks), fe_order 2, dt 0.02, d 3.5, 4-side
#  sponges L/R 6 / B/T 4, mu_max 12, periods 12, save_every 10.
#  (Flat bed only: a ring has no meaningful shore-parallel bar variant.)
#
#  LAUNCH (px*py MUST equal -n): see run/dist_small/run_nl_ring_flat_small.sh
# ==============================================================

include(joinpath(@__DIR__, "..", "distributed", "_dist_common.jl"))

get!(ENV, "LFEM_REGIME", "nonlinear")
get!(ENV, "LFEM_NL_PRESSURE", "full")
get!(ENV, "LFEM_FLAT_BED", "1")

M       = genv_i("LFEM_M", 2)
px, py  = genv_i("LFEM_PX", 8), genv_i("LFEM_PY", 8)      # 8*8 = 64 ranks
nx, ny  = genv_i("LFEM_NX", 160), genv_i("LFEM_NY", 160)   # square cells 0.25x0.25
feord   = genv_i("LFEM_FE_ORDER", 2)
Lx, Ly  = genv_f("LFEM_LX", 40.0), genv_f("LFEM_LY", 40.0)
d       = genv_f("LFEM_D", 3.5)
Twave   = genv_f("LFEM_TWAVE", 2.0)
Awave   = genv_f("LFEM_AWAVE", 0.1)
x_wm    = genv_f("LFEM_XWM", Lx/2)
y_wm    = genv_f("LFEM_YWM", Ly/2)
spX     = genv_f("LFEM_SPONGE_X", 4.0)
spY     = genv_f("LFEM_SPONGE_Y", 4.0)
mumax   = genv_f("LFEM_MUMAX", 40.0)   # strong: kill the reflected/boundary mode fast
dt      = genv_f("LFEM_DT", 0.02)
periods = genv_f("LFEM_PERIODS", 12.0)
Tfinal  = haskey(ENV, "LFEM_TFINAL") ? genv_f("LFEM_TFINAL", 0.0) : periods * Twave
save_ev = genv_i("LFEM_SAVE_EVERY", 10)

tag     = "$(regime_sym())_$(nl_pressure_sym())_A$(Awave)_T$(Twave)"
outdir  = genv("LFEM_OUTDIR", joinpath(ROOT, "output", "small_ring_$(tag)_M$(M)"))

banner("SMALL | ring wave (point source, flat bed) | $(regime_sym()) $(nl_pressure_sym()) A=$Awave",
       M, (px,py), (nx,ny), nx*ny, outdir)

diags, vert, prob = setup_and_run_distributed(
    cpu_grid=(px,py), M=M, c_bdy=cbdy_override(), p_horizontal=feord,
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
    print_every=genv_i("LFEM_PRINT_EVERY", 10))

is_rank0() && @printf("ring [%s] done: %d steps, %d snapshots to %s\n",
                      tag, length(diags), length(diags) ÷ max(save_ev,1), outdir)
