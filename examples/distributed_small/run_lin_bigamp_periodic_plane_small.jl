# ==============================================================
#  run_lin_bigamp_periodic_plane_small.jl — SMALL-DOMAIN comparison run
#
#  CASE 9/12:  LINEAR · flat bed · y-periodic · BIG amplitude A=0.1
#  The LINEAR dispersive model driven at the SAME big amplitude as the nonlinear
#  case 2. Compared with case 2 (nonlinear, A=0.1) the difference is exactly the
#  finite-amplitude physics (crest–trough asymmetry, bound-harmonic generation,
#  amplitude dispersion). Compared with case 1 (linear, A=0.001) it also checks
#  that the linear model scales EXACTLY with amplitude (×100 in ⇒ ×100 out).
#
#  Geometry identical to cases 1 & 2. d=3.5, T=2.0 ⇒ kd≈3.5, λ≈6.25 m.
#
#  LAUNCH:  LFEM_PX=8 LFEM_PY=4 mpiexecjl --project=. -n 32 julia --project=. \
#             GridapLFEM.jl/examples/distributed_small/run_lin_bigamp_periodic_plane_small.jl
#  (SLURM: run/dist_small/run_lin_bigamp_periodic_plane_small.sh)
# ==============================================================

include(joinpath(@__DIR__, "..", "distributed", "_dist_common.jl"))

# ---- CASE 9 physics config (env still overrides; keeps banner ⇔ solver consistent) ----
get!(ENV, "LFEM_REGIME", "linear"); get!(ENV, "LFEM_NL_PRESSURE", "none")
get!(ENV, "LFEM_FLAT_BED", "1")

M        = genv_i("LFEM_M", 2)
px, py   = genv_i("LFEM_PX", 8), genv_i("LFEM_PY", 4)
nx, ny   = genv_i("LFEM_NX", 200), genv_i("LFEM_NY", 40)
feord    = genv_i("LFEM_FE_ORDER", 2)
Lx, Ly   = genv_f("LFEM_LX", 50.0), genv_f("LFEM_LY", 20.0)
d        = genv_f("LFEM_D", 3.5)
Twave    = genv_f("LFEM_TWAVE", 2.0)
Awave    = genv_f("LFEM_AWAVE", 0.1)            # ★ big amplitude, LINEAR model
x_wm     = genv_f("LFEM_XWM", 12.0)
spL      = genv_f("LFEM_SPONGE_L", 6.0)
spR      = genv_f("LFEM_SPONGE_R", 10.0)
mumax    = genv_f("LFEM_MUMAX", 10.0)
dt       = genv_f("LFEM_DT", 0.02)
periods  = genv_f("LFEM_PERIODS", 12.0)
Tfinal   = haskey(ENV, "LFEM_TFINAL") ? genv_f("LFEM_TFINAL", 0.0) : periods * Twave
save_ev  = genv_i("LFEM_SAVE_EVERY", 20)
outdir   = genv("LFEM_OUTDIR", joinpath(ROOT, "output", "small_lin_bigamp_periodic_plane_M$(M)"))

banner("SMALL | LINEAR periodic plane wave at BIG amplitude (flat bed, A=$(Awave))",
       M, (px,py), (nx,ny), nx*ny, outdir)

diags, vert, prob = setup_and_run_distributed(
    cpu_grid=(px,py), M=M, c_bdy=cbdy_override(), p_horizontal=feord,
    domain=(0.0,Lx,0.0,Ly), partition=(nx,ny),
    h_val=d, T_wave=Twave, A_wave=Awave, x_wm=x_wm, y_wm=nothing,   # line source
    sponge_wL=spL, sponge_wR=spR, sponge_wB=0.0, sponge_wT=0.0, mu_max=mumax,
    T_final=Tfinal, dt=dt,
    regime=regime_sym(), nl_pressure=nl_pressure_sym(), flat_bed=flat_bed_flag(),  # ★ CASE 9 (defaults above)
    y_wall_bc=:periodic, x_wall_bc=false,
    output_dir=outdir, save_every=save_ev,
    write_w=write_w_flag(), write_pressure=write_p_flag(), rho=rho_val(),
    solver_type=solver_sym(), tableau=tableau_sym(),
    nl_iter=nl_iter_val(), nl_tol=nl_tol_val(),
    ls_rtol=ls_rtol_val(), ls_maxiter=ls_maxiter_val(),
    print_every=genv_i("LFEM_PRINT_EVERY", 10))

is_rank0() && @printf("small_lin_bigamp_periodic_plane done: %d steps, %d snapshots to %s\n",
                      length(diags), length(diags) ÷ max(save_ev,1), outdir)
