# ==============================================================
#  run_lin_periodic_plane_small.jl — SMALL-DOMAIN observation run
#
#  CASE 1/6:  linear · flat bed · y-periodic · small amplitude
#  Long-crested plane wave (Gaussian LINE source, y-invariant) in a small
#  50×20 m periodic-width flume. LINEAR dispersive model (regime=:linear,
#  nl_pressure=:none, flat_bed=true), tiny amplitude A=0.001 m — the clean,
#  crash-free reference for linear wave propagation / dispersion.
#
#  Small-domain sizing (see building_files/LFEM_runs.md): left sponge [0,6],
#  wavemaker at x=12, working region 12→40 m (~4.5 λ), right sponge [40,50];
#  lateral (y) edges PERIODIC ⇒ no top/bottom sponge. d=3.5, T=2.0 ⇒ kd≈3.5,
#  λ≈6.25 m, c≈3.13 m/s (fills the working region in ~5 periods).
#
#  LAUNCH (px·py MUST equal -n):
#    LFEM_PX=8 LFEM_PY=4 \
#    ~/.julia/bin/mpiexecjl --project=. -n 32 julia --project=. \
#        GridapLFEM.jl/examples/distributed_small/run_lin_periodic_plane_small.jl
#  (SLURM: run/dist_small/run_lin_periodic_plane_small.sh)
#
#  Case-specific env vars (shared ones in ../distributed/_dist_common.jl):
#    LFEM_LX,LFEM_LY  domain [m]        50 × 20
#    LFEM_D           depth [m]         3.5
#    LFEM_TWAVE       period [s]        2.0
#    LFEM_AWAVE       amplitude [m]     0.001
#    LFEM_XWM         wavemaker x [m]   12
#    LFEM_SPONGE_L/R  x-end sponges [m] 6 / 10
#    LFEM_MUMAX       sponge strength   10
#    LFEM_PERIODS     run length        12
# ==============================================================

include(joinpath(@__DIR__, "..", "distributed", "_dist_common.jl"))

# ---- CASE 1 physics config (env still overrides; keeps banner ⇔ solver consistent) ----
get!(ENV, "LFEM_REGIME", "linear"); get!(ENV, "LFEM_NL_PRESSURE", "none")
get!(ENV, "LFEM_FLAT_BED", "1")

M        = genv_i("LFEM_M", 2)
px, py   = genv_i("LFEM_PX", 8), genv_i("LFEM_PY", 4)
nx, ny   = genv_i("LFEM_NX", 200), genv_i("LFEM_NY", 40)
feord    = genv_i("LFEM_FE_ORDER", 2)
Lx, Ly   = genv_f("LFEM_LX", 50.0), genv_f("LFEM_LY", 20.0)
d        = genv_f("LFEM_D", 3.5)
Twave    = genv_f("LFEM_TWAVE", 2.0)
Awave    = genv_f("LFEM_AWAVE", 0.001)          # ★ small amplitude (linear)
x_wm     = genv_f("LFEM_XWM", 12.0)
spL      = genv_f("LFEM_SPONGE_L", 6.0)
spR      = genv_f("LFEM_SPONGE_R", 10.0)
mumax    = genv_f("LFEM_MUMAX", 10.0)
dt       = genv_f("LFEM_DT", 0.02)
periods  = genv_f("LFEM_PERIODS", 12.0)
Tfinal   = haskey(ENV, "LFEM_TFINAL") ? genv_f("LFEM_TFINAL", 0.0) : periods * Twave
save_ev  = genv_i("LFEM_SAVE_EVERY", 20)
outdir   = genv("LFEM_OUTDIR", joinpath(ROOT, "output", "small_lin_periodic_plane_M$(M)"))

banner("SMALL | LINEAR periodic plane wave (flat bed, A=$(Awave))",
       M, (px,py), (nx,ny), nx*ny, outdir)

diags, vert, prob = setup_and_run_distributed(
    cpu_grid=(px,py), M=M, c_bdy=cbdy_override(), p_horizontal=feord,
    domain=(0.0,Lx,0.0,Ly), partition=(nx,ny),
    h_val=d, T_wave=Twave, A_wave=Awave, x_wm=x_wm, y_wm=nothing,   # line source
    sponge_wL=spL, sponge_wR=spR, sponge_wB=0.0, sponge_wT=0.0, mu_max=mumax,
    T_final=Tfinal, dt=dt,
    regime=regime_sym(), nl_pressure=nl_pressure_sym(), flat_bed=flat_bed_flag(),  # ★ CASE 1 (defaults above)
    y_wall_bc=:periodic, x_wall_bc=false,                          # ★ periodic lateral edges
    output_dir=outdir, save_every=save_ev,
    write_w=write_w_flag(), write_pressure=write_p_flag(), rho=rho_val(),
    solver_type=solver_sym(), tableau=tableau_sym(),
    nl_iter=nl_iter_val(), nl_tol=nl_tol_val(),
    ls_rtol=ls_rtol_val(), ls_maxiter=ls_maxiter_val(),
    print_every=genv_i("LFEM_PRINT_EVERY", 10))

is_rank0() && @printf("small_lin_periodic_plane done: %d steps, %d snapshots to %s\n",
                      length(diags), length(diags) ÷ max(save_ev,1), outdir)
