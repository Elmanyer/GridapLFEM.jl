# ==============================================================
#  run_plane_wave_alg_dist.jl — DISTRIBUTED long-crested plane wave (algebraic)
#
#  Gaussian LINE-source wavemaker → plane waves down a long flume, absorbed by
#  sponge layers at both x-ends. Writes η, per-node u/v components AND the
#  reconstructed vertical velocity (w_s*) and total pressure (p_s*) fields.
#  Runs the FULL nonlinear physics distributed (one Gridap path).
#
#  LAUNCH (px·py MUST equal -n):
#    LFEM_M=2 LFEM_PX=8 LFEM_PY=1 \
#    ~/.julia/bin/mpiexecjl --project=. -n 8 julia --project=. \
#        LFEM_2D/examples/distributed/run_plane_wave_alg_dist.jl
#
#  Case-specific env vars (shared ones in _dist_common_alg.jl):
#    LFEM_LX,LFEM_LY  domain size [m]         default 400 × 20
#    LFEM_D           still-water depth [m]   3.5
#    LFEM_TWAVE       wave period [s]         1.6   (→ kd≈5.5 at d=3.5)
#    LFEM_AWAVE       wave amplitude [m]      0.001
#    LFEM_XWM         wavemaker x-position    40
#    LFEM_SPONGE      sponge width [m]        40
#    LFEM_MUMAX       sponge strength         5
#    LFEM_PERIODS     run length in wave periods (if LFEM_TFINAL unset)  50
# ==============================================================

include(joinpath(@__DIR__, "_dist_common_alg.jl"))

M        = genv_i("LFEM_M", 2)
px, py   = genv_i("LFEM_PX", 2), genv_i("LFEM_PY", 1)
nx, ny   = genv_i("LFEM_NX", 2000), genv_i("LFEM_NY", 100)
feord    = genv_i("LFEM_FE_ORDER", 2)
Lx, Ly   = genv_f("LFEM_LX", 400.0), genv_f("LFEM_LY", 20.0)
d        = genv_f("LFEM_D", 3.5)
Twave    = genv_f("LFEM_TWAVE", 1.6)
Awave    = genv_f("LFEM_AWAVE", 0.001)
x_wm     = genv_f("LFEM_XWM", 40.0)
sponge   = genv_f("LFEM_SPONGE", 40.0)
mumax    = genv_f("LFEM_MUMAX", 5.0)
dt       = genv_f("LFEM_DT", 0.02)
periods  = genv_f("LFEM_PERIODS", 50.0)
Tfinal   = haskey(ENV, "LFEM_TFINAL") ? genv_f("LFEM_TFINAL", 0.0) : periods * Twave
save_ev  = genv_i("LFEM_SAVE_EVERY", 20)
outdir   = genv("LFEM_OUTDIR", joinpath(ROOT, "output", "plane_wave_alg_dist_M$(M)"))

banner("PLANE WAVE (line source) — distributed", M, (px,py), (nx,ny), nx*ny, outdir)

diags, vert, prob = setup_and_run_alg_distributed(
    cpu_grid=(px,py), M=M, c_bdy=cbdy_override(), fe_order=feord,
    domain=(0.0,Lx,0.0,Ly), partition=(nx,ny),
    d_val=d, T_wave=Twave, A_wave=Awave, x_wm=x_wm, y_wm=nothing,   # line source
    sponge_wL=sponge, sponge_wR=sponge, sponge_wB=0.0, sponge_wT=0.0, mu_max=mumax,
    T_final=Tfinal, dt=dt,
    linearised=lin_flag(), advection=adv_flag(),
    P_full=pfull_flag(), nl_pressure68=nlp68_flag(),
    y_wall_bc=true, x_wall_bc=false,
    output_dir=outdir, save_every=save_ev,
    write_w=write_w_flag(), write_pressure=write_p_flag(), rho=rho_val(),
    print_dt=genv_f("LFEM_PRINT_DT", Twave))

is_rank0() && @printf("plane_wave_alg_dist done: %d steps, %d snapshots to %s\n",
                      length(diags), length(diags) ÷ max(save_ev,1), outdir)
