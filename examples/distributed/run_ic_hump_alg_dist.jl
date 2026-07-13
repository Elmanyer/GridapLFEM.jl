# ==============================================================
#  run_ic_hump_alg_dist.jl — DISTRIBUTED Gaussian-hump release (algebraic)
#
#  A Gaussian free-surface hump released from rest in a CLOSED basin
#  (all four walls solid). No wavemaker, no sponge — energy stays inside;
#  the hump collapses into rings that reflect off the walls.
#
#  ⚠ IC problems REQUIRE x_wall_bc=true (closed basin): free x-walls plus the
#    dispersion term form a spurious-forcing instability that an IC
#    perturbation excites directly (root CLAUDE.md rule).
#
#  LAUNCH (px·py MUST equal -n):
#    LFEM_M=2 LFEM_PX=4 LFEM_PY=4 \
#    ~/.julia/bin/mpiexecjl --project=. -n 16 julia --project=. \
#        LFEM_2D/examples/distributed/run_ic_hump_alg_dist.jl
#
#  Case-specific env vars:
#    LFEM_L        basin side [m]           100
#    LFEM_D        depth [m]                3.5
#    LFEM_A0       hump amplitude [m]       0.01
#    LFEM_SIGMA0   hump half-width [m]      4.0
#    LFEM_TFINAL   final time [s]           40
# ==============================================================

include(joinpath(@__DIR__, "_dist_common_alg.jl"))

M        = genv_i("LFEM_M", 2)
px, py   = genv_i("LFEM_PX", 2), genv_i("LFEM_PY", 2)
L        = genv_f("LFEM_L", 100.0)
nx, ny   = genv_i("LFEM_NX", 100), genv_i("LFEM_NY", 100)
feord    = genv_i("LFEM_FE_ORDER", 2)
d        = genv_f("LFEM_D", 3.5)
A0       = genv_f("LFEM_A0", 0.01)
sig0     = genv_f("LFEM_SIGMA0", 4.0)
dt       = genv_f("LFEM_DT", 0.02)
Tfinal   = genv_f("LFEM_TFINAL", 40.0)
save_ev  = genv_i("LFEM_SAVE_EVERY", 25)
outdir   = genv("LFEM_OUTDIR", joinpath(ROOT, "output", "ic_hump_alg_dist_M$(M)"))

xc, yc = L/2, L/2
eta0(x) = A0 * exp(-((x[1]-xc)^2 + (x[2]-yc)^2) / sig0^2)

banner("IC HUMP RELEASE (closed basin) — distributed", M, (px,py), (nx,ny), nx*ny, outdir)

diags, vert, prob = setup_and_run_alg_distributed(
    cpu_grid=(px,py), M=M, c_bdy=cbdy_override(), fe_order=feord,
    domain=(0.0,L,0.0,L), partition=(nx,ny),
    d_val=d, T_wave=1.6, A_wave=0.0,                 # no wavemaker (A=0 → zero source)
    x_wm=-1e6, y_wm=nothing,
    sponge_wL=0.0, sponge_wR=0.0, sponge_wB=0.0, sponge_wT=0.0, mu_max=0.0,
    T_final=Tfinal, dt=dt,
    eta0_func=eta0,                                  # ★ initial condition release
    linearised=lin_flag(), advection=adv_flag(),
    P_full=pfull_flag(), nl_pressure68=nlp68_flag(),
    y_wall_bc=true, x_wall_bc=true,                  # ★ CLOSED basin (mandatory for IC)
    output_dir=outdir, save_every=save_ev,
    write_w=write_w_flag(), write_pressure=write_p_flag(), rho=rho_val(),
    print_dt=genv_f("LFEM_PRINT_DT", 1.0))

is_rank0() && @printf("ic_hump_alg_dist done: %d steps, %d snapshots to %s\n",
                      length(diags), length(diags) ÷ max(save_ev,1), outdir)
