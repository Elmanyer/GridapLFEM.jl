# ==============================================================
#  run_periodic_plane_wave_dist.jl — DISTRIBUTED plane wave, PERIODIC y edges
#
#  Long-crested plane wave (Gaussian LINE source, y-invariant) in a flume whose
#  lateral (y) boundaries are PERIODIC (y_wall_bc=:periodic) — the numerical
#  infinite-width flume. The top/bottom edges are identified in the mesh, so no
#  lateral walls and no lateral sponges are used; the x-ends keep sponges.
#
#  LAUNCH (px·py MUST equal -n):
#    LFEM_M=2 LFEM_PX=8 LFEM_PY=1 \
#    ~/.julia/bin/mpiexecjl --project=. -n 8 julia --project=. \
#        GridapLFEM.jl/examples/distributed/run_periodic_plane_wave_dist.jl
#
#  Case-specific env vars (shared ones in _dist_common.jl):
#    LFEM_LX,LFEM_LY  domain size [m]         default 400 × 20
#    LFEM_D           still-water depth [m]   3.5
#    LFEM_TWAVE       wave period [s]         1.6   (→ kd≈5.5 at d=3.5)
#    LFEM_AWAVE       wave amplitude [m]      0.001
#    LFEM_XWM         wavemaker x-position    40
#    LFEM_SPONGE      x-end sponge width [m]  40
#    LFEM_MUMAX       sponge strength         5
#    LFEM_PERIODS     run length in periods   50
# ==============================================================

include(joinpath(@__DIR__, "_dist_common.jl"))

M        = genv_i("LFEM_M", 2)
px, py   = genv_i("LFEM_PX", 32), genv_i("LFEM_PY", 4)
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
save_ev  = genv_i("LFEM_SAVE_EVERY", 10)
outdir   = genv("LFEM_OUTDIR", joinpath(ROOT, "output", "periodic_plane_wave_dist_M$(M)"))

banner("PERIODIC PLANE WAVE (line source, periodic y) — distributed",
       M, (px,py), (nx,ny), nx*ny, outdir)

diags, vert, prob = setup_and_run_distributed(
    cpu_grid=(px,py), M=M, c_bdy=cbdy_override(), p_horizontal=feord,
    domain=(0.0,Lx,0.0,Ly), partition=(nx,ny),
    h_val=d, T_wave=Twave, A_wave=Awave, x_wm=x_wm, y_wm=nothing,   # line source
    sponge_wL=sponge, sponge_wR=sponge, sponge_wB=0.0, sponge_wT=0.0, mu_max=mumax,
    T_final=Tfinal, dt=dt,
    regime=regime_sym(), nl_pressure=nl_pressure_sym(), flat_bed=flat_bed_flag(),          # flat sea bed (∇h≡0); set LFEM_FLAT_BED=0 for variable bathymetry
    y_wall_bc=:periodic, x_wall_bc=false,   # ★ periodic lateral (y) boundaries
    output_dir=outdir, save_every=save_ev,
    write_w=write_w_flag(), write_pressure=write_p_flag(), rho=rho_val(),
    solver_type=solver_sym(), tableau=tableau_sym(),
    nl_iter=nl_iter_val(), nl_tol=nl_tol_val(),
    ls_rtol=ls_rtol_val(), ls_maxiter=ls_maxiter_val(),
    print_every=genv_i("LFEM_PRINT_EVERY", 10))

is_rank0() && @printf("periodic_plane_wave_dist done: %d steps, %d snapshots to %s\n",
                      length(diags), length(diags) ÷ max(save_ev,1), outdir)
