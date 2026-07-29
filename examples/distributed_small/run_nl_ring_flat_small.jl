# ==============================================================
#  run_nl_ring_flat_small.jl — SMALL-DOMAIN observation run
#
#  CASE 4/6:  nonlinear · FULL pressure · flat bed · ring wave · BIG amplitude
#  Gaussian POINT source at the domain centre (25,10) → circular wavefronts,
#  4-side sponge absorption. Fully nonlinear, all pressure tiers, flat bed,
#  A=0.1 m — observe radial spreading and nonlinear ring dynamics in a compact
#  50×20 box. NB: the 20 m width clips the ring at r≈6 m in y (the top/bottom
#  sponges); this is a small observation box, not a spreading-law benchmark.
#
#  4-side sponges: L/R 6 m, B/T 4 m (μ_max=12 to compensate the thinner lateral
#  layers). y_wall_bc=:wall behind the sponges (the sponge shrinks the edge
#  field so wall-vs-open is immaterial there).
#
#  LAUNCH (px·py MUST equal -n):
#    LFEM_PX=4 LFEM_PY=2 \
#    ~/.julia/bin/mpiexecjl --project=. -n 8 julia --project=. \
#        GridapLFEM.jl/examples/distributed_small/run_nl_ring_flat_small.jl
#  (SLURM: run/dist_small/run_nl_ring_flat_small.sh)
#
#  Case-specific env vars:
#    LFEM_LX,LFEM_LY 50×20 · LFEM_D 3.5 · LFEM_TWAVE 2.0 · LFEM_AWAVE 0.1
#    LFEM_SPONGE_X 6 · LFEM_SPONGE_Y 4 · LFEM_MUMAX 12 · LFEM_PERIODS 12
# ==============================================================

include(joinpath(@__DIR__, "..", "distributed", "_dist_common.jl"))

# ---- CASE 4 physics config (env still overrides; keeps banner ⇔ solver consistent) ----
get!(ENV, "LFEM_REGIME", "nonlinear"); get!(ENV, "LFEM_NL_PRESSURE", "full")
get!(ENV, "LFEM_FLAT_BED", "1")

M        = genv_i("LFEM_M", 2)
px, py   = genv_i("LFEM_PX", 4), genv_i("LFEM_PY", 2)
nx, ny   = genv_i("LFEM_NX", 200), genv_i("LFEM_NY", 80)
feord    = genv_i("LFEM_FE_ORDER", 2)
Lx, Ly   = genv_f("LFEM_LX", 50.0), genv_f("LFEM_LY", 20.0)
d        = genv_f("LFEM_D", 3.5)
Twave    = genv_f("LFEM_TWAVE", 2.0)
Awave    = genv_f("LFEM_AWAVE", 0.1)            # ★ big amplitude (nonlinear)
spX      = genv_f("LFEM_SPONGE_X", 6.0)
spY      = genv_f("LFEM_SPONGE_Y", 4.0)
mumax    = genv_f("LFEM_MUMAX", 12.0)
dt       = genv_f("LFEM_DT", 0.02)
periods  = genv_f("LFEM_PERIODS", 12.0)
Tfinal   = haskey(ENV, "LFEM_TFINAL") ? genv_f("LFEM_TFINAL", 0.0) : periods * Twave
save_ev  = genv_i("LFEM_SAVE_EVERY", 20)
outdir   = genv("LFEM_OUTDIR", joinpath(ROOT, "output", "small_nl_ring_flat_M$(M)"))

banner("SMALL | NONLINEAR full-pressure RING wave (point source, flat bed, A=$(Awave))",
       M, (px,py), (nx,ny), nx*ny, outdir)

diags, vert, prob = setup_and_run_distributed(
    cpu_grid=(px,py), M=M, c_bdy=cbdy_override(), p_horizontal=feord,
    domain=(0.0,Lx,0.0,Ly), partition=(nx,ny),
    h_val=d, T_wave=Twave, A_wave=Awave,
    x_wm=Lx/2, y_wm=Ly/2,                                          # ★ point source → ring waves
    sponge_wL=spX, sponge_wR=spX, sponge_wB=spY, sponge_wT=spY, mu_max=mumax,
    T_final=Tfinal, dt=dt,
    regime=regime_sym(), nl_pressure=nl_pressure_sym(), flat_bed=flat_bed_flag(),  # ★ CASE 4 (defaults above)
    y_wall_bc=:wall, x_wall_bc=false,
    output_dir=outdir, save_every=save_ev,
    write_w=write_w_flag(), write_pressure=write_p_flag(), rho=rho_val(),
    solver_type=solver_sym(), tableau=tableau_sym(),
    nl_iter=nl_iter_val(), nl_tol=nl_tol_val(),
    ls_rtol=ls_rtol_val(), ls_maxiter=ls_maxiter_val(),
    print_every=genv_i("LFEM_PRINT_EVERY", 10))

is_rank0() && @printf("small_nl_ring_flat done: %d steps, %d snapshots to %s\n",
                      length(diags), length(diags) ÷ max(save_ev,1), outdir)
