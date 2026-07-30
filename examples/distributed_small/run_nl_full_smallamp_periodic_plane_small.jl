# ==============================================================
#  run_nl_full_smallamp_periodic_plane_small.jl — SMALL-DOMAIN comparison run
#
#  CASE 10/12:  nonlinear · FULL pressure · flat bed · y-periodic · SMALL amplitude A=0.001
#  The FULL nonlinear model driven at the SAME small amplitude as the linear
#  reference (case 1). Key correctness check: as A→0 the full nonlinear model
#  must reproduce the linear solution (case 1) to O(A²) — the nonlinear→linear
#  consistency limit. Compared with case 2 (full, A=0.1) it gives the amplitude
#  scaling of the bound second harmonic (∝A²).
#
#  Geometry identical to cases 1 & 2. d=3.5, T=2.0 ⇒ kd≈3.5, λ≈6.25 m.
#
#  LAUNCH:  LFEM_PX=8 LFEM_PY=4 mpiexecjl --project=. -n 32 julia --project=. \
#             GridapLFEM.jl/examples/distributed_small/run_nl_full_smallamp_periodic_plane_small.jl
#  (SLURM: run/dist_small/run_nl_full_smallamp_periodic_plane_small.sh)
# ==============================================================

include(joinpath(@__DIR__, "..", "distributed", "_dist_common.jl"))

# ---- CASE 10 physics config (env still overrides; keeps banner ⇔ solver consistent) ----
get!(ENV, "LFEM_REGIME", "nonlinear"); get!(ENV, "LFEM_NL_PRESSURE", "full")
get!(ENV, "LFEM_FLAT_BED", "1")

M        = genv_i("LFEM_M", 2)
px, py   = genv_i("LFEM_PX", 8), genv_i("LFEM_PY", 4)
nx, ny   = genv_i("LFEM_NX", 200), genv_i("LFEM_NY", 40)
feord    = genv_i("LFEM_FE_ORDER", 2)
Lx, Ly   = genv_f("LFEM_LX", 50.0), genv_f("LFEM_LY", 20.0)
d        = genv_f("LFEM_D", 3.5)
Twave    = genv_f("LFEM_TWAVE", 2.0)
Awave    = genv_f("LFEM_AWAVE", 0.001)          # ★ small amplitude, FULL nonlinear model
x_wm     = genv_f("LFEM_XWM", 12.0)
spL      = genv_f("LFEM_SPONGE_L", 6.0)
spR      = genv_f("LFEM_SPONGE_R", 10.0)
mumax    = genv_f("LFEM_MUMAX", 10.0)
dt       = genv_f("LFEM_DT", 0.02)
periods  = genv_f("LFEM_PERIODS", 12.0)
Tfinal   = haskey(ENV, "LFEM_TFINAL") ? genv_f("LFEM_TFINAL", 0.0) : periods * Twave
save_ev  = genv_i("LFEM_SAVE_EVERY", 20)
outdir   = genv("LFEM_OUTDIR", joinpath(ROOT, "output", "small_nl_full_smallamp_periodic_plane_M$(M)"))

banner("SMALL | NONLINEAR full-pressure periodic plane wave at SMALL amplitude (flat bed, A=$(Awave))",
       M, (px,py), (nx,ny), nx*ny, outdir)

diags, vert, prob = setup_and_run_distributed(
    cpu_grid=(px,py), M=M, c_bdy=cbdy_override(), p_horizontal=feord,
    domain=(0.0,Lx,0.0,Ly), partition=(nx,ny),
    h_val=d, T_wave=Twave, A_wave=Awave, x_wm=x_wm, y_wm=nothing,   # line source
    sponge_wL=spL, sponge_wR=spR, sponge_wB=0.0, sponge_wT=0.0, mu_max=mumax,
    T_final=Tfinal, dt=dt,
    regime=regime_sym(), nl_pressure=nl_pressure_sym(), flat_bed=flat_bed_flag(),  # ★ CASE 10 (defaults above)
    y_wall_bc=:periodic, x_wall_bc=false,
    output_dir=outdir, save_every=save_ev,
    write_w=write_w_flag(), write_pressure=write_p_flag(), rho=rho_val(),
    solver_type=solver_sym(), tableau=tableau_sym(),
    nl_iter=nl_iter_val(), nl_tol=nl_tol_val(),
    ls_rtol=ls_rtol_val(), ls_maxiter=ls_maxiter_val(),
    print_every=genv_i("LFEM_PRINT_EVERY", 10))

is_rank0() && @printf("small_nl_full_smallamp_periodic_plane done: %d steps, %d snapshots to %s\n",
                      length(diags), length(diags) ÷ max(save_ev,1), outdir)
