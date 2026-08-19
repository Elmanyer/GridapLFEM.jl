# ==============================================================
#  run_ic_hump_dist.jl — DISTRIBUTED Gaussian-hump release (algebraic)
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
#    BALFEM_M=2 BALFEM_PX=4 BALFEM_PY=4 \
#    ~/.julia/bin/mpiexecjl --project=. -n 16 julia --project=. \
#        GridapBALFEM.jl/examples/distributed/run_ic_hump_dist.jl
#
#  Case-specific env vars:
#    BALFEM_L        basin side [m]           100
#    BALFEM_D        depth [m]                3.5
#    BALFEM_A0       hump amplitude [m]       0.01
#    BALFEM_SIGMA0   hump half-width [m]      4.0
#    BALFEM_TFINAL   final time [s]           40
# ==============================================================

include(joinpath(@__DIR__, "_dist_common.jl"))

M        = genv_i("BALFEM_M", 2)

#  Vertical BASIS ORDER. The model is named P{p_vert}LFE-{M}: `Pp` is the

#  vertical Lagrange order and `M` the number of vertical elements, so the run

#  says which member of the BALFE-M family it actually exercises. Default p=1

#  reproduces the piecewise-linear models of Yang & Liu.

p_vert  = genv_i("BALFEM_P_VERT", 1)

model_name = "P$(p_vert)LFE-$(M)"
px, py   = genv_i("BALFEM_PX", 2), genv_i("BALFEM_PY", 2)
L        = genv_f("BALFEM_L", 100.0)
nx, ny   = genv_i("BALFEM_NX", 100), genv_i("BALFEM_NY", 100)
feord    = genv_i("BALFEM_FE_ORDER", 2)
#  p_eta = 0 keeps the historical EQUAL-ORDER spaces (unchanged default).
#  Set BALFEM_P_ETA = BALFEM_FE_ORDER-1 for the Taylor-Hood-like pairing, which is
#  the only one measured to reach the theoretical order in BOTH fields. It is
#  NOT automatically the better production choice: at a GIVEN mesh the
#  equal-order spaces were 40x more accurate, because eta sits in a richer
#  space. Compare error-vs-DOF before switching.
p_eta    = genv_i("BALFEM_P_ETA", 0)
d        = genv_f("BALFEM_D", 3.5)
A0       = genv_f("BALFEM_A0", 0.01)
sig0     = genv_f("BALFEM_SIGMA0", 4.0)
dt       = genv_f("BALFEM_DT", 0.02)
Tfinal   = genv_f("BALFEM_TFINAL", 40.0)
save_ev  = genv_i("BALFEM_SAVE_EVERY", 25)
outdir   = genv("BALFEM_OUTDIR", joinpath(ROOT, "output", "ic_hump_dist_$(model_name)"))

xc, yc = L/2, L/2
eta0(x) = A0 * exp(-((x[1]-xc)^2 + (x[2]-yc)^2) / sig0^2)

banner("IC HUMP RELEASE (closed basin) — distributed", M, (px,py), (nx,ny), nx*ny, outdir)

diags, vert, prob = setup_and_run_distributed(
    cpu_grid=(px,py), M=M, p_vertical=p_vert, c_bdy=cbdy_override(), p_horizontal=feord, p_eta=p_eta,
    domain=(0.0,L,0.0,L), partition=(nx,ny),
    h_val=d, T_wave=1.6, A_wave=0.0,                 # no wavemaker (A=0 → zero source)
    x_wm=-1e6, y_wm=nothing,
    sponge_wL=0.0, sponge_wR=0.0, sponge_wB=0.0, sponge_wT=0.0, mu_max=0.0,
    T_final=Tfinal, dt=dt,
    eta0_func=eta0,                                  # ★ initial condition release
    regime=regime_sym(), nl_pressure=nl_pressure_sym(), flat_bed=flat_bed_flag(),          # flat sea bed (∇h≡0); set BALFEM_FLAT_BED=0 for variable bathymetry
    y_wall_bc=:wall, x_wall_bc=true,                  # ★ CLOSED basin (mandatory for IC)
    output_dir=outdir, save_every=save_ev,
    write_w=write_w_flag(), write_pressure=write_p_flag(), rho=rho_val(),
    solver_type=solver_sym(), tableau=tableau_sym(),
    nl_iter=nl_iter_val(), nl_tol=nl_tol_val(),
    ls_rtol=ls_rtol_val(), ls_maxiter=ls_maxiter_val(), krylov_m=krylov_m_val(), precond=precond_sym(),
    diag_every=diag_every_val(), diag_csv=diag_csv_flag(),
    div_factor=div_factor_val(), eta_ref=eta_ref_val(),
    print_every=genv_i("BALFEM_PRINT_EVERY", 10))

is_rank0() && @printf("ic_hump_dist done: %d steps, %d snapshots to %s\n",
                      length(diags), length(diags) ÷ max(save_ev,1), outdir)
