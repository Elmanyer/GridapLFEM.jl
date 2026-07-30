# ==============================================================
#  run_lin_varbed_periodic_plane_small.jl — SMALL-DOMAIN comparison run
#
#  CASE 12/12:  LINEAR · VARIABLE bathymetry · y-periodic · SMALL amplitude A=0.001
#  The LINEAR model shoaling over the SAME submerged bar as the nonlinear case 3
#  (flat_bed=false). Compared with case 3 (nonlinear, A=0.1) it isolates the
#  nonlinear shoaling physics: the linear case shows pure depth-induced shoaling
#  (amplitude/wavelength change over the bar) with NO harmonic release, whereas
#  case 3 adds the bound/free super-harmonics generated on the crest. Also
#  exercises the flat_bed=false path in the LINEAR regime, which activates the
#  A/K linear slope-pressure package (the bed-slope pressure correction).
#
#  Bar identical to case 3: x_b=26, w=6, h_bar=1.5 ⇒ d:3.5→2.0 on the crest.
#  d0=3.5, T=2.0 ⇒ kd≈3.5 offshore. y-invariant bar ⇒ periodic-compatible.
#
#  LAUNCH:  LFEM_PX=8 LFEM_PY=4 mpiexecjl --project=. -n 32 julia --project=. \
#             GridapLFEM.jl/examples/distributed_small/run_lin_varbed_periodic_plane_small.jl
#  (SLURM: run/dist_small/run_lin_varbed_periodic_plane_small.sh)
# ==============================================================

include(joinpath(@__DIR__, "..", "distributed", "_dist_common.jl"))

# ---- CASE 12 physics config (env still overrides; keeps banner ⇔ solver consistent) ----
get!(ENV, "LFEM_REGIME", "linear"); get!(ENV, "LFEM_NL_PRESSURE", "none")
get!(ENV, "LFEM_FLAT_BED", "0")   # ★ variable bathymetry: ∇h terms ON (linear shoaling)

M        = genv_i("LFEM_M", 2)
px, py   = genv_i("LFEM_PX", 8), genv_i("LFEM_PY", 4)
nx, ny   = genv_i("LFEM_NX", 200), genv_i("LFEM_NY", 40)
feord    = genv_i("LFEM_FE_ORDER", 2)
Lx, Ly   = genv_f("LFEM_LX", 50.0), genv_f("LFEM_LY", 20.0)
d0       = genv_f("LFEM_D0", 3.5)
hbar     = genv_f("LFEM_HBAR", 1.5)             # bar height (d0 → d0−hbar on crest)
xbar     = genv_f("LFEM_XBAR", 26.0)
wbar     = genv_f("LFEM_WBAR", 6.0)
Twave    = genv_f("LFEM_TWAVE", 2.0)
Awave    = genv_f("LFEM_AWAVE", 0.001)          # ★ small amplitude, LINEAR shoaling
x_wm     = genv_f("LFEM_XWM", 12.0)
spL      = genv_f("LFEM_SPONGE_L", 6.0)
spR      = genv_f("LFEM_SPONGE_R", 10.0)
mumax    = genv_f("LFEM_MUMAX", 10.0)
dt       = genv_f("LFEM_DT", 0.02)
periods  = genv_f("LFEM_PERIODS", 12.0)
Tfinal   = haskey(ENV, "LFEM_TFINAL") ? genv_f("LFEM_TFINAL", 0.0) : periods * Twave
save_ev  = genv_i("LFEM_SAVE_EVERY", 20)
outdir   = genv("LFEM_OUTDIR", joinpath(ROOT, "output", "small_lin_varbed_periodic_plane_M$(M)"))

# smooth y-invariant submerged bar (analytic ⇒ exact ∇h, ∇²h) — same as case 3
sramp   = wbar / 3.0
h_bathy(x) = d0 - 0.5*hbar*(tanh((x[1]-(xbar-wbar))/sramp) - tanh((x[1]-(xbar+wbar))/sramp))

banner("SMALL | LINEAR shoaling periodic plane wave (VARIABLE bed, A=$(Awave))",
       M, (px,py), (nx,ny), nx*ny, outdir)

diags, vert, prob = setup_and_run_distributed(
    cpu_grid=(px,py), M=M, c_bdy=cbdy_override(), p_horizontal=feord,
    domain=(0.0,Lx,0.0,Ly), partition=(nx,ny),
    h_val=d0, T_wave=Twave, A_wave=Awave, x_wm=x_wm, y_wm=nothing,  # line source
    sponge_wL=spL, sponge_wR=spR, sponge_wB=0.0, sponge_wT=0.0, mu_max=mumax,
    T_final=Tfinal, dt=dt,
    h_bathy=h_bathy,                                               # ★ variable bathymetry (the bar)
    regime=regime_sym(), nl_pressure=nl_pressure_sym(), flat_bed=flat_bed_flag(0),  # ★ CASE 12 (∇h ON; defaults above)
    y_wall_bc=:periodic, x_wall_bc=false,
    output_dir=outdir, save_every=save_ev,
    write_w=write_w_flag(), write_pressure=write_p_flag(), rho=rho_val(),
    solver_type=solver_sym(), tableau=tableau_sym(),
    nl_iter=nl_iter_val(), nl_tol=nl_tol_val(),
    ls_rtol=ls_rtol_val(), ls_maxiter=ls_maxiter_val(),
    print_every=genv_i("LFEM_PRINT_EVERY", 10))

is_rank0() && @printf("small_lin_varbed_periodic_plane done: %d steps, %d snapshots to %s\n",
                      length(diags), length(diags) ÷ max(save_ev,1), outdir)
