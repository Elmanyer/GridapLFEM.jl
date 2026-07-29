# ==============================================================
#  run_nl_periodic_plane_varbed_small.jl — SMALL-DOMAIN observation run
#
#  CASE 3/6:  nonlinear · FULL pressure · VARIABLE bathymetry · y-periodic · BIG amplitude
#  Same long-crested plane wave as case 2 but over a smooth submerged bar
#  (flat_bed=false ⇒ ∇h terms ON): the wave shoals onto the bar crest and
#  releases higher harmonics, then de-shoals — the bathymetric counterpart to
#  the flat-bed nonlinear case. y-invariant bar ⇒ compatible with periodic y.
#
#  Bar: d(x)=d0−½h_bar[tanh((x−(x_b−w))/s)−tanh((x−(x_b+w))/s)], analytic ∇h,∇²h,
#  x_b=26, w=6, h_bar=1.5, s=w/3 ⇒ d:3.5→2.0 on the crest (~20–32 m), inside the
#  working region (12→40). A/d≈0.05 on the crest — still safely below breaking.
#
#  LAUNCH (px·py MUST equal -n):
#    LFEM_PX=4 LFEM_PY=1 \
#    ~/.julia/bin/mpiexecjl --project=. -n 4 julia --project=. \
#        GridapLFEM.jl/examples/distributed_small/run_nl_periodic_plane_varbed_small.jl
#  (SLURM: run/dist_small/run_nl_periodic_plane_varbed_small.sh)
#
#  Case-specific env vars:
#    LFEM_LX,LFEM_LY 50×20 · LFEM_D0 3.5 · LFEM_HBAR 1.5 · LFEM_XBAR 26 · LFEM_WBAR 6
#    LFEM_TWAVE 2.0 · LFEM_AWAVE 0.1 · LFEM_XWM 12 · LFEM_SPONGE_L/R 6/10
#    LFEM_MUMAX 10 · LFEM_PERIODS 12
# ==============================================================

include(joinpath(@__DIR__, "..", "distributed", "_dist_common.jl"))

# ---- CASE 3 physics config (env still overrides; keeps banner ⇔ solver consistent) ----
get!(ENV, "LFEM_REGIME", "nonlinear"); get!(ENV, "LFEM_NL_PRESSURE", "full")
get!(ENV, "LFEM_FLAT_BED", "0")   # ★ variable bathymetry: ∇h terms ON

M        = genv_i("LFEM_M", 2)
px, py   = genv_i("LFEM_PX", 4), genv_i("LFEM_PY", 1)
nx, ny   = genv_i("LFEM_NX", 200), genv_i("LFEM_NY", 20)
feord    = genv_i("LFEM_FE_ORDER", 2)
Lx, Ly   = genv_f("LFEM_LX", 50.0), genv_f("LFEM_LY", 20.0)
d0       = genv_f("LFEM_D0", 3.5)
hbar     = genv_f("LFEM_HBAR", 1.5)             # bar height (d0 → d0−hbar on crest)
xbar     = genv_f("LFEM_XBAR", 26.0)
wbar     = genv_f("LFEM_WBAR", 6.0)
Twave    = genv_f("LFEM_TWAVE", 2.0)
Awave    = genv_f("LFEM_AWAVE", 0.1)            # ★ big amplitude (nonlinear)
x_wm     = genv_f("LFEM_XWM", 12.0)
spL      = genv_f("LFEM_SPONGE_L", 6.0)
spR      = genv_f("LFEM_SPONGE_R", 10.0)
mumax    = genv_f("LFEM_MUMAX", 10.0)
dt       = genv_f("LFEM_DT", 0.02)
periods  = genv_f("LFEM_PERIODS", 12.0)
Tfinal   = haskey(ENV, "LFEM_TFINAL") ? genv_f("LFEM_TFINAL", 0.0) : periods * Twave
save_ev  = genv_i("LFEM_SAVE_EVERY", 20)
outdir   = genv("LFEM_OUTDIR", joinpath(ROOT, "output", "small_nl_periodic_plane_varbed_M$(M)"))

# smooth y-invariant submerged bar (analytic ⇒ exact ∇h, ∇²h)
sramp   = wbar / 3.0
h_bathy(x) = d0 - 0.5*hbar*(tanh((x[1]-(xbar-wbar))/sramp) - tanh((x[1]-(xbar+wbar))/sramp))

banner("SMALL | NONLINEAR full-pressure periodic plane wave (VARIABLE bed, A=$(Awave))",
       M, (px,py), (nx,ny), nx*ny, outdir)

diags, vert, prob = setup_and_run_distributed(
    cpu_grid=(px,py), M=M, c_bdy=cbdy_override(), p_horizontal=feord,
    domain=(0.0,Lx,0.0,Ly), partition=(nx,ny),
    h_val=d0, T_wave=Twave, A_wave=Awave, x_wm=x_wm, y_wm=nothing,  # line source
    sponge_wL=spL, sponge_wR=spR, sponge_wB=0.0, sponge_wT=0.0, mu_max=mumax,
    T_final=Tfinal, dt=dt,
    h_bathy=h_bathy,                                               # ★ variable bathymetry (the bar)
    regime=regime_sym(), nl_pressure=nl_pressure_sym(), flat_bed=flat_bed_flag(0),  # ★ CASE 3 (∇h ON; defaults above)
    y_wall_bc=:periodic, x_wall_bc=false,                          # ★ periodic lateral edges
    output_dir=outdir, save_every=save_ev,
    write_w=write_w_flag(), write_pressure=write_p_flag(), rho=rho_val(),
    solver_type=solver_sym(), tableau=tableau_sym(),
    nl_iter=nl_iter_val(), nl_tol=nl_tol_val(),
    ls_rtol=ls_rtol_val(), ls_maxiter=ls_maxiter_val(),
    print_every=genv_i("LFEM_PRINT_EVERY", 10))

is_rank0() && @printf("small_nl_periodic_plane_varbed done: %d steps, %d snapshots to %s\n",
                      length(diags), length(diags) ÷ max(save_ev,1), outdir)
