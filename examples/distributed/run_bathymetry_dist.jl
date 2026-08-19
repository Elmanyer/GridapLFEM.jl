# ==============================================================
#  run_bathymetry_dist.jl — DISTRIBUTED shoaling over a submerged bar
#                               (variable bathymetry, algebraic solver)
#
#  Plane wave generated over depth d0 shoals onto a smooth (tanh-profile)
#  submerged bar of height h_bar and returns to deep water; sponge layers at
#  both x-ends. The variable bed activates the slope-pressure physics:
#  BALFEM_LINP=1 (default here) turns on the A/K linear slope-pressure package,
#  and the bed slope also feeds the ∇h parts of the leading pressure when
#  BALFEM_PFULL=1 and the nonlinear comps 6–8 when BALFEM_NLP68=1.
#
#  LAUNCH (px·py MUST equal -n):
#    BALFEM_M=2 BALFEM_PX=8 BALFEM_PY=1 \
#    ~/.julia/bin/mpiexecjl --project=. -n 8 julia --project=. \
#        GridapBALFEM.jl/examples/distributed/run_bathymetry_dist.jl
#
#  Case-specific env vars:
#    BALFEM_LX,BALFEM_LY  domain size [m]        300 × 20
#    BALFEM_D0          offshore depth [m]     3.5
#    BALFEM_HBAR        bar height [m]         2.0
#    BALFEM_XBAR        bar centre x [m]       150
#    BALFEM_WBAR        bar half-width [m]     25
#    BALFEM_TWAVE       wave period [s]        2.5
#    BALFEM_AWAVE       amplitude [m]          0.001
#    BALFEM_XWM         wavemaker x [m]        40
#    BALFEM_SPONGE      sponge width [m]       35
#    BALFEM_PERIODS     run length in periods  40
# ==============================================================

include(joinpath(@__DIR__, "_dist_common.jl"))

M        = genv_i("BALFEM_M", 2)

#  Vertical BASIS ORDER. The model is named P{p_vert}LFE-{M}: `Pp` is the

#  vertical Lagrange order and `M` the number of vertical elements, so the run

#  says which member of the BALFE-M family it actually exercises. Default p=1

#  reproduces the piecewise-linear models of Yang & Liu.

p_vert  = genv_i("BALFEM_P_VERT", 1)

model_name = "P$(p_vert)LFE-$(M)"
px, py   = genv_i("BALFEM_PX", 2), genv_i("BALFEM_PY", 1)
nx, ny   = genv_i("BALFEM_NX", 1500), genv_i("BALFEM_NY", 100)
feord    = genv_i("BALFEM_FE_ORDER", 2)
#  p_eta = 0 keeps the historical EQUAL-ORDER spaces (unchanged default).
#  Set BALFEM_P_ETA = BALFEM_FE_ORDER-1 for the Taylor-Hood-like pairing, which is
#  the only one measured to reach the theoretical order in BOTH fields. It is
#  NOT automatically the better production choice: at a GIVEN mesh the
#  equal-order spaces were 40x more accurate, because eta sits in a richer
#  space. Compare error-vs-DOF before switching.
p_eta    = genv_i("BALFEM_P_ETA", 0)
Lx, Ly   = genv_f("BALFEM_LX", 300.0), genv_f("BALFEM_LY", 20.0)
d0       = genv_f("BALFEM_D0", 3.5)
hbar     = genv_f("BALFEM_HBAR", 2.0)
xbar     = genv_f("BALFEM_XBAR", 150.0)
wbar     = genv_f("BALFEM_WBAR", 25.0)
Twave    = genv_f("BALFEM_TWAVE", 2.5)
Awave    = genv_f("BALFEM_AWAVE", 0.001)
x_wm     = genv_f("BALFEM_XWM", 40.0)
sponge   = genv_f("BALFEM_SPONGE", 35.0)
mumax    = genv_f("BALFEM_MUMAX", 5.0)
dt       = genv_f("BALFEM_DT", 0.02)
periods  = genv_f("BALFEM_PERIODS", 40.0)
Tfinal   = haskey(ENV, "BALFEM_TFINAL") ? genv_f("BALFEM_TFINAL", 0.0) : periods * Twave
save_ev  = genv_i("BALFEM_SAVE_EVERY", 25)
outdir   = genv("BALFEM_OUTDIR", joinpath(ROOT, "output", "bathymetry_dist_$(model_name)"))

# smooth submerged bar: d(x) = d0 − (h_bar/2)·[tanh((x−x_L)/s) − tanh((x−x_R)/s)]
# (difference → 2 on the bar top ⇒ d = d0 − h_bar there; analytic ⇒ exact ∇h, ∇²h)
sramp   = wbar / 3.0
h_bathy(x) = d0 - 0.5*hbar*(tanh((x[1]-(xbar-wbar))/sramp) - tanh((x[1]-(xbar+wbar))/sramp))

banner("SHOALING OVER SUBMERGED BAR — distributed", M, (px,py), (nx,ny), nx*ny, outdir)

diags, vert, prob = setup_and_run_distributed(
    cpu_grid=(px,py), M=M, p_vertical=p_vert, c_bdy=cbdy_override(), p_horizontal=feord, p_eta=p_eta,
    domain=(0.0,Lx,0.0,Ly), partition=(nx,ny),
    h_val=d0, T_wave=Twave, A_wave=Awave, x_wm=x_wm, y_wm=nothing,
    sponge_wL=sponge, sponge_wR=sponge, sponge_wB=0.0, sponge_wT=0.0, mu_max=mumax,
    T_final=Tfinal, dt=dt,
    h_bathy=h_bathy,                                   # ★ variable bathymetry
    regime=regime_sym(), nl_pressure=nl_pressure_sym(),
    flat_bed=flat_bed_flag(0),                         # ★ variable bathymetry on the bar: ∇h terms ON (default)
    y_wall_bc=:wall, x_wall_bc=false,
    output_dir=outdir, save_every=save_ev,
    write_w=write_w_flag(), write_pressure=write_p_flag(), rho=rho_val(),
    solver_type=solver_sym(), tableau=tableau_sym(),
    nl_iter=nl_iter_val(), nl_tol=nl_tol_val(),
    ls_rtol=ls_rtol_val(), ls_maxiter=ls_maxiter_val(), krylov_m=krylov_m_val(), precond=precond_sym(),
    diag_every=diag_every_val(), diag_csv=diag_csv_flag(),
    div_factor=div_factor_val(), eta_ref=eta_ref_val(),
    print_every=genv_i("BALFEM_PRINT_EVERY", 10))

is_rank0() && @printf("bathymetry_dist done: %d steps, %d snapshots to %s\n",
                      length(diags), length(diags) ÷ max(save_ev,1), outdir)
