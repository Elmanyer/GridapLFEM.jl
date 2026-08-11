# ==============================================================
#  run_periodic_plane_small.jl — PARAMETRIC small-domain periodic plane wave
#
#  ONE script for every interior-line-source, y-periodic plane-wave case on the
#  50x20 m small domain. The physics is selected entirely by environment
#  variables (set in the launcher); the geometry/mesh/partition are fixed here.
#  Covers the observation + comparison cases 1,2,3,7-12 (see run/dist_small/).
#
#  Config via env (launcher overrides ONLY what changes; base = nonlinear / full
#  pressure / flat bed / A=0.1 / T=2.0):
#    LFEM_REGIME       linear | nonlinear                    (default nonlinear)
#    LFEM_NL_PRESSURE  none | native | full                  (default full)
#    LFEM_FLAT_BED     1 flat | 0 variable (=> submerged bar built here)  (1)
#    LFEM_AWAVE        wave amplitude [m]                    (default 0.1)
#    LFEM_TWAVE        wave period [s]                       (default 2.0)
#    LFEM_HBAR/XBAR/WBAR   bar shape, used only when FLAT_BED=0   1.5 / 26 / 6
#
#  Fixed defaults baked in here (rarely overridden): domain 50x20, mesh 200x40,
#  partition 8x4 (32 ranks), fe_order 2, dt 0.02, d 3.5, x_wm 12, sponges 6/10,
#  mu_max 10, periods 12, save_every 10. All still env-overridable.
#
#  LAUNCH (px*py MUST equal -n): see run/dist_small/*.sh
# ==============================================================

include(joinpath(@__DIR__, "..", "distributed", "_dist_common.jl"))

# base-case physics (launcher overrides; get! => banner and solver stay consistent)
get!(ENV, "LFEM_REGIME", "nonlinear")
get!(ENV, "LFEM_NL_PRESSURE", "full")
get!(ENV, "LFEM_FLAT_BED", "1")

# ---- fixed geometry / numerics defaults (common to all periodic-plane cases) ----
M       = genv_i("LFEM_M", 2)
px, py  = genv_i("LFEM_PX", 8), genv_i("LFEM_PY", 4)      # 8*4 = 32 ranks
nx, ny  = genv_i("LFEM_NX", 200), genv_i("LFEM_NY", 40)   # dx=0.25, dy=0.5
feord   = genv_i("LFEM_FE_ORDER", 2)
Lx, Ly  = genv_f("LFEM_LX", 50.0), genv_f("LFEM_LY", 20.0)
d       = genv_f("LFEM_D", 3.5)
x_wm    = genv_f("LFEM_XWM", 12.0)
spL     = genv_f("LFEM_SPONGE_L", 10.0)
spR     = genv_f("LFEM_SPONGE_R", 10.0)
mumax   = genv_f("LFEM_MUMAX", 40.0)   # strong: kill the reflected/boundary mode fast
dt      = genv_f("LFEM_DT", 0.02)
save_ev = genv_i("LFEM_SAVE_EVERY", 10)
# ---- per-case knobs (defaults = base case) ----
Twave   = genv_f("LFEM_TWAVE", 2.0)
Awave   = genv_f("LFEM_AWAVE", 0.1)
periods = genv_f("LFEM_PERIODS", 12.0)
Tfinal  = haskey(ENV, "LFEM_TFINAL") ? genv_f("LFEM_TFINAL", 0.0) : periods * Twave

# bathymetry: flat_bed=false => variable bathymetry => build the y-invariant bar
usebar  = !flat_bed_flag(1)
hbar    = genv_f("LFEM_HBAR", 1.5); xbar = genv_f("LFEM_XBAR", 26.0); wbar = genv_f("LFEM_WBAR", 6.0)
sramp   = wbar / 3.0
h_bathy = usebar ?
    (x -> d - 0.5*hbar*(tanh((x[1]-(xbar-wbar))/sramp) - tanh((x[1]-(xbar+wbar))/sramp))) : nothing

bedtag  = usebar ? "bar" : "flat"
tag     = "$(regime_sym())_$(nl_pressure_sym())_$(bedtag)_A$(Awave)_T$(Twave)"
outdir  = genv("LFEM_OUTDIR", joinpath(ROOT, "output", "small_plane_$(tag)_M$(M)"))

banner("SMALL | periodic plane wave | $(regime_sym()) $(nl_pressure_sym()) $bedtag A=$Awave T=$Twave",
       M, (px,py), (nx,ny), nx*ny, outdir)

diags, vert, prob = setup_and_run_distributed(
    cpu_grid=(px,py), M=M, c_bdy=cbdy_override(), p_horizontal=feord,
    domain=(0.0,Lx,0.0,Ly), partition=(nx,ny),
    wave_gen=:inner_res,                                           # interior line source
    h_val=d, T_wave=Twave, A_wave=Awave, x_wm=x_wm, y_wm=nothing,   # line source
    sponge_wL=spL, sponge_wR=spR, sponge_wB=0.0, sponge_wT=0.0, mu_max=mumax,
    T_final=Tfinal, dt=dt, h_bathy=h_bathy,
    regime=regime_sym(), nl_pressure=nl_pressure_sym(), flat_bed=flat_bed_flag(1),
    y_wall_bc=:periodic, x_wall_bc=false,
    output_dir=outdir, save_every=save_ev,
    write_w=write_w_flag(), write_pressure=write_p_flag(), rho=rho_val(),
    solver_type=solver_sym(), tableau=tableau_sym(),
    nl_iter=nl_iter_val(), nl_tol=nl_tol_val(),
    ls_rtol=ls_rtol_val(), ls_maxiter=ls_maxiter_val(), krylov_m=krylov_m_val(), precond=precond_sym(),
    diag_every=diag_every_val(), diag_csv=diag_csv_flag(),
    div_factor=div_factor_val(), eta_ref=eta_ref_val(),
    print_every=genv_i("LFEM_PRINT_EVERY", 10))

is_rank0() && @printf("periodic_plane [%s] done: %d steps, %d snapshots to %s\n",
                      tag, length(diags), length(diags) ÷ max(save_ev,1), outdir)
