# ==============================================================
#  run_bc_plane_small.jl — PARAMETRIC small-domain BC periodic plane wave
#
#  Periodic plane wave GENERATED FROM THE LEFT BOUNDARY (Dirichlet data, no
#  interior source): η/𝖴x prescribed from a parametrised regular Airy component
#  (wave_gen=:bc_gen), a generation+absorption relaxation zone at the
#  inflow, and a strong sponge at the far (right) end. y-periodic ⇒ a clean plane
#  wave with NO two-open-ends resonance — this is the stable replacement for the
#  interior-line-source plane wave that seeded the open-boundary instability.
#
#  Config via env (base = nonlinear / full pressure / flat bed / A=0.1 / T=2.0):
#    BALFEM_REGIME       linear | nonlinear                    (default nonlinear)
#    BALFEM_NL_PRESSURE  none | native | full                  (default full)
#    BALFEM_FLAT_BED     1 flat | 0 variable (=> submerged bar built here)   (1)
#    BALFEM_AWAVE        wave amplitude [m]                    (default 0.1)
#    BALFEM_TWAVE        wave period [s]                       (default 2.0)
#    BALFEM_WAVE_DIR     propagation angle vs +x [rad]         (default 0.0)
#    BALFEM_BC_PROFILE   model | airy  (vertical polarization) (default model)
#    BALFEM_HBAR/XBAR/WBAR   bar shape, used only when FLAT_BED=0   1.5 / 26 / 6
#
#  Fixed defaults baked in here: domain 50x20, mesh 200x40, partition 8x4 (32
#  ranks), p_horizontal 2, dt 0.02, d 3.5, LEFT relaxation zone (width ~1λ) + right
#  sponge 12, mu_max 40, periods 12, save_every 10. All still env-overridable.
#
#  LAUNCH (px*py MUST equal -n): see run/dist_small/*bc_plane*.sh
# ==============================================================

include(joinpath(@__DIR__, "..", "distributed", "_dist_common.jl"))

# base-case physics (launcher overrides; get! => banner and solver stay consistent)
get!(ENV, "BALFEM_REGIME", "nonlinear")
get!(ENV, "BALFEM_NL_PRESSURE", "full")
get!(ENV, "BALFEM_FLAT_BED", "1")
get!(ENV, "BALFEM_RELAX", "1"); get!(ENV, "BALFEM_RELAX_W", "6")

# ---- fixed geometry / numerics defaults (common to all bc-plane cases) ----
M       = genv_i("BALFEM_M", 2)
#  Vertical BASIS ORDER. The model is named P{p_vert}LFE-{M}: `Pp` is the
#  vertical Lagrange order and `M` the number of vertical elements, so the run
#  says which member of the BALFE-M family it actually exercises. Default p=1
#  reproduces the piecewise-linear models of Yang & Liu.
p_vert  = genv_i("BALFEM_P_VERT", 1)
model_name = "P$(p_vert)LFE-$(M)"
px, py  = genv_i("BALFEM_PX", 8), genv_i("BALFEM_PY", 4)      # 8*4 = 32 ranks
nx, ny  = genv_i("BALFEM_NX", 200), genv_i("BALFEM_NY", 40)   # dx=0.25, dy=0.5
feord   = genv_i("BALFEM_FE_ORDER", 2)
#  p_eta = 0 keeps the historical EQUAL-ORDER spaces (unchanged default).
#  Set BALFEM_P_ETA = BALFEM_FE_ORDER-1 for the Taylor-Hood-like pairing, which is
#  the only one measured to reach the theoretical order in BOTH fields. It is
#  NOT automatically the better production choice: at a GIVEN mesh the
#  equal-order spaces were 40x more accurate, because eta sits in a richer
#  space. Compare error-vs-DOF before switching.
p_eta   = genv_i("BALFEM_P_ETA", 0)
Lx, Ly  = genv_f("BALFEM_LX", 50.0), genv_f("BALFEM_LY", 20.0)
d       = genv_f("BALFEM_D", 3.5)
spR     = genv_f("BALFEM_SPONGE_R", 12.0)
mumax   = genv_f("BALFEM_MUMAX", 40.0)   # strong: kill the outgoing/boundary mode fast
dt      = genv_f("BALFEM_DT", 0.02)
save_ev = genv_i("BALFEM_SAVE_EVERY", 10)
# ---- per-case knobs (defaults = base case) ----
Twave   = genv_f("BALFEM_TWAVE", 2.0)
Awave   = genv_f("BALFEM_AWAVE", 0.1)
wdir    = genv_f("BALFEM_WAVE_DIR", 0.0)
periods = genv_f("BALFEM_PERIODS", 12.0)
Tfinal  = haskey(ENV, "BALFEM_TFINAL") ? genv_f("BALFEM_TFINAL", 0.0) : periods * Twave

# bathymetry: flat_bed=false => variable bathymetry => build the y-invariant bar
usebar  = !flat_bed_flag(1)
hbar    = genv_f("BALFEM_HBAR", 1.5); xbar = genv_f("BALFEM_XBAR", 26.0); wbar = genv_f("BALFEM_WBAR", 6.0)
sramp   = wbar / 3.0
h_bathy = usebar ?
    (x -> d - 0.5*hbar*(tanh((x[1]-(xbar-wbar))/sramp) - tanh((x[1]-(xbar+wbar))/sramp))) : nothing

bedtag  = usebar ? "bar" : "flat"
tag     = "$(regime_sym())_$(nl_pressure_sym())_$(bedtag)_A$(Awave)_T$(Twave)"
outdir  = genv("BALFEM_OUTDIR", joinpath(ROOT, "output", "small_bcplane_$(tag)_$(model_name)"))

banner("SMALL | BC plane wave (Dirichlet left) | $(regime_sym()) $(nl_pressure_sym()) $bedtag A=$Awave T=$Twave",
       M, (px,py), (nx,ny), nx*ny, outdir)

diags, vert, prob = setup_and_run_distributed(
    cpu_grid=(px,py), M=M, p_vertical=p_vert, c_bdy=cbdy_override(), p_horizontal=feord, p_eta=p_eta,
    domain=(0.0,Lx,0.0,Ly), partition=(nx,ny),
    wave_gen=:bc_gen, bc_side=:left, wave_dir=wdir, bc_profile=bc_profile_sym(),
    h_val=d, T_wave=Twave, A_wave=Awave,
    T_ramp=tramp_val(), relax_bc=relax_flag(), relax_width=relax_w_val(),
    sponge_wL=0.0, sponge_wR=spR, sponge_wB=0.0, sponge_wT=0.0, mu_max=mumax,
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
    print_every=genv_i("BALFEM_PRINT_EVERY", 10))

is_rank0() && @printf("bc_plane [%s] done: %d steps, %d snapshots to %s\n",
                      tag, length(diags), length(diags) ÷ max(save_ev,1), outdir)
