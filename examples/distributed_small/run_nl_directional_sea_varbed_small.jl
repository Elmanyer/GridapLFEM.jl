# ==============================================================
#  run_nl_directional_sea_varbed_small.jl — SMALL-DOMAIN sea-state comparison run
#
#  CASE 14/16:  nonlinear · FULL pressure · VARIABLE bathymetry · directional sea (BC) · Hs=0.2
#  Short-crested JONSWAP × cosine-power sea shoaling (and refracting) over a
#  shore-parallel submerged bar (flat_bed=false ⇒ ∇h terms ON) — the coastal
#  "real directional sea over a shoal" scenario. Compared with case 5 (flat bed,
#  same sea) it isolates the bathymetric effects (depth-induced harmonic release,
#  refraction of the oblique components). Generation stays at the offshore depth
#  d0=3.5; the sea shoals to d≈2.0 on the crest.
#
#  Bar identical to case 3 (y-invariant, analytic ∇h,∇²h): x_b=26, w=6, h_bar=1.5.
#  Geometry otherwise identical to case 5. Hs=0.2, Tp=2.0. y_wall_bc=:open.
#
#  NOTE: uses build_airy_state ⇒ depends on the WaveSpec change_seed! fix.
#
#  LAUNCH (px·py MUST equal -n):
#    LFEM_PX=4 LFEM_PY=2 mpiexecjl --project=. -n 8 julia --project=. \
#        GridapLFEM.jl/examples/distributed_small/run_nl_directional_sea_varbed_small.jl
#  (SLURM: run/dist_small/run_nl_directional_sea_varbed_small.sh)
# ==============================================================

include(joinpath(@__DIR__, "..", "distributed", "_dist_common.jl"))

# ---- CASE 14 physics + sea + small-box spreading defaults (env still overrides) ----
get!(ENV, "LFEM_REGIME", "nonlinear"); get!(ENV, "LFEM_NL_PRESSURE", "full")
get!(ENV, "LFEM_FLAT_BED", "0")   # ★ variable bathymetry: ∇h terms ON
get!(ENV, "LFEM_HS", "0.2"); get!(ENV, "LFEM_TP", "2.0")
get!(ENV, "LFEM_NFREQ", "15"); get!(ENV, "LFEM_NTHETA", "7")
get!(ENV, "LFEM_SPREAD_STD", "15"); get!(ENV, "LFEM_THETA_MAX", "40")
get!(ENV, "LFEM_RELAX", "1"); get!(ENV, "LFEM_RELAX_W", "6")

M        = genv_i("LFEM_M", 2)
px, py   = genv_i("LFEM_PX", 4), genv_i("LFEM_PY", 2)
nx, ny   = genv_i("LFEM_NX", 200), genv_i("LFEM_NY", 80)
feord    = genv_i("LFEM_FE_ORDER", 2)
Lx, Ly   = genv_f("LFEM_LX", 50.0), genv_f("LFEM_LY", 20.0)
d0       = genv_f("LFEM_D0", 3.5)
hbar     = genv_f("LFEM_HBAR", 1.5)
xbar     = genv_f("LFEM_XBAR", 26.0)
wbar     = genv_f("LFEM_WBAR", 6.0)
spR      = genv_f("LFEM_SPONGE_R", 12.0)
spY      = genv_f("LFEM_SPONGE_Y", 4.0)
mumax    = genv_f("LFEM_MUMAX", 8.0)
dt       = genv_f("LFEM_DT", 0.02)
periods  = genv_f("LFEM_PERIODS", 15.0)
Tp       = tp_val()
Tfinal   = haskey(ENV, "LFEM_TFINAL") ? genv_f("LFEM_TFINAL", 0.0) : periods * Tp
save_ev  = genv_i("LFEM_SAVE_EVERY", 15)
outdir   = genv("LFEM_OUTDIR", joinpath(ROOT, "output", "small_nl_directional_sea_varbed_M$(M)"))

# smooth y-invariant (shore-parallel) submerged bar — same as case 3
sramp   = wbar / 3.0
h_bathy(x) = d0 - 0.5*hbar*(tanh((x[1]-(xbar-wbar))/sramp) - tanh((x[1]-(xbar+wbar))/sramp))

state = build_airy_state(d0; directional=true)

banner("SMALL | NONLINEAR full-pressure DIRECTIONAL sea over a BAR (Dirichlet BC, variable bed)",
       M, (px,py), (nx,ny), nx*ny, outdir)
is_rank0() && @printf("#   Hs=%.4g m Tp=%.3g s | bar d:%.1f→%.1f m | nθ=%d σθ=%.1f° seed=%d\n",
                      hs_val(), Tp, d0, d0-hbar, genv_i("LFEM_NTHETA", 7),
                      genv_f("LFEM_SPREAD_STD", 15.0), genv_i("LFEM_SEED", 20260723))

diags, vert, prob = setup_and_run_distributed(
    cpu_grid=(px,py), M=M, c_bdy=cbdy_override(), p_horizontal=feord,
    domain=(0.0,Lx,0.0,Ly), partition=(nx,ny),
    h_val=d0, T_wave=Tp, A_wave=hs_val()/2,
    wave_bc=state, bc_side=bc_side_sym(), bc_profile=bc_profile_sym(),
    T_ramp=tramp_val(), relax_bc=relax_flag(), relax_width=relax_w_val(),
    sponge_wL=0.0, sponge_wR=spR, sponge_wB=spY, sponge_wT=spY, mu_max=mumax,
    T_final=Tfinal, dt=dt,
    h_bathy=h_bathy,                                               # ★ variable bathymetry (shore-parallel bar)
    regime=regime_sym(), nl_pressure=nl_pressure_sym(), flat_bed=flat_bed_flag(0),  # ★ CASE 14 (∇h ON; defaults above)
    y_wall_bc=:open, x_wall_bc=false,                              # REQUIRED: 𝖴y ≠ 0 at the inflow
    output_dir=outdir, save_every=save_ev,
    write_w=genv_b("LFEM_WRITE_W", 0), write_pressure=genv_b("LFEM_WRITE_PRESSURE", 0),
    rho=rho_val(),
    solver_type=solver_sym(), tableau=tableau_sym(),
    nl_iter=nl_iter_val(), nl_tol=nl_tol_val(),
    ls_rtol=ls_rtol_val(), ls_maxiter=ls_maxiter_val(),
    print_every=genv_i("LFEM_PRINT_EVERY", 10))

is_rank0() && @printf("small_nl_directional_sea_varbed done: %d steps, %d snapshots to %s\n",
                      length(diags), save_ev > 0 ? length(diags) ÷ save_ev : 0, outdir)
