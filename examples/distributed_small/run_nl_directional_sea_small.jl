# ==============================================================
#  run_nl_directional_sea_small.jl — SMALL-DOMAIN observation run
#
#  CASE 5/6:  nonlinear · FULL pressure · flat bed · directional sea (BC gen) · BIG amplitude
#  Short-crested JONSWAP × cosine-power sea generated purely by time-varying
#  Dirichlet data (η, 𝖴x, 𝖴y) on the LEFT boundary — NO interior wavemaker.
#  Fully nonlinear, all pressure tiers, flat bed. Hs=0.2 m ⇒ characteristic
#  amplitude A=Hs/2=0.1 m. Directional seas prescribe 𝖴y ≠ 0 at the inflow, so
#  y_wall_bc=:open (lateral sponges), NOT walls.
#
#  Small-domain sizing: Dirichlet left (sponge_wL=0) with an inflow relaxation
#  zone (6 m ≈ 1 λ, absorbs backscatter without damping the incident wave);
#  right sponge 12 m; lateral sponges 4 m; μ_max=8. Spreading is narrowed
#  (σθ=15°, θmax=40°, 7 angles) so components stay forward in the short box.
#  Band kept modest (15 freq bins) for a light, dynamic run.
#
#  LAUNCH (px·py MUST equal -n):
#    LFEM_PX=8 LFEM_PY=8 \
#    ~/.julia/bin/mpiexecjl --project=. -n 64 julia --project=. \
#        GridapLFEM.jl/examples/distributed_small/run_nl_directional_sea_small.jl
#  (SLURM: run/dist_small/run_nl_directional_sea_small.sh)
#
#  Case-specific env vars (sea-state + shared ones in ../distributed/_dist_common.jl):
#    LFEM_LX,LFEM_LY 50×20 · LFEM_D 3.5 · LFEM_HS 0.2 · LFEM_TP 2.0
#    LFEM_SPONGE_R 12 · LFEM_SPONGE_Y 4 · LFEM_MUMAX 8 · LFEM_RELAX_W 6
#    LFEM_NTHETA 7 · LFEM_SPREAD_STD 15 · LFEM_THETA_MAX 40 · LFEM_PERIODS 15
# ==============================================================

include(joinpath(@__DIR__, "..", "distributed", "_dist_common.jl"))

# ---- CASE 5 physics + big-amplitude sea + small-box spreading defaults (env still overrides) ----
get!(ENV, "LFEM_REGIME", "nonlinear"); get!(ENV, "LFEM_NL_PRESSURE", "full")
get!(ENV, "LFEM_FLAT_BED", "1")
get!(ENV, "LFEM_HS", "0.2"); get!(ENV, "LFEM_TP", "2.0")
get!(ENV, "LFEM_NFREQ", "15"); get!(ENV, "LFEM_NTHETA", "7")
get!(ENV, "LFEM_SPREAD_STD", "15"); get!(ENV, "LFEM_THETA_MAX", "40")
get!(ENV, "LFEM_RELAX", "1"); get!(ENV, "LFEM_RELAX_W", "6")

M        = genv_i("LFEM_M", 2)
px, py   = genv_i("LFEM_PX", 8), genv_i("LFEM_PY", 8)
nx, ny   = genv_i("LFEM_NX", 200), genv_i("LFEM_NY", 80)
feord    = genv_i("LFEM_FE_ORDER", 2)
Lx, Ly   = genv_f("LFEM_LX", 50.0), genv_f("LFEM_LY", 20.0)
d        = genv_f("LFEM_D", 3.5)
spR      = genv_f("LFEM_SPONGE_R", 12.0)
spY      = genv_f("LFEM_SPONGE_Y", 4.0)
mumax    = genv_f("LFEM_MUMAX", 8.0)
dt       = genv_f("LFEM_DT", 0.02)
periods  = genv_f("LFEM_PERIODS", 15.0)
Tp       = tp_val()
Tfinal   = haskey(ENV, "LFEM_TFINAL") ? genv_f("LFEM_TFINAL", 0.0) : periods * Tp
save_ev  = genv_i("LFEM_SAVE_EVERY", 15)
outdir   = genv("LFEM_OUTDIR", joinpath(ROOT, "output", "small_nl_directional_sea_M$(M)"))

state = build_airy_state(d; directional=true)

banner("SMALL | NONLINEAR full-pressure DIRECTIONAL sea (Dirichlet BC, flat bed)",
       M, (px,py), (nx,ny), nx*ny, outdir)
is_rank0() && @printf("#   Hs=%.4g m Tp=%.3g s (A=Hs/2=%.4g) | nθ=%d σθ=%.1f° θmax=%.1f° seed=%d\n",
                      hs_val(), Tp, hs_val()/2, genv_i("LFEM_NTHETA", 7),
                      genv_f("LFEM_SPREAD_STD", 15.0), genv_f("LFEM_THETA_MAX", 40.0),
                      genv_i("LFEM_SEED", 20260723))

diags, vert, prob = setup_and_run_distributed(
    cpu_grid=(px,py), M=M, c_bdy=cbdy_override(), p_horizontal=feord,
    domain=(0.0,Lx,0.0,Ly), partition=(nx,ny),
    h_val=d, T_wave=Tp, A_wave=hs_val()/2,                         # ★ big-amplitude sea (Hs=0.2 ⇒ A=0.1)
    wave_bc=state, bc_side=bc_side_sym(), bc_profile=bc_profile_sym(),
    T_ramp=tramp_val(), relax_bc=relax_flag(), relax_width=relax_w_val(),
    sponge_wL=0.0, sponge_wR=spR, sponge_wB=spY, sponge_wT=spY, mu_max=mumax,
    T_final=Tfinal, dt=dt,
    regime=regime_sym(), nl_pressure=nl_pressure_sym(), flat_bed=flat_bed_flag(),  # ★ CASE 5 (defaults above)
    y_wall_bc=:open, x_wall_bc=false,                              # REQUIRED: 𝖴y ≠ 0 at the inflow
    output_dir=outdir, save_every=save_ev,
    write_w=genv_b("LFEM_WRITE_W", 0), write_pressure=genv_b("LFEM_WRITE_PRESSURE", 0),
    rho=rho_val(),
    solver_type=solver_sym(), tableau=tableau_sym(),
    nl_iter=nl_iter_val(), nl_tol=nl_tol_val(),
    ls_rtol=ls_rtol_val(), ls_maxiter=ls_maxiter_val(),
    print_every=genv_i("LFEM_PRINT_EVERY", 10))

is_rank0() && @printf("small_nl_directional_sea done: %d steps, %d snapshots to %s\n",
                      length(diags), save_ev > 0 ? length(diags) ÷ save_ev : 0, outdir)
