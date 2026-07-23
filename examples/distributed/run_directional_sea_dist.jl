# ==============================================================
#  run_directional_sea_dist.jl — DISTRIBUTED short-crested (directional)
#                                JONSWAP sea via Dirichlet BCs (PRODUCTION)
#
#  JONSWAP × cosine-power angular spreading: every (ωᵢ,θⱼ) bin is a plane-wave
#  component with its own direction. The left Dirichlet boundary prescribes
#  η, 𝖴x AND 𝖴y (directional seas REQUIRE y_wall_bc=false — the lateral
#  boundaries are sponge-absorbed instead of solid walls). Result: a genuinely
#  2D short-crested sea surface evolving under the LFE-M dynamics.
#
#  LAUNCH (px·py MUST equal -n):
#    LFEM_M=2 LFEM_PX=16 LFEM_PY=8 \
#    ~/.julia/bin/mpiexecjl --project=. -n 128 julia --project=. \
#        GridapLFEM.jl/examples/distributed/run_directional_sea_dist.jl
#
#  Case-specific env vars (sea-state + shared ones in _dist_common.jl):
#    LFEM_LX,LFEM_LY  domain size [m]              default 400 × 200
#    LFEM_D           still-water depth [m]        3.5
#    LFEM_SPONGE      right/lateral sponge [m]     40
#    LFEM_MUMAX       sponge strength              5
#    LFEM_PERIODS     run length in Tp units       100
#    LFEM_NTHETA      angle samples (bins = n-1)   7 (short-crested default)
#    LFEM_SPREAD_STD  spreading σθ [deg]           20
#
#  Oblique components travel a longer lateral path — keep Ly wide enough that
#  the ±θmax components clear the working area before the lateral sponges.
# ==============================================================

include(joinpath(@__DIR__, "_dist_common.jl"))

M        = genv_i("LFEM_M", 2)
px, py   = genv_i("LFEM_PX", 16), genv_i("LFEM_PY", 8)
nx, ny   = genv_i("LFEM_NX", 1200), genv_i("LFEM_NY", 600)
feord    = genv_i("LFEM_FE_ORDER", 2)
Lx, Ly   = genv_f("LFEM_LX", 400.0), genv_f("LFEM_LY", 200.0)
d        = genv_f("LFEM_D", 3.5)
sponge   = genv_f("LFEM_SPONGE", 40.0)
mumax    = genv_f("LFEM_MUMAX", 5.0)
dt       = genv_f("LFEM_DT", 0.02)
periods  = genv_f("LFEM_PERIODS", 100.0)
Tp       = tp_val()
Tfinal   = haskey(ENV, "LFEM_TFINAL") ? genv_f("LFEM_TFINAL", 0.0) : periods * Tp
save_ev  = genv_i("LFEM_SAVE_EVERY", 40)
outdir   = genv("LFEM_OUTDIR", joinpath(ROOT, "output", "directional_sea_dist_M$(M)"))

state = build_airy_state(d; directional=true)

banner("DIRECTIONAL SEA (JONSWAP × spreading, Dirichlet BCs) — distributed",
       M, (px,py), (nx,ny), nx*ny, outdir)
is_rank0() && @printf("#   Hs=%.4g m Tp=%.3g s | nθ=%d σθ=%.1f° | bc=%s/%s seed=%d\n",
                      hs_val(), Tp, genv_i("LFEM_NTHETA", 7),
                      genv_f("LFEM_SPREAD_STD", 20.0), string(bc_side_sym()),
                      string(bc_profile_sym()), genv_i("LFEM_SEED", 20260723))

diags, vert, prob = setup_and_run_distributed(
    cpu_grid=(px,py), M=M, c_bdy=cbdy_override(), fe_order=feord,
    domain=(0.0,Lx,0.0,Ly), partition=(nx,ny),
    d_val=d, T_wave=Tp, A_wave=hs_val()/2,
    wave_bc=state, bc_side=bc_side_sym(), bc_profile=bc_profile_sym(),
    T_ramp=tramp_val(), relax_bc=relax_flag(), relax_width=relax_w_val(),
    sponge_wL=0.0, sponge_wR=sponge, sponge_wB=sponge, sponge_wT=sponge,
    mu_max=mumax,
    T_final=Tfinal, dt=dt,
    linearised=lin_flag(), advection=adv_flag(),
    P_full=pfull_flag(), nl_pressure68=nlp68_flag(),
    y_wall_bc=false, x_wall_bc=false,       # REQUIRED: v ≠ 0 at the inflow
    output_dir=outdir, save_every=save_ev,
    write_w=genv_b("LFEM_WRITE_W", 0), write_pressure=genv_b("LFEM_WRITE_PRESSURE", 0),
    rho=rho_val(),
    solver_type=solver_sym(), tableau=tableau_sym(),
    nl_iter=nl_iter_val(), nl_tol=nl_tol_val(),
    ls_rtol=ls_rtol_val(), ls_maxiter=ls_maxiter_val(),
    print_every=genv_i("LFEM_PRINT_EVERY", 10))

is_rank0() && @printf("directional_sea_dist done: %d steps, %d snapshots to %s\n",
                      length(diags), save_ev > 0 ? length(diags) ÷ save_ev : 0, outdir)
