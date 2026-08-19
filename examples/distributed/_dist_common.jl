# ==============================================================
#  _dist_common.jl — shared config for the distributed example scripts
#                        (algebraic solver, GridapBALFEM)
#
#  All example scripts read their parameters from environment variables so the
#  SAME script serves any M, any core count, any mesh size — ideal for a
#  cluster job array. Every knob has a sensible default; override via `export`.
#
#  Common environment variables (case-specific ones in each script):
#    BALFEM_M           vertical layers (elements)         default per script
#    BALFEM_PX,BALFEM_PY  MPI process grid (px×py)           MUST satisfy px·py == mpiexec -n
#    BALFEM_NX,BALFEM_NY  horizontal elements                (nx divisible by px, ny by py)
#    BALFEM_FE_ORDER    VELOCITY Q-order (≥2 required)     2
#    BALFEM_P_ETA       SURFACE Q-order; 0 = equal order   0
#                     Set to BALFEM_FE_ORDER−1 for the Taylor-Hood-like pairing.
#                     η enters momentum undifferentiated (via ∇·v after IBP), so
#                     equal-order spaces are inf-sup deficient and the analytic
#                     MMS measures order p there, not p+1. But a better RATE is
#                     not a better ANSWER at a given mesh — equal order was 40x
#                     more accurate at nx=24. Compare error-vs-DOF first.
#    BALFEM_DT          time step [s]                      0.02
#    BALFEM_TFINAL      final time [s] (overrides periods) —
#    BALFEM_SAVE_EVERY  VTK snapshot every N steps         (script default)
#    BALFEM_OUTDIR      output directory                   (script default)
#    BALFEM_WRITE_W         write w_s<σ> fields (1/0)       1
#    BALFEM_WRITE_PRESSURE  write p_s<σ> fields (1/0)       1
#    BALFEM_CBDY        comma-sep σ node boundaries        (else optimised M≤4 / uniform)
#    BALFEM_RHO         water density [kg/m³]              1025
#    BALFEM_PRINT_EVERY solver progress report every N steps    (default 10)
#    BALFEM_REGIME      physics regime: linear|nonlinear   nonlinear
#    BALFEM_NL_PRESSURE nonlinear pressure: none|native|full  none
#    BALFEM_FLAT_BED    flat sea bed ∇h≡0 (1/0)  1 flat scripts / 0 bathymetry script
#
#  The algebraic solver runs the FULL physics distributed through the one
#  Gridap path (GMRES+Jacobi+Newton) — no owned V⊗H loop, no linear-core
#  restriction (unlike the old solver's distributed drivers).
#
#  c_bdy: paper-optimised vertical nodes exist for M≤4 (Yang & Liu 2024
#  Table 1); for larger M the driver falls back to a uniform σ-grid.
# ==============================================================

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
using GridapBALFEM
using Printf

genv(k, d)   = get(ENV, k, string(d))
genv_i(k, d) = parse(Int,     genv(k, d))
genv_f(k, d) = parse(Float64, genv(k, d))
genv_b(k, d) = lowercase(genv(k, d)) in ("1", "true", "yes", "on")

# c_bdy override (BALFEM_CBDY="0,0.7,1"), else nothing → driver picks optimised/uniform.
cbdy_override() = haskey(ENV, "BALFEM_CBDY") ?
    parse.(Float64, split(ENV["BALFEM_CBDY"], ",")) : nothing

# shared knobs
write_w_flag()  = genv_b("BALFEM_WRITE_W", 1)
write_p_flag()  = genv_b("BALFEM_WRITE_PRESSURE", 1)
rho_val()       = genv_f("BALFEM_RHO", 1025.0)
regime_sym()         = Symbol(genv("BALFEM_REGIME", "nonlinear"))
nl_pressure_sym()    = Symbol(genv("BALFEM_NL_PRESSURE", "none"))
# Sea-bed geometry: false = variable bathymetry (∇h≠0), true = flat bed (∇h≡0).
# Default per script (BALFEM_FLAT_BED): flat-bed cases pass 1, the bathymetry case passes 0.
flat_bed_flag(default::Int=1) = genv_b("BALFEM_FLAT_BED", default)

# ---- Time integrator + nonlinear/linear solver controls ---------------------
#  Default integrator is the fully-implicit RungeKutta :SDIRK_2_2 (L-stable,
#  more robust than Crank–Nicolson). Override caps/tolerances per cluster job:
#    BALFEM_SOLVER      integrator: sdirk|theta        sdirk
#    BALFEM_TABLEAU     RK tableau (solver=sdirk)       SDIRK_2_2
#    BALFEM_NL_ITER     max Newton iterations / stage    50
#    BALFEM_NL_TOL      Newton tolerance (production)    1e-5
#    BALFEM_LS_MAXITER  max GMRES iterations / Newton    1000   (TIME bound)
#    BALFEM_LS_RTOL     GMRES relative tolerance         1e-5  (measured: 1e-9→1e-6→1e-5
#                       each cut GMRES iterations with max|η| unmoved. NOTE this now
#                       EQUALS BALFEM_NL_TOL rather than sitting one order below it —
#                       see the ladder discussion in LOCAL_TESTS_RESULTS.md §5.4)
#    BALFEM_KRYLOV_M    GMRES basis size, restart=true    100   (MEMORY bound)
#      NOTE BALFEM_KRYLOV_M and BALFEM_LS_MAXITER are different things: the basis
#      costs m+1 distributed vectors plus a dense (m+1)×m Hessenberg PER RANK at
#      every numerical setup, so it drives memory; BALFEM_LS_MAXITER only decides
#      how long GMRES may iterate. Conflating them (passing the iteration cap as
#      the basis size) is what OOM-killed the 32-rank runs.
# ---- Field diagnostics (monitor.jl) -----------------------------------------
#    BALFEM_DIAG_EVERY  sample every N steps            0 → follow BALFEM_PRINT_EVERY
#                                                     −1 → disable the whole block
#    BALFEM_DIAG_CSV    write <outdir>/diagnostics.csv  1
#    BALFEM_DIV_FACTOR  abort at div_factor·eta_ref     20
#    BALFEM_ETA_REF     override the amplitude scale    (unset → inferred from the
#                                                      forcing: A_wave / Hs / peak η₀)
#  The diagnostics are ON by default and cost within noise (measured −0.8 %/+2.2 %).
#  They are what makes a failed cluster run diagnosable from its log alone: WHERE
#  max|η| sits, its interior/damped split, |u|/|η|, mass & energy, GMRES
#  saturation, and per-rank RSS. Read one with examples/inspect_run.jl.
diag_every_val() = genv_i("BALFEM_DIAG_EVERY", 0)
diag_csv_flag()  = genv_b("BALFEM_DIAG_CSV", 1)
div_factor_val() = genv_f("BALFEM_DIV_FACTOR", 20.0)
eta_ref_val()    = haskey(ENV, "BALFEM_ETA_REF") ? genv_f("BALFEM_ETA_REF", 0.0) : nothing

solver_sym()     = Symbol(genv("BALFEM_SOLVER", "sdirk"))
tableau_sym()    = Symbol(genv("BALFEM_TABLEAU", "SDIRK_2_2"))
nl_iter_val()    = genv_i("BALFEM_NL_ITER", 50)
nl_tol_val()     = genv_f("BALFEM_NL_TOL", 1e-5)
ls_maxiter_val() = genv_i("BALFEM_LS_MAXITER", 1000)
ls_rtol_val()    = genv_f("BALFEM_LS_RTOL", 1e-5)
krylov_m_val()   = genv_i("BALFEM_KRYLOV_M", 100)
#    BALFEM_PRECOND   GMRES preconditioner: jacobi|schwarz|gs   jacobi
#      Jacobi is weak for this operator (450-760 iters/solve measured); :schwarz
#      does an exact LU on each rank's own block. See build_preconditioner.
precond_sym()    = Symbol(genv("BALFEM_PRECOND", "jacobi"))

# rank-0 detection BEFORE MPI.Init (OpenMPI / MPICH set these per rank).
is_rank0() = get(ENV, "OMPI_COMM_WORLD_RANK",
                 get(ENV, "PMI_RANK", "0")) == "0"

# ---- Dirichlet boundary wave generation (WaveSpec.jl sea states) -------------
#  Environment variables (used by run_irregular_sea_dist / run_directional_sea_dist):
#    BALFEM_HS          significant wave height [m]        0.002 (linear regime)
#    BALFEM_TP          peak period [s]                    1.6
#    BALFEM_GAMMA       JONSWAP peakedness (0 = estimate)  3.3
#    BALFEM_NFREQ       frequency samples (bins = nf-1)    21
#    BALFEM_FMIN_FAC    fmin = 1/(FMIN_FAC·Tp)             2.5
#    BALFEM_FMAX_FAC    fmax = 1/(FMAX_FAC·Tp)             0.75  (keep kd ≲ kd_app AND
#                                                        ≥6 cells per shortest λ — see README)
#    BALFEM_SAMPLING    bin spacing: uniform|energy        uniform (leakage-free analysis)
#    BALFEM_NTHETA      angle samples (≤1 → long-crested)  0
#    BALFEM_SPREAD_STD  cosine-power spreading σθ [deg]    20
#    BALFEM_THETA_MAX   angular truncation ±θmax [deg]     60
#    BALFEM_SEED        phase seed (reproducible)          20260723
#    BALFEM_BC_SIDE     generation boundary left|right     left
#    BALFEM_BC_PROFILE  vertical polarization model|airy   model
#    BALFEM_TRAMP       Hann ramp [s] (unset → 2·Tp)       —
#    BALFEM_RELAX       relaxation zone at the inflow      0
#    BALFEM_RELAX_W     zone width [m] (0 → one peak λ)    0
#  The WaveInput conversion snapshots the SEEDED phases into plain arrays, so
#  every MPI rank builds an identical component table — no communication needed.
hs_val()         = genv_f("BALFEM_HS", 0.002)
tp_val()         = genv_f("BALFEM_TP", 1.6)
bc_side_sym()    = Symbol(genv("BALFEM_BC_SIDE", "left"))
bc_profile_sym() = Symbol(genv("BALFEM_BC_PROFILE", "model"))
tramp_val()      = haskey(ENV, "BALFEM_TRAMP") ? genv_f("BALFEM_TRAMP", 0.0) : nothing
relax_flag()     = genv_b("BALFEM_RELAX", 0)
relax_w_val()    = genv_f("BALFEM_RELAX_W", 0.0)

"""
    build_airy_state(h_val; directional=false) → WaveSpec.AiryWaves.AiryState

Env-configured stochastic sea state: JONSWAP spectrum, uniform-frequency (or
equal-energy) bins, optional cosine-power angular spreading, seeded phases.
`directional=true` switches the default spreading on (`BALFEM_NTHETA` ≥ 2).
"""
function build_airy_state(h_val; directional::Bool=false)
    Hs, Tp = hs_val(), tp_val()
    γ      = genv_f("BALFEM_GAMMA", 3.3)
    nf     = genv_i("BALFEM_NFREQ", 21)
    fmin   = 1.0/(genv_f("BALFEM_FMIN_FAC", 2.5)*Tp)
    fmax   = 1.0/(genv_f("BALFEM_FMAX_FAC", 0.75)*Tp)
    dom    = lowercase(genv("BALFEM_SAMPLING", "uniform")) == "energy" ?
             WaveSpec.SpectralSampling.Energy : WaveSpec.SpectralSampling.Frequency
    spec   = γ > 0 ? WaveSpec.ContinuousSpectrums.JONSWAP(Hs, Tp, γ) :
                     WaveSpec.ContinuousSpectrums.JONSWAP(Hs, Tp)
    dspec  = WaveSpec.SpectralSpreading.DiscreteSpectralSpreading(
                 spec, WaveSpec.SpectralSampling.UniformSampling(),
                 fmin, fmax, nf; domain=dom, mess=is_rank0())
    nθ = genv_i("BALFEM_NTHETA", directional ? 7 : 0)
    spread = nθ <= 1 ? WaveSpec.AngularSpreading.DiscreteAngularSpreading(0.0) :
             WaveSpec.AngularSpreading.DiscreteAngularSpreading(
                 :cosinepow, 0.0, genv_f("BALFEM_SPREAD_STD", 20.0)*pi/180,
                 -genv_f("BALFEM_THETA_MAX", 60.0)*pi/180,
                  genv_f("BALFEM_THETA_MAX", 60.0)*pi/180, nθ)
    state = WaveSpec.AiryWaves.AiryState(dspec, spread, h_val)
    return WaveSpec.AiryWaves.change_seed!(state, genv_i("BALFEM_SEED", 20260723))
end

function banner(title, M, cpu_grid, partition, ncells, outdir)
    is_rank0() || return
    @printf("############################################################\n")
    @printf("# %s  [ALGEBRAIC solver, stacked layout]\n", title)
    @printf("#   M=%d layers | cpu_grid=%s (%d ranks) | mesh=%s = %d cells\n",
            M, string(cpu_grid), prod(cpu_grid), string(partition), ncells)
    @printf("#   regime=%s nl_pressure=%s flat_bed=%s\n",
            string(regime_sym()), string(nl_pressure_sym()), string(flat_bed_flag()))
    @printf("#   write_w=%s write_pressure=%s | out=%s\n",
            string(write_w_flag()), string(write_p_flag()), outdir)
    @printf("############################################################\n")
    flush(stdout)
end
