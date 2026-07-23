# ==============================================================
#  _dist_common.jl — shared config for the distributed example scripts
#                        (algebraic solver, GridapLFEM)
#
#  All example scripts read their parameters from environment variables so the
#  SAME script serves any M, any core count, any mesh size — ideal for a
#  cluster job array. Every knob has a sensible default; override via `export`.
#
#  Common environment variables (case-specific ones in each script):
#    LFEM_M           vertical layers (elements)         default per script
#    LFEM_PX,LFEM_PY  MPI process grid (px×py)           MUST satisfy px·py == mpiexec -n
#    LFEM_NX,LFEM_NY  horizontal elements                (nx divisible by px, ny by py)
#    LFEM_FE_ORDER    Q-order (≥2 required)              2
#    LFEM_DT          time step [s]                      0.02
#    LFEM_TFINAL      final time [s] (overrides periods) —
#    LFEM_SAVE_EVERY  VTK snapshot every N steps         (script default)
#    LFEM_OUTDIR      output directory                   (script default)
#    LFEM_WRITE_W         write w_s<σ> fields (1/0)       1
#    LFEM_WRITE_PRESSURE  write p_s<σ> fields (1/0)       1
#    LFEM_CBDY        comma-sep σ node boundaries        (else optimised M≤4 / uniform)
#    LFEM_RHO         water density [kg/m³]              1025
#    LFEM_PRINT_EVERY solver progress report every N steps    (default 10)
#    LFEM_LINEARISED  linearised physics (1/0)           0  (fully nonlinear)
#    LFEM_ADVECTION   nonlinear advection (1/0)          1
#    LFEM_LINP        linear slope pressure A/K (1/0)    script default
#    LFEM_PFULL       full P¹L¹+P²L²+P³L³ leading press. 0  (oracle P³L³ form)
#    LFEM_NLP68       nonlinear pressure comps 6–8 (1/0) 0
#
#  The algebraic solver runs the FULL physics distributed through the one
#  Gridap path (GMRES+Jacobi+Newton) — no owned V⊗H loop, no linear-core
#  restriction (unlike the old solver's distributed drivers).
#
#  c_bdy: paper-optimised vertical nodes exist for M≤4 (Yang & Liu 2024
#  Table 1); for larger M the driver falls back to a uniform σ-grid.
# ==============================================================

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(ROOT, "src", "GridapLFEM.jl"))
using .GridapLFEM
using Printf

genv(k, d)   = get(ENV, k, string(d))
genv_i(k, d) = parse(Int,     genv(k, d))
genv_f(k, d) = parse(Float64, genv(k, d))
genv_b(k, d) = lowercase(genv(k, d)) in ("1", "true", "yes", "on")

# c_bdy override (LFEM_CBDY="0,0.7,1"), else nothing → driver picks optimised/uniform.
cbdy_override() = haskey(ENV, "LFEM_CBDY") ?
    parse.(Float64, split(ENV["LFEM_CBDY"], ",")) : nothing

# shared knobs
write_w_flag()  = genv_b("LFEM_WRITE_W", 1)
write_p_flag()  = genv_b("LFEM_WRITE_PRESSURE", 1)
rho_val()       = genv_f("LFEM_RHO", 1025.0)
lin_flag()      = genv_b("LFEM_LINEARISED", 0)
adv_flag()      = genv_b("LFEM_ADVECTION", 1)
pfull_flag()    = genv_b("LFEM_PFULL", 0)
nlp68_flag()    = genv_b("LFEM_NLP68", 0)

# rank-0 detection BEFORE MPI.Init (OpenMPI / MPICH set these per rank).
is_rank0() = get(ENV, "OMPI_COMM_WORLD_RANK",
                 get(ENV, "PMI_RANK", "0")) == "0"

# ---- Dirichlet boundary wave generation (WaveSpec.jl sea states) -------------
#  Environment variables (used by run_irregular_sea_dist / run_directional_sea_dist):
#    LFEM_HS          significant wave height [m]        0.002 (linear regime)
#    LFEM_TP          peak period [s]                    1.6
#    LFEM_GAMMA       JONSWAP peakedness (0 = estimate)  3.3
#    LFEM_NFREQ       frequency samples (bins = nf-1)    21
#    LFEM_FMIN_FAC    fmin = 1/(FMIN_FAC·Tp)             2.5
#    LFEM_FMAX_FAC    fmax = 1/(FMAX_FAC·Tp)             0.75  (keep kd ≲ kd_app AND
#                                                        ≥6 cells per shortest λ — see README)
#    LFEM_SAMPLING    bin spacing: uniform|energy        uniform (leakage-free analysis)
#    LFEM_NTHETA      angle samples (≤1 → long-crested)  0
#    LFEM_SPREAD_STD  cosine-power spreading σθ [deg]    20
#    LFEM_THETA_MAX   angular truncation ±θmax [deg]     60
#    LFEM_SEED        phase seed (reproducible)          20260723
#    LFEM_BC_SIDE     generation boundary left|right     left
#    LFEM_BC_PROFILE  vertical polarization model|airy   model
#    LFEM_TRAMP       Hann ramp [s] (unset → 2·Tp)       —
#    LFEM_RELAX       relaxation zone at the inflow      0
#    LFEM_RELAX_W     zone width [m] (0 → one peak λ)    0
#  The WaveInput conversion snapshots the SEEDED phases into plain arrays, so
#  every MPI rank builds an identical component table — no communication needed.
hs_val()         = genv_f("LFEM_HS", 0.002)
tp_val()         = genv_f("LFEM_TP", 1.6)
bc_side_sym()    = Symbol(genv("LFEM_BC_SIDE", "left"))
bc_profile_sym() = Symbol(genv("LFEM_BC_PROFILE", "model"))
tramp_val()      = haskey(ENV, "LFEM_TRAMP") ? genv_f("LFEM_TRAMP", 0.0) : nothing
relax_flag()     = genv_b("LFEM_RELAX", 0)
relax_w_val()    = genv_f("LFEM_RELAX_W", 0.0)

"""
    build_airy_state(d_val; directional=false) → WaveSpec.AiryWaves.AiryState

Env-configured stochastic sea state: JONSWAP spectrum, uniform-frequency (or
equal-energy) bins, optional cosine-power angular spreading, seeded phases.
`directional=true` switches the default spreading on (`LFEM_NTHETA` ≥ 2).
"""
function build_airy_state(d_val; directional::Bool=false)
    Hs, Tp = hs_val(), tp_val()
    γ      = genv_f("LFEM_GAMMA", 3.3)
    nf     = genv_i("LFEM_NFREQ", 21)
    fmin   = 1.0/(genv_f("LFEM_FMIN_FAC", 2.5)*Tp)
    fmax   = 1.0/(genv_f("LFEM_FMAX_FAC", 0.75)*Tp)
    dom    = lowercase(genv("LFEM_SAMPLING", "uniform")) == "energy" ?
             WaveSpec.SpectralSampling.Energy : WaveSpec.SpectralSampling.Frequency
    spec   = γ > 0 ? WaveSpec.ContinuousSpectrums.JONSWAP(Hs, Tp, γ) :
                     WaveSpec.ContinuousSpectrums.JONSWAP(Hs, Tp)
    dspec  = WaveSpec.SpectralSpreading.DiscreteSpectralSpreading(
                 spec, WaveSpec.SpectralSampling.UniformSampling(),
                 fmin, fmax, nf; domain=dom, mess=is_rank0())
    nθ = genv_i("LFEM_NTHETA", directional ? 7 : 0)
    spread = nθ <= 1 ? WaveSpec.AngularSpreading.DiscreteAngularSpreading(0.0) :
             WaveSpec.AngularSpreading.DiscreteAngularSpreading(
                 :cosinepow, 0.0, genv_f("LFEM_SPREAD_STD", 20.0)*pi/180,
                 -genv_f("LFEM_THETA_MAX", 60.0)*pi/180,
                  genv_f("LFEM_THETA_MAX", 60.0)*pi/180, nθ)
    state = WaveSpec.AiryWaves.AiryState(dspec, spread, d_val)
    return WaveSpec.AiryWaves.change_seed!(state, genv_i("LFEM_SEED", 20260723))
end

function banner(title, M, cpu_grid, partition, ncells, outdir)
    is_rank0() || return
    @printf("############################################################\n")
    @printf("# %s  [ALGEBRAIC solver, stacked layout]\n", title)
    @printf("#   M=%d layers | cpu_grid=%s (%d ranks) | mesh=%s = %d cells\n",
            M, string(cpu_grid), prod(cpu_grid), string(partition), ncells)
    @printf("#   linearised=%s advection=%s Pfull=%s nlP68=%s\n",
            string(lin_flag()), string(adv_flag()), string(pfull_flag()), string(nlp68_flag()))
    @printf("#   write_w=%s write_pressure=%s | out=%s\n",
            string(write_w_flag()), string(write_p_flag()), outdir)
    @printf("############################################################\n")
    flush(stdout)
end
