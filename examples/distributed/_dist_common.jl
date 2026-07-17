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
