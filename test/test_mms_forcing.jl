# ==============================================================
#  test_mms_forcing.jl — analytic-MMS FORCING gates (no FE solve; seconds)
#
#  This is the cheap tier that catches a wrong forcing BEFORE any expensive
#  convergence run. If G1 fails, nothing downstream is worth running.
#
#  WHY THIS TEST IS DIFFERENT FROM test_selfconsistency.jl. That test builds its
#  forcing by assembling the solver's OWN residual, so writing R = R_true + E the
#  error cancels identically and it passes for ANY R. Here the forcing comes from
#  the governing equations in closed form, so it does not cancel — which is what
#  makes the convergence rate in test_mms_convergence.jl able to detect a residual
#  that is self-consistently wrong. G5 mechanically enforces that independence.
#
#  RUN:  julia --project=. test/test_mms_forcing.jl
# ==============================================================

using GridapBALFEM
using Gridap.TensorValues: VectorValue      # NOT re-exported by GridapBALFEM
using Printf

println("="^64)
println("  test_mms_forcing.jl — analytic MMS, forcing gates")
println("="^64)

n_pass = 0; n_fail = 0
function check(name, cond)
    global n_pass, n_fail
    if cond; println("  PASS  $name"); n_pass += 1
    else;    println("  FAIL  $name"); n_fail += 1; end
end

const D_DEPTH = 1.0
const G_ACC   = 9.81
vert = assemble_vertical_tensors(2, 1, [0.0, 0.728, 1.0])
Nσ   = vert.N_dof

# ---------------------------------------------------------------------------
#  G1 — the free check: on a plane-wave EIGENMODE, 𝓛(u*) ≡ 0
#  This is the load-bearing gate. It validates the strong-form evaluator against
#  the same closed-form dispersion relation that anchors the Yang & Liu
#  comparison, and it is what PINS THE SIGN CONVENTION OF B: a sign error makes
#  the forcing fail to vanish on an exact solution.
# ---------------------------------------------------------------------------
println("\n--- G1: eigenmode ⇒ 𝓛(u*) = 0 ---")
worst_eig = 0.0
for k in (0.5, 1.0, 2.0, 5.0)
    cbs, ω, û = eigenmode_callables(vert, D_DEPTH, G_ACC, k)
    m = 0.0
    for x in (0.13, 0.77, 2.10), t in (0.0, 0.37, 1.90)
        Lη, Lx, Ly = strong_residual_stage1(cbs, vert, D_DEPTH, G_ACC, x, 0.31, t)
        m = max(m, abs(Lη), maximum(abs.(Lx)), maximum(abs.(Ly)))
    end
    @printf("    k=%-5.2f Cm=%.6f ω=%.6f  max|𝓛(u*)|=%.3e\n",
            k, model_celerity(vert, D_DEPTH, G_ACC, k), ω, m)
    global worst_eig = max(worst_eig, m)
end
check("G1 eigenmode forcing vanishes (<1e-12): $(@sprintf("%.3e", worst_eig))",
      worst_eig < 1e-12)

# ---------------------------------------------------------------------------
#  G2 — closed-form forcing ≡ generic AD strong-form evaluator.
#  Two independent codings of the same operator: one by hand, one by ForwardDiff.
# ---------------------------------------------------------------------------
println("\n--- G2: closed form ≡ AD ---")
f   = MMSField(Nσ; Lx=1.7, Ly=1.1, omega=1.3)
src = mms_forcing_stage1(f, vert, D_DEPTH, G_ACC)
cbs = field_callables(f)
worst_cf = 0.0
for x in (0.11, 0.63, 1.29), y in (0.07, 0.55, 0.98), t in (0.0, 0.41, 1.70)
    Lη, Lx, Ly = strong_residual_stage1(cbs, vert, D_DEPTH, G_ACC, x, y, t)
    p = VectorValue(x, y)
    Sη = src.Seta(p, t); Sx = src.Sx(p, t); Sy = src.Sy(p, t)
    global worst_cf = max(worst_cf, abs(Sη - Lη),
        maximum(abs.([Sx[i] - Lx[i] for i in 1:Nσ])),
        maximum(abs.([Sy[i] - Ly[i] for i in 1:Nσ])))
end
@printf("    max|S_closed − S_AD| = %.3e\n", worst_cf)
check("G2 closed-form forcing matches AD (<1e-10)", worst_cf < 1e-10)

# ---------------------------------------------------------------------------
#  G3 — a zero manufactured field must produce exactly zero forcing.
# ---------------------------------------------------------------------------
println("\n--- G3: zero field ⇒ zero forcing ---")
f0 = MMSField(Nσ; a_eta=0.0, alpha=zeros(Nσ), beta=zeros(Nσ),
              phi=zeros(Nσ), psi=zeros(Nσ), Lx=1.7, Ly=1.1)
s0 = mms_forcing_stage1(f0, vert, D_DEPTH, G_ACC)
p0 = VectorValue(0.3, 0.4)
z  = max(abs(s0.Seta(p0, 0.7)),
         maximum(abs.([s0.Sx(p0, 0.7)[i] for i in 1:Nσ])),
         maximum(abs.([s0.Sy(p0, 0.7)[i] for i in 1:Nσ])))
check("G3 zero field ⇒ S ≡ 0 exactly", z == 0.0)

# ---------------------------------------------------------------------------
#  G4 — the manufactured velocity satisfies the solid-wall conditions exactly.
#  This is what makes all three IBP boundary integrals vanish, so that no
#  boundary forcing is needed and the test is a pure interior-operator check.
# ---------------------------------------------------------------------------
println("\n--- G4: wall conditions ---")
wall = 0.0
for t in (0.0, 0.6, 1.4), yy in (0.0, 0.4, 1.1), j in 1:Nσ
    global wall = max(wall, abs(mms_ux(f, 0.0, yy, t, j)), abs(mms_ux(f, 1.7, yy, t, j)))
end
for t in (0.0, 0.6, 1.4), xx in (0.0, 0.9, 1.7), j in 1:Nσ
    global wall = max(wall, abs(mms_uy(f, xx, 0.0, t, j)), abs(mms_uy(f, xx, 1.1, t, j)))
end
@printf("    max|u*| on walls = %.3e\n", wall)
check("G4 u*ˣ=0 on x-walls, u*ʸ=0 on y-walls (<1e-14)", wall < 1e-14)

# ---------------------------------------------------------------------------
#  G5 — INDEPENDENCE, enforced mechanically.
#  The forcing module must never reach into the residual code; if it did, an
#  error in the residual would appear in the forcing too and cancel, silently
#  turning this whole exercise back into a consistency check.
# ---------------------------------------------------------------------------
println("\n--- G5: independence from problem.jl ---")
mms_src_path = joinpath(@__DIR__, "..", "src", "mms.jl")
body = read(mms_src_path, String)
banned = ["global_residual", "jacobian_u", "jacobian_u_t",
          "build_ode_operator", "nlp_native_contrib", "BALFEMProblem"]
hits = String[]
for b in banned
    # ignore prose mentions inside comment lines; flag real code references
    for ln in split(body, '\n')
        s = strip(ln)
        startswith(s, "#") && continue
        occursin(b, ln) && push!(hits, "$b  in:  $(strip(ln))")
    end
end
isempty(hits) || (println("    offending references:"); foreach(h -> println("      ", h), hits))
check("G5 src/mms.jl references no residual/assembly symbol", isempty(hits))

println()
println("="^64)
@printf("  Results: %d PASS,  %d FAIL\n", n_pass, n_fail)
println("="^64)
n_fail > 0 ? error("test_mms_forcing: $n_fail failed!") :
             println("  Analytic-MMS forcing verified (independent of the residual code).")
