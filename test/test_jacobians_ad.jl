# ==============================================================
#  test_jacobians_ad.jl — the hand-written Jacobians vs AUTOMATIC DIFFERENTIATION
#                         of the same residual, for EVERY model the solver builds
#
#  WHAT IT DOES.
#  For each physics configuration, it assembles the two Jacobian MATRICES the
#  time integrator actually uses —
#      A = ∂R/∂u    (jacobian_u)        B = ∂R/∂u̇   (jacobian_u_t)
#  — and compares them entry by entry against ForwardDiff differentiation of
#  `global_residual` itself, on the same state, same mesh, same quadrature.
#
#  WHY MATRICES AND NOT SOLUTIONS. Comparing end-to-end solutions is both far
#  slower (a whole time loop) and far blunter: Newton drives the RESIDUAL to
#  zero, so a converged run agrees to the tolerance even when the Jacobian is
#  wrong — a wrong Jacobian costs iterations, not accuracy. The matrix is where
#  the discrepancy actually lives, so this is the sharp instrument, and it needs
#  no time stepping at all.
#
#  HOW THE AD REFERENCE IS BUILT. `∂t(u)` is held FIXED as an ordinary CellField
#  while `u` is differentiated (and vice versa), so each derivative is taken at
#  frozen opposite argument — exactly the split the integrator forms as
#  J = ∂R/∂u + (1/aΔt)·∂R/∂u̇. This uses only STEADY multifield AD, so the
#  comparison is valid regardless of the state of the vendored Gridap fork
#  (which patches the TRANSIENT path). That independence is deliberate — but it
#  also means these gates do NOT cover the fork, so gate A0 checks it separately
#  and directly.
#
#  ⚠⚠ WHAT A PASS HERE DOES **NOT** MEAN.
#  AD differentiates the SAME assembled residual the hand Jacobians are written
#  for. Agreement proves residual↔Jacobian CONSISTENCY and says NOTHING about
#  whether the residual is the intended operator. That exact inference was made
#  in this project, recorded as fact, and retracted on 2026-08-15 — the residual
#  was at the time double-counting the 𝓐/𝓚 slope package, and AD agreed with the
#  hand Jacobian all the way. Residual CORRECTNESS is the analytic MMS's job
#  (test_mms_convergence*.jl), because only its forcing is derived independently
#  of problem.jl.
#
#  ⚠ THE CRITERION IS NOT THE SAME FOR EVERY MODEL, AND IT MUST NOT BE.
#  `src/problem.jl` documents the coverage explicitly:
#    * the LINEAR branch is EXACT — the residual is affine in (u,u̇) and every
#      assembled row has its exact derivative. Here we demand equality to
#      round-off. (This is the matrix-level counterpart of the one-Newton-
#      iteration gate in test_linear_newton_gate.jl.)
#    * the NONLINEAR branch is QUASI-NEWTON BY CHOICE — advection is
#      differentiated in full, but the leading- and slope-pressure packages
#      contribute no η-derivative, the 𝓐/𝓚 package is absent from ∂R/∂u̇
#      entirely, and the 𝓝 blocks add to the residual but not the Jacobian.
#      Demanding equality there would assert something the code deliberately
#      does not do, and the test would fail by design rather than on a defect.
#      So the gate is the one property that MUST hold anyway:
#
#          the hand↔AD discrepancy must VANISH as the state amplitude → 0.
#
#      Every omitted term is higher order in amplitude, so ‖ΔJ‖/‖J‖ ~ O(εᵏ) with
#      k ≥ 1. If instead a LEADING-ORDER term were wrong — a wrong coefficient,
#      a missing linear contribution — the relative discrepancy would tend to a
#      NONZERO CONSTANT. That is a real, non-brittle correctness gate on the
#      nonlinear branch, and it pins no magic numbers.
#
#  COST — READ THIS BEFORE RUNNING. The AD JIT compile is paid PER DISTINCT
#  BRANCH COMBINATION of the residual, NOT once per process: turning on
#  `flat_bed=false`, advection, or a 𝓝 tier each routes through code the previous
#  model never touched, so each pays its own compile. Measured on this machine:
#  M1 ~12 min, M1+M2 ~20 min, M1–M3 ~38 min. Budget an hour or more for the full
#  sweep. (An earlier version of this note claimed "~700 s once, then ~1 s each";
#  that was an extrapolation from a single model and it was wrong.)
#  Consequences: this file sits in the :slow tier of runtests.jl, progress is
#  flushed after every model so a long run is visibly alive, and
#  BALFEM_JAC_MODELS=M5,M6 finishes a clipped tail without redoing what passed.
#
#  RUN:  julia --project=. test/test_jacobians_ad.jl
#        BALFEM_JAC_MODELS=M5,M6,M7,M8 julia --project=. test/test_jacobians_ad.jl
# ==============================================================

using GridapBALFEM
using Gridap
using Gridap.ODEs
using Gridap.FESpaces
using Gridap.Algebra
using LinearAlgebra, SparseArrays, Printf

println("=" ^ 78)
println("  test_jacobians_ad.jl — hand Jacobians vs AD, every model")
println("=" ^ 78)
flush(stdout)

const G   = GridapBALFEM.g
const D0  = 2.5      # ≠ 1 deliberately: d = 1 makes the h-weighting invisible
const AB  = 0.2      # bed-slope amplitude; > 0 ⇒ a genuinely varying bed
const KBX = 1.3
const T0  = 0.37     # a generic time; nothing in the residual is special at t=0

#  A SLOPING bed is used throughout. Every bed-slope term (rows C3, M3-IBP, M10b,
#  M14 of the term classification) vanishes identically when ∇h ≡ 0, so a
#  flat-bed-only comparison is structurally incapable of testing that half of
#  either Jacobian — which is exactly the blind spot that hid a real defect for
#  weeks. `flat_bed=true` models still use this bed; the switch zeroes ∇h inside
#  the residual, which is the thing being tested.
h_bathy(x) = D0 * (1 + AB * sin(KBX * x[1]))

n_pass = 0; n_fail = 0
function check(name, cond, extra = "")
    global n_pass, n_fail
    cond ? (println("  PASS  $name $extra"); n_pass += 1) :
           (println("  FAIL  $name $extra"); n_fail += 1)
    flush(stdout)
end

# ---------------------------------------------------------------------------
#  Discretisation — deliberately tiny. This test compares matrices, so mesh size
#  buys nothing: a 3x3 mesh already exercises every term, interior and boundary,
#  and keeps the dense-ish comparison instant.
# ---------------------------------------------------------------------------
const NX, NY = 3, 3
const LX, LY = 1.7, 1.1

vert         = assemble_vertical_tensors(2, 1, [0.0, 0.728, 1.0])
const Nσ     = vert.N_dof
model, trian = build_horizontal_model(((0.0, LX), (0.0, LY)), (NX, NY))
U, V         = build_fe_spaces(model, 2, Nσ; y_wall_bc = :wall, x_wall_bc = true,
                               p_eta = 1)
dΩh          = Measure(trian, 6)

@printf("  mesh %dx%d   Nσ=%d   free DOFs=%d   bed h=%.2f(1+%.2f sin(%.1f x))\n",
        NX, NY, Nσ, num_free_dofs(U), D0, AB, KBX)
flush(stdout)

"""
Smooth, non-degenerate state scaled by `a`. Every field is nonzero and varies in
both x and y, and the per-layer entries differ from each other, so no term can
be accidentally annihilated by a symmetry of the test data.
"""
state_funs(a) = [
    x -> a * 0.30 * cos(1.1x[1]) * cos(0.7x[2]),
    x -> VectorValue(ntuple(j -> a * (0.20 + 0.05j) * sin(0.9x[1] + 0.3j) * cos(0.5x[2]), Nσ)...),
    x -> VectorValue(ntuple(j -> a * (0.15 - 0.03j) * cos(0.8x[1]) * sin(0.6x[2] + 0.2j), Nσ)...),
]

"""
Assemble (A_hand, B_hand, A_ad, B_ad) for `prob` at state amplitude `a`.

`u` and `u̇` are independent fields here, as they are for the integrator: the
two Jacobians are partial derivatives at frozen opposite argument.
"""
function jac_pair(prob, a)
    uh  = interpolate_everywhere(state_funs(a),       U)
    uth = interpolate_everywhere(state_funs(0.6 * a), U)
    tu  = TransientCellField(uh, (uth,))

    A_hand = assemble_matrix((du,  v) -> jacobian_u(  T0, tu, du,  v, prob, trian, dΩh), U, V)
    B_hand = assemble_matrix((dut, v) -> jacobian_u_t(T0, tu, dut, v, prob, trian, dΩh), U, V)

    #  AD reference. Differentiating w.r.t. the FIRST argument of the
    #  TransientCellField leaves ∂t(u) = uth frozen, giving ∂R/∂u; differentiating
    #  the stored derivative instead gives ∂R/∂u̇. Same residual function both times.
    A_ad = jacobian(FEOperator((x, v) -> global_residual(T0, TransientCellField(x, (uth,)),
                                                         v, prob, trian, dΩh), U, V), uh)
    B_ad = jacobian(FEOperator((w, v) -> global_residual(T0, TransientCellField(uh, (w,)),
                                                         v, prob, trian, dΩh), U, V), uth)
    return A_hand, B_hand, A_ad, B_ad
end

absdiff(H, D) = norm(Matrix(H) - Matrix(D))
reldiff(H, D) = absdiff(H, D) / max(norm(Matrix(D)), 1e-300)

#  ⚠ THE AMPLITUDE ORDER MUST BE MEASURED ON THE **ABSOLUTE** DIFFERENCE.
#  Using the relative one silently biases the exponent low whenever the gap is
#  large, because the denominator ‖A_ad‖ contains the very terms that are missing
#  and therefore grows with amplitude too: with ‖ΔA‖ ~ cε and ‖A_ad‖ ~ a + bε the
#  ratio behaves like cε/(a+bε), which flattens as bε approaches a. Measured
#  first-hand: the :full tier reported order 0.59/0.54 on the relative metric —
#  read as a FAIL — while the small-gap models reported a correct ~1.1 because
#  their denominator barely moved. The numerator alone scales as O(εᵏ) cleanly
#  and is what the gate is actually about. Relative values are still printed,
#  because they say how BIG the gap is; the order says whether it VANISHES.

# ---------------------------------------------------------------------------
#  The models. Every combination `resolve_physics` accepts:
#  `regime=:linear` with `nl_pressure≠:none` is rejected by the solver itself,
#  so the linear row has only the `:none` tier.
#
#  `exact = true` marks the configurations whose hand Jacobian is CLAIMED to be
#  the exact derivative; those are gated on equality. The rest are gated on the
#  amplitude-vanishing property (see the header).
# ---------------------------------------------------------------------------
all_models = [
    (tag="M1", name="M1  linear    / flat bed / :none  ", regime=:linear,    nlp=:none,   flat=true,  exact=true),
    (tag="M2", name="M2  linear    / var  bed / :none  ", regime=:linear,    nlp=:none,   flat=false, exact=true),
    (tag="M3", name="M3  nonlinear / flat bed / :none  ", regime=:nonlinear, nlp=:none,   flat=true,  exact=false),
    (tag="M4", name="M4  nonlinear / var  bed / :none  ", regime=:nonlinear, nlp=:none,   flat=false, exact=false),
    (tag="M5", name="M5  nonlinear / flat bed / :native", regime=:nonlinear, nlp=:native, flat=true,  exact=false),
    (tag="M6", name="M6  nonlinear / var  bed / :native", regime=:nonlinear, nlp=:native, flat=false, exact=false),
    (tag="M7", name="M7  nonlinear / flat bed / :full  ", regime=:nonlinear, nlp=:full,   flat=true,  exact=false),
    (tag="M8", name="M8  nonlinear / var  bed / :full  ", regime=:nonlinear, nlp=:full,   flat=false, exact=false),
]

#  Model selection, e.g. BALFEM_JAC_MODELS=M5,M6,M7,M8.
#  This exists because the AD compile is paid PER BRANCH COMBINATION, not once
#  per process (measured — see the cost note in the header), so the full sweep is
#  long enough that being able to finish a clipped tail without re-deriving the
#  models that already passed is worth the six lines.
const SEL = get(ENV, "BALFEM_JAC_MODELS", "all")
models = SEL == "all" ? all_models :
         let want = strip.(split(SEL, ","))
             bad = setdiff(want, [m.tag for m in all_models])
             isempty(bad) || error("test_jacobians_ad: unknown model tag(s): $(join(bad, ", "))")
             [m for m in all_models if m.tag in want]
         end
SEL == "all" || println("  MODEL SUBSET: $SEL  (partial run — not a full verification)")

const TOL_EXACT = 1e-10   # round-off is ~1e-15; this is a generous ceiling
const AMP       = 1.0     # reference state amplitude for the nonlinear scaling gate

#  `nl_pressure=:full` reads frozen L²-projections (π𝖲, π𝖻) off prob.nlp_state[].
#  With the state unset those blocks are silently SKIPPED, so half the tier would
#  go untested. Build the context and seed the state so both halves are live —
#  and note the frozen fields are constants to hand and AD alike, which is
#  exactly how the integrator sees them (they are lagged one step by design).
nlp_ctx = build_nlp_ctx(model, 2, Nσ, trian, dΩh)

# ---------------------------------------------------------------------------
#  A0 — the vendored Gridap fork is still in place.
#
#  This costs nothing and is checked HERE because the gates below do NOT cover
#  it. They differentiate at frozen opposite argument, which uses only STEADY
#  multifield AD; the fork patches the TRANSIENT path. Concretely, the fork adds
#  one additive outer constructor accepting a TUPLE as the third argument —
#  `time_derivative(::TransientMultiFieldCellField)` builds that argument with
#  `map(cellfield, derivatives...)`, and `map` yields a Tuple when fed a Tuple
#  (the AD path) but a Vector when fed an array-like MultiField (the hand path),
#  while the struct field is declared `Vector{<:TransientCellField}`.
#  Stock Gridap has only the Vector method, so this predicate is exactly the
#  fork's presence. See building_files/CONFIGURATION.md §2 — which is why
#  this one-line check, not that document, is the durable record.
# ---------------------------------------------------------------------------
println()
let T = Gridap.ODEs.TransientMultiFieldCellField
    check("A0  vendored Gridap fork present (Tuple-arg TransientMultiFieldCellField ctor)",
          hasmethod(T, Tuple{Any, Tuple, Tuple}),
          "\n        ⇒ if this fails, Manifest.toml no longer pins Elmanyer/Gridap.jl @" *
          "\n          fix-transient-multifield-ad and use_ad=true will MethodError.")
end

for m in models
    println("-" ^ 78)
    println("  $(m.name)   [", m.exact ? "EXACT: equality gate" :
                                         "QUASI-NEWTON: amplitude-vanishing gate", "]")
    flush(stdout)

    prob = build_problem(vert; g = G, h_bathy = h_bathy,
                         regime = m.regime, nl_pressure = m.nlp, flat_bed = m.flat,
                         mu_sponge = (x -> 0.0), wm_src = ((x, t) -> 0.0))

    #  Re-seed the frozen projections AT THE AMPLITUDE BEING TESTED. Leaving them
    #  at the reference amplitude while the state is halved would break the very
    #  scaling the nonlinear gate measures: the 𝓝-frozen contribution would stay
    #  O(1) while everything else shrank, and the measured order would be wrong.
    seed_nlp!(a) = m.nlp === :full &&
        update_nlp_state!(prob, nlp_ctx, interpolate_everywhere(state_funs(a), U))

    seed_nlp!(AMP)
    A_h, B_h, A_d, B_d = jac_pair(prob, AMP)
    rA = reldiff(A_h, A_d)
    rB = reldiff(B_h, B_d)
    @printf("      ‖ΔA‖/‖A‖ = %.3e      ‖ΔB‖/‖B‖ = %.3e\n", rA, rB)
    flush(stdout)

    if m.exact
        check("$(m.name) ∂R/∂u  ≡ AD",  rA < TOL_EXACT, @sprintf("(rel %.2e)", rA))
        check("$(m.name) ∂R/∂u̇ ≡ AD",  rB < TOL_EXACT, @sprintf("(rel %.2e)", rB))
    else
        #  Halving the amplitude must shrink the discrepancy. A wrong LEADING-order
        #  term would leave it flat; the deliberate quasi-Newton omissions are all
        #  higher order and must decay.
        seed_nlp!(AMP / 2)
        A_h2, B_h2, A_d2, B_d2 = jac_pair(prob, AMP / 2)
        #  Absolute norms for the ORDER (see the note at `absdiff`), relative for scale.
        aA, aB   = absdiff(A_h,  A_d),  absdiff(B_h,  B_d)
        aA2, aB2 = absdiff(A_h2, A_d2), absdiff(B_h2, B_d2)
        rA2, rB2 = reldiff(A_h2, A_d2), reldiff(B_h2, B_d2)
        kA  = aA2 > 0 ? log2(aA / aA2) : Inf
        kB  = aB2 > 0 ? log2(aB / aB2) : Inf
        @printf("      halving the amplitude (ABS):  ΔA %.3e → %.3e (order %.2f)   ΔB %.3e → %.3e (order %.2f)\n",
                aA, aA2, kA, aB, aB2, kB)
        @printf("      (relative, for scale:         ΔA %.3e → %.3e             ΔB %.3e → %.3e)\n",
                rA, rA2, rB, rB2)
        flush(stdout)
        #  order ≥ 0.8 ⇒ genuinely vanishing (allow slack: several omitted terms
        #  of different orders superpose, so the measured exponent is a mixture).
        check("$(m.name) ∂R/∂u  gap vanishes with amplitude",
              kA > 0.8 || rA < TOL_EXACT, @sprintf("(order %.2f)", kA))
        check("$(m.name) ∂R/∂u̇ gap vanishes with amplitude",
              kB > 0.8 || rB < TOL_EXACT, @sprintf("(order %.2f)", kB))
    end
end

println()
println("=" ^ 78)
@printf("  Results: %d PASS,  %d FAIL\n", n_pass, n_fail)
println("=" ^ 78)
println("  Reminder: this is residual↔Jacobian CONSISTENCY, not residual correctness.")
println("  For correctness see test_mms_convergence.jl / test_mms_convergence_nonlinear.jl.")
n_fail > 0 ? error("test_jacobians_ad: $n_fail failed!") :
             println("  Hand Jacobians are consistent with the residual in every model.")
