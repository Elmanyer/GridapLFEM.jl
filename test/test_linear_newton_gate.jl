# ==============================================================
#  test_linear_newton_gate.jl — "a linear problem converges in ONE Newton
#  iteration per stage" (SMALL & FAST)
#
#  WHY THIS TEST EXISTS.
#  Under `regime=:linear` the residual is AFFINE in (u, u̇). If the hand
#  Jacobians are its exact derivatives, Newton reaches round-off in exactly ONE
#  iteration per implicit stage, from any initial guess, at any amplitude, over
#  any bathymetry. Any excess iteration count is therefore PROOF of a
#  residual↔Jacobian inconsistency — a free, sharp gate that needs no reference
#  value and no oracle.
#
#  WHAT IT PROTECTS.
#  The sloping-bed cases below are the ones the rest of the suite cannot reach.
#  Every bed-slope term of the model (rows C3, M3-IBP, M10b, M14 of the term
#  classification in GridapImplementation.tex `tab: term classification`)
#  vanishes identically when ∇h ≡ 0, so a flat-bed regression is STRUCTURALLY
#  incapable of testing them. That blind spot let a real defect survive the
#  whole suite: the 𝓐/𝓚 slope-pressure package was assembled TWICE under
#  `regime=:linear, flat_bed=false` (once in its linearised form M14, once in
#  its nonlinear form M14–M18), which stalled Newton at 50 iterations.
#  See building_files/RESIDUAL_TERM_AUDIT_PLAN.md.
#
#  Expected: nl_iters == n_stages per step (θ: 1 stage, SDIRK_2_2: 2 stages),
#  with the final nonlinear residual at round-off (~1e-15).
#
#  RUN:  julia --project=. test/test_linear_newton_gate.jl
# ==============================================================

using GridapBALFEM
using Gridap
using Printf

println("=" ^ 72)
println("  test_linear_newton_gate.jl — linear regime ⇒ 1 Newton iteration/stage")
println("=" ^ 72)

const G   = GridapBALFEM.g
const D0  = 2.5      # ≠ 1 deliberately: d = 1 makes the h-weighting invisible
const AB  = 0.2      # bed-slope amplitude; > 0 ⇒ an actually varying bed
const KBX = 1.3

h_slope(x) = D0 * (1 + AB * sin(KBX * x[1]))
h_flat(x)  = D0

"""
Run `nsteps` of a forced linear MMS problem and return the per-step Newton
iteration counts and the worst final nonlinear residual.
"""
function linear_newton_counts(; hfun, flat_bed, solver_type, nsteps = 4, dt = 2e-3,
                                nx = 6, ny = 6, p_u = 2, p_e = 1,
                                Lx = 1.7, Ly = 1.1)
    vert = assemble_vertical_tensors(2, 1, [0.0, 0.728, 1.0])
    f    = MMSField(vert.N_dof; Lx = Lx, Ly = Ly, omega = 1.3, ky = 0.0)

    model, trian = build_horizontal_model(((0.0, Lx), (0.0, Ly)), (nx, ny))
    U, V = build_fe_spaces(model, p_u, vert.N_dof;
                           y_wall_bc = :wall, x_wall_bc = true, p_eta = p_e)
    dΩh  = Measure(trian, 2 * max(p_u, p_e) + 2)

    #  One bathymetry object feeds BOTH the forcing and the solver, and the same
    #  flat_bed symbol selects both, so the two cannot describe different models.
    src  = mms_forcing(f, vert, (x, y) -> hfun(VectorValue(x, y)), G;
                       regime = :linear, flat_bed = flat_bed)
    prob = build_problem(vert; g = G, h_bathy = hfun,
                         regime = :linear, nl_pressure = :none, flat_bed = flat_bed,
                         mu_sponge = (x -> 0.0), wm_src = ((x, t) -> 0.0),
                         mms_src = src)

    op  = build_ode_operator(prob, U, V, trian, dΩh)     # HAND Jacobians (the point)
    mon = SolverMonitor()
    slv = build_ode_solver(dt; solver_type = solver_type, nl_iter = 50,
                           nl_tol = 1e-12, monitor = mon)
    u0  = interpolate_everywhere([mms_exact_eta(f, 0.0),
                                  mms_exact_ux(f, 0.0),
                                  mms_exact_uy(f, 0.0)], U)

    diags = run_time_loop(op, slv, u0, 0.0, dt * nsteps;
                          output_dir = mktempdir(), save_every = 0,
                          trian = trian, Nσ = vert.N_dof,
                          print_every = typemax(Int), dt = dt,
                          monitor = mon, diag_every = -1, check_every = 0)

    return (iters = [d.nl_iters for d in diags],
            res   = maximum(d.res_nl for d in diags))
end

#  (label, bathymetry, flat_bed, solver, stages/step)
cases = [
    ("flat bed,    flat_bed=true ", h_flat,  true,  :theta, 1),
    ("flat bed,    flat_bed=true ", h_flat,  true,  :sdirk, 2),
    ("flat bed,    flat_bed=false", h_flat,  false, :theta, 1),
    ("sloping bed, flat_bed=false", h_slope, false, :theta, 1),
    ("sloping bed, flat_bed=false", h_slope, false, :sdirk, 2),
]

const RES_TOL = 1e-10        # round-off is ~1e-15; this is a generous ceiling
npass = 0; ntot = 0

for (label, hfun, flat_bed, st, nstage) in cases
    r = linear_newton_counts(hfun = hfun, flat_bed = flat_bed, solver_type = st)

    ok_iters = all(==(nstage), r.iters)
    ok_res   = r.res < RES_TOL
    global ntot += 2
    global npass += ok_iters + ok_res

    @printf("  %-28s %-6s  iters/step=%-12s (expect %d)  %s\n",
            label, st, string(r.iters), nstage, ok_iters ? "PASS" : "FAIL")
    @printf("  %-28s %-6s  max residual = %.3e                  %s\n",
            "", "", r.res, ok_res ? "PASS" : "FAIL")

    if !ok_iters
        println("    ⚠ excess Newton iterations ⇒ the hand Jacobian is NOT the exact")
        println("      derivative of the residual in this configuration.")
    end
end

println("-" ^ 72)
@printf("  %d/%d checks passed\n", npass, ntot)
println(npass == ntot ? "  test_linear_newton_gate.jl: PASS" :
                        "  test_linear_newton_gate.jl: FAIL")
println("=" ^ 72)
npass == ntot || error("test_linear_newton_gate.jl FAILED")
