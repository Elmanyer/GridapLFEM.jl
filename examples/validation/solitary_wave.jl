# ==============================================================
#  solitary_wave.jl — solitary-wave propagation (nonlinearity–dispersion
#                     balance), algebraic-package port of the oracle case
#
#  The defining property of a solitary wave is a PERMANENT balance of
#  nonlinearity and dispersion: a sech² hump of amplitude A over depth d
#  propagates at celerity c ≈ √(g(d+A)) with preserved shape. This exercises
#  the fully-nonlinear momentum (advection ON) natively in the stacked layout —
#  no matrix-free V⊗H loop is needed. An initial-condition problem, so a CLOSED
#  x-wall is MANDATORY (x_wall_bc=true): free x-walls + the non-hydrostatic
#  B-term is a spurious-forcing instability an IC perturbation excites. A sponge
#  absorbs the soliton before the solid wall.
#
#  We track the crest position and peak amplitude along a dense gauge line and
#  report the measured celerity and amplitude retention.
#  Oracle reference (A/d=0.05): celerity err ≈1.3%, ~100% amplitude retained.
#
#  RUN:  julia --project=. GridapLFEM.jl/examples/validation/solitary_wave.jl
# ==============================================================

if !isdefined(Main, :GridapLFEM)
    include(joinpath(@__DIR__, "..", "..", "src", "GridapLFEM.jl"))
end
using .GridapLFEM
using Gridap
using Printf, LinearAlgebra

println("=" ^ 64)
println("  solitary_wave.jl — sech² soliton, celerity & shape retention")
println("=" ^ 64)

g = 9.81; d = 1.0; A = 0.05; x0 = 12.0
c_th = sqrt(g*(d + A))                       # solitary-wave celerity
κ    = sqrt(3A/(4d^3))                        # classical sech² width
@printf("  A/d=%.2f  c_th=%.3f m/s  width 1/κ=%.1f m\n", A/d, c_th, 1/κ)

vert = assemble_vertical_tensors(2, 1, [0.0, 0.728, 1.0]); Nσ = vert.N_dof
domain = ((0.0, 50.0), (0.0, 2.0)); p_horizontal = 2
model, trian = build_horizontal_model(domain, (120, 4))     # ~12 cells across the soliton
dO = Measure(trian, 2*p_horizontal + 2)
U, V = build_fe_spaces(model, p_horizontal, Nσ; y_wall_bc=:wall, x_wall_bc=true)   # closed x!

sponge = make_sponge(domain, 0.0, 12.0, 0.0, 0.0, 8.0)          # absorb at the far end
prob = build_problem(vert; g=g, h_bathy=x -> d, linearised=false, advection=true,
                     mu_sponge=sponge, wm_src=(x,t) -> 0.0)

# IC: η = A sech²(κ(x−x0)); uⱼˣ = c η/(d+η) (uniform over modes); uⱼʸ = 0
sech2(z) = 1/cosh(z)^2
etaf(x)  = A*sech2(κ*(x[1]-x0))
uxf(x)   = c_th*etaf(x)/(d + etaf(x))
zvv      = VectorValue(ntuple(_->0.0, Nσ)...)
u0 = interpolate_everywhere([etaf, x -> VectorValue(ntuple(_->uxf(x), Nσ)...),
                                    x -> zvv], U)

xg = collect(3.0:0.5:40.0); gauges = [(xx, 1.0) for xx in xg]
dt = 0.025; T_final = 4.0                                    # fully nonlinear ⇒ multi-minute run
op     = build_ode_operator(prob, U, V, trian, dO)
solver = build_ode_solver(dt)
diags  = run_time_loop(op, solver, u0, 0.0, T_final;
                       trian=trian, Nσ=Nσ, gauges=gauges, dt=dt,
                       save_every=0, print_every=25)

# crest position & amplitude from the gauge line
ts = [d.t for d in diags]; cx = Float64[]; cA = Float64[]
for dd in diags
    gv = dd.gauge_vals; im = argmax(gv)
    push!(cx, xg[im]); push!(cA, gv[im])
end
# celerity from a linear fit of crest position over the mid part of the run (pre-sponge)
w = findall(t -> 0.5 < t < T_final-0.8, ts)
c_meas = (cx[w[end]] - cx[w[1]]) / (ts[w[end]] - ts[w[1]])
A_ret  = cA[w[end]] / A
@printf("\n  measured celerity = %.3f m/s (theory %.3f, err %.1f%%)\n",
        c_meas, c_th, 100*abs(c_meas/c_th - 1))
@printf("  amplitude retention = %.1f%%  (crest %.5f → %.5f m)\n",
        100*A_ret, cA[w[1]], cA[w[end]])

n_fail = 0
(abs(c_meas/c_th - 1) < 0.05) ? println("  PASS  celerity within 5% of √(g(d+A))") :
                                (println("  FAIL  celerity off"); global n_fail+=1)
(0.85 < A_ret < 1.15)         ? println("  PASS  amplitude retained (shape preserved)") :
                                (println("  FAIL  amplitude not retained"); global n_fail+=1)
println("=" ^ 64)
n_fail == 0 ? println("  Solitary wave propagates with balanced nonlinearity–dispersion.") :
              println("  (refine mesh/dt; check x_wall_bc and sponge placement)")
