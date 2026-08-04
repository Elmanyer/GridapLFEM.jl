# ==============================================================
#  bichromatic_sideband.jl — deep-water bichromatic wave group
#                            (Yang & Liu 2024 §4, cases 2–3: bichromatic
#                             groups / sideband instability) — ADVANCED
#
#  A two-frequency wavemaker (ω₁,ω₂ close) generates a wave GROUP whose envelope
#  evolves through sum/difference nonlinear interactions. In deep water a single
#  carrier seeded with small sidebands additionally exhibits Benjamin–Feir
#  modulational growth. These are the most demanding LFE-M validations: deep
#  water (large kd — use LFE-3/LFE-4 to stay inside the applicable band), fine
#  resolution, LONG domains and runs, and the full nonlinear pressure.
#
#  This script is a SCAFFOLD: it builds a custom bichromatic mass source via the
#  low-level API (setup_and_run only exposes a single frequency), runs a modest
#  case, and checks that BOTH forcing frequencies appear in a downstream gauge
#  and the run stays bounded. For a quantitative envelope / BF-growth study,
#  enlarge the domain to many group lengths, extend the run, and (for sidebands)
#  replace the source by a carrier + seeded ±δω perturbation.
#
#  RUN:  julia --project=. GridapLFEM.jl/examples/validation/bichromatic_sideband.jl
# ==============================================================

using GridapLFEM
using Gridap
using Printf, LinearAlgebra

println("=" ^ 64)
println("  bichromatic_sideband.jl — deep-water two-frequency group (scaffold)")
println("=" ^ 64)

g = 9.81; d = 3.5
# two close periods → carrier + slow group envelope; deep water (kd large)
T1, T2 = 1.10, 1.25; A1, A2 = 0.0006, 0.0006
ω1, ω2 = 2π/T1, 2π/T2
k1 = find_wavenumber(ω1, d, g); k2 = find_wavenumber(ω2, d, g)
lam = 2π/max(k1,k2); kd1 = k1*d
@printf("  T₁=%.2f (kd=%.1f)  T₂=%.2f  Δω=%.3f  group T=%.1f s\n",
        T1, kd1, T2, abs(ω1-ω2), 2π/abs(ω1-ω2))
# LFE-3 needed for deep water (LFE-2 applicable only to kd≈10.9)
M = kd1 > 9 ? 3 : 2
c_bdy = M == 3 ? [0.0,0.726,0.925,1.0] : [0.0,0.728,1.0]

x_wm = 12.0; Lx = x_wm + 12lam + 15.0; Ly = 2.0; σ_wm = 1.5
vert = assemble_vertical_tensors(M, 1, c_bdy); Nσ = vert.N_dof
p_horizontal = 2
model, trian = build_horizontal_model(((0.0,Lx),(0.0,Ly)),
                                      (round(Int, Lx/(lam/12)), 2))
dΩh = Measure(trian, 2*p_horizontal + 2)
U, V = build_fe_spaces(model, p_horizontal, Nσ; y_wall_bc=:wall, x_wall_bc=false)

# custom bichromatic Gaussian mass source (two frequencies superposed)
bwm(x, t) = 2*exp(-((x[1]-x_wm)/σ_wm)^2)*(A1*ω1*cos(ω1*t) + A2*ω2*cos(ω2*t))
sponge = make_sponge(((0.0,Lx),(0.0,Ly)), 15.0, 15.0, 0.0, 0.0, 8.0)
prob = build_problem(vert; g=g, h_bathy=x -> d, regime=:nonlinear,
                     nl_pressure=:native, flat_bed=true,   # constant-depth (flat bed)
                     mu_sponge=sponge, wm_src=bwm)

x_g = x_wm + 6lam
u0 = make_initial_conditions(U, Nσ)
op     = build_ode_operator(prob, U, V, trian, dΩh)
solver = build_ode_solver(0.02)
diags  = run_time_loop(op, solver, u0, 0.0, 25*max(T1,T2);
                       trian=trian, Nσ=Nσ, gauges=[(x_g, Ly/2)], dt=0.02,
                       save_every=0, print_every=100)

ts = [d.t for d in diags]; gs = [d.gauge_vals[1] for d in diags]
i0 = length(ts) ÷ 2; tv = ts[i0:end]; gv = gs[i0:end]
amp(w) = 2*abs(sum(gv .* exp.(-im .* w .* tv)))/length(gv)
a1 = amp(ω1); a2 = amp(ω2); emax = maximum(d.eta_max for d in diags)
@printf("\n  gauge x=%.1f:  amp(ω₁)=%.5f  amp(ω₂)=%.5f  max η=%.5f\n", x_g, a1, a2, emax)

n_fail = 0
(a1 > 0.2*A1 && a2 > 0.2*A2) ? println("  PASS  both forcing frequencies present in the group") :
                               (println("  FAIL  a component missing"); global n_fail+=1)
(!isnan(emax) && emax < 20*max(A1,A2)) ? println("  PASS  bounded / no NaN") :
                               (println("  FAIL  unbounded"); global n_fail+=1)
println("=" ^ 64)
println("  Scaffold OK. For quantitative envelope / Benjamin–Feir growth, extend")
println("  domain (many group lengths) & run length, and seed sidebands on a carrier.")
n_fail == 0 || error("bichromatic_sideband: $n_fail failed!")
