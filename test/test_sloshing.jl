# ==============================================================
#  test_sloshing.jl — standing-wave dynamics (SMALL & FAST)
#
#  Port of ../../LFE-M_2D_solver/test/test_sloshing_lfem2D.jl.
#  Closed basin [0,L], first sloshing mode η₀ = A cos(πx/L), u = 0 IC.
#  The standing wave must oscillate at the LFE-M dispersion frequency at
#  k = π/L:  c² = g d · Φᵀ(M − (kd)²B)⁻¹Φ,  T = 2π/(kc). We measure the
#  numerical period at the antinode gauge (DFT peak) and compare (<5%).
#
#  RUN:  julia --project=. GridapLFEM.jl/test/test_sloshing.jl
# ==============================================================

using GridapLFEM
using Printf, LinearAlgebra

println("=" ^ 60)
println("  test_sloshing.jl — standing wave vs dispersion theory")
println("=" ^ 60)

g = 9.81; h_val = 1.0; L = 2.0; Ly = 0.5
vert = assemble_vertical_tensors(2, 1, [0.0, 0.728, 1.0])

# theory: ω from the LFE-M dispersion at k = π/L
k = π / L; kd = k * h_val
Meff = vert.Mmat .- (kd^2) .* vert.B                 # M − (kd)²B,  B ≤ 0
c2 = g * h_val * dot(vert.Phi, Meff \ vert.Phi)
omega_th = k * sqrt(c2); T_th = 2π / omega_th
@printf("  k=%.4f  kd=%.3f  c=%.4f m/s  ω_th=%.4f  T_th=%.4f s\n",
        k, kd, sqrt(c2), omega_th, T_th)

A0 = 0.005
eta0(x) = A0 * cos(π * x[1] / L)
dt = T_th / 40; Tf = 3 * T_th

# closed basin (x_wall_bc MANDATORY for IC problems); no wavemaker, no sponge
diags, _, _ = setup_and_run(
    M=2, c_bdy=[0.0, 0.728, 1.0], h_val=h_val, g=g,
    domain=((0.0, L), (0.0, Ly)), partition=(24, 2), p_horizontal=2,
    T_wave=1.0, A_wave=0.0, x_wm=-1e6,
    sponge_wL=0.0, sponge_wR=0.0, sponge_wB=0.0, sponge_wT=0.0, mu_max=0.0,
    T_final=Tf, dt=dt, eta0_func=eta0,
    regime=:linear,
    y_wall_bc=:wall, x_wall_bc=true,
    gauges=[(0.05, Ly/2)], save_every=0, nl_tol=1e-8)

ts = [d.t for d in diags]; gs = [d.gauge_vals[1] for d in diags]

# ω_num: peak of |Σ η exp(−iωt)| in a sweep around ω_th
ωsweep = LinRange(0.3*omega_th, 2.0*omega_th, 4000)
amp(ω) = abs(sum(gs .* exp.(-im .* ω .* ts)))
ω_num = ωsweep[argmax(amp.(ωsweep))]
T_num = 2π / ω_num
err = abs(T_num/T_th - 1)
@printf("  steps=%d  T_num=%.4f s  err=%.2f%%\n", length(ts), T_num, 100*err)

n_fail = 0
(err < 0.05) ? println("  PASS  standing-wave period matches dispersion theory (<5%)") :
               (println("  FAIL  period err $(round(100err, digits=2))%"); global n_fail += 1)
mx = maximum(abs.(gs))
(!isnan(mx) && mx < 5*A0) ? println("  PASS  bounded / no NaN") :
               (println("  FAIL  unbounded ($mx)"); global n_fail += 1)

println("=" ^ 60)
n_fail > 0 ? error("test_sloshing: $n_fail failed!") :
             println("  Standing-wave dynamics validated.")
