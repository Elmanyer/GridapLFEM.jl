# ==============================================================
#  stokes_harmonics.jl — nonlinear Stokes wave: bound-harmonic generation
#                        (Yang & Liu 2024 §4, deep-/intermediate-water case)
#
#  A finite-amplitude regular wave is not a pure sinusoid: nonlinear self-
#  interaction generates a bound (phase-locked) second harmonic. To second
#  order (Stokes),
#     η = a₁cosψ + a₂cos2ψ,   a₂ = a₁²k · cosh(kd)(2+cosh2kd)/(4 sinh³kd),
#  and the bound 2ω component travels at the PRIMARY celerity (not the free
#  2ω celerity). This exercises the full nonlinear pressure package. We DFT a
#  downstream gauge at ω and 2ω and compare the measured a₂/a₁ to Stokes theory.
#
#  QUICK default (short flume, ~15 periods). Production: longer flume, more
#  periods, finer mesh, and a resolved second harmonic (≥8 cells per λ/2).
#
#  RUN:  julia --project=. GridapLFEM.jl/examples/validation/stokes_harmonics.jl
# ==============================================================

if !isdefined(Main, :GridapLFEM)
    include(joinpath(@__DIR__, "..", "..", "src", "GridapLFEM.jl"))
end
using .GridapLFEM
using Printf, LinearAlgebra

println("=" ^ 64)
println("  stokes_harmonics.jl — bound second harmonic vs Stokes theory")
println("=" ^ 64)

g = 9.81; d = 3.5; T = 2.0; A = 0.02          # finite amplitude (kA ≈ 0.05)
omega = 2π/T; k = find_wavenumber(omega, d, g); lam = 2π/k; kd = k*d
@printf("  d=%.1f  T=%.1f  A=%.3f  kd=%.2f  λ=%.2f  kA=%.3f\n", d,T,A,kd,lam,k*A)

# Stokes 2nd-order bound-harmonic ratio a₂/a₁
a2_over_a1_stokes = A*k * cosh(kd)*(2 + cosh(2kd)) / (4*sinh(kd)^3)
@printf("  Stokes 2nd-order  a₂/a₁ = %.4f\n", a2_over_a1_stokes)

x_wm = 12.0; Lx = x_wm + 8lam + 12.0; Ly = 2.0
nx = round(Int, Lx/(lam/12)); ny = 2               # ≥12 cells/λ (resolves 2ω)
dt = T/40; T_final = 15*T
x_g = x_wm + 4lam; y_g = Ly/2
@printf("  domain %.0f×%.0f, %d×%d cells, gauge at x=%.1f\n", Lx,Ly,nx,ny,x_g)

diags, _, _ = setup_and_run(
    M=2, c_bdy=[0.0,0.728,1.0], domain=((0.0,Lx),(0.0,Ly)), partition=(nx,ny),
    fe_order=2, d_val=d, T_wave=T, A_wave=A, x_wm=x_wm, y_wm=nothing,
    sponge_wL=12.0, sponge_wR=12.0, mu_max=20.0, T_final=T_final, dt=dt,
    save_every=0, gauges=[(x_g, y_g)],
    linearised=false, advection=true,
    lin_pressure=true, P_full=true, nl_pressure68=true, nl_pressure_full=true,
    print_every=200)

# DFT the steady (second-half) gauge signal at ω and 2ω
ts = [d.t for d in diags]; gs = [d.gauge_vals[1] for d in diags]
i0 = length(ts) ÷ 2; tv = ts[i0:end]; gv = gs[i0:end]
amp(w) = 2*abs(sum(gv .* exp.(-im .* w .* tv))) / length(gv)
a1 = amp(omega); a2 = amp(2omega)
ratio = a2/a1
@printf("\n  measured: a₁=%.5f  a₂=%.5f  a₂/a₁=%.4f  (Stokes %.4f)\n",
        a1, a2, ratio, a2_over_a1_stokes)

n_fail = 0
(a1 > 0.5A && a1 < 2A) ? println("  PASS  primary harmonic amplitude ≈ A") :
                         (println("  FAIL  primary amplitude off ($a1)"); global n_fail+=1)
rel = abs(ratio/a2_over_a1_stokes - 1)
(rel < 0.6) ? println("  PASS  bound 2nd harmonic within 60% of Stokes ($(round(100rel))%)") :
              (println("  FAIL  a₂/a₁ far from Stokes ($(round(100rel))%)"); global n_fail+=1)
println("=" ^ 64)
n_fail == 0 ? println("  Nonlinear bound-harmonic generation reproduced.") :
              println("  (tune amplitude/mesh/length; near-field & generation bias a₂/a₁)")
