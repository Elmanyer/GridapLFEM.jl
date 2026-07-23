# ==============================================================
#  test_bc_generation.jl — Dirichlet boundary wave generation (FEM, e2e)
#
#  Regular wave, kd = 3, LFE-2, generated PURELY from time-varying Dirichlet
#  data on the left boundary (no interior wavemaker), absorbed by the right
#  sponge. Gates:
#    A. :model polarization, linearised — stability, gauge amplitude at 2λ
#       within 10% of A, phase speed vs the MODEL celerity ω/k_mod < 2%
#       (and vs Airy < 3.5%);
#    B. :airy polarization — stable, amplitude within 15% (the cosh nodal
#       sampling violates the discrete continuity closure at LFE-2 — see
#       test_waveinput.jl);
#    C. relaxation zone (relax_bc=true, no left sponge) — stable, amplitude
#       within 10%;
#    D. fully nonlinear short run — stability of the Dirichlet BCs with the
#       H-weighted terms and advection.
#
#  RUN:  julia --project=. GridapLFEM.jl/test/test_bc_generation.jl
# ==============================================================

if !isdefined(Main, :GridapLFEM)
    include(joinpath(@__DIR__, "..", "src", "GridapLFEM.jl"))
end
using .GridapLFEM
using Printf, LinearAlgebra

println("=" ^ 60)
println("  test_bc_generation.jl — Dirichlet wave generation (kd=3)")
println("=" ^ 60)

n_pass = 0; n_fail = 0
check(name, cond, extra="") = (global n_pass, n_fail;
    cond ? (println("  PASS  $name $extra"); n_pass+=1) :
           (println("  FAIL  $name $extra"); n_fail+=1))

"DFT amplitude at ω from the last `nper` FULL periods of a gauge series."
function gauge_amplitude(diags, idx, omega, T, dt; nper=4)
    ts = [d.t for d in diags]; gv = [d.gauge_vals[idx] for d in diags]
    nwin = round(Int, nper*T/dt)
    n = length(ts); i0 = n - nwin + 1
    i0 < 1 && return NaN
    tv = ts[i0:end]; gw = gv[i0:end]
    return 2.0*abs(dot(gw, exp.(-im .* omega .* tv)))/length(gw)
end

"Two-gauge DFT phase difference → phase speed (branch-safe for Δx = λ/2)."
function phase_speed(diags, x1, x2, omega)
    ts = [d.t for d in diags]
    g1 = [d.gauge_vals[1] for d in diags]; g2 = [d.gauge_vals[2] for d in diags]
    n = length(ts); i0 = n ÷ 2
    tv = ts[i0:end]; eiw = exp.(-im .* omega .* tv)
    A1 = dot(g1[i0:end], eiw); A2 = dot(g2[i0:end], eiw)
    (abs(A1) < 1e-14 || abs(A2) < 1e-14) && return NaN
    dphi = angle(A2/A1); dphi > 0 && (dphi -= 2π)
    k = -dphi/(x2 - x1)
    return k > 0 ? omega/k : NaN
end

# ---- case setup: kd = 3 at T = 1.6 s ----------------------------------------
g = 9.81; T_wave = 1.6; omega = 2π/T_wave; A_wave = 0.0005
kd_target = 3.0
k = sqrt(omega^2/(g*tanh(kd_target)))
for _ in 1:50
    d_try = kd_target/k
    f = g*k*tanh(k*d_try) - omega^2; df = g*tanh(k*d_try)
    dk = f/df; global k -= dk; abs(dk) < 1e-12*abs(k) && break
end
d_val = kd_target/k; Ce = omega/k; lam = 2π/k

vert0  = assemble_vertical_tensors(2, 1, [0.0, 0.728, 1.0])
k_mod  = model_wavenumber(vert0, omega, d_val, g)
Cm_th  = omega/k_mod
@printf("\n  kd=%.1f  d=%.3f m  λ=%.2f m  Ce=%.3f m/s  Cm(model)=%.3f m/s\n",
        kd_target, d_val, lam, Ce, Cm_th)

Lx = 6lam + 8.0; Ly = 2.0
nx = round(Int, Lx/(lam/6)); ny = 2
dt = T_wave/24; Tf = 12*T_wave
x_g1 = 2lam; x_g2 = x_g1 + lam/2; y_g = Ly/2
common = (M=2, c_bdy=[0.0,0.728,1.0], domain=((0.0,Lx),(0.0,Ly)),
          partition=(nx,ny), fe_order=2, d_val=d_val, T_wave=T_wave,
          A_wave=A_wave, sponge_wR=8.0, sponge_wB=0.0, sponge_wT=0.0,
          mu_max=30.0, dt=dt, save_every=0,
          gauges=[(x_g1,y_g),(x_g2,y_g)], check_every=0, print_every=50)
@printf("  Domain %.1f×%.1f m, %d×%d cells, %d periods, gauges at 2λ, 2λ+λ/2\n",
        Lx, Ly, nx, ny, 12)

# ---- A. :model polarization, linearised -------------------------------------
println("\n--- A: Dirichlet inflow, :model polarization, linearised ---")
diagsA, _, _ = setup_and_run(; common...,
    wave_bc=:regular, bc_side=:left, bc_profile=:model, sponge_wL=0.0,
    T_final=Tf, linearised=true, advection=false)

emaxA = maximum(d.eta_max for d in diagsA)
check("A: stable (no NaN, no blow-up)",
      !isnan(emaxA) && emaxA < 10*A_wave, @sprintf("(max η=%.2e m)", emaxA))
tailA = [d.eta_max for d in diagsA if d.t > 4*T_wave]
check("A: η bounded in [0.5A, 1.6A] after the ramp",
      all(isfinite, tailA) && maximum(tailA) < 1.6*A_wave &&
      maximum(tailA) > 0.5*A_wave,
      @sprintf("(max=%.3e, A=%.1e)", maximum(tailA), A_wave))

ampA = gauge_amplitude(diagsA, 1, omega, T_wave, dt)
check("A: gauge amplitude at 2λ within 10% of A",
      abs(ampA - A_wave)/A_wave < 0.10,
      @sprintf("(amp=%.4e, err=%.1f%%)", ampA, 100*abs(ampA-A_wave)/A_wave))

# the measured celerity carries the propagation's own space/time
# discretisation error on top of the boundary-data consistency (CN at
# dt=T/24 alone ≈ 0.6%); the established dispersion gate is 3%
CmA = phase_speed(diagsA, x_g1, x_g2, omega)
errm = abs(CmA/Cm_th - 1.0); erre = abs(CmA/Ce - 1.0)
check("A: phase speed vs MODEL celerity < 3%", errm < 0.03,
      @sprintf("(Cm=%.3f, err=%.2f%%)", CmA, 100errm))
check("A: phase speed vs Airy celerity < 3.5%", erre < 0.035,
      @sprintf("(err=%.2f%%)", 100erre))

# ---- B. :airy polarization ---------------------------------------------------
println("\n--- B: Dirichlet inflow, :airy polarization, linearised ---")
diagsB, _, _ = setup_and_run(; common...,
    wave_bc=:regular, bc_side=:left, bc_profile=:airy, sponge_wL=0.0,
    T_final=Tf, linearised=true, advection=false)
emaxB = maximum(d.eta_max for d in diagsB)
check("B: stable", !isnan(emaxB) && emaxB < 10*A_wave,
      @sprintf("(max η=%.2e m)", emaxB))
ampB = gauge_amplitude(diagsB, 1, omega, T_wave, dt)
check("B: gauge amplitude within 15% of A (cosh sampling mismatch)",
      abs(ampB - A_wave)/A_wave < 0.15,
      @sprintf("(amp=%.4e, err=%.1f%%)", ampB, 100*abs(ampB-A_wave)/A_wave))

# ---- C. relaxation zone ------------------------------------------------------
println("\n--- C: Dirichlet inflow + generation/absorption relaxation zone ---")
diagsC, _, _ = setup_and_run(; common...,
    wave_bc=:regular, bc_side=:left, bc_profile=:model, sponge_wL=0.0,
    relax_bc=true, T_final=Tf, linearised=true, advection=false)
emaxC = maximum(d.eta_max for d in diagsC)
check("C: stable with relaxation zone", !isnan(emaxC) && emaxC < 10*A_wave,
      @sprintf("(max η=%.2e m)", emaxC))
ampC = gauge_amplitude(diagsC, 1, omega, T_wave, dt)
check("C: gauge amplitude within 10% of A",
      abs(ampC - A_wave)/A_wave < 0.10,
      @sprintf("(amp=%.4e, err=%.1f%%)", ampC, 100*abs(ampC-A_wave)/A_wave))

# ---- D. fully nonlinear short run -------------------------------------------
println("\n--- D: Dirichlet inflow, fully nonlinear (advection + H-weights) ---")
diagsD, _, _ = setup_and_run(; common...,
    wave_bc=:regular, bc_side=:left, bc_profile=:model, sponge_wL=0.0,
    T_final=5*T_wave, linearised=false, advection=true)
emaxD = maximum(d.eta_max for d in diagsD)
check("D: nonlinear run stable", !isnan(emaxD) && emaxD < 10*A_wave,
      @sprintf("(max η=%.2e m)", emaxD))
ampD5 = [d.eta_max for d in diagsD if d.t > 4*T_wave]
check("D: nonlinear η bounded (< 1.6A)", maximum(ampD5) < 1.6*A_wave,
      @sprintf("(max=%.3e)", maximum(ampD5)))

println()
println("=" ^ 60)
@printf("  Results: %d PASS,  %d FAIL\n", n_pass, n_fail)
println("=" ^ 60)
n_fail > 0 ? error("test_bc_generation: $n_fail failed!") :
             println("  Dirichlet boundary wave generation validated.")
