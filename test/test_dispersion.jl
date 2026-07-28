# ==============================================================
#  test_dispersion.jl — FEM dispersion gate (SMALL & FAST)
#
#  Port of ../../LFE-M_2D_solver/test/test_dispersion_lfem2D.jl.
#  Time-domain check that the algebraic solver's R_P (leading pressure)
#  term reproduces the linear dispersion end-to-end: kd = 3, gauges 2λ from
#  the wavemaker separated λ/2, DFT phase at the forcing frequency.
#
#  RUN:  julia --project=. GridapLFEM.jl/test/test_dispersion.jl
# ==============================================================

if !isdefined(Main, :GridapLFEM)
    include(joinpath(@__DIR__, "..", "src", "GridapLFEM.jl"))
end
using .GridapLFEM
using Printf, LinearAlgebra

println("=" ^ 60)
println("  test_dispersion.jl — algebraic solver dispersion (kd=3)")
println("=" ^ 60)

n_pass = 0; n_fail = 0
check(name, cond, extra="") = (global n_pass, n_fail;
    cond ? (println("  PASS  $name $extra"); n_pass+=1) :
           (println("  FAIL  $name $extra"); n_fail+=1))

function estimate_phase_speed(diags, x1, x2, omega; idx1=1, idx2=2)
    ts = [d.t for d in diags]
    g1 = [d.gauge_vals[idx1] for d in diags]
    g2 = [d.gauge_vals[idx2] for d in diags]
    n = length(ts); i0 = n ÷ 2
    tv = ts[i0:end]; g1 = g1[i0:end]; g2 = g2[i0:end]
    eiw = exp.(-im .* omega .* tv)
    A1 = dot(g1, eiw); A2 = dot(g2, eiw)
    (abs(A1) < 1e-14 || abs(A2) < 1e-14) && return NaN
    dphi = angle(A2 / A1); dphi > 0 && (dphi -= 2π)
    k = -dphi / (x2 - x1)
    return k > 0 ? omega / k : NaN
end

g = 9.81; T_wave = 1.6; omega = 2π/T_wave
kd_target = 3.0

# solve (k, d) for this kd at fixed T
k = sqrt(omega^2 / (g*tanh(kd_target)))
for _ in 1:50
    d_try = kd_target/k
    f = g*k*tanh(k*d_try) - omega^2; df = g*tanh(k*d_try)
    dk = f/df; global k -= dk; abs(dk) < 1e-12*abs(k) && break
end
h_val = kd_target/k; Ce = omega/k; lam = 2π/k
@printf("\n  kd=%.1f  d=%.3f m  λ=%.2f m  Ce=%.3f m/s\n", kd_target, h_val, lam, Ce)

x_wm = 6.0
Lx = x_wm + 6lam + 8.0; Ly = 2.0
nx = round(Int, Lx/(lam/6)); ny = 2
dt = T_wave/24; Tf = 8*T_wave
x_g1 = x_wm + 2lam; x_g2 = x_g1 + lam/2; y_g = Ly/2
@printf("  Domain %.0f×%.0f m, %d×%d cells, %d periods\n", Lx, Ly, nx, ny, 8)

diags, _, _ = setup_and_run(
    M=2, c_bdy=[0.0,0.728,1.0], domain=((0.0,Lx),(0.0,Ly)), partition=(nx,ny),
    p_horizontal=2, h_val=h_val, T_wave=T_wave, A_wave=0.0005, x_wm=x_wm,
    y_wm=nothing, sponge_wL=8.0, sponge_wR=8.0, sponge_wB=0.0, sponge_wT=0.0,
    mu_max=30.0, T_final=Tf, dt=dt, save_every=0,
    gauges=[(x_g1,y_g),(x_g2,y_g)], linearised=true, advection=false,
    nl_tol=1e-8,
)
Cm  = estimate_phase_speed(diags, x_g1, x_g2, omega)
err = isnan(Cm) ? Inf : abs(Cm/Ce - 1.0)
@printf("  Cm=%.3f  Ce=%.3f  err=%.2f%%\n", isnan(Cm) ? 0.0 : Cm, Ce, 100*min(err,999))
check("kd=$kd_target: |Cm/Ce−1| < 3%", err < 0.03,
      "(got $(round(100*min(err,999),digits=2))%)")

println()
println("=" ^ 60)
@printf("  Results: %d PASS,  %d FAIL\n", n_pass, n_fail)
println("=" ^ 60)
n_fail > 0 ? error("test_dispersion: $n_fail failed!") :
             println("  Algebraic solver reproduces dispersion.")
