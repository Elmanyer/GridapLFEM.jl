# ==============================================================
#  test_shallow_water.jl — shallow-water (long-wave) limit
#
#  The other asymptotic limit of the LFE-M: as kd → 0 the non-hydrostatic
#  dispersion term vanishes and the model must reduce to the NON-DISPERSIVE
#  shallow-water equations, celerity c → √(gd) independent of k.
#
#  Two checks:
#   (1) CLOSED FORM (exact identity). Cm²(kd) = g d · Φᵀ(M−(kd)²B)⁻¹Φ, so at
#       kd→0, Cm² = g d · ΦᵀM⁻¹Φ. Because the vertical Lagrange basis is a
#       partition of unity (Σⱼφⱼ≡1 ⇒ M·𝟙=Φ) and ΣΦ=1, one has ΦᵀM⁻¹Φ = Φᵀ𝟙 = 1
#       EXACTLY, so Cm(kd→0) = √(gd) exactly — the SWE celerity. We verify the
#       identity and the ratio Cm/Ce→1 at tiny kd.
#   (2) TIME DOMAIN. The FULL nonlinear solver run with a long wave (small kd)
#       on a flat bed must propagate at c ≈ √(gd) (and match Airy Ce), confirming
#       the model is in the shallow-water regime.
#
#  RUN:  julia --project=. GridapLFEM.jl/test/test_shallow_water.jl
# ==============================================================

if !isdefined(Main, :GridapLFEM)
    include(joinpath(@__DIR__, "..", "src", "GridapLFEM.jl"))
end
using .GridapLFEM
using Printf, LinearAlgebra

println("=" ^ 64)
println("  test_shallow_water.jl — long-wave (SWE) limit  c → √(gd)")
println("=" ^ 64)

n_pass = 0; n_fail = 0
check(name, cond, extra="") = (global n_pass, n_fail;
    cond ? (println("  PASS  $name $extra"); n_pass += 1) :
           (println("  FAIL  $name $extra"); n_fail += 1))

g = 9.81

# ---- (1) closed-form identity ΦᵀM⁻¹Φ = 1 ⇒ Cm(kd→0) = √(gd) --------------------
for (M, cb, name) in ((2,[0.0,0.728,1.0],"LFE-2"), (3,[0.0,0.726,0.925,1.0],"LFE-3"))
    vert = assemble_vertical_tensors(M, 1, cb)
    id   = dot(vert.Phi, vert.Mmat \ vert.Phi)           # ΦᵀM⁻¹Φ
    @printf("  %s:  ΦᵀM⁻¹Φ = %.12f   (partition-of-unity ⇒ 1)\n", name, id)
    check("$name: ΦᵀM⁻¹Φ = 1 (SWE limit Cm→√(gd))", abs(id - 1.0) < 1e-10)
    # dispersion ratio Cm/Ce → 1 as kd → 0
    r0 = dispersion_ratio(vert, g, 1.0, [0.05])[1]
    check("$name: Cm/Ce → 1 as kd→0 (at kd=0.05)", abs(r0 - 1.0) < 1e-3,
          "($(round(r0,digits=6)))")
end

# ---- (2) time-domain long-wave run, full nonlinear, c ≈ √(gd) -------------------
d = 1.0; c_sw = sqrt(g*d); T_wave = 8.0; omega = 2π/T_wave    # long wave ⇒ small kd
k = omega^2/(g)                                               # initial guess (shallow)
for _ in 1:100                                               # refine ω²=gk tanh(kd)
    f = g*k*tanh(k*d) - omega^2; df = g*(tanh(k*d)+k*d*(1-tanh(k*d)^2))
    global k -= f/df; abs(f) < 1e-14 && break
end
kd = k*d; lam = 2π/k; Ce = omega/k
@printf("\n  long wave: T=%.1f s  kd=%.3f  λ=%.1f m  √(gd)=%.3f  Airy Ce=%.3f m/s\n",
        T_wave, kd, lam, c_sw, Ce)

x_wm = 8.0; Lx = x_wm + 5lam + 30.0; Ly = 3.0
nx = max(24, round(Int, Lx/(lam/6))); ny = 2
dt = T_wave/24; Tf = 8*T_wave
x_g1 = x_wm + 2lam; x_g2 = x_g1 + lam/2; y_g = Ly/2
@printf("  domain %.0f m, %d cells, %d periods\n", Lx, nx, 8)

diags, _, _ = setup_and_run(
    M=2, c_bdy=[0.0,0.728,1.0], domain=((0.0,Lx),(0.0,Ly)), partition=(nx,ny),
    fe_order=2, d_val=d, T_wave=T_wave, A_wave=5e-4, x_wm=x_wm, y_wm=nothing,
    sponge_wL=15.0, sponge_wR=15.0, mu_max=5.0, T_final=Tf, dt=dt, save_every=0,
    gauges=[(x_g1,y_g),(x_g2,y_g)], linearised=false, advection=true,   # ★ full nonlinear
    print_every=10_000)

ts = [d.t for d in diags]
g1 = [d.gauge_vals[1] for d in diags]; g2 = [d.gauge_vals[2] for d in diags]
i0 = length(ts)÷2; tv = ts[i0:end]
eiw = exp.(-im .* omega .* tv)
A1 = dot(g1[i0:end], eiw); A2 = dot(g2[i0:end], eiw)
dphi = angle(A2/A1); dphi > 0 && (dphi -= 2π)
k_meas = -dphi/(x_g2-x_g1); Cm = omega/k_meas
@printf("  measured c = %.3f m/s   (√(gd)=%.3f, Airy Ce=%.3f)\n", Cm, c_sw, Ce)
@printf("  c/√(gd) = %.4f   c/Ce = %.4f\n", Cm/c_sw, Cm/Ce)

check("long-wave celerity ≈ √(gd) (SWE limit, <5%)", abs(Cm/c_sw - 1) < 0.05,
      "($(round(100*abs(Cm/c_sw-1),digits=2))%)")
check("long-wave celerity matches Airy Ce (<3%)", abs(Cm/Ce - 1) < 0.03,
      "($(round(100*abs(Cm/Ce-1),digits=2))%)")

println()
println("=" ^ 64)
@printf("  Results: %d PASS,  %d FAIL\n", n_pass, n_fail)
println("=" ^ 64)
n_fail > 0 ? error("test_shallow_water: $n_fail failed!") :
             println("  Model reduces to non-dispersive shallow water (c → √(gd)).")
