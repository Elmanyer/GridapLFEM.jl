# ==============================================================
#  test_bc_spectrum.jl — multichromatic Dirichlet generation (FEM, e2e)
#
#  Deterministic 3-component sea (hand-built WaveInput — the same machinery
#  the WaveSpec converter feeds) with commensurate frequencies
#  f = {0.500, 0.625, 0.750} Hz (common cycle 8 s ⇒ a 16 s measurement
#  window holds an INTEGER number of periods of every component: the
#  rectangular DFT is leakage-free at all component and intermodulation
#  lines). Amplitudes are measured with the two-gauge GODA–SUZUKI
#  incident/reflected decomposition — a single-point DFT is biased by the
#  partial standing wave from residual sponge reflection (measured ~12% on
#  the shortest component). Gates:
#    * each component's INCIDENT amplitude within 10% of its prescribed A_c;
#    * reflection coefficient of every component < 15%;
#    * no spurious intermodulation lines (f_i±f_j, 2f_i) above 5% of the
#      largest component (linearised run ⇒ exact superposition);
#    * stability.
#
#  RUN:  julia --project=. GridapLFEM.jl/test/test_bc_spectrum.jl
# ==============================================================

if !isdefined(Main, :GridapLFEM)
    include(joinpath(@__DIR__, "..", "src", "GridapLFEM.jl"))
end
using .GridapLFEM
using Printf, LinearAlgebra

println("=" ^ 60)
println("  test_bc_spectrum.jl — 3-component Dirichlet sea")
println("=" ^ 60)

n_pass = 0; n_fail = 0
check(name, cond, extra="") = (global n_pass, n_fail;
    cond ? (println("  PASS  $name $extra"); n_pass+=1) :
           (println("  FAIL  $name $extra"); n_fail+=1))

"Complex DFT coefficient (2/N · Σ v e^{−iωt}) over the LAST `Twin` seconds."
function window_dft(diags, idx, omega, Twin)
    ts = [d.t for d in diags]; gv = [d.gauge_vals[idx] for d in diags]
    t1 = ts[end] - Twin
    sel = ts .> t1 .+ 1e-9
    tv = ts[sel]; gw = gv[sel]
    return 2.0*dot(gw, exp.(-im .* omega .* tv))/length(gw)
end

window_amplitude(diags, idx, omega, Twin) = abs(window_dft(diags, idx, omega, Twin))

"""
Goda–Suzuki two-gauge incident/reflected decomposition at frequency ω with
wavenumber k: gauges at x1, x2 give C_j = a_I e^{−i(kx_j+φ_I)} + a_R e^{+i(kx_j+φ_R)};
    a_I = |e^{ikx₂}C₁ − e^{ikx₁}C₂| / (2|sin k(x₂−x₁)|)
    a_R = |e^{−ikx₂}C₁ − e^{−ikx₁}C₂| / (2|sin k(x₂−x₁)|)
Valid away from k(x₂−x₁) ≈ nπ.
"""
function goda_suzuki(diags, omega, k, x1, x2, Twin)
    C1 = window_dft(diags, 1, omega, Twin)
    C2 = window_dft(diags, 2, omega, Twin)
    s  = 2.0*abs(sin(k*(x2 - x1)))
    aI = abs(exp(im*k*x2)*C1 - exp(im*k*x1)*C2)/s
    aR = abs(exp(-im*k*x2)*C1 - exp(-im*k*x1)*C2)/s
    return aI, aR
end

# ---- component table ---------------------------------------------------------
g = 9.81
Ts     = [2.0, 1.6, 4/3]                       # f = 0.500, 0.625, 0.750 Hz
omegas = 2π ./ Ts
amps   = [0.0004, 0.0005, 0.0003]
phases = [0.3, 1.1, 2.5]
d_val  = 2.387                                 # kd ≈ 2.2 / 3.0 / 3.9 — in-band

vert0 = assemble_vertical_tensors(2, 1, [0.0, 0.728, 1.0])
wi = WaveInput(vert0, amps, omegas; d=d_val, g=g, phases=phases,
               T_ramp=4.0, profile=:model)
kds = wi.ks .* d_val
@printf("\n  components: f=%s Hz, kd=%s\n",
        string(round.(1 ./ Ts; digits=3)), string(round.(kds; digits=2)))

# ---- domain / numerics -------------------------------------------------------
lam_max = 2π/minimum(wi.ks)
Lx = 4*lam_max + 10.0; Ly = 2.0
nx = round(Int, Lx/(2π/maximum(wi.ks)/6)); ny = 2
dt = 1/15                                      # 16 s window = 240 samples exactly
Tf = 32.0
# two gauges for the Goda–Suzuki split: Δx=0.9 m keeps kΔx ∈ (0.3, 2.8) rad
# for every component (away from the nπ singularities)
x_g1 = 2*lam_max; x_g2 = x_g1 + 0.9; y_g = Ly/2
@printf("  domain %.1f×%.1f m, %d×%d cells, T=%.0f s, gauges at x=%.1f, %.1f m\n",
        Lx, Ly, nx, ny, Tf, x_g1, x_g2)

diags, _, _ = setup_and_run(
    M=2, c_bdy=[0.0,0.728,1.0], domain=((0.0,Lx),(0.0,Ly)), partition=(nx,ny),
    fe_order=2, d_val=d_val, T_wave=Ts[2], A_wave=maximum(amps),
    wave_bc=wi, bc_side=:left, sponge_wL=0.0, sponge_wR=10.0,
    sponge_wB=0.0, sponge_wT=0.0, mu_max=30.0, T_final=Tf, dt=dt,
    save_every=0, gauges=[(x_g1, y_g), (x_g2, y_g)], linearised=true,
    advection=false, check_every=0, print_every=100)

emax = maximum(d.eta_max for d in diags)
check("stable (no NaN, bounded)", !isnan(emax) && emax < 10*sum(amps),
      @sprintf("(max η=%.2e m)", emax))

# ---- component amplitudes (Goda–Suzuki incident/reflected split) -------------
Twin = 16.0
for c in 1:3
    aI, aR = goda_suzuki(diags, omegas[c], wi.ks[c], x_g1, x_g2, Twin)
    err = abs(aI - amps[c])/amps[c]
    check(@sprintf("component f=%.3f Hz incident amplitude within 10%%", 1/Ts[c]),
          err < 0.10, @sprintf("(A=%.2e, a_I=%.2e, err=%.1f%%)",
                               amps[c], aI, 100err))
    check(@sprintf("component f=%.3f Hz reflection < 15%%", 1/Ts[c]),
          aR/aI < 0.15, @sprintf("(a_R/a_I=%.1f%%)", 100*aR/aI))
end

# ---- spurious intermodulation lines ------------------------------------------
f_lines = [1/Ts[1]+1/Ts[2], 1/Ts[2]+1/Ts[3], 1/Ts[1]+1/Ts[3],
           2/Ts[1], 2/Ts[2], 2/Ts[3],
           1/Ts[2]-1/Ts[1], 1/Ts[3]-1/Ts[2]]
worst = maximum(window_amplitude(diags, 1, 2π*f, Twin) for f in f_lines)
check("no spurious intermodulation lines > 5% of max component",
      worst < 0.05*maximum(amps),
      @sprintf("(worst=%.2e = %.1f%% of A_max)", worst, 100*worst/maximum(amps)))

println()
println("=" ^ 60)
@printf("  Results: %d PASS,  %d FAIL\n", n_pass, n_fail)
println("=" ^ 60)
n_fail > 0 ? error("test_bc_spectrum: $n_fail failed!") :
             println("  Multichromatic Dirichlet generation validated.")
