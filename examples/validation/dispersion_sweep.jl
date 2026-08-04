# ==============================================================
#  dispersion_sweep.jl — trace the dispersion curve with the FULL solver
#
#  Runs the full nonlinear solver (regime=:nonlinear) at a tiny
#  amplitude on a flat bed, sweeping kd across the range, and measures the phase
#  celerity Cm at each point via DFT phase differencing. Produces the measured
#  dispersion curve Cm/Ce(kd) — the time-domain counterpart of the closed-form
#  test_dispersion_curve.jl, but using the actual production code path. The
#  measured curve should track Airy (Cm/Ce ≈ 1) up to the model's applicable kd
#  and fall away beyond it (LFE-2 ≈ 10.9).
#
#  Writes output/dispersion_sweep_M<M>.csv (kd, Cm, Ce, Cm/Ce, err) for plotting
#  with the postprocessing library.
#
#  ANALYSIS SCRIPT (many FEM runs — tens of minutes). RUN:
#    julia --project=. GridapLFEM.jl/examples/validation/dispersion_sweep.jl
#  Env: LFEM_M (vertical layers, default 2), LFEM_KD (comma-sep kd list).
# ==============================================================

using GridapLFEM
using Printf, LinearAlgebra

M      = parse(Int, get(ENV, "LFEM_M", "2"))
c_bdy  = M == 3 ? [0.0,0.726,0.925,1.0] :
         M == 4 ? [0.0,0.745,0.923,0.977,1.0] : [0.0,0.728,1.0]
kd_list = haskey(ENV,"LFEM_KD") ? parse.(Float64, split(ENV["LFEM_KD"],",")) :
          [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]

println("=" ^ 66)
@printf("  dispersion_sweep.jl — LFE-%d, full nonlinear, %d kd points\n", M, length(kd_list))
println("=" ^ 66)

g = 9.81; T_wave = 1.6; omega = 2π/T_wave; A_wave = 5e-4
outdir = joinpath(@__DIR__, "..", "..", "output"); mkpath(outdir)
csv = joinpath(outdir, "dispersion_sweep_M$(M).csv")
open(csv, "w") do io; println(io, "kd,k,d,lambda,Ce,Cm,ratio,rel_err"); end

# Robust celerity: temporal DFT at ω per gauge, then a continuous spatial k-scan
# over a far-field gauge line (avoids the λ/2 branch-cut + near-field bias of a
# two-gauge phase difference — see test_dispersion_nonlinear.jl).
function celerity_spatial(diags, xs, w, k0)
    ts=[d.t for d in diags]; i0=length(ts)÷2; tv=ts[i0:end]; eiw=exp.(-im.*w.*tv)
    Ĉ=ComplexF64[ sum([d.gauge_vals[gi] for d in diags][i0:end].*eiw) for gi in eachindex(xs)]
    ks=range(0.4k0,1.8k0;length=8000)
    w / ks[argmax([abs(sum(Ĉ.*exp.(im.*k.*xs))) for k in ks])]
end

@printf("\n  %5s %8s %8s %8s %8s %8s\n", "kd", "d[m]", "λ[m]", "Ce", "Cm", "Cm/Ce")
for kd in kd_list
    k = omega^2/(g*tanh(kd)); d = kd/k; Ce = omega/k; lam = 2π/k
    x_wm = 6.0
    xg = collect(range(x_wm+2.5lam, x_wm+5.5lam; length=20)); Ly = 2.0
    Lx = last(xg) + 8.0
    nx = max(24, round(Int, Lx/(lam/6)))
    diags,_,_ = setup_and_run(
        M=M, c_bdy=c_bdy, domain=((0.0,Lx),(0.0,Ly)), partition=(nx,2), p_horizontal=2,
        h_val=d, T_wave=T_wave, A_wave=A_wave, x_wm=x_wm, y_wm=nothing,
        sponge_wL=6.0, sponge_wR=8.0, mu_max=30.0, T_final=14*T_wave, dt=T_wave/24,
        save_every=0, gauges=[(x,Ly/2) for x in xg],
        regime=:nonlinear, print_every=10_000)
    Cm = celerity_spatial(diags, xg, omega, k)
    r  = isnan(Cm) ? NaN : Cm/Ce; err = isnan(Cm) ? NaN : abs(r-1)
    @printf("  %5.1f %8.3f %8.2f %8.3f %8.3f %8.4f\n", kd, d, lam, Ce, Cm, r)
    open(csv,"a") do io; @printf(io,"%.4f,%.6f,%.6f,%.4f,%.6f,%.6f,%.6f,%.6e\n",
                                 kd,k,d,lam,Ce,Cm,r,err); end
end
@printf("\n  curve written to %s\n", csv)
println("  (plot Cm/Ce vs kd with the postprocessing library; overlay the 2%% band)")
