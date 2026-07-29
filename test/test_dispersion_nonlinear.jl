# ==============================================================
#  test_dispersion_nonlinear.jl — FULL nonlinear solver reproduces LINEAR
#                                 dispersion in the small-amplitude limit
#
#  The linear dispersion relation ω²=gk tanh(kd) is a property of infinitesimal
#  waves. Rather than run the *linearised* solver (test_dispersion.jl), here the
#  FULL nonlinear solver (regime=:nonlinear) is run on a FLAT bed
#  at a tiny amplitude (kA~1e-4, so the O((kA)²) Stokes phase-speed correction is
#  ~1e-8): the production code path must reproduce Airy theory — the asymptotic-
#  consistency principle (a nonlinear model reduces to the simpler linear model
#  in the appropriate limit).
#
#  Measurement. Two-gauge phase differencing over λ/2 is ill-conditioned (the
#  phase sits on the ±π branch cut and picks up wavemaker near-field). Instead we
#  fit the wavenumber ROBUSTLY: take the complex temporal DFT at ω at each of many
#  gauges spread over 3λ in the FAR field (past the evanescent near-field), then
#  scan k continuously to maximise |Σ Ĉ(xⱼ) e^{-ik xⱼ}| — a clean, unbiased fit.
#  Swept over kd = 1, 3, 5 (LFE-2); each gated |Cm/Ce−1| < 4%.
#
#  Tolerance. kd = 1 and 3 land at 0.9% and 0.4%. Deep water (kd = 5) reaches
#  ~3%: its group velocity is low, so in a finite record the far-field gauges are
#  only marginally established — a TIME-DOMAIN measurement floor, not a model
#  error (the closed-form test_dispersion_curve.jl gates the MODEL's Cm at 2% up
#  to kd≈10.9). The 4% gate reflects that floor and still catches regressions (a
#  broken solver reads ~25% off). CSV written for plotting.
#
#  RUN:  julia --project=. GridapLFEM.jl/test/test_dispersion_nonlinear.jl
# ==============================================================

if !isdefined(Main, :GridapLFEM)
    include(joinpath(@__DIR__, "..", "src", "GridapLFEM.jl"))
end
using .GridapLFEM
using Printf, LinearAlgebra

println("=" ^ 66)
println("  test_dispersion_nonlinear.jl — full nonlinear solver vs Airy dispersion")
println("=" ^ 66)

n_pass = 0; n_fail = 0
check(name, cond, extra="") = (global n_pass, n_fail;
    cond ? (println("  PASS  $name $extra"); n_pass += 1) :
           (println("  FAIL  $name $extra"); n_fail += 1))

"Robust celerity: temporal DFT at ω per gauge, then a continuous spatial k-scan."
function celerity_spatial(diags, xs, ω, k0)
    ts = [d.t for d in diags]; i0 = length(ts) ÷ 2; tv = ts[i0:end]
    eiw = exp.(-im .* ω .* tv)                       # temporal DFT kernel Σ η e^{-iωt}
    Ĉ = ComplexF64[ sum([d.gauge_vals[gi] for d in diags][i0:end] .* eiw) for gi in eachindex(xs)]
    # a rightward wave A cos(kx−ωt) gives Ĉⱼ ∝ e^{-ik xⱼ}, so |Σ Ĉⱼ e^{+ik xⱼ}| peaks at the true k
    ks = range(0.4k0, 1.8k0; length=8000)
    obj = [abs(sum(Ĉ .* exp.(im .* k .* xs))) for k in ks]
    return ω / ks[argmax(obj)]
end

const g = 9.81; const T_wave = 1.6; const omega = 2π/T_wave; const A_wave = 5e-4
kd_targets = [1.0, 3.0, 5.0]
outdir = joinpath(@__DIR__, "..", "output"); mkpath(outdir)
csv = joinpath(outdir, "dispersion_nonlinear.csv")
open(csv, "w") do io; println(io, "kd,k,d,lambda,Ce,Cm,rel_err"); end

"Full nonlinear run at fixed kd (flat bed); measure Cm by robust wavenumber fit."
function run_kd(kd)
    k = omega^2/(g*tanh(kd)); d = kd/k; Ce = omega/k; lam = 2π/k
    x_wm = 6.0
    # 20-gauge line over 3λ, starting 2.5λ past the maker (clear of the near-field);
    # deep waves have low group velocity ⇒ run long enough to establish the far field.
    xg = collect(range(x_wm + 2.5lam, x_wm + 5.5lam; length=20))
    Lx = last(xg) + 8.0; Ly = 2.0
    nx = max(24, round(Int, Lx/(lam/6))); ny = 2
    dt = T_wave/24; Tf = 14*T_wave
    @printf("\n  kd=%.1f  d=%.3f m  λ=%.2f m  Ce=%.3f m/s | domain %.0f m, %d cells, 14 periods\n",
            kd, d, lam, Ce, Lx, nx)
    diags, _, _ = setup_and_run(
        M=2, c_bdy=[0.0,0.728,1.0], domain=((0.0,Lx),(0.0,Ly)), partition=(nx,ny),
        p_horizontal=2, h_val=d, T_wave=T_wave, A_wave=A_wave, x_wm=x_wm, y_wm=nothing,
        sponge_wL=6.0, sponge_wR=8.0, sponge_wB=0.0, sponge_wT=0.0, mu_max=30.0,
        T_final=Tf, dt=dt, save_every=0, gauges=[(x, Ly/2) for x in xg],
        regime=:nonlinear, nl_tol=1e-8, print_every=10_000)
    Cm = celerity_spatial(diags, xg, omega, k)
    return k, d, lam, Ce, Cm
end

"Run one kd point, apply the 3% gate, and append the row to the CSV. Returns err."
function gate_kd(kd)
    k, d, lam, Ce, Cm = run_kd(kd)
    err = isnan(Cm) ? Inf : abs(Cm/Ce - 1.0)
    @printf("  Cm=%.3f m/s  Ce=%.3f m/s  err=%.2f%%\n",
            isnan(Cm) ? 0.0 : Cm, Ce, 100*min(err,999))
    check("kd=$kd: full nonlinear Cm matches Airy (|Cm/Ce−1| < 4%)", err < 0.04,
          "($(round(100*min(err,999),digits=2))%)")
    open(csv, "a") do io
        @printf(io, "%.4f,%.6f,%.6f,%.4f,%.6f,%.6f,%.6e\n",
                kd, k, d, lam, Ce, isnan(Cm) ? NaN : Cm, err)
    end
    return err
end

function main()
    for kd in kd_targets; gate_kd(kd); end
    println()
    println("=" ^ 66)
    @printf("  Results: %d PASS,  %d FAIL   (curve data → %s)\n", n_pass, n_fail, csv)
    println("=" ^ 66)
    n_fail > 0 ? error("test_dispersion_nonlinear: $n_fail failed!") :
                 println("  Full nonlinear solver reproduces linear dispersion (small-A limit).")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
