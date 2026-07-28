# ==============================================================
#  spectral_fidelity.jl — spectral fidelity of Dirichlet sea-state generation
#
#  VALIDATION RUN (long): a JONSWAP long-crested sea (WaveSpec.jl) enters
#  through the left Dirichlet boundary and is measured at two gauges. Because
#  the input is a KNOWN discrete component table (the WaveInput), fidelity is
#  checked component-by-component without any spectral estimator:
#
#    1. AMPLITUDE TRANSFER — DFT of each gauge record at every component ω_c:
#       measured amplitude vs prescribed A_c (target: within ~10% for
#       components carrying >20% of the peak amplitude, in-band kd);
#    2. DISPERSION TRANSFER — inter-gauge DFT phase difference at each ω_c
#       vs the MODEL wavenumber k_c: Δφ = −k_c·Δx (target: within 5%,
#       branch-safe gauge separation);
#    3. INTEGRAL — Hs from the summed measured component energies vs the
#       target Hs of the spectrum.
#
#  Results are printed as a table and appended to spectral_fidelity.csv.
#  Quick default: 60 Tp, 20 bins. Production: 200+ Tp, 40+ bins.
#
#  RUN:  julia --project=. GridapLFEM.jl/examples/validation/spectral_fidelity.jl
# ==============================================================

if !isdefined(Main, :GridapLFEM)
    include(joinpath(@__DIR__, "..", "..", "src", "GridapLFEM.jl"))
end
using .GridapLFEM
using Printf, LinearAlgebra

println("=" ^ 64)
println("  spectral_fidelity.jl — Dirichlet sea-state generation fidelity")
println("=" ^ 64)

# ── Sea state ─────────────────────────────────────────────────
h_val = 3.5
Hs    = 0.002
Tp    = 1.6
g     = 9.81
nfreq = 21
seed  = 20260723

# Frequency band and bin count are MEASUREMENT-DRIVEN (first-run lesson,
# 2026-07-23: energy-domain bins spaced 0.02 Hz leaked into each other over
# the 64 s window, and the 1/(0.55Tp) tail reached kd = 21.7 — outside the
# LFE-2 band (10.9) and mesh-unresolved):
#   * uniform FREQUENCY sampling, spacing ≥ 3/window (leakage-free-ish DFT);
#   * fmax = 1/(0.75 Tp) → kd ≤ ~10, inside the band and ≥ 6 cells/λ.
spec  = WaveSpec.ContinuousSpectrums.JONSWAP(Hs, Tp)
dspec = WaveSpec.SpectralSpreading.DiscreteSpectralSpreading(
            spec, WaveSpec.SpectralSampling.UniformSampling(),
            1.0/(2.5*Tp), 1.0/(0.75*Tp), nfreq;
            domain=WaveSpec.SpectralSampling.Frequency)
spread = WaveSpec.AngularSpreading.DiscreteAngularSpreading(0.0)
state  = WaveSpec.AiryWaves.AiryState(dspec, spread, h_val)
state  = WaveSpec.AiryWaves.change_seed!(state, seed)   # reproducible phases

vert0 = assemble_vertical_tensors(2, 1, [0.0, 0.728, 1.0])
wi    = WaveInput(vert0, state; d=h_val, g=g, profile=:model)

# ── Domain / numerics ─────────────────────────────────────────
Lx, Ly  = 100.0, 6.0
nx, ny  = 260, 4
T_final = 60.0 * Tp                  # production: 200+ Tp
dt      = Tp / 32
# g1+g1b: close pair (Δx=0.9 m, kΔx ∈ (0.6, 2.4) for the whole band) for the
# Goda–Suzuki incident/reflected split; g2: far gauge for dispersion transfer
x_g1, x_g1b, x_g2 = 30.0, 30.9, 42.0
gauges  = [(x_g1, Ly/2), (x_g1b, Ly/2), (x_g2, Ly/2)]

diags, vert, prob = setup_and_run(
    M=2, c_bdy=[0.0,0.728,1.0], domain=((0.0,Lx),(0.0,Ly)), partition=(nx,ny),
    p_horizontal=2, h_val=h_val, g=g, T_wave=Tp, A_wave=Hs/2,
    wave_bc=wi, bc_side=:left, bc_profile=:model,
    sponge_wL=0.0, sponge_wR=25.0, sponge_wB=0.0, sponge_wT=0.0, mu_max=5.0,
    T_final=T_final, dt=dt, linearised=true, advection=false,
    save_every=0, gauges=gauges, print_every=200, check_every=0)

# ── Component-wise analysis ───────────────────────────────────
ts  = [d.t for d in diags]
g1  = [d.gauge_vals[1] for d in diags]
g1b = [d.gauge_vals[2] for d in diags]
g2  = [d.gauge_vals[3] for d in diags]
i0 = findfirst(ts .> max(2*wi.T_ramp, ts[end]/3))   # steady window
tv = ts[i0:end]; v1 = g1[i0:end]; v1b = g1b[i0:end]; v2 = g2[i0:end]

dftat(v, ω) = 2*dot(v, exp.(-im .* ω .* tv))/length(tv)   # complex amplitude
Amax = maximum(wi.amps)
dx   = x_g2 - x_g1
dxb  = x_g1b - x_g1

println("\n   f [Hz]    kd     A_in [m]   a_I [m]    err%   a_R/a_I   k_meas    k_model   err%")
println("  " * "-"^84)
csv_rows = String[]
n_amp_ok = 0; n_amp_tot = 0; n_k_ok = 0
E_meas = 0.0
for c in sort(1:wi.ncomp; by=i->wi.omegas[i])
    ω = wi.omegas[c]; k = wi.ks[c]; kd = k*h_val
    C1 = dftat(v1, ω); C1b = dftat(v1b, ω); C2 = dftat(v2, ω)
    # Goda–Suzuki incident/reflected split on the close pair (g1, g1b)
    s  = 2*abs(sin(k*dxb))
    aI = abs(exp(im*k*x_g1b)*C1 - exp(im*k*x_g1)*C1b)/s
    aR = abs(exp(-im*k*x_g1b)*C1 - exp(-im*k*x_g1)*C1b)/s
    global E_meas += aI^2/2
    # far-pair phase difference → measured k (branch nearest k_model)
    dphi = angle(C2/C1)
    k_raw = -dphi/dx
    nbr  = round((k - k_raw)*dx/(2π))
    k_m  = k_raw + nbr*2π/dx
    aerr = abs(aI - wi.amps[c])/wi.amps[c]
    kerr = abs(k_m - k)/k
    major = wi.amps[c] > 0.2*Amax
    if major
        global n_amp_tot += 1
        aerr < 0.10 && (global n_amp_ok += 1)
        kerr < 0.05 && (global n_k_ok += 1)
    end
    @printf("  %7.4f  %5.2f  %9.6f  %9.6f  %5.1f%%  %6.1f%%  %8.4f  %8.4f  %5.1f%%%s\n",
            ω/2π, kd, wi.amps[c], aI, 100aerr, 100*aR/max(aI,1e-30), k_m, k, 100kerr,
            major ? "  *" : "")
    push!(csv_rows, join(string.([ω/2π, kd, wi.amps[c], aI, aR, k_m, k]), ","))
end
Hs_meas = 4*sqrt(E_meas)
println("  " * "-"^76)
@printf("  (* = major components, A_c > 0.2 A_max)\n")
@printf("\n  amplitude transfer : %d/%d major components within 10%%\n",
        n_amp_ok, n_amp_tot)
@printf("  dispersion transfer: %d/%d major components within 5%% of model k\n",
        n_k_ok, n_amp_tot)
@printf("  Hs: input=%.5f m  measured(g1)=%.5f m  (ratio %.3f)\n",
        Hs, Hs_meas, Hs_meas/Hs)

outdir = joinpath(@__DIR__, "..", "..", "output", "spectral_fidelity")
mkpath(outdir)
open(joinpath(outdir, "spectral_fidelity.csv"), "w") do io
    println(io, "f,kd,A_in,a_I,a_R,k_meas,k_model")
    foreach(r -> println(io, r), csv_rows)
end
println("\n  CSV: output/spectral_fidelity/spectral_fidelity.csv")
