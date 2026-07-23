# ==============================================================
#  test_waveinput.jl — Dirichlet boundary wave generation data (FAST, no FEM)
#
#  Checks on waveinput.jl:
#    σ-node extraction, the model-dispersion consistency identity
#    (dk/ω)(gk/ω)ΦᵀM_eff⁻¹Φ = 1, model-vs-Airy wavenumber agreement inside
#    the applicable band, the :airy nodal profile vs the closed cosh formula,
#    the deep-water guard, the Hann ramp, the regular-wave closures vs the
#    closed-form Airy solution, the ForwardDiff safety of the closures (t),
#    and the WaveSpec.AiryState converter (amplitude/phase snapshot, Hs
#    recovery, seed reproducibility).
#
#  RUN:  julia --project=. GridapLFEM.jl/test/test_waveinput.jl
# ==============================================================

if !isdefined(Main, :GridapLFEM)
    include(joinpath(@__DIR__, "..", "src", "GridapLFEM.jl"))
end
using .GridapLFEM
using LinearAlgebra, Printf
using Gridap: VectorValue
import Gridap

println("=" ^ 60)
println("  test_waveinput.jl — boundary wave generation data")
println("=" ^ 60)

n_pass = 0; n_fail = 0
function check(name, cond)
    global n_pass, n_fail
    if cond; println("  PASS  $name"); n_pass += 1
    else;    println("  FAIL  $name"); n_fail += 1; end
end

g = 9.81; d = 3.5
vert = assemble_vertical_tensors(2, 1, [0.0, 0.728, 1.0])   # LFE-2, Nσ=3

# ---- σ-nodes -----------------------------------------------------------------
sig = sigma_dof_nodes(vert)
check("σ-nodes: sorted copy equals c_bdy (p=1)",
      isapprox(sort(sig), [0.0, 0.728, 1.0]; atol=1e-12))

# ---- model dispersion --------------------------------------------------------
T = 2.4                                    # kd ≈ 1.6 — well inside the LFE-2 band
omega = 2.0*pi/T
k_airy = find_wavenumber(omega, d, g)
k_mod  = model_wavenumber(vert, omega, d, g)
M_eff  = vert.Mmat .- vert.B .* (k_mod*d)^2
consistency = (d*k_mod/omega) * (g*k_mod/omega) * dot(vert.Phi, M_eff \ vert.Phi)
check("model dispersion: continuity closure identity = 1",
      abs(consistency - 1.0) < 1e-10)
check("model k within 2% of Airy k inside the band",
      abs(k_mod - k_airy)/k_airy < 0.02)

# out-of-band frequency falls back to the Airy k (with a warning)
omega_hi = 2.0*pi/0.2                      # kd_airy ≈ 353 >> band
k_hi = model_wavenumber(vert, omega_hi, d, g)
check("out-of-band ω falls back to a finite Airy k",
      isfinite(k_hi) && k_hi > 0)

# ---- regular-wave WaveInput: :airy profile -----------------------------------
A = 0.001
wia = WaveInput(vert; A=A, T=T, d=d, g=g, T_ramp=0.0, profile=:airy)
check("regular :airy — 1 component, k = Airy k",
      wia.ncomp == 1 && abs(wia.ks[1] - k_airy) < 1e-12)
uamp_ref = [A*omega*cosh(k_airy*d*sig[j])/sinh(k_airy*d) for j in 1:3]
check("regular :airy — nodal amplitudes = A ω cosh(kdσ)/sinh(kd)",
      isapprox(vec(wia.Uamp), uamp_ref; rtol=1e-12))

# deep-water guard finite
wideep = WaveInput(vert; A=A, T=0.4, d=100.0, g=g, T_ramp=0.0, profile=:airy)
check("deep-water guard (kd>20): finite nodal amplitudes",
      all(isfinite, wideep.Uamp))

# ---- regular-wave WaveInput: :model profile ----------------------------------
wim = WaveInput(vert; A=A, T=T, d=d, g=g, T_ramp=0.0, profile=:model)
m_ref = A*(g*wim.ks[1]/omega) .* ((vert.Mmat .- vert.B .* (wim.ks[1]*d)^2) \ vert.Phi)
check("regular :model — nodal amplitudes = A(gk/ω)M_eff⁻¹Φ",
      isapprox(vec(wim.Uamp), m_ref; rtol=1e-12))
# depth-integrated transport: continuity closure requires Φ·Uamp = Aω/(kd).
# Exact (machine precision) for the :model eigenmode; interpolation-accurate
# for the :airy cosh sampling.
check(":model — transport identity Φ·Uamp = Aω/(kd) (machine precision)",
      abs(dot(vert.Phi, vec(wim.Uamp)) - A*omega/(wim.ks[1]*d)) <
      1e-12 * A*omega/(wim.ks[1]*d))
# The :airy nodal sampling does NOT satisfy the discrete continuity closure —
# the Φ-weighted quadrature of the P1 interpolant of cosh on the 2-element
# optimised σ-mesh is ~15% off at kd≈2.5. This mismatch (spurious boundary
# radiation) is precisely why :model is the default profile.
check(":airy — transport within 20% of Aω/(kd) (coarse σ-mesh cosh quadrature)",
      abs(dot(vert.Phi, vec(wia.Uamp)) - A*omega/(k_airy*d)) <
      0.20 * A*omega/(k_airy*d))
jsurf = argmax(sig)
check(":model vs :airy surface-node amplitude within 2%",
      abs(wim.Uamp[1, jsurf] - wia.Uamp[1, jsurf]) / wia.Uamp[1, jsurf] < 0.02)

# ---- closures vs closed form -------------------------------------------------
xp = (13.7, 4.2); tp = 3.3
psi = k_airy*xp[1] - omega*tp
eta_ref = A*cos(psi)
etaf = eta_bc(wia)
check("η closure = A cos(kx − ωt) (no ramp)",
      abs(etaf(tp)(VectorValue(xp...)) - eta_ref) < 1e-14)
uxf = ux_bc(wia)
uref = VectorValue((uamp_ref .* cos(psi))...)
check("𝖴x closure = nodal Airy velocities",
      norm(uxf(tp)(VectorValue(xp...)) - uref) < 1e-14)
uyf = uy_bc(wia)
check("𝖴y closure = 0 for θ=0", norm(uyf(tp)(VectorValue(xp...))) < 1e-16)

inc = incident_fields(wia)
check("incident_fields consistent with the BC closures",
      abs(inc.eta(VectorValue(xp...), tp) - eta_ref) < 1e-14 &&
      norm(inc.ux(VectorValue(xp...), tp) - uref) < 1e-14)

# ---- ramp --------------------------------------------------------------------
Tr = 2.0*T
check("ramp: r(0)=0, r(T_ramp/2)=1/2, r(≥T_ramp)=1",
      abs(ramp_value(Tr, 0.0)) < 1e-15 &&
      abs(ramp_value(Tr, Tr/2) - 0.5) < 1e-15 &&
      ramp_value(Tr, Tr + 0.1) == 1.0 && ramp_value(0.0, 0.3) == 1.0)
wr = WaveInput(vert; A=A, T=T, d=d, g=g, T_ramp=Tr, profile=:airy)
check("ramped η closure = r(t)·η",
      abs(eta_bc(wr)(tp)(VectorValue(xp...)) -
          ramp_value(Tr, tp)*eta_ref) < 1e-14)

# ---- AD safety in t: Gridap's own Dirichlet time-derivative machinery --------
tpl   = 1.5*Tr                                        # beyond the ramp: ṙ=0
psipl = k_airy*xp[1] - omega*tpl
detadt_ad = Gridap.ODEs.time_derivative(eta_bc(wr))(tpl)(VectorValue(xp...))
check("∂t η by Gridap time_derivative (AD) = analytic (ramp plateau)",
      abs(detadt_ad - A*omega*sin(psipl)) < 1e-12)
duxdt_ad = Gridap.ODEs.time_derivative(ux_bc(wr))(tpl)(VectorValue(xp...))
check("∂t 𝖴x by Gridap time_derivative (AD) = analytic",
      abs(duxdt_ad[1] - uamp_ref[1]*omega*sin(psipl)) < 1e-12)
# inside the ramp, the product rule must appear: ∂t(r·η) = ṙη + r·∂tη
tin  = 0.7*Tr
rdot = 0.5*pi/Tr*sin(pi*tin/Tr)
psin = k_airy*xp[1] - omega*tin
d_in = Gridap.ODEs.time_derivative(eta_bc(wr))(tin)(VectorValue(xp...))
check("∂t η inside the ramp includes the ṙ product term",
      abs(d_in - (rdot*A*cos(psin) + ramp_value(Tr, tin)*A*omega*sin(psin))) < 1e-12)

# ---- WaveSpec AiryState converter --------------------------------------------
using WaveSpec
spec   = WaveSpec.ContinuousSpectrums.JONSWAP(0.002, T)     # Hs=2 mm, Tp=T
ds     = WaveSpec.SpectralSpreading.DiscreteSpectralSpreading(
             spec, WaveSpec.SpectralSampling.UniformSampling(),
             1.0/(2.5*T), 1.0/(0.55*T), 13; mess=false)
spread = WaveSpec.AngularSpreading.DiscreteAngularSpreading(0.0)  # long-crested
state  = WaveSpec.AiryWaves.AiryState(ds, spread, d)

wis = WaveInput(vert, state; d=d, g=g, profile=:model)
A_ij = WaveSpec.AiryWaves.get_amplitudes(state)
check("AiryState converter: ncomp = nω·nθ bins", wis.ncomp == state.nω*state.nθ)
check("AiryState converter: amplitudes snapshot get_amplitudes",
      isapprox(sort(wis.amps), sort(vec(A_ij)); rtol=1e-12))
Hs_in = 4.0*sqrt(sum(A_ij.^2)/2.0)
Hs_wi = 4.0*sqrt(sum(wis.amps.^2)/2.0)
check("AiryState converter: Hs preserved", abs(Hs_wi - Hs_in)/Hs_in < 1e-12)
check("AiryState converter: T_ramp defaults to 2 peak periods",
      abs(wis.T_ramp - 2.0*2.0*pi/wis.omegas[argmax(wis.amps)]) < 1e-12)
check("AiryState converter: long-crested ⇒ not directional", !wis.directional)

# seed reproducibility: converting the same state twice gives identical phases
wis2 = WaveInput(vert, state; d=d, g=g, profile=:model)
check("AiryState converter: phases reproducible (seeded)",
      wis.phases == wis2.phases)

# k re-solved with the SOLVER's g (WaveSpec uses g=9.80665)
c = argmax(wis.amps)
check("AiryState converter: k re-solved from the model dispersion",
      abs(wis.ks[c] - model_wavenumber(vert, wis.omegas[c], d, g)) < 1e-12)

# multi-component η closure = direct series sum
etafs = eta_bc(wis)
x1, x2, tt = 7.3, 2.1, 11.7
eta_direct = ramp_value(wis.T_ramp, tt) *
    sum(wis.amps[c]*cos(wis.ks[c]*(x1*wis.cos_th[c] + x2*wis.sin_th[c]) -
                        wis.omegas[c]*tt + wis.phases[c]) for c in 1:wis.ncomp)
check("stochastic η closure = direct component series",
      abs(etafs(tt)(VectorValue(x1, x2)) - eta_direct) < 1e-13)

# directional state: uy prescribed flag
spread_dir = WaveSpec.AngularSpreading.DiscreteAngularSpreading(
                 :cosinepow, 0.0, 0.35, -pi/2, pi/2, 5)
state_dir  = WaveSpec.AiryWaves.AiryState(ds, spread_dir, d)
wid = WaveInput(vert, state_dir; d=d, g=g, profile=:model)
check("directional AiryState ⇒ directional flag set", wid.directional)
check("directional: ncomp = nω·nθ", wid.ncomp == state_dir.nω*state_dir.nθ)

println("-" ^ 60)
@printf("  %d PASS, %d FAIL\n", n_pass, n_fail)
n_fail == 0 || error("test_waveinput.jl: $n_fail test(s) failed")
