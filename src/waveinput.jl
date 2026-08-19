# ==============================================================
#  waveinput.jl — Dirichlet boundary wave generation + WaveSpec coupling
#
#  Turns a wave description (regular Airy wave, or a WaveSpec.jl AiryState
#  stochastic sea state) into the time-dependent Dirichlet data the stacked
#  solver needs on a generation boundary:
#      η(x,t)              scalar
#      𝖴x(x,t), 𝖴y(x,t)    VectorValue{Nσ} nodal velocity traces
#
#  Everything is snapshotted into plain Float64 arrays at construction
#  (WaveSpec types never reach the residual/spaces; seeded phases make the
#  data deterministic — identical on every MPI rank and across sessions).
#
#  Vertical polarization per component (ω_c, θ_c):
#    :model (default) — the DISCRETE BALFE-M plane-wave eigenmode:
#         u_j = (g k/ω) [ (Mmat − k²d²B)⁻¹ Φ ]_j · A_c,
#         k from the model dispersion  ω² = g d k² Φᵀ(Mmat − k²d²B)⁻¹Φ.
#      The boundary data is then an exact solution of the linearised
#      discrete system — minimal spurious radiation at the boundary.
#    :airy — sample linear Airy theory at the still-water σ-levels
#         u_j = A_c ω cosh(k d σ_j)/sinh(k d),  k from ω² = gk tanh(kd),
#      with the deep-water guard  cosh(kdσ)/sinh(kd) → exp(kd(σ−1))  (kd>20).
#
#  A Hann ramp r(t) = ½(1−cos(πt/T_ramp)) multiplies all BC data so a cold
#  (zero) start is compatible. Closures are ForwardDiff-safe in t (Gridap
#  computes the Dirichlet time derivative by AD over t).
# ==============================================================

"""
    WaveInput

Immutable component table for Dirichlet boundary wave generation. Built by
the regular-wave constructor or the `WaveSpec.AiryWaves.AiryState` converter;
consumed by `eta_bc`/`ux_bc`/`uy_bc` (transient Dirichlet closures) and
`incident_fields` (plain `(x,t)` functions for hot starts / relaxation zones).
"""
struct WaveInput
    ncomp   :: Int
    Nσ      :: Int
    amps    :: Vector{Float64}   # A_c  [m]
    omegas  :: Vector{Float64}   # ω_c  [rad/s]
    ks      :: Vector{Float64}   # k_c  [rad/m]  (re-solved with the SOLVER's g)
    cos_th  :: Vector{Float64}   # cos θ_c
    sin_th  :: Vector{Float64}   # sin θ_c
    phases  :: Vector{Float64}   # φ_c  [rad]
    Uamp    :: Matrix{Float64}   # ncomp × Nσ nodal speed amplitudes along k̂ (A_c folded in)
    d       :: Float64           # still-water depth at the generation boundary [m]
    g       :: Float64
    T_ramp  :: Float64           # Hann ramp duration [s] (≤0 = off)
    sigma_nodes :: Vector{Float64}
    directional :: Bool          # any θ_c ≠ 0  ⇒ 𝖴y must be prescribed too
    profile :: Symbol            # :model | :airy
end

"""
    sigma_dof_nodes(vert) → Vector{Float64}

σ-coordinate of every vertical DOF of the σ-mesh Lagrange space, in the SAME
order as the tensor mode index j (the tensors are assembled from the unit DOF
vectors of `vert.U_phi`, so free-DOF order ≡ mode order). Exact for nodal
Lagrange bases: interpolate the identity and read the DOF values.
"""
function sigma_dof_nodes(vert)
    return copy(get_free_dof_values(interpolate_everywhere(x -> x[1], vert.U_phi)))
end

"""
    model_wavenumber(vert, omega, d, g) → k

Wavenumber of the DISCRETE BALFE-M dispersion relation
    ω² = g d k² Φᵀ (Mmat − k²d²B)⁻¹ Φ
(secant iteration seeded at the Airy k). The model frequency saturates above
the applicable-kd band; if ω lies beyond it (or the iteration fails) the Airy
wavenumber is returned with a warning — the component is outside the band the
vertical mesh resolves anyway.
"""
function model_wavenumber(vert, omega::Float64, d::Float64, g::Float64)
    Mmat, B, Phi = vert.Mmat, vert.B, vert.Phi
    f(k) = g * d * k^2 * dot(Phi, (Mmat .- B .* (k * d)^2) \ Phi) - omega^2

    k_airy = find_wavenumber(omega, d, g)
    # saturation check: model ω is increasing in k and bounded; if even a very
    # large k cannot reach ω, there is no model solution for this component.
    if f(100.0 * k_airy) < 0.0
        @warn "model_wavenumber: ω=$(omega) rad/s is beyond the model dispersion band " *
              "(kd_airy=$(k_airy*d)); falling back to the Airy wavenumber"
        return k_airy
    end
    k0, k1 = 0.9 * k_airy, k_airy
    f0, f1 = f(k0), f(k1)
    for _ in 1:100
        df = (f1 - f0) / (k1 - k0)
        k2 = k1 - f1 / df
        k2 = k2 > 0 ? k2 : 0.5 * k1          # keep positive
        abs(k2 - k1) < 1e-13 * abs(k1) && return k2
        k0, f0 = k1, f1
        k1, f1 = k2, f(k2)
    end
    @warn "model_wavenumber: secant iteration did not converge for ω=$(omega); " *
          "using the Airy wavenumber"
    return k_airy
end

# nodal speed-amplitude vector (along k̂) for one component
function _component_uamp(vert, sigma_nodes::Vector{Float64}, profile::Symbol,
                         A::Float64, omega::Float64, k::Float64,
                         d::Float64, g::Float64)
    Nσ = vert.N_dof
    if profile == :model
        M_eff = vert.Mmat .- vert.B .* (k * d)^2
        m = M_eff \ vert.Phi
        return A * (g * k / omega) .* m
    elseif profile == :airy
        kd = k * d
        return [kd > 20.0 ? A * omega * exp(kd * (sigma_nodes[j] - 1.0)) :
                            A * omega * cosh(kd * sigma_nodes[j]) / sinh(kd)
                for j in 1:Nσ]
    else
        error("WaveInput: profile must be :model or :airy (got :$profile)")
    end
end

function _build_waveinput(vert, amps, omegas, thetas, phases,
                          d::Float64, g::Float64, T_ramp::Float64,
                          profile::Symbol)
    ncomp = length(amps)
    @assert length(omegas) == ncomp && length(thetas) == ncomp && length(phases) == ncomp
    ncomp > 0 || error("WaveInput: no wave components (all amplitudes zero?)")
    sig = sigma_dof_nodes(vert)
    Nσ  = vert.N_dof
    ks  = zeros(ncomp)
    Uamp = zeros(ncomp, Nσ)
    for c in 1:ncomp
        ks[c] = profile == :model ? model_wavenumber(vert, omegas[c], d, g) :
                                    find_wavenumber(omegas[c], d, g)
        Uamp[c, :] .= _component_uamp(vert, sig, profile, amps[c], omegas[c],
                                      ks[c], d, g)
    end
    # long-crested tolerance: WaveSpec's no-spreading dummy bin carries |θ| ≤ 1e-3
    directional = any(t -> abs(sin(t)) > 1e-2, thetas)
    return WaveInput(ncomp, Nσ, collect(Float64, amps), collect(Float64, omegas),
                     ks, cos.(thetas), sin.(thetas), collect(Float64, phases),
                     Uamp, d, g, T_ramp, sig, directional, profile)
end

"""
    WaveInput(vert; A, T, d, g=9.81, theta=0.0, phase=0.0,
                    T_ramp=2T, profile=:model)

Regular (monochromatic) wave of amplitude `A` [m], period `T` [s],
propagation direction `theta` [rad] (vs +x).
"""
function WaveInput(vert;
                   A::Float64, T::Float64, d::Float64, g::Float64=g,
                   theta::Float64=0.0, phase::Float64=0.0,
                   T_ramp::Float64=2.0*T, profile::Symbol=:model)
    return _build_waveinput(vert, [A], [2.0*pi/T], [theta], [phase],
                            d, g, T_ramp, profile)
end

"""
    WaveInput(vert, amps, omegas; d, g=9.81, thetas=0…, phases=0…,
                                  T_ramp=nothing, profile=:model)

Generic multi-component constructor from plain vectors (deterministic
multichromatic seas, hand-built spectra, tests). `T_ramp` defaults to twice
the peak (largest-amplitude) period.
"""
function WaveInput(vert, amps::Vector{Float64}, omegas::Vector{Float64};
                   d::Float64, g::Float64=g,
                   thetas::Vector{Float64}=zeros(length(amps)),
                   phases::Vector{Float64}=zeros(length(amps)),
                   T_ramp=nothing, profile::Symbol=:model)
    Tr = isnothing(T_ramp) ? 2.0 * 2.0*pi / omegas[argmax(amps)] : Float64(T_ramp)
    return _build_waveinput(vert, amps, omegas, thetas, phases, d, g, Tr, profile)
end

"""
    WaveInput(vert, state; d, g=9.81, T_ramp=nothing, profile=:model, amp_tol=0.0)

Converter from a `WaveSpec.AiryWaves.AiryState` stochastic sea state.
Snapshots the discrete bins into plain arrays: amplitudes
`WaveSpec.AiryWaves.get_amplitudes(state)` (energy-preserving normalisation),
seeded random phases (deterministic per `state.seed`), directions `state.θ`.

The wavenumbers stored in the state (solved with WaveSpec's g=9.80665 at
`state.h`) are DISCARDED and re-solved with the solver's `g` and the boundary
depth `d` (`:airy`) or the model dispersion (`:model`). Warns when
`state.h ≠ d`. `T_ramp` defaults to twice the peak period (`2·2π/ω_peak`);
`amp_tol` drops bins with `A_c ≤ amp_tol · max(A)`.
"""
function WaveInput(vert, state::WaveSpec.AiryWaves.AiryState;
                   d::Float64, g::Float64=g, T_ramp=nothing,
                   profile::Symbol=:model, amp_tol::Float64=0.0)
    if !isapprox(state.h, d; rtol=1e-6)
        @warn "WaveInput: AiryState depth h=$(state.h) m ≠ boundary depth d=$(d) m — " *
              "the spectrum was sampled at a different depth"
    end
    A_ij = WaveSpec.AiryWaves.get_amplitudes(state)        # nω × nθ
    ph_ij = WaveSpec.AiryWaves.get_random_phases(state)    # nω × nθ (seeded)
    nω, nθ = state.nω, state.nθ
    Amax = maximum(A_ij)
    amps   = Float64[]; omegas = Float64[]
    thetas = Float64[]; phases = Float64[]
    for i in 1:nω, j in 1:nθ
        A_ij[i, j] > amp_tol * Amax || continue
        push!(amps, A_ij[i, j]); push!(omegas, state.ω[i])
        push!(thetas, state.θ[j]); push!(phases, ph_ij[i, j])
    end
    isempty(amps) && error("WaveInput: every bin was dropped (amp_tol too large?)")
    Tr = isnothing(T_ramp) ? 2.0 * 2.0*pi / omegas[argmax(amps)] : Float64(T_ramp)
    return _build_waveinput(vert, amps, omegas, thetas, phases, d, g, Tr, profile)
end

# --------------------------------------------------------------
#  Evaluation core.
#
#  TYPE-GENERICITY RULE (load-bearing): Gridap computes the Dirichlet time
#  derivative ġ(t) by ForwardDiff over t (`Gridap.ODEs.time_derivative`), so
#  `t` may arrive as a Dual number. Every accumulator below is therefore
#  seeded with `zero(t)` (Dual-aware zero) and no intermediate is annotated
#  `::Float64`. The `t < T_ramp` branch is Dual-safe (comparison uses the
#  value part) and the ramp is C¹ at both ends, so the AD derivative is
#  continuous. Verified against the analytic ∂t in test_waveinput.jl.
# --------------------------------------------------------------

"Hann start-up ramp r(t): 0 → 1 over T_ramp (1 when T_ramp ≤ 0). C¹ everywhere."
function ramp_value(T_ramp::Float64, t)
    T_ramp <= 0.0 && return one(t)
    return t < T_ramp ? 0.5 * (1.0 - cos(pi * t / T_ramp)) : one(t)
end

@inline function _eta_val(wi::WaveInput, x1, x2, t)
    acc = zero(t) + 0.0
    @inbounds for c in 1:wi.ncomp
        psi = wi.ks[c] * (x1 * wi.cos_th[c] + x2 * wi.sin_th[c]) -
              wi.omegas[c] * t + wi.phases[c]
        acc += wi.amps[c] * cos(psi)
    end
    return ramp_value(wi.T_ramp, t) * acc
end

# Stacked nodal velocity component: dir = wi.cos_th (x) or wi.sin_th (y).
# The ntuple over Val(N) builds the VectorValue{Nσ} without allocation; the
# per-point cost is ncomp·Nσ trig evaluations — negligible for boundary-DOF
# counts (the Dirichlet data is interpolated on one boundary only).
@inline function _uvec_val(wi::WaveInput, x1, x2, t, dir::Vector{Float64},
                           ::Val{N}) where {N}
    rt = ramp_value(wi.T_ramp, t)
    vals = ntuple(Val(N)) do j
        acc = zero(t) + 0.0
        @inbounds for c in 1:wi.ncomp
            psi = wi.ks[c] * (x1 * wi.cos_th[c] + x2 * wi.sin_th[c]) -
                  wi.omegas[c] * t + wi.phases[c]
            acc += wi.Uamp[c, j] * dir[c] * cos(psi)
        end
        rt * acc
    end
    return VectorValue(vals...)
end

"""
    eta_bc(wi) → t -> x -> η_inc(x,t)

Transient Dirichlet closure for the free surface (Gridap
`TransientTrialFESpace` format).
"""
eta_bc(wi::WaveInput) = t -> (x -> _eta_val(wi, x[1], x[2], t))

"""
    ux_bc(wi) → t -> x -> VectorValue{Nσ}  (nodal x-velocities)
"""
ux_bc(wi::WaveInput) = t -> (x -> _uvec_val(wi, x[1], x[2], t, wi.cos_th, Val(wi.Nσ)))

"""
    uy_bc(wi) → t -> x -> VectorValue{Nσ}  (nodal y-velocities; directional seas)
"""
uy_bc(wi::WaveInput) = t -> (x -> _uvec_val(wi, x[1], x[2], t, wi.sin_th, Val(wi.Nσ)))

"""
    incident_fields(wi) → (eta, ux, uy)

The same incident-wave data as plain `(x, t)` functions (valid in the whole
domain under linear theory) — hot-start initial conditions and
relaxation-zone targets.
"""
function incident_fields(wi::WaveInput)
    return (eta = (x, t) -> _eta_val(wi, x[1], x[2], t),
            ux  = (x, t) -> _uvec_val(wi, x[1], x[2], t, wi.cos_th, Val(wi.Nσ)),
            uy  = (x, t) -> _uvec_val(wi, x[1], x[2], t, wi.sin_th, Val(wi.Nσ)))
end

"""
    waveinput_summary(wi) — banner block describing the generation data.
"""
function waveinput_summary(wi::WaveInput; io::IO=stdout)
    m0 = sum(wi.amps .^ 2) / 2.0
    Hs = 4.0 * sqrt(m0)
    ipk = argmax(wi.amps)
    Tp  = 2.0 * pi / wi.omegas[ipk]
    kds = wi.ks .* wi.d
    println(io, "=== Dirichlet wave generation (WaveInput) ===")
    @printf(io, "  components: %d | profile: %s | directional: %s\n",
            wi.ncomp, string(wi.profile), string(wi.directional))
    @printf(io, "  Hs=%.4f m | Tp=%.3f s | kd ∈ [%.3f, %.3f] | d=%.3f m\n",
            Hs, Tp, minimum(kds), maximum(kds), wi.d)
    @printf(io, "  ramp: T_ramp=%.3f s (Hann)\n", wi.T_ramp)
    flush(io)
    return nothing
end
