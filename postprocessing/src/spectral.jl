# ==============================================================
#  spectral.jl — DFT, spectra, harmonic amplitudes, celerity
# ==============================================================

"Second-half (steady) window indices of a time record."
_steady(t) = (length(t) ÷ 2):length(t)

"""
    dft_at(t, v, ω) -> Complex

Discrete Fourier coefficient of signal `v(t)` at angular frequency `ω`, over the
steady (second-half) window: `Σ v·e^{-iωt}`.
"""
function dft_at(t::AbstractVector, v::AbstractVector, ω::Real)
    w = _steady(t); tv = @view t[w]; vv = @view v[w]
    return sum(vv .* exp.(-im .* ω .* tv))
end

"Physical amplitude of the `ω`-component: `2|DFT|/N` over the steady window."
function amplitude_at(t, v, ω)
    w = _steady(t)
    return 2 * abs(dft_at(t, v, ω)) / length(w)
end

"Phase (radians) of the `ω`-component over the steady window."
phase_at(t, v, ω) = angle(dft_at(t, v, ω))

"""
    spectrum(t, v; window=:hann) -> (f, amp)

One-sided amplitude spectrum of a uniformly-sampled signal (FFTW). `f` in Hz
(cycles per unit time), `amp` the single-sided amplitude.
"""
function spectrum(t::AbstractVector, v::AbstractVector; window::Symbol=:hann)
    n = length(v); dt = (t[end]-t[1])/(n-1)
    w = window === :hann ? (0.5 .* (1 .- cos.(2π .* (0:n-1) ./ (n-1)))) : ones(n)
    x = (v .- mean(v)) .* w
    V = rfft(x); f = rfftfreq(n, 1/dt)
    amp = (2/sum(w)) .* abs.(V)
    return f, amp
end

"""
    harmonic_amplitudes(t, v, ω; n=3) -> Vector

Amplitudes of the first `n` harmonics (ω, 2ω, …, nω) via DFT at each — the
Stokes bound-harmonic content.
"""
harmonic_amplitudes(t, v, ω; n::Int=3) = [amplitude_at(t, v, m*ω) for m in 1:n]

"Peak amplitude over the steady window (envelope proxy)."
function steady_amplitude(t, v)
    w = _steady(t)
    return maximum(abs, @view v[w])
end

"""
    celerity(sim, field, g1::Gauge, g2::Gauge, ω) -> Cm

Phase celerity from two downstream gauges by DFT phase differencing at ω:
`k = -arg(A2/A1)/Δx`, `Cm = ω/k`. Robust for any gauge separation.
"""
function celerity(sim::WaveSimulation, field::AbstractString,
                  g1::Gauge, g2::Gauge, ω::Real)
    t, v1 = timeseries(sim, field, g1)
    _, v2 = timeseries(sim, field, g2)
    A1 = dft_at(t, v1, ω); A2 = dft_at(t, v2, ω)
    (abs(A1) < 1e-14 || abs(A2) < 1e-14) && return NaN
    dphi = angle(A2/A1); dphi > 0 && (dphi -= 2π)
    dx = hypot(g2.x - g1.x, g2.y - g1.y)
    k = -dphi / dx
    return k > 0 ? ω/k : NaN
end
