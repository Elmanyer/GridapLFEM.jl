# ==============================================================
#  seastate.jl — stochastic sea-state analysis (Dirichlet-BC generation runs)
#
#  Analysis for irregular-sea simulations driven by WaveSpec.jl boundary
#  data: Welch PSD estimation of gauge records, spectral moments and Hs,
#  the closed-form JONSWAP target density for overlays (the SAME
#  energy-normalised form WaveSpec.jl uses, so measured-vs-target plots are
#  consistent by construction), zero-upcrossing wave statistics and the
#  Rayleigh exceedance check. Self-contained — no WaveSpec dependency.
# ==============================================================

"""
    jonswap_density(f; Hs, Tp, gamma=3.3) -> S(f) [m²s]

Energy-normalised JONSWAP spectral density (m₀ = (Hs/4)²), identical to
WaveSpec.jl's `get_density(JONSWAP(Hs,Tp,γ), f)`:
`S = Ag (Hs/4)²/fp · 5 fr⁻⁵ exp(−1.25 fr⁻⁴) γ^b`, `b = exp(−(fr−1)²/2σ²)`,
σ = 0.07 (f≤fp) / 0.09 (f>fp), `Ag` from unit-shape integration.
"""
function jonswap_density(f::Real; Hs::Real, Tp::Real, gamma::Real=3.3)
    f <= 1e-6 && return 0.0
    fp = 1.0/Tp
    fr = f*Tp
    return _jonswap_Ag(gamma) * ((Hs/4)^2/fp) * _jonswap_shape(fr, gamma)
end

function _jonswap_shape(fr::Real, gamma::Real)
    fr <= 1e-6 && return 0.0
    σ = fr <= 1.0 ? 0.07 : 0.09
    b = exp(-0.5*((fr - 1.0)/σ)^2)
    return 5.0*fr^-5*exp(-1.25*fr^-4)*gamma^b
end

const _JONSWAP_AG_CACHE = Dict{Float64,Float64}()
function _jonswap_Ag(gamma::Real)
    get!(_JONSWAP_AG_CACHE, Float64(gamma)) do
        frs = range(1e-4, 20.0; length=20000)
        vals = _jonswap_shape.(frs, gamma)
        1.0 / (sum(vals[1:end-1] .+ vals[2:end])/2 * step(frs))
    end
end

"""
    psd_welch(t, v; nseg=8, overlap=0.5, window=:hann) -> (f, S)

Welch power spectral density of a uniformly-sampled record: split into
`nseg` segments with fractional `overlap`, Hann-window and detrend (mean)
each, average the periodograms. One-sided density scaling
`S = 2|X|²/(fs·Σw²)` so that `∫S df = var(v)` — directly comparable to a
wave spectrum S(f) in m²s. Returns frequencies in Hz.
"""
function psd_welch(t::AbstractVector, v::AbstractVector;
                   nseg::Int=8, overlap::Real=0.5, window::Symbol=:hann)
    n  = length(v)
    dt = (t[end]-t[1])/(n-1)
    fs = 1.0/dt
    # segment length from the requested count/overlap; floor at 16 samples so a
    # short record degrades to fewer (longer) segments instead of erroring
    L  = Int(floor(n / (1 + (nseg-1)*(1-overlap))))
    L  = max(L, 16)
    step_ = max(1, Int(floor(L*(1-overlap))))
    w  = window === :hann ? 0.5 .* (1 .- cos.(2π .* (0:L-1) ./ (L-1))) : ones(L)
    U  = sum(abs2, w)
    f  = collect(rfftfreq(L, fs))
    S  = zeros(length(f))
    m  = 0
    i0 = 1
    while i0 + L - 1 <= n
        seg = v[i0:i0+L-1]
        seg = (seg .- mean(seg)) .* w
        X = rfft(seg)
        S .+= (2.0/(fs*U)) .* abs2.(X)
        m += 1
        i0 += step_
    end
    m == 0 && error("psd_welch: record too short for the requested segmentation")
    S ./= m
    if length(f) > 1                       # DC and Nyquist carry no factor 2
        S[1] /= 2; iseven(L) && (S[end] /= 2)
    end
    return f, S
end

"""
    spectral_moment(f, S, n) -> mₙ = ∫ fⁿ S df   (trapezoid)
"""
function spectral_moment(f::AbstractVector, S::AbstractVector, n::Int)
    acc = 0.0
    for i in 1:length(f)-1
        acc += 0.5*(f[i]^n*S[i] + f[i+1]^n*S[i+1])*(f[i+1]-f[i])
    end
    return acc
end

"Significant wave height from a PSD: Hs = 4√m₀."
significant_height(f::AbstractVector, S::AbstractVector) =
    4.0*sqrt(spectral_moment(f, S, 0))

"""
    zero_crossing_heights(t, v) -> Vector{Float64}

Individual wave heights (crest-to-trough) by zero-UPcrossing analysis of a
surface-elevation record (mean removed first).
"""
function zero_crossing_heights(t::AbstractVector, v::AbstractVector)
    x  = v .- mean(v)
    ups = Int[]
    for i in 1:length(x)-1
        x[i] < 0 && x[i+1] >= 0 && push!(ups, i)
    end
    H = Float64[]
    for j in 1:length(ups)-1
        seg = @view x[ups[j]:ups[j+1]]
        push!(H, maximum(seg) - minimum(seg))
    end
    return H
end

"""
    plot_sea_spectrum(t, v; target=nothing, nseg=8, fmax=nothing) -> Plot

Measured Welch PSD of a gauge record with an optional JONSWAP target overlay
`target = (Hs=…, Tp=…)` (optionally `gamma`). Annotates measured vs target Hs.
"""
function plot_sea_spectrum(t::AbstractVector, v::AbstractVector;
                           target=nothing, nseg::Int=8, fmax=nothing)
    f, S = psd_welch(t, v; nseg=nseg)
    Hs_m = significant_height(f, S)
    fm   = fmax === nothing ? (target === nothing ? f[end] : 3.0/target.Tp) : fmax
    sel  = f .<= fm
    p = plot(f[sel], S[sel]; lw=2, label=@sprintf("measured (Hs=%.4g m)", Hs_m),
             xlabel="f [Hz]", ylabel="S(f) [m² s]", title="sea-state spectrum")
    if target !== nothing
        γ  = hasproperty(target, :gamma) ? target.gamma : 3.3
        fg = range(max(f[2], 1e-3), fm; length=400)
        Sg = [jonswap_density(fq; Hs=target.Hs, Tp=target.Tp, gamma=γ) for fq in fg]
        plot!(p, fg, Sg; lw=2, ls=:dash, c=:black,
              label=@sprintf("JONSWAP target (Hs=%.4g m)", target.Hs))
    end
    return p
end

"""
    plot_sea_spectrum(sim, field, x, y; kwargs...) -> Plot

Convenience overload: gauge the simulation at (x,y) first.
"""
plot_sea_spectrum(sim::WaveSimulation, field::AbstractString, x::Real, y::Real;
                  kwargs...) =
    plot_sea_spectrum(timeseries(sim, field, gauge(sim, x, y))...; kwargs...)

"""
    plot_exceedance(t, v; Hs=nothing) -> Plot

Empirical exceedance probability of the zero-upcrossing wave heights vs the
Rayleigh law `P(H>h) = exp(−2h²/Hs²)` of a linear (Gaussian) sea. `Hs`
defaults to `4·std(v)`.
"""
function plot_exceedance(t::AbstractVector, v::AbstractVector; Hs=nothing)
    H  = sort(zero_crossing_heights(t, v); rev=true)
    isempty(H) && error("plot_exceedance: no complete waves in the record")
    Pe = (1:length(H)) ./ (length(H) + 1)
    Hs_v = Hs === nothing ? 4.0*std(v .- mean(v)) : Float64(Hs)
    p = plot(H, Pe; seriestype=:scatter, yscale=:log10, ms=3,
             label=@sprintf("measured (%d waves)", length(H)),
             xlabel="wave height H [m]", ylabel="P(H > h)",
             title="wave-height exceedance")
    hg = range(0, maximum(H)*1.05; length=200)
    plot!(p, hg, exp.(-2 .* hg.^2 ./ Hs_v^2); lw=2, ls=:dash, c=:black,
          label=@sprintf("Rayleigh (Hs=%.4g m)", Hs_v))
    return p
end
