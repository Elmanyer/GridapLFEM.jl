# ==============================================================
#  diagnostics.jl — physical diagnostics + linear (Airy) theory overlays
# ==============================================================

# ---- linear (Airy) theory (for validation overlays) --------------------------
"Airy phase celerity Ce = √(g tanh(kd)/k)."
airy_Ce(k, d; g=9.81) = sqrt(g * tanh(k*d) / k)

"Solve ω² = g k tanh(kd) for k (Newton), returning k and kd."
function airy_kd(ω, d; g=9.81)
    k = ω^2/g
    for _ in 1:100
        f = g*k*tanh(k*d) - ω^2
        df = g*(tanh(k*d) + k*d*(1 - tanh(k*d)^2))
        k -= f/df; abs(f) < 1e-13 && break
    end
    return k, k*d
end

"Normalised Airy vertical-velocity profile shape sinh(kd·σ)/sinh(kd) (0 at bed, 1 at surface)."
airy_w_shape(σ, kd) = sinh(kd*σ) / sinh(kd)

"Normalised Airy dynamic-pressure profile shape cosh(kd·σ)/cosh(kd) (→1 at surface)."
airy_p_shape(σ, kd) = cosh(kd*σ) / cosh(kd)

# ---- ring waves: radial amplitude profile ------------------------------------
"""
    radial_profile(sim, field, center, radii; ω, along=:+x) -> (r, amp)

Steady amplitude of `field` at gauges placed at the given `radii` from `center`
(default along +x). For a point source, `amp·√r` should be ~constant in the far
field (cylindrical spreading). If `ω` is given, the amplitude is the DFT amplitude
at ω; otherwise the peak steady amplitude.
"""
function radial_profile(sim::WaveSimulation, field::AbstractString,
                        center::Tuple, radii::AbstractVector; ω=nothing, along=:x)
    amp = similar(collect(Float64, radii))
    for (i, r) in enumerate(radii)
        pt = along === :x ? (center[1]+r, center[2]) :
             along === :y ? (center[1], center[2]+r) :
                            (center[1]+r/√2, center[2]+r/√2)
        g = gauge(sim, pt...)
        t, v = timeseries(sim, field, g)
        amp[i] = ω === nothing ? steady_amplitude(t, v) : amplitude_at(t, v, ω)
    end
    return collect(Float64, radii), amp
end

# ---- shoal / bar: harmonic growth along a transect ---------------------------
"""
    harmonic_growth(sim, y0, ω; xs, n=3, field="eta") -> (xs, H)

DFT amplitude of the first `n` harmonics of `field` at gauges along `y=y0` for
each x in `xs`. `H` is an `n × length(xs)` matrix (row m = m-th harmonic H_m(x)).
Shows harmonic generation over a submerged bar/shoal.
"""
function harmonic_growth(sim::WaveSimulation, y0::Real, ω::Real;
                         xs::AbstractVector, n::Int=3, field::AbstractString="eta")
    H = zeros(n, length(xs))
    for (j, x) in enumerate(xs)
        t, v = timeseries(sim, field, gauge(sim, x, y0))
        H[:, j] .= harmonic_amplitudes(t, v, ω; n=n)
    end
    return collect(Float64, xs), H
end

# ---- integral diagnostics (grid quadrature) ----------------------------------
"""
    mass_integral(sim, field, it) -> Float64

∫ field dΩ at snapshot `it`, by trapezoidal quadrature on the regularised
Cartesian node grid (requires `sim.grid`). A quick conservation proxy.
"""
function mass_integral(sim::WaveSimulation, field::AbstractString, it::Int)
    sim.grid === nothing && error("mass_integral needs a regularised grid (regularize!)")
    F = grid_field(sim, field, it)                 # Nx × Ny
    gv = sim.grid
    # trapezoid weights in x and y
    wx = _trap_weights(gv.xs); wy = _trap_weights(gv.ys)
    return sum(wx[i]*wy[j]*F[i,j] for i in eachindex(gv.xs), j in eachindex(gv.ys))
end

function _trap_weights(x::AbstractVector)
    n = length(x); w = zeros(n)
    for i in 1:n
        lo = i == 1 ? x[1] : (x[i-1]+x[i])/2
        hi = i == n ? x[n] : (x[i]+x[i+1])/2
        w[i] = hi - lo
    end
    return w
end
