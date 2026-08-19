# ==============================================================
#  errors.jl — L² error norms and convergence-rate fitting
#
#  Template: GridapUtilities.jl (GridapSWE.jl), which defines
#      l2(u,dΩ) = sqrt(sum( ∫( u⊙u )*dΩ ))
#  and forms the error as `Ua(tn) - Uh` before taking that norm. The same idea is
#  used here, with three adaptations the BALFE-M layout and the MMS demand:
#
#  1. STACKED FIELDS. Velocity is two VectorValue{Nσ} fields, so `ΔU⊙ΔU` already
#     contracts over the layer index j. e_u = (Σⱼ‖uⱼ−u*ⱼ‖²)^{1/2} therefore comes
#     out of a single ∫ per horizontal component — no loop over layers.
#
#  2. REFERENCE IS THE EXACT FIELD, NEVER ITS INTERPOLANT. Comparing against
#     Π_h u* would subtract exactly the interpolation error that carries the
#     convergence rate, and the measured slope would be meaningless (this is an
#     explicit requirement in ValidationTests.tex §"Implementation requirements").
#     `exact` is therefore always wrapped as an analytic CellField.
#
#  3. ELEVATED QUADRATURE. The error integral must be computed on a measure of
#     HIGHER degree than the one used to assemble the operator: at degree 2p the
#     quadrature error is itself O(h^{2p}) and would saturate a 3rd-order rate.
#     Build the error measure with `error_measure(trian, p)` below.
# ==============================================================

"""
    l2(u, dΩ) → Float64

L² norm `sqrt(∫ u⊙u dΩ)`. `u` may be scalar- or (stacked) vector-valued; `⊙`
contracts every component, so a `VectorValue{Nσ}` field sums over the layers.
"""
l2(u, dΩ) = sqrt(sum( ∫( u ⊙ u ) * dΩ ))

"""
    error_measure(trian, p_horizontal; extra=2) → Measure

Quadrature for error evaluation: degree `2(p+1)+extra`, strictly higher than the
assembly degree, so the reported error is discretisation error and not quadrature
error. Saturating a rate on quadrature is a classic false negative.
"""
error_measure(trian, p_horizontal::Int; extra::Int = 2) =
    Measure(trian, 2*(p_horizontal + 1) + extra)

"""
    l2_error(uh, exact, trian, dΩ) → Float64

`‖uh − exact‖_{L²}` with `exact` an analytic function of `x` (NOT an FEFunction and
NOT an interpolant). Returned in the same units as the field.
"""
function l2_error(uh, exact, trian, dΩ)
    e_cf = CellField(exact, trian)
    return l2(uh - e_cf, dΩ)
end

"""
    l2_norm_exact(exact, trian, dΩ) → Float64

`‖exact‖_{L²}`, the denominator for a relative error.
"""
l2_norm_exact(exact, trian, dΩ) = l2(CellField(exact, trian), dΩ)

"""
    convergence_rate(hs, errs) → (slope, pairwise)

Least-squares slope of `log(err)` against `log(h)` — the verification result — plus
the PAIRWISE rates `log(e_i/e_{i+1}) / log(h_i/h_{i+1})`.

Both are reported deliberately: a least-squares slope averages a rate that is
drifting (pre-asymptotic coarse meshes, or an error floor at the fine end) into a
single plausible-looking number, whereas the pairwise sequence shows it. Read the
pairwise values first; quote the fit only when they are flat.
"""
function convergence_rate(hs::AbstractVector, errs::AbstractVector)
    length(hs) == length(errs) ||
        error("convergence_rate: hs and errs must have equal length")
    length(hs) ≥ 2 || error("convergence_rate: need at least two refinement levels")
    all(>(0), errs) ||
        error("convergence_rate: a non-positive error ($(minimum(errs))) — the exact " *
              "solution is probably being compared against its own interpolant")
    x = log.(hs); y = log.(errs)
    x̄ = sum(x)/length(x); ȳ = sum(y)/length(y)
    slope = sum((x .- x̄) .* (y .- ȳ)) / sum((x .- x̄).^2)
    pairwise = [ log(errs[i]/errs[i+1]) / log(hs[i]/hs[i+1]) for i in 1:length(hs)-1 ]
    return slope, pairwise
end

"""
    refinement_table(io, label, hs, e_eta, e_u; expected=3.0) → (p_eta, p_u)

Print a refinement table with pairwise and fitted rates, and return the fitted
slopes. `hs` is the refinement parameter (mesh size `h`, or `Δt` for a temporal
study — the fit is the same, only the expected value changes).
"""
#  `expected_u` defaults to `expected`, so existing single-expectation callers are
#  unchanged. It exists because THE TWO FIELDS HAVE DIFFERENT OPTIMA under a mixed
#  (Taylor-Hood-like) FE pairing: η ∈ Q_{p_e} converges at p_e+1 while u ∈ Q_{p_u}
#  converges at p_u+1. Printing one number for both invites reading a perfectly
#  optimal velocity rate as a failure.
function refinement_table(io::IO, label::AbstractString,
                          hs::AbstractVector, e_eta::AbstractVector, e_u::AbstractVector;
                          expected::Float64 = 3.0,
                          expected_u::Float64 = expected)
    p_eta, pw_eta = convergence_rate(hs, e_eta)
    p_u,   pw_u   = convergence_rate(hs, e_u)
    println(io, "\n", "="^74)
    println(io, "  $label — expected slope: eta $(expected), u $(expected_u)")
    println(io, "="^74)
    @printf(io, "  %-12s %-16s %-10s %-16s %-10s\n", "h", "e_eta", "rate", "e_u", "rate")
    for i in eachindex(hs)
        re = i == 1 ? "  —   " : @sprintf("%6.3f", pw_eta[i-1])
        ru = i == 1 ? "  —   " : @sprintf("%6.3f", pw_u[i-1])
        @printf(io, "  %-12.6g %-16.8e %-10s %-16.8e %-10s\n",
                hs[i], e_eta[i], re, e_u[i], ru)
    end
    @printf(io, "\n  fitted slope:  p_eta = %.3f (expected %.1f)   p_u = %.3f (expected %.1f)\n",
            p_eta, expected, p_u, expected_u)
    return p_eta, p_u
end

refinement_table(label, hs, e_eta, e_u; expected=3.0, expected_u=expected) =
    refinement_table(stdout, label, hs, e_eta, e_u;
                     expected=expected, expected_u=expected_u)
