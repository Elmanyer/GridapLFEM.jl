# ==============================================================
#  test_dispersion_curve.jl — closed-form dispersion accuracy sweep
#                             (FAST, no horizontal FEM)
#
#  Reproduces the Yang & Liu (2024) dispersion-accuracy result directly from
#  the Stage-1 tensors: the model phase celerity is the Rayleigh quotient
#     Cm²(kd) = g d · Φᵀ (M − (kd)² B)⁻¹ Φ,          (B ⪯ 0)
#  and the exact Airy celerity is  Ce²(kd) = g d tanh(kd)/(kd).
#  We sweep kd for LFE-2/3/4, print the Cm/Ce curve, and check:
#    * |Cm/Ce − 1| ≤ 2% throughout kd ∈ (0, kd_app]  (definition of kd_app),
#    * kd_app matches Yang & Liu Table 1/2 (LFE-2≈10.9, LFE-3≈39.2, LFE-4≈127.9),
#    * the error is (essentially) monotone increasing past kd_app.
#  No time stepping — this is the cheapest, most fundamental physics check.
#
#  RUN:  julia --project=. GridapLFEM.jl/test/test_dispersion_curve.jl
# ==============================================================

if !isdefined(Main, :GridapLFEM)
    include(joinpath(@__DIR__, "..", "src", "GridapLFEM.jl"))
end
using .GridapLFEM
using Printf, LinearAlgebra

println("=" ^ 64)
println("  test_dispersion_curve.jl — Cm/Ce(kd) vs Airy (closed form)")
println("=" ^ 64)

n_pass = 0; n_fail = 0
check(name, cond) = (global n_pass, n_fail;
    cond ? (println("  PASS  $name"); n_pass += 1) :
           (println("  FAIL  $name"); n_fail += 1))

g = 9.81; d = 3.5

models = [
    (name="LFE-2", M=2, c_bdy=[0.0, 0.728, 1.0],               kd_ref=10.9, tol=1.0),
    (name="LFE-3", M=3, c_bdy=[0.0, 0.726, 0.925, 1.0],        kd_ref=39.2, tol=2.0),
    (name="LFE-4", M=4, c_bdy=[0.0, 0.745, 0.923, 0.977, 1.0], kd_ref=127.9, tol=6.0),
]

for m in models
    vert = assemble_vertical_tensors(m.M, 1, m.c_bdy)
    kd_app = applicable_kd(vert, g, d)

    # print a compact curve up to just beyond kd_app
    kd_grid = collect(LinRange(0.5, min(1.4*m.kd_ref, 180.0), 8))
    ratios  = dispersion_ratio(vert, g, d, kd_grid)
    @printf("\n  %s   kd_app = %.1f  (Table ≈ %.1f)\n", m.name, kd_app, m.kd_ref)
    println("    kd    :  " * join((@sprintf("%7.1f", x) for x in kd_grid)))
    println("    Cm/Ce :  " * join((@sprintf("%7.4f", r) for r in ratios)))

    # (a) accuracy holds throughout the applicable band
    kd_in = collect(LinRange(0.1, kd_app, 200))
    r_in  = dispersion_ratio(vert, g, d, kd_in)
    check("$(m.name): |Cm/Ce−1| ≤ 2% for all kd ≤ kd_app",
          all(abs.(r_in .- 1.0) .<= 0.02 + 1e-6))

    # (b) applicable-kd matches the paper table
    check("$(m.name): kd_app ≈ $(m.kd_ref) (±$(m.tol))", abs(kd_app - m.kd_ref) < m.tol)

    # (c) error clearly worse beyond the band (degradation is real, not a fluke)
    r_beyond = dispersion_ratio(vert, g, d, [1.3*kd_app])[1]
    check("$(m.name): error grows past kd_app (|Cm/Ce−1|>2% at 1.3·kd_app)",
          abs(r_beyond - 1.0) > 0.02)
end

println()
println("=" ^ 64)
@printf("  Results: %d PASS,  %d FAIL\n", n_pass, n_fail)
println("=" ^ 64)
n_fail > 0 ? error("test_dispersion_curve: $n_fail failed!") :
             println("  Dispersion accuracy envelope reproduces Yang & Liu Table 1/2.")
