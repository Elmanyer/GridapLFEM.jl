# ==============================================================
#  test_vertical_profile.jl — vertical velocity profile vs Airy (FAST)
#
#  Semi-analytical analogue of Yang & Liu (2024) §3.2 (Figs 6–8): the model
#  vertical-velocity profile must match the linear (Airy) profile up to the
#  applicable kd, and degrade beyond it.
#
#  For a linear progressive wave on a flat bed, w ≈ −d Σⱼ (∇·uⱼ) φ̄ⱼ(σ) with
#  the modal amplitudes given by the dispersion eigenvector
#     û ∝ (M − (kd)² B)⁻¹ Φ,
#  so the model profile shape (normalised to the surface) is
#     ŵ_model(σ) = Σⱼ ûⱼ φ̄ⱼ(σ) / (Φ·û),        φ̄ⱼ = ∫₀^σ φⱼ,  φ̄ⱼ(0)=0.
#  The exact Airy shape (z+d = σd) normalised the same way is
#     ŵ_Airy(σ)  = sinh(k σ d)/sinh(kd)  ÷  1  =  sinh(kd σ)/sinh(kd).
#  Both vanish at the bed (σ=0) and equal 1 at the surface (σ=1). We compare the
#  L∞ profile error over σ∈[0,1] at several kd and require it small inside the
#  applicable band and clearly larger past it.
#
#  RUN:  julia --project=. GridapLFEM.jl/test/test_vertical_profile.jl
# ==============================================================

if !isdefined(Main, :GridapLFEM)
    include(joinpath(@__DIR__, "..", "src", "GridapLFEM.jl"))
end
using .GridapLFEM
using Gridap
using Printf, LinearAlgebra

println("=" ^ 64)
println("  test_vertical_profile.jl — w(σ) profile vs Airy (semi-analytical)")
println("=" ^ 64)

n_pass = 0; n_fail = 0
check(name, cond, extra="") = (global n_pass, n_fail;
    cond ? (println("  PASS  $name $extra"); n_pass += 1) :
           (println("  FAIL  $name $extra"); n_fail += 1))

g = 9.81; d = 3.5

"L∞ error between the model and Airy normalised w-profiles at wavenumber-depth kd."
function profile_error(vert, kd; nσ=41)
    Meff = vert.Mmat .- (kd^2) .* vert.B          # M − (kd)²B ≻ 0
    û    = Meff \ vert.Phi                         # dispersion eigenvector (∝ modal amps)
    surf = dot(vert.Phi, û)                        # Σⱼ ûⱼ φ̄ⱼ(1) = Φ·û
    σs   = LinRange(0.0, 1.0, nσ)
    err  = 0.0
    for σ in σs
        # model profile: Σⱼ ûⱼ φ̄ⱼ(σ) / surface
        wm = 0.0
        for j in 1:vert.N_dof
            wm += û[j] * vert.phi_int_fns[j](VectorValue(Float64(σ)))
        end
        wm /= surf
        # Airy profile (sinh), same normalisation (=1 at σ=1)
        wa = sinh(kd*σ) / sinh(kd)
        err = max(err, abs(wm - wa))
    end
    return err
end

# ---- LFE-2 (applicable to kd≈10.9): accurate at kd=1,3,5; degraded at kd=14 ----
vert2 = assemble_vertical_tensors(2, 1, [0.0, 0.728, 1.0])
println("\n  LFE-2  (kd_app ≈ $(round(applicable_kd(vert2,g,d),digits=1)))")
for kd in (1.0, 3.0, 5.0)
    e = profile_error(vert2, kd)
    @printf("    kd=%4.1f   L∞ w-profile error = %.4f\n", kd, e)
    check("LFE-2 w-profile matches Airy at kd=$kd (L∞ < 5%)", e < 0.05,
          "($(round(100e,digits=2))%)")
end
e_deep2 = profile_error(vert2, 14.0)
@printf("    kd=14.0   L∞ w-profile error = %.4f (beyond band)\n", e_deep2)
check("LFE-2 w-profile degrades past kd_app (kd=14 error > 5%)", e_deep2 > 0.05,
      "($(round(100e_deep2,digits=2))%)")

# ---- LFE-3 (applicable to kd≈39): accurate at kd=5,15,30 -------------------------
vert3 = assemble_vertical_tensors(3, 1, [0.0, 0.726, 0.925, 1.0])
println("\n  LFE-3  (kd_app ≈ $(round(applicable_kd(vert3,g,d),digits=1)))")
for kd in (5.0, 15.0, 30.0)
    e = profile_error(vert3, kd)
    @printf("    kd=%4.1f   L∞ w-profile error = %.4f\n", kd, e)
    check("LFE-3 w-profile matches Airy at kd=$kd (L∞ < 6%)", e < 0.06,
          "($(round(100e,digits=2))%)")
end

println()
println("=" ^ 64)
@printf("  Results: %d PASS,  %d FAIL\n", n_pass, n_fail)
println("=" ^ 64)
n_fail > 0 ? error("test_vertical_profile: $n_fail failed!") :
             println("  Vertical velocity profile reproduces Airy within the applicable band.")
