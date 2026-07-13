# ==============================================================
#  test_vertical_alg.jl — vertical tensor set (FAST, no horizontal FEM)
#
#  Checks on assemble_vertical_tensors_alg:
#    identities (Mmat sym/PSD, ΣΦ=1, φ_int BCs), the leading-pressure
#    identities P[:,:,3] = −B and Kcal = Pcal − ∫σΘφᵢ, and the dispersion
#    bridge vs Yang & Liu (2024) Table 1 applicable-kd values.
#
#  RUN:  julia --project=. LFEM_2D/test/test_vertical_alg.jl
# ==============================================================

if !isdefined(Main, :LFEModelAlg)
    include(joinpath(@__DIR__, "..", "src", "LFEModelAlg.jl"))
end
using .LFEModelAlg
using LinearAlgebra, Printf

println("=" ^ 60)
println("  test_vertical_alg.jl — vertical tensor set")
println("=" ^ 60)

n_pass = 0; n_fail = 0
function check(name, cond)
    global n_pass, n_fail
    if cond; println("  PASS  $name"); n_pass += 1
    else;    println("  FAIL  $name"); n_fail += 1; end
end

g = 9.81; d = 3.5

# ---- LFE-2 -------------------------------------------------------------------
vert2 = assemble_vertical_tensors_alg(2, 1, [0.0, 0.728, 1.0])
check("LFE-2: N_dof = 3", vert2.N_dof == 3)
check("LFE-2: ΣΦ = 1", abs(sum(vert2.Phi) - 1.0) < 1e-12)
check("LFE-2: Mmat symmetric", norm(vert2.Mmat - vert2.Mmat') < 1e-13)
check("LFE-2: Mmat positive definite", all(eigvals(Symmetric(vert2.Mmat)) .> 0))
check("LFE-2: B symmetric ≤ 0", norm(vert2.B - vert2.B') < 1e-13 &&
                                all(eigvals(Symmetric(vert2.B)) .< 1e-14))

# antiderivative boundary values: φⱼ_int(0)=0, φⱼ_int(1)=Φⱼ
p0 = VectorValue(0.0); p1 = VectorValue(1.0)
check("LFE-2: φⱼ_int(0) = 0",
      all(abs(vert2.phi_int_fns[j](p0)) < 1e-12 for j in 1:vert2.N_dof))
check("LFE-2: φⱼ_int(1) = Φⱼ",
      all(abs(vert2.phi_int_fns[j](p1) - vert2.Phi[j]) < 1e-12 for j in 1:vert2.N_dof))

# leading-pressure identities
check("LFE-2: P[:,:,3] = −B (dispersion carrier identity)",
      norm(vert2.P[:,:,3] + vert2.B) < 1e-12)
check("LFE-2: Mcal fully symmetric in (i,k,j)",
      all(abs(vert2.Mcal[i,k,j] - vert2.Mcal[k,i,j]) < 1e-13 &&
          abs(vert2.Mcal[i,k,j] - vert2.Mcal[i,j,k]) < 1e-13
          for i in 1:3, k in 1:3, j in 1:3))

# Fubini split: Kcal = Pcal − ∫σΘφᵢ  ⇒  Pcal − Kcal = ∫σΘφᵢ; for the strictly
# positive components c=1..5 (Θ ≥ 0, σφᵢ ≥ 0) this difference must be ≥ 0.
dPK = vert2.Pcal .- vert2.Kcal
check("LFE-2: Pcal − Kcal = ∫σΘφᵢ ≥ 0 for components 1–5",
      all(dPK[:, :, :, 1:5] .> -1e-12))
check("LFE-2: Pcal assembled (nonzero)", maximum(abs.(vert2.Pcal)) > 1e-6)

# dispersion bridge — Yang & Liu (2024) Table 1
kd2 = applicable_kd_alg(vert2, g, d)
@printf("  LFE-2 applicable kd = %.1f  (Table 1: ~10.9)\n", kd2)
check("LFE-2: applicable kd ≈ 10.9 (±1.0)", abs(kd2 - 10.9) < 1.0)

# ---- LFE-3 -------------------------------------------------------------------
vert3 = assemble_vertical_tensors_alg(3, 1, [0.0, 0.726, 0.925, 1.0])
kd3 = applicable_kd_alg(vert3, g, d)
@printf("  LFE-3 applicable kd = %.1f  (Table 1: ~39.2)\n", kd3)
check("LFE-3: ΣΦ = 1", abs(sum(vert3.Phi) - 1.0) < 1e-12)
check("LFE-3: applicable kd ≈ 39.2 (±2.0)", abs(kd3 - 39.2) < 2.0)
check("LFE-3: P[:,:,3] = −B", norm(vert3.P[:,:,3] + vert3.B) < 1e-12)

println()
println("=" ^ 60)
@printf("  Results: %d PASS,  %d FAIL\n", n_pass, n_fail)
println("=" ^ 60)
n_fail > 0 ? error("test_vertical_alg: $n_fail failed!") :
             println("  Vertical tensor set validated.")
