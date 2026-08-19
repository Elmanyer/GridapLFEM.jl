# ==============================================================
#  test_primitives.jl — algebraic primitives (FAST)
#
#  Verifies the load-bearing conventions of the stacked formulation:
#    * alg_to_tensor2/alg_to_tensor3 preserve index order,
#    * double_contraction contracts the TRAILING two indices,
#    * W ⋅ ThirdOrder contracts the FIRST index,
#    * ∂x/∂y = e⋅∇f orientation on VectorValue{Nσ} FE fields,
#    * outer product (a⊗b)[k,j] = a[k]b[j].
#
#  RUN:  julia --project=. GridapBALFEM.jl/test/test_primitives.jl
# ==============================================================

using GridapBALFEM
using Gridap
using LinearAlgebra, Printf

println("=" ^ 60)
println("  test_primitives.jl — stacked-formulation primitives")
println("=" ^ 60)

n_pass = 0; n_fail = 0
function check(name, cond)
    global n_pass, n_fail
    if cond; println("  PASS  $name"); n_pass += 1
    else;    println("  FAIL  $name"); n_fail += 1; end
end

# ---- constant-tensor constructors ---------------------------------------------
let N = 3
    Mtest = reshape(collect(1.0:9.0), N, N)
    T2 = alg_to_tensor2(Mtest)
    check("alg_to_tensor2 index order", all(T2[i,j] == Mtest[i,j] for i in 1:N, j in 1:N))

    Ttest = reshape(collect(1.0:27.0), N, N, N)
    T3 = alg_to_tensor3(Ttest)
    check("alg_to_tensor3 index order",
          all(T3[i,j,k] == Ttest[i,j,k] for i in 1:N, j in 1:N, k in 1:N))

    Stest = reshape(collect(2.0:2.0:18.0), N, N)
    S2 = alg_to_tensor2(Stest)
    dc = Gridap.TensorValues.double_contraction(T3, S2)
    ref = [sum(Ttest[i,k,j]*Stest[k,j] for k in 1:N, j in 1:N) for i in 1:N]
    check("double_contraction contracts TRAILING two indices",
          all(abs(dc[i]-ref[i]) < 1e-12 for i in 1:N))

    w = VectorValue(1.0, 2.0, 3.0)
    wT = w ⋅ T3
    refT = [sum(w[i]*Ttest[i,k,j] for i in 1:N) for k in 1:N, j in 1:N]
    check("W ⋅ ThirdOrder contracts FIRST index",
          all(abs(wT[k,j]-refT[k,j]) < 1e-12 for k in 1:N, j in 1:N))
end

# ---- FE-field primitives -------------------------------------------------------
let
    model = CartesianDiscreteModel((0.0,1.0,0.0,1.0), (4,4))
    reffe = ReferenceFE(lagrangian, VectorValue{3,Float64}, 2)
    Vs = FESpace(model, reffe; conformity=:H1)
    f = interpolate_everywhere(x -> VectorValue(x[1], 2*x[2], x[1]+x[2]), Vs)
    pt = VectorValue(0.3, 0.6)
    check("∂x orientation: f=(x,2y,x+y) → ∂x f=(1,0,1)",
          norm(alg_dx(f)(pt) - VectorValue(1.0,0.0,1.0)) < 1e-12)
    check("∂y orientation: f=(x,2y,x+y) → ∂y f=(0,2,1)",
          norm(alg_dy(f)(pt) - VectorValue(0.0,2.0,1.0)) < 1e-12)

    g2 = interpolate_everywhere(x -> VectorValue(1.0+x[1], x[2], 2.0-x[1]), Vs)
    ab = alg_outer(f, g2)(pt)
    fa = f(pt); gb = g2(pt)
    check("outer product (a⊗b)[k,j] = a[k]b[j]",
          all(abs(ab[k,j] - fa[k]*gb[j]) < 1e-12 for k in 1:3, j in 1:3))

    # matvec + dot helpers on FE fields
    Mc = alg_to_tensor2([2.0 0.0 1.0; 0.0 3.0 0.0; 1.0 0.0 4.0])
    mv = alg_mul(Mc, f)(pt)
    check("alg_mul: (𝗠⋅f)(pt) exact", norm(mv - Mc ⋅ fa) < 1e-12)
    a  = VectorValue(1.0, -2.0, 0.5)
    check("alg_dot: (a⋅f)(pt) exact", abs(alg_dot(a, f)(pt) - a ⋅ fa) < 1e-12)
end

println()
println("=" ^ 60)
@printf("  Results: %d PASS,  %d FAIL\n", n_pass, n_fail)
println("=" ^ 60)
n_fail > 0 ? error("test_primitives: $n_fail failed!") :
             println("  Primitives validated.")
