# ==============================================================
#  tensors.jl — constant-tensor constructors + pointwise algebra helpers
#
#  Constructors (build time only). Gridap TensorValue data is column-major
#  (first index fastest):
#    TensorValue{N,N}(data...)            → T[i,j]   = data[i + (j-1)N]
#    ThirdOrderTensorValue{N,N,N}(data..) → T[i,j,k] = data[i + (j-1)N + (k-1)N²]
#  (verified by test/test_primitives.jl)
#
#  Pointwise helpers wrap constants in Operation closures (closing over
#  constants is fine — only CellFields must not be closed over in loops).
#  Verified semantics (see building_files/ARCHITECTURE.md §1):
#    ∇ of a VectorValue{Nσ} field = TensorValue{2,Nσ} (spatial index FIRST)
#      ⇒ ∂x f = e_x ⋅ ∇f   (use e⋅∇f, never ∇f⋅e)
#    double_contraction(𝗧3, S) contracts the TRAILING two indices:
#      result_i = Σ_{k,j} 𝗧3[i,k,j] S[k,j]
# ==============================================================

"Vector{Float64} → VectorValue{N} constant."
alg_to_vec(v::AbstractVector{Float64}) = VectorValue{length(v),Float64}(v...)

"Matrix → TensorValue{N,N} constant (index-preserving)."
function alg_to_tensor2(M::AbstractMatrix{Float64})
    N = size(M, 1); @assert size(M, 2) == N
    data = ntuple(k -> M[(k-1) % N + 1, (k-1) ÷ N + 1], N * N)
    return TensorValue{N,N,Float64}(data...)
end

"Array{Float64,3} → ThirdOrderTensorValue{N,N,N} constant (index-preserving)."
function alg_to_tensor3(T::AbstractArray{Float64,3})
    N = size(T, 1); @assert size(T, 2) == N && size(T, 3) == N
    data = ntuple(N^3) do k
        k0 = k - 1
        i = k0 % N + 1
        j = (k0 ÷ N) % N + 1
        l = k0 ÷ (N * N) + 1
        T[i, j, l]
    end
    return ThirdOrderTensorValue{N,N,N,Float64}(data...)
end

"∂x of a scalar or VectorValue{Nσ} CellField: e_x ⋅ ∇f (spatial index first)."
alg_dx(f) = Operation(g -> Ex ⋅ g)(∇(f))

"∂y of a scalar or VectorValue{Nσ} CellField: e_y ⋅ ∇f."
alg_dy(f) = Operation(g -> Ey ⋅ g)(∇(f))

"Constant TensorValue{N,N} ⋅ VectorValue{N}-CellField → VectorValue{N}-CellField (matvec)."
alg_mul(A::TensorValue, u) = Operation(x -> A ⋅ x)(u)

"Constant VectorValue{N} ⋅ VectorValue{N}-CellField → scalar CellField."
alg_dot(a::VectorValue, u) = Operation(x -> a ⋅ x)(u)

"double_contraction(constant ThirdOrderTensorValue, TensorValue-CellField):
contracts the TRAILING two indices → VectorValue{N}-CellField."
alg_dc3(T::ThirdOrderTensorValue, S) =
    Operation(s -> Gridap.TensorValues.double_contraction(T, s))(S)

"Outer product of two VectorValue{N}-CellFields → TensorValue{N,N}-CellField,
(a⊗b)[k,j] = a[k]·b[j]."
alg_outer(a, b) = Operation(Gridap.TensorValues.outer)(a, b)

"Fuse two scalar CellFields into a VectorValue{2}-CellField."
alg_vec2(a, b) = Operation(VectorValue)(a, b)
