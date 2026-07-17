# ==============================================================
#  horizontal.jl — Stage 2: 2D mesh and the STACKED FE spaces
#
#  Stacked layout: MultiFieldFESpace with THREE fields
#      [η, 𝖴x, 𝖴y],   𝖴x,𝖴y ∈ VectorValue{Nσ}
#  (vs the old solver's 1+2Nσ scalar fields). Solid-wall Dirichlet BCs act on
#  the WHOLE stacked field (all Nσ components zero at once).
#
#  Boundary tags (CartesianDiscreteModel, 2D):
#    tag_1..tag_4 = corners, tag_5/6 = bottom/top edges, tag_7/8 = left/right.
#  Corner tags are MANDATORY in any wall BC (omitting them leaves corner DOFs
#  unconstrained → exponential instability; root CLAUDE.md rule 5).
#
#  All fields H1-conforming Lagrange, fe_order ≥ 2 (linear elements zero the
#  dispersion term and disable all non-hydrostatic physics).
# ==============================================================

"""
    build_horizontal_model(domain, partition) → (model, trian)

2D Cartesian mesh on ((x0,x1),(y0,y1)) (or flat (x0,x1,y0,y1)) with
`partition = (nx, ny)` cells. Create the measure as `Measure(trian, 2*fe_order+2)`.
"""
function build_horizontal_model(domain::Tuple, partition::Tuple)
    if domain isa Tuple{Tuple,Tuple}
        (x0,x1), (y0,y1) = domain
        dom_flat = (x0, x1, y0, y1)
    else
        dom_flat = domain
    end
    model = CartesianDiscreteModel(dom_flat, partition)
    trian = Triangulation(model)
    return model, trian
end

"""
    build_fe_spaces(model, fe_order, Nσ; y_wall_bc=true, x_wall_bc=false) → (U, V)

Stacked 3-field MultiFieldFESpace `[η, 𝖴x, 𝖴y]`.

- `y_wall_bc=true` (default): solid wall `u_j^y = 0 ∀j` on y-boundaries —
  Dirichlet zero `VectorValue{Nσ}` on the whole 𝖴y field (tags 1–6).
- `x_wall_bc=true`: closed basin — additionally `u_j^x = 0 ∀j` on x-boundaries
  (tags 1–4,7,8). REQUIRED for any initial-condition problem (soliton, sloshing,
  IC hump); flumes keep it false (open x-ends, sponge-absorbed).
"""
function build_fe_spaces(model, fe_order::Int, Nσ::Int;
                             y_wall_bc::Bool = true, x_wall_bc::Bool = false)
    reffe_eta = ReferenceFE(lagrangian, Float64, fe_order)
    reffe_U   = ReferenceFE(lagrangian, VectorValue{Nσ,Float64}, fe_order)
    zvv       = VectorValue(ntuple(_ -> 0.0, Nσ)...)
    y_tags = y_wall_bc ? ["tag_1","tag_2","tag_3","tag_4","tag_5","tag_6"] : String[]
    x_tags = x_wall_bc ? ["tag_1","tag_2","tag_3","tag_4","tag_7","tag_8"] : String[]

    V_eta = FESpace(model, reffe_eta; conformity=:H1)
    U_eta = TrialFESpace(V_eta)
    V_Ux  = isempty(x_tags) ? FESpace(model, reffe_U; conformity=:H1) :
                              FESpace(model, reffe_U; conformity=:H1, dirichlet_tags=x_tags)
    U_Ux  = isempty(x_tags) ? TrialFESpace(V_Ux) : TrialFESpace(V_Ux, zvv)
    V_Uy  = isempty(y_tags) ? FESpace(model, reffe_U; conformity=:H1) :
                              FESpace(model, reffe_U; conformity=:H1, dirichlet_tags=y_tags)
    U_Uy  = isempty(y_tags) ? TrialFESpace(V_Uy) : TrialFESpace(V_Uy, zvv)

    # The element type of the space vectors must be an abstract FE-space supertype
    # so that a mixed-type vector (unconstrained + Dirichlet fields) is accepted by
    # the sequential Vector{<:SingleFieldFESpace} resp. distributed
    # Vector{<:DistributedSingleFieldFESpace} MultiFieldFESpace dispatch (same
    # runtime-isa trick as the old solver). Keep ConsecutiveMultiFieldStyle —
    # BlockMultiFieldStyle breaks the distributed Jacobi preconditioner's `diag`.
    if isa(V_eta, GridapDistributed.DistributedSingleFieldFESpace)
        Vspaces = GridapDistributed.DistributedSingleFieldFESpace[V_eta, V_Ux, V_Uy]
        Uspaces = GridapDistributed.DistributedSingleFieldFESpace[U_eta, U_Ux, U_Uy]
    else
        Vspaces = Gridap.FESpaces.SingleFieldFESpace[V_eta, V_Ux, V_Uy]
        Uspaces = Gridap.FESpaces.SingleFieldFESpace[U_eta, U_Ux, U_Uy]
    end
    U = MultiFieldFESpace(Uspaces)
    V = MultiFieldFESpace(Vspaces)
    return U, V
end
