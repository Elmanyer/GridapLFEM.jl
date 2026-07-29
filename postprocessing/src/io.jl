# ==============================================================
#  io.jl — read Gridap VTK (.pvd/.vtu) and CSV into Julia structures
# ==============================================================

"Extract the point-data field arrays of a VTKFile as name => Vector{Float64}.

For a VTK file (snapshot at time step tn), extract the values of each field (eta, ux0, uy0, ux1, uy1...) as 
(flat) vectors of Float64 and store them in a dictionary keyed by the corresponding field name."
function _snapshot_fields(vtk)
    pd = get_point_data(vtk)
    out = Dict{String,Vector{Float64}}()
    for name in keys(pd)
        arr = get_data(pd[name])
        out[String(name)] = vec(Float64.(arr))
    end
    return out
end

"""
    load_snapshot(vtu) -> (points::Matrix{n×2}, fields::Dict, )

Load a single `.vtu`: node coordinates (x,y) and every point-data field.
"""
function load_snapshot(vtu::AbstractString)
    vtk = VTKFile(vtu)
    P = get_points(vtk)                       # returns matrix with spatial coordinates of DoFs grid [3 × n_points] = [x1, x2, x3... ; y1, y2, y3... ; z1, z2, z3...]
    points = permutedims(P[1:2, :])           # drop z coordinate and transpose so that [3 × n_points] -> [n_points × 2] = [x1, y1; x2, y2; x3, y3...]
    return points, _snapshot_fields(vtk)
end

"Recover the snapshot time from a `sol_t_<t>.vtu` filename (`sol_t_1_6000`→1.6); NaN if unmatched."
function _time_from_name(f::AbstractString)
    m = match(r"sol_t_(.+)\.vtu$", basename(f))
    m === nothing && return NaN
    x = tryparse(Float64, replace(m.captures[1], "_" => "."))
    return x === nothing ? NaN : x
end

"Parse a ParaView .pvd collection → (vtu_paths::Vector, times::Vector), paths absolute.

It extracts all referenced .vtu snapshot file paths and their corresponding simulation timestamps, 
converts relative paths to absolute paths, and returns them sorted chronologically by time.
"
function _parse_pvd(pvd::AbstractString)
    dir = dirname(abspath(pvd))
    txt = read(pvd, String)
    files = String[]; times = Float64[]
    for m in eachmatch(r"<DataSet\b[^>]*>", txt)
        tag = m.match
        tm = match(r"timestep\s*=\s*\"([^\"]+)\"", tag)
        fm = match(r"file\s*=\s*\"([^\"]+)\"", tag)
        fm === nothing && continue
        push!(files, isabspath(fm.captures[1]) ? fm.captures[1] : joinpath(dir, fm.captures[1]))
        push!(times, tm === nothing ? length(times)+0.0 : parse(Float64, tm.captures[1]))
    end
    order = sortperm(times)
    return files[order], times[order]
end

"""
    load_simulation(path; fields=:all, regularize=true) -> WaveSimulation

Load a whole run. `path` is a `.pvd` collection or a directory containing one
(or a set of `sol_t_*.vtu`). All snapshots share the mesh; each field becomes a
`[n_points × n_times]` matrix. If `regularize` and the mesh is a Cartesian node
grid, a `GridView` is attached (enables heatmaps/interpolation/quadrature).

returns a sim=`WaveSimulation` with fields, times, points, and metadata. 
"""
function load_simulation(path::AbstractString; fields=:all, regularize::Bool=true)
    # resolve to a list of (vtu, time)
    if isdir(path)
        pvds = filter(f -> endswith(f, ".pvd"), readdir(path; join=true))
        if !isempty(pvds)
            vtus, times = _parse_pvd(first(pvds))
        else
            # no collection file (e.g. run interrupted) — glob the snapshots and
            # recover the time from the "sol_t_<t>.vtu" filename (dots → underscores)
            vtus  = filter(f -> endswith(f, ".vtu"), readdir(path; join=true))
            times = _time_from_name.(vtus)
            any(isnan, times) && (times = Float64.(eachindex(vtus)))
            order = sortperm(times); vtus = vtus[order]; times = times[order]
        end
    elseif endswith(path, ".pvd")
        vtus, times = _parse_pvd(path)
    else
        vtus, times = [path], [0.0]
    end
    isempty(vtus) && error("no VTK snapshots found at $path")

    points, first_fields = load_snapshot(vtus[1])
    wanted = fields === :all ? collect(keys(first_fields)) : String.(fields)
    np = size(points, 1); nt = length(vtus)
    F = Dict(name => Matrix{Float64}(undef, np, nt) for name in wanted)
    for name in wanted
        F[name][:, 1] .= first_fields[name]
    end
    for it in 2:nt
        _, fl = load_snapshot(vtus[it])
        for name in wanted
            F[name][:, it] .= fl[name]
        end
    end

    # Generate simulation structure with results and metadata. 
    # WaveSimulation:
    #     times  :: Vector{Float64}                 # vector of times tn
    #     points :: Matrix{Float64}                 # horizontal grid coordinates [n_points × 2]
    #     fields :: Dict{String,FieldSeries}        # solution fields (eta, ux0, uy0...) dictionary   name → [n_points × Nt]
    #     grid   :: Union{Nothing,GridView}         # optional regular-grid view of the node cloud (for reshape/interpolation)
    #     meta   :: Dict{Symbol,Any}                # metadata dictionary (run directory, number of snapshots, and bounding box)

    sim = WaveSimulation(collect(Float64, times),   # vector of times tn
                                 points,            # horizontal grid coordinates 
                                 F,                 # solution fields dictionary  
                                 nothing,           # optional regular-grid view
                                 Dict{Symbol,Any}(:dir => dirname(abspath(vtus[1])),
                                          :nt => nt,
                                          :bbox => (extrema(points[:,1]), extrema(points[:,2]))))
                                          
    # Add optional regular-grid view if the mesh is a Cartesian node grid.
    regularize && regularize!(sim)
    return sim
end

"""
    regularize!(sim) -> sim

Detect a Cartesian node grid (unique x's × unique y's cover all nodes exactly)
and attach a `GridView`. No-op (leaves `grid=nothing`) if the cloud is not a
full Cartesian grid.
"""
function regularize!(sim::WaveSimulation)
    x = sim.points[:, 1]; y = sim.points[:, 2]
    tol = 1e-6 * max(maximum(abs, x), maximum(abs, y), 1.0)
    xs = _unique_sorted(x, tol); ys = _unique_sorted(y, tol)
    # Gridap writes VTK per higher-order cell, so shared nodes are DUPLICATED: the
    # cloud has ≥ Nx·Ny points. We therefore do NOT require Nx·Ny == n_points; we
    # map each grid node (i,j) to any point with those coordinates (duplicates carry
    # identical CG values) and only require every grid node to be covered.
    length(xs)*length(ys) > 50*length(x) && return sim    # clearly not a tensor grid
    ix = Dict(xs[i] => i for i in eachindex(xs)); iy = Dict(ys[j] => j for j in eachindex(ys))
    idx = zeros(Int, length(xs), length(ys))
    @inbounds for p in axes(sim.points, 1)
        i = ix[_snap(x[p], xs, tol)]; j = iy[_snap(y[p], ys, tol)]
        idx[i, j] = p
    end
    all(!=(0), idx) || return sim                         # missing nodes ⇒ not a full grid
    sim.grid = GridView(xs, ys, idx)
    return sim
end

function _unique_sorted(v, tol)
    s = sort(v); out = Float64[]
    for x in s
        (isempty(out) || x - out[end] > tol) && push!(out, x)
    end
    return out
end
_snap(x, grid, tol) = grid[clamp(searchsortedlast(grid, x + tol/2), 1, length(grid))]

"""
    load_csv(path) -> CsvTable

Read a headered numeric CSV (the cluster/convergence/sweep outputs) into columns.
"""
function load_csv(path::AbstractString)
    data, header = readdlm(path, ','; header=true)
    names = String.(vec(header))
    cols = Dict{String,Vector{Float64}}()
    for (k, name) in enumerate(names)
        cols[name] = Float64[_tofloat(v) for v in data[:, k]]
    end
    return CsvTable(names, cols)
end
_tofloat(v::Number) = Float64(v)
_tofloat(v) = (x = tryparse(Float64, strip(string(v))); x === nothing ? NaN : x)
