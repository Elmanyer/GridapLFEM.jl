# ==============================================================
#  probes.jl — gauges, time series, transects, σ-profiles
# ==============================================================

"Field names present in the simulation."
fieldnames_of(sim::WaveSimulation) = sort!(collect(keys(sim.fields)))
"Number of snapshots."
nsnapshots(sim::WaveSimulation) = length(sim.times)

"""
    gauge(sim, x, y; bilinear=true) -> Gauge

A point probe at (x,y). Records the nearest mesh node and, when a Cartesian grid
view is available, a bilinear interpolation stencil.
"""
function gauge(sim::WaveSimulation, x::Real, y::Real; bilinear::Bool=true)
    inode = _nearest_node(sim.points, x, y)
    stencil = (bilinear && sim.grid !== nothing) ? _bilinear_stencil(sim.grid, x, y) : nothing
    return Gauge(float(x), float(y), inode, stencil)
end

function _nearest_node(P::AbstractMatrix, x, y)
    best = 1; bd = Inf
    @inbounds for i in axes(P,1)
        d = (P[i,1]-x)^2 + (P[i,2]-y)^2
        d < bd && (bd = d; best = i)
    end
    return best
end

function _bilinear_stencil(gv::GridView, x, y)
    ix = clamp(searchsortedlast(gv.xs, x), 1, length(gv.xs)-1)
    iy = clamp(searchsortedlast(gv.ys, y), 1, length(gv.ys)-1)
    x0,x1 = gv.xs[ix], gv.xs[ix+1]; y0,y1 = gv.ys[iy], gv.ys[iy+1]
    tx = (x-x0)/(x1-x0); ty = (y-y0)/(y1-y0)
    return [ (gv.idx[ix,  iy  ], (1-tx)*(1-ty)),
             (gv.idx[ix+1,iy  ], tx*(1-ty)),
             (gv.idx[ix,  iy+1], (1-tx)*ty),
             (gv.idx[ix+1,iy+1], tx*ty) ]
end

"""
    timeseries(sim, field, g::Gauge) -> (t, v)

Time series of `field` at gauge `g` (bilinear on the grid if available, else
nearest node).
"""
function timeseries(sim::WaveSimulation, field::AbstractString, g::Gauge)
    F = sim.fields[field]
    if g.stencil === nothing
        return sim.times, F[g.inode, :]
    else
        v = zeros(length(sim.times))
        for (node, wt) in g.stencil
            v .+= wt .* @view F[node, :]
        end
        return sim.times, v
    end
end

"Field values at all nodes at snapshot `it`."
field_at(sim::WaveSimulation, field::AbstractString, it::Int) = sim.fields[field][:, it]

"""
    grid_field(sim, field, it) -> Matrix (Nx × Ny)

The field at snapshot `it` reshaped onto the Cartesian node grid (requires a
regularised grid view). Rows index x, columns index y.
"""
function grid_field(sim::WaveSimulation, field::AbstractString, it::Int)
    gv = sim.grid
    gv === nothing && error("grid_field needs a regularised grid (call regularize!)")
    v = sim.fields[field][:, it]
    return [ v[gv.idx[i,j]] for i in eachindex(gv.xs), j in eachindex(gv.ys) ]
end

"""
    transect(sim, field, p0, p1; n=200, it) -> (s, v)

Sample `field` at `n` points along the segment p0→p1 at snapshot `it`. `s` is
arclength from p0. Uses bilinear sampling on the grid (nearest node otherwise).
"""
function transect(sim::WaveSimulation, field::AbstractString,
                  p0::Tuple, p1::Tuple; n::Int=200, it::Int)
    xs = range(p0[1], p1[1]; length=n); ys = range(p0[2], p1[2]; length=n)
    s = [hypot(xs[i]-p0[1], ys[i]-p0[2]) for i in 1:n]
    v = zeros(n)
    for i in 1:n
        g = gauge(sim, xs[i], ys[i])
        if g.stencil === nothing
            v[i] = sim.fields[field][g.inode, it]
        else
            v[i] = sum(wt * sim.fields[field][node, it] for (node,wt) in g.stencil)
        end
    end
    return s, v
end

"""
    sigma_profile(sim, x, y; kind=:w, ω) -> SigmaProfile

Vertical profile at station (x,y) of the reconstructed `w`/`p` σ-level fields
(`w_s<σ>` / `p_s<σ>`). Value = first-harmonic amplitude at ω (steady) if `ω`
given, else peak steady amplitude, at each σ-level. Levels sorted ascending.
"""
function sigma_profile(sim::WaveSimulation, x::Real, y::Real; kind::Symbol=:w, ω=nothing)
    prefix = kind === :w ? "w_s" : "p_s"
    levels = Tuple{Float64,String}[]
    for name in keys(sim.fields)
        startswith(name, prefix) || continue
        σ = parse(Float64, replace(name[length(prefix)+1:end], "_" => "."))
        push!(levels, (σ, name))
    end
    isempty(levels) && error("no $prefix<σ> fields in the simulation (run with write_$(kind)=true)")
    sort!(levels; by=first)
    g = gauge(sim, x, y)
    val = map(levels) do (σ, name)
        t, v = timeseries(sim, name, g)
        ω === nothing ? steady_amplitude(t, v) : amplitude_at(t, v, ω)
    end
    return SigmaProfile(first.(levels), collect(val), kind)
end
