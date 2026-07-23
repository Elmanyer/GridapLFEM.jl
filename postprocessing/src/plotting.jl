# ==============================================================
#  plotting.jl — Plots.jl visualisations
# ==============================================================

"Symmetric colour limits around zero from the max |value| of a field over the run."
function _symclims(sim, field; frac=1.0)
    m = maximum(abs, sim.fields[field]) * frac
    m = m == 0 ? 1e-12 : m
    return (-m, m)
end

"""
    plot_field(sim, field="eta"; it=nsnapshots(sim), clims=:sym, kw...) -> Plot

Heatmap of `field(x,y)` at snapshot `it` (needs a regularised Cartesian grid).
"""
function plot_field(sim::WaveSimulation, field::AbstractString="eta";
                    it::Int=nsnapshots(sim), clims=:sym, title=nothing, kw...)
    gv = sim.grid
    gv === nothing && error("plot_field needs a regularised grid (regularize!)")
    F = grid_field(sim, field, it)
    cl = clims === :sym ? _symclims(sim, field) : clims
    heatmap(gv.xs, gv.ys, permutedims(F);
            aspect_ratio=:equal, c=:balance, clims=cl,
            xlabel="x [m]", ylabel="y [m]",
            title = title === nothing ? @sprintf("%s   t = %.3f s", field, sim.times[it]) : title,
            colorbar_title=field, kw...)
end

"""
    animate_field(sim, field="eta"; fps=15, out="wave.gif", clims=:sym, every=1) -> path

Animate `field(x,y,t)` over the run and save a GIF — the wave-propagation visual.
"""
function animate_field(sim::WaveSimulation, field::AbstractString="eta";
                       fps::Int=15, out::AbstractString="wave.gif",
                       clims=:sym, every::Int=1)
    gv = sim.grid
    gv === nothing && error("animate_field needs a regularised grid (regularize!)")
    cl = clims === :sym ? _symclims(sim, field) : clims
    anim = @animate for it in 1:every:nsnapshots(sim)
        heatmap(gv.xs, gv.ys, permutedims(grid_field(sim, field, it));
                aspect_ratio=:equal, c=:balance, clims=cl,
                xlabel="x [m]", ylabel="y [m]", colorbar_title=field,
                title=@sprintf("%s   t = %.3f s", field, sim.times[it]))
    end
    gif(anim, out; fps=fps)
    return out
end

"""
    plot_gauge(sim, field, x, y; ω=nothing) -> Plot

Time series of `field` at (x,y); annotates the DFT amplitude at ω if given.
"""
function plot_gauge(sim::WaveSimulation, field::AbstractString, x::Real, y::Real; ω=nothing)
    t, v = timeseries(sim, field, gauge(sim, x, y))
    p = plot(t, v; xlabel="t [s]", ylabel=field, legend=false,
             title=@sprintf("%s at (%.1f, %.1f)", field, x, y))
    ω !== nothing && annotate!(p, t[end], maximum(v),
        text(@sprintf("a(ω)=%.4g", amplitude_at(t, v, ω)), 8, :right))
    return p
end

"""
    plot_hovmoller(sim, field, p0, p1; n=200) -> Plot

Space–time (Hovmöller) diagram: `field` along the transect p0→p1 vs time.
Straight crest bands whose slope is the celerity.
"""
function plot_hovmoller(sim::WaveSimulation, field::AbstractString,
                        p0::Tuple, p1::Tuple; n::Int=200, clims=:sym)
    S = zeros(n, nsnapshots(sim))
    local s
    for it in 1:nsnapshots(sim)
        s, S[:, it] = transect(sim, field, p0, p1; n=n, it=it)
    end
    cl = clims === :sym ? (-maximum(abs,S), maximum(abs,S)) : clims
    heatmap(sim.times, s, S; c=:balance, clims=cl,
            xlabel="t [s]", ylabel="s along transect [m]",
            title="$field  (Hovmöller)", colorbar_title=field)
end

"""
    plot_dispersion(tbl; kd_app=nothing, band=0.02) -> Plot

Cm/Ce vs kd from a dispersion CSV (`ratio` or Cm/Ce columns), with the ±band and
the applicable-kd marker.
"""
function plot_dispersion(tbl::CsvTable; kd_app=nothing, band::Real=0.02)
    kd = tbl["kd"]
    ratio = haskey(tbl.cols,"ratio") ? tbl["ratio"] : tbl["Cm"] ./ tbl["Ce"]
    p = plot(kd, ratio; marker=:circle, lw=2, label="C_m/C_e (measured)",
             xlabel="kd", ylabel="C_m / C_e", title="Dispersion accuracy")
    hline!(p, [1.0]; ls=:solid, c=:black, label="Airy")
    hline!(p, [1-band, 1+band]; ls=:dash, c=:gray, label="±$(round(Int,100band))%")
    kd_app !== nothing && vline!(p, [kd_app]; ls=:dot, c=:red,
                                 label=@sprintf("kd_app=%.1f", kd_app))
    return p
end

"""
    plot_vertical_profile(prof::SigmaProfile; kd=nothing) -> Plot

Normalised model σ-profile vs the Airy shape (sinh for w, cosh for p) if `kd`
given. Plots value vs σ (σ up the vertical axis).
"""
function plot_vertical_profile(prof::SigmaProfile; kd=nothing)
    v = prof.value ./ (prof.value[end] == 0 ? 1 : prof.value[end])   # normalise at surface
    p = plot(v, prof.sigma; marker=:circle, lw=2, label="model",
             xlabel="normalised $(prof.kind)", ylabel="σ = (z+h)/H",
             title="vertical $(prof.kind) profile")
    if kd !== nothing
        σg = range(0, 1; length=100)
        shape = prof.kind === :w ? airy_w_shape.(σg, kd) : airy_p_shape.(σg, kd)
        plot!(p, shape, σg; ls=:dash, c=:black, label="Airy (kd=$(round(kd,digits=1)))")
    end
    return p
end

"""
    plot_harmonic_growth(xs, H; depth=nothing) -> Plot

Harmonic amplitudes H₁,₂,₃(x) along a transect (bar/shoal). If `depth=(xd,dd)` is
given, the bathymetry is drawn (scaled) for context.
"""
function plot_harmonic_growth(xs, H; labels=nothing)
    n = size(H, 1)
    lab = labels === nothing ? ["H$m" for m in 1:n] : labels
    p = plot(; xlabel="x [m]", ylabel="harmonic amplitude [m]", title="Harmonic generation")
    for m in 1:n
        plot!(p, xs, H[m, :]; lw=2, label=lab[m])
    end
    return p
end

"""
    plot_radial_decay(r, amp) -> Plot

`amp·√r` vs `r` (should be flat in the far field) plus the raw amplitude and a
1/√r reference anchored at the first point.
"""
function plot_radial_decay(r, amp)
    ref = amp[1] .* sqrt.(r[1] ./ r)
    p = plot(r, amp; marker=:circle, lw=2, label="amplitude",
             xlabel="r [m]", ylabel="amplitude [m]", title="Cylindrical spreading")
    plot!(p, r, ref; ls=:dash, c=:black, label="1/√r reference")
    plot!(twinx(p), r, amp .* sqrt.(r); marker=:square, c=:red, lw=1,
          ylabel="amp·√r", label="amp·√r")
    return p
end

"""
    plot_csv(tbl, x, ys...; logy=false, kw...) -> Plot

Generic CSV plotter: column `x` vs one or more `ys` (conservation/convergence).
"""
function plot_csv(tbl::CsvTable, x::AbstractString, ys::AbstractString...; logy=false, kw...)
    p = plot(; xlabel=x, yscale = logy ? :log10 : :identity, kw...)
    for y in ys
        plot!(p, tbl[x], logy ? abs.(tbl[y]) .+ 1e-300 : tbl[y]; lw=2, marker=:circle, label=y)
    end
    return p
end
