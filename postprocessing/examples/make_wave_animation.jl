# ==============================================================
#  make_wave_animation.jl — the flagship "observe wave propagation" visual
#
#  Loads a VTK run, animates η(x,y,t) to a GIF, and saves a final snapshot
#  heatmap and a Hovmöller (space–time) transect.
#
#  RUN (from the postprocessing/ folder):
#    julia --project=. examples/make_wave_animation.jl [pvd-or-dir] [outdir]
# ==============================================================
include(joinpath(@__DIR__, "..", "src", "GridapLFEMPost.jl"))
using .GridapLFEMPost

src = length(ARGS) ≥ 1 ? ARGS[1] : joinpath(@__DIR__, "..", "..", "output", "pp_demo")
out = length(ARGS) ≥ 2 ? ARGS[2] : joinpath(@__DIR__, "..", "..", "output", "pp_plots")
mkpath(out)

sim = load_simulation(src)
println("loaded $(nsnapshots(sim)) snapshots, fields: ", join(fieldnames_of(sim), ", "))

# η(x,y) animation (symmetric colour scale about 0)
animate_field(sim, "eta"; fps=12, out=joinpath(out, "wave.gif"))
# final snapshot + a mid-domain space–time transect
savefig(plot_field(sim, "eta"), joinpath(out, "eta_snapshot.png"))
(x0,x1) = extrema(sim.points[:,1]); ymid = sum(extrema(sim.points[:,2]))/2
savefig(plot_hovmoller(sim, "eta", (x0, ymid), (x1, ymid); n=200),
        joinpath(out, "hovmoller.png"))
println("wrote wave.gif, eta_snapshot.png, hovmoller.png to $out")
