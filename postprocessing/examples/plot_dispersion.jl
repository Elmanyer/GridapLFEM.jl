# ==============================================================
#  plot_dispersion.jl — dispersion curve Cm/Ce(kd) from a sweep CSV
#
#  Plots the measured dispersion accuracy (from dispersion_sweep_M*.csv or
#  dispersion_nonlinear.csv) with the Airy line and ±2% band; marks kd_app.
#
#  RUN:  julia --project=. examples/plot_dispersion.jl [csv] [kd_app]
# ==============================================================
include(joinpath(@__DIR__, "..", "src", "GridapBALFEMPost.jl"))
using .GridapBALFEMPost

csv = length(ARGS) ≥ 1 ? ARGS[1] :
      joinpath(@__DIR__, "..", "..", "output", "dispersion_nonlinear.csv")
kd_app = length(ARGS) ≥ 2 ? parse(Float64, ARGS[2]) : 10.9   # P1LFE-2 default
out = joinpath(@__DIR__, "..", "..", "output", "pp_plots"); mkpath(out)

tbl = load_csv(csv)
p = plot_dispersion(tbl; kd_app=kd_app, band=0.02)
savefig(p, joinpath(out, "dispersion.png"))
println("wrote dispersion.png to $out  (points: ", length(tbl["kd"]), ")")
