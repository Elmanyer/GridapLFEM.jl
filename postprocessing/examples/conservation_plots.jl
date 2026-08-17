# ==============================================================
#  conservation_plots.jl — cluster/convergence CSV → validation traces
#
#  Plots the CSV time series written by the cluster validations:
#    conservation.csv → mass/energy drift + amplitude envelope vs t
#    mms.csv          → rel_err vs t (flat at machine precision = pass)
#
#  RUN:  julia --project=. examples/conservation_plots.jl [csv]
# ==============================================================
include(joinpath(@__DIR__, "..", "src", "GridapLFEMPost.jl"))
using .GridapLFEMPost

out = joinpath(@__DIR__, "..", "..", "output", "pp_plots"); mkpath(out)
csvs = isempty(ARGS) ?
    filter(isfile, [joinpath(@__DIR__,"..","..","output","cluster_conservation","conservation.csv"),
                    joinpath(@__DIR__,"..","..","output","cluster_selfconsistency","mms.csv")]) : ARGS

isempty(csvs) && (println("no CSV found — run a cluster validation first, or pass a path"); exit())

for csv in csvs
    tbl = load_csv(csv); cols = keys(tbl.cols)
    base = splitext(basename(csv))[1]
    if "dmass_rel" in cols            # conservation.csv
        savefig(plot_csv(tbl, "t", "dmass_rel", "denergy_rel"; logy=true,
                         title="conservation drift", ylabel="relative drift"),
                joinpath(out, "$(base)_drift.png"))
        "etaL2" in cols && savefig(plot_csv(tbl, "t", "etaL2"; title="amplitude envelope"),
                                   joinpath(out, "$(base)_envelope.png"))
        println("wrote $(base)_drift.png (+envelope)")
    elseif "rel_err" in cols          # mms.csv
        savefig(plot_csv(tbl, "t", "rel_err"; logy=true,
                         title="MMS recovery error", ylabel="‖u_n − u*‖ / ‖u*‖"),
                joinpath(out, "$(base)_error.png"))
        println("wrote $(base)_error.png")
    else
        println("unrecognised columns in $csv: ", join(cols, ","))
    end
end
