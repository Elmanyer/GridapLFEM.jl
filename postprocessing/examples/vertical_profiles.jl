# ==============================================================
#  vertical_profiles.jl — w(σ) / p(σ) at a station vs Airy theory
#
#  Two independent paths, both plotted here (reproduces Yang & Liu §3.2 Figs 6–8):
#    (1) STORED — read the solver's w_s<σ>/p_s<σ> fields (needs a run written with
#        write_w=true / write_pressure=true); sampled only at the Nσ σ-nodes.
#    (2) FROM MODES — reconstruct w(σ)/p_nh(σ) at ANY σ purely from the stored
#        velocity modes u{j}x,u{j}y (works even without w_s/p_s), via
#        reconstruct_profile. Needs the σ-mesh (c_bdy,p) and still-water depth d.
#
#  RUN:  julia --project=. examples/vertical_profiles.jl [pvd-or-dir] [x] [y] [kd] [d]
# ==============================================================
include(joinpath(@__DIR__, "..", "src", "GridapBALFEMPost.jl"))
using .GridapBALFEMPost

src = length(ARGS) ≥ 1 ? ARGS[1] : joinpath(@__DIR__, "..", "..", "output", "pp_demo")
x   = length(ARGS) ≥ 2 ? parse(Float64, ARGS[2]) : 24.0
y   = length(ARGS) ≥ 3 ? parse(Float64, ARGS[3]) : 5.0
kd  = length(ARGS) ≥ 4 ? parse(Float64, ARGS[4]) : 5.5
d   = length(ARGS) ≥ 5 ? parse(Float64, ARGS[5]) : 3.5
c_bdy = [0.0, 0.728, 1.0]        # P1LFE-2 σ-mesh (match the run's M/c_bdy)
out = joinpath(@__DIR__, "..", "..", "output", "pp_plots"); mkpath(out)

sim = load_simulation(src)
ω = 2π/1.6
has(pre) = any(startswith(n, pre) for n in fieldnames_of(sim))

# (1) stored w_s/p_s (σ-node samples)
has("w_s") && savefig(plot_vertical_profile(sigma_profile(sim, x, y; kind=:w, ω=ω); kd=kd),
                      joinpath(out, "wprofile_stored.png"))
has("p_s") && savefig(plot_vertical_profile(sigma_profile(sim, x, y; kind=:p, ω=ω); kd=kd),
                      joinpath(out, "pprofile_stored.png"))

# (2) reconstructed from the velocity modes (continuous σ) — always available
savefig(plot_vertical_profile(
            reconstruct_profile(sim, x, y; kind=:w, c_bdy=c_bdy, depth=d, ω=ω, nσ=81); kd=kd),
        joinpath(out, "wprofile_recon.png"))
savefig(plot_vertical_profile(
            reconstruct_profile(sim, x, y; kind=:pnh, c_bdy=c_bdy, depth=d, ω=ω, nσ=81); kd=kd),
        joinpath(out, "pprofile_recon.png"))

println("wrote wprofile_recon.png, pprofile_recon.png",
        has("w_s") ? " (+ stored)" : "")
