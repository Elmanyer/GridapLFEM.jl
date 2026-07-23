# ==============================================================
#  GridapLFEMPost.jl — postprocessing library for the GridapLFEM solver
#
#  Reads the solver's VTK (solution.pvd + sol_t_*.vtu) and CSV outputs into
#  Julia data structures, provides wave-analysis operations (gauges, transects,
#  DFT/harmonics, celerity, radial/harmonic diagnostics), and visualises them
#  with Plots.jl. Self-contained: no dependency on the solver.
#
#  Load:  include("postprocessing/src/GridapLFEMPost.jl"); using .GridapLFEMPost
#  (activate the postprocessing/ environment first).
# ==============================================================
module GridapLFEMPost

using ReadVTK
using FFTW
using Interpolations
using Plots
import Plots: savefig, gif, plot, plot!    # re-exported for convenience
using DelimitedFiles
using Statistics
using LinearAlgebra
using Printf

# ---- data model --------------------------------------------------------------
"One field over the whole run on a fixed (Eulerian) mesh: [n_points × n_times]."
const FieldSeries = Matrix{Float64}

"Regular-grid view of a Cartesian Lagrange node cloud (for reshape/interpolation)."
struct GridView
    xs  :: Vector{Float64}          # unique sorted node x  (Nx)
    ys  :: Vector{Float64}          # unique sorted node y  (Ny)
    idx :: Matrix{Int}              # [Nx × Ny] → row index into points / field data
end

"A loaded simulation: fixed mesh + time series of every field."
mutable struct WaveSimulation
    times  :: Vector{Float64}
    points :: Matrix{Float64}                 # [n_points × 2]
    fields :: Dict{String,FieldSeries}        # name → [n_points × Nt]
    grid   :: Union{Nothing,GridView}
    meta   :: Dict{Symbol,Any}
end

"A virtual gauge: point probe with a precomputed nearest-node (and optional bilinear)."
struct Gauge
    x :: Float64
    y :: Float64
    inode :: Int
    stencil :: Union{Nothing,Vector{Tuple{Int,Float64}}}   # (node,weight) bilinear
end

"A σ-level vertical profile (from w_s<σ>/p_s<σ> fields) at a station."
struct SigmaProfile
    sigma :: Vector{Float64}
    value :: Vector{Float64}
    kind  :: Symbol                 # :w or :p
end

"A tabulated CSV result."
struct CsvTable
    names :: Vector{String}
    cols  :: Dict{String,Vector{Float64}}
end
Base.getindex(t::CsvTable, name::AbstractString) = t.cols[String(name)]

# ---- includes ----------------------------------------------------------------
include("io.jl")
include("probes.jl")
include("spectral.jl")
include("diagnostics.jl")
include("reconstruct.jl")
include("plotting.jl")

# ---- exports -----------------------------------------------------------------
export WaveSimulation, GridView, Gauge, SigmaProfile, CsvTable
export load_simulation, load_snapshot, regularize!, load_csv, fieldnames_of, nsnapshots
export gauge, timeseries, field_at, grid_field, transect, sigma_profile
export dft_at, amplitude_at, phase_at, spectrum, harmonic_amplitudes, steady_amplitude, celerity
export radial_profile, harmonic_growth, mass_integral
export airy_Ce, airy_kd, airy_w_shape, airy_p_shape
export SigmaBasis, sigma_basis, phi, phi_int, pi3, unit_w
export reconstruct_profile, reconstruct_w, reconstruct_pressure
export plot_field, animate_field, plot_gauge, plot_hovmoller, plot_dispersion
export plot_vertical_profile, plot_harmonic_growth, plot_radial_decay, plot_csv
export savefig, gif, plot, plot!          # re-exported from Plots

end # module
