# `GridapBALFEMPost` — postprocessing library

Turns the solver's **VTK** (`solution.pvd` + `sol_t_*.vtu`, fields `eta, u1x, …, w_s<σ>, p_s<σ>`) and
**CSV** outputs into Julia data structures you can sample, transform (DFT / harmonics / celerity /
integrals) and visualise with **Plots.jl**. Self‑contained — no dependency on the solver.

The full design rationale is in `../building_files/ARCHITECTURE.md` §7.

## Setup

Its own environment (separate from the solver, keeps the plotting/VTK deps out of the solver):

```julia
import Pkg; Pkg.activate("postprocessing"); Pkg.instantiate()   # ReadVTK, Plots, GR, FFTW, Interpolations
```

Load it by file path (it is not a registered package):

```julia
include("postprocessing/src/GridapBALFEMPost.jl")
using .GridapBALFEMPost
```

## Producing input for it

- **VTK** — run the solver with `save_every > 0` (and `write_w=true, write_pressure=true` for σ‑profiles):
  ```julia
  setup_and_run(...; save_every=6, write_w=true, write_pressure=true, output_dir="output/run1")
  ```
  → `output/run1/solution.pvd` (+ `sol_t_*.vtu`). If a run is interrupted before the `.pvd` is
  finalised, `load_simulation` still works from the directory (times recovered from the filenames).
- **CSV** — the sweep/cluster tests write CSVs (`dispersion_sweep_M*.csv`, `conservation.csv`,
  `mms.csv`, …) read by `load_csv`.

## Quick start

```julia
sim = load_simulation("output/run1")          # → WaveSimulation (mesh + fields over time)
fieldnames_of(sim); nsnapshots(sim)

# observe the wave propagation
animate_field(sim, "eta"; fps=12, out="wave.gif")        # η(x,y,t) → GIF
savefig(plot_field(sim, "eta"), "snap.png")               # snapshot heatmap
savefig(plot_hovmoller(sim, "eta", (8,5), (40,5)), "hov.png")   # space–time transect

# probe + analyse
t, η = timeseries(sim, "eta", gauge(sim, 24.0, 5.0))
Cm = celerity(sim, "eta", gauge(sim,20,5), gauge(sim,22,5), 2π/1.6)
H  = harmonic_amplitudes(t, η, 2π/1.6; n=3)               # bound-harmonic content

# vertical profile vs Airy — two paths:
prof = sigma_profile(sim, 24.0, 5.0; kind=:w, ω=2π/1.6)             # (1) read stored w_s<σ>
prof = reconstruct_profile(sim, 24.0, 5.0; kind=:w,                  # (2) rebuild from the modes
                           c_bdy=[0,0.728,1], depth=3.5, ω=2π/1.6)   #     (any σ, no w_s needed)
savefig(plot_vertical_profile(prof; kd=5.5), "wprofile.png")

# dispersion curve from a sweep CSV
savefig(plot_dispersion(load_csv("output/dispersion_sweep_M2.csv"); kd_app=10.9), "disp.png")
```

## API at a glance

| group | functions |
|-------|-----------|
| **IO** | `load_simulation`, `load_snapshot`, `regularize!`, `load_csv` |
| **probes** | `gauge`, `timeseries`, `field_at`, `grid_field`, `transect`, `sigma_profile` |
| **reconstruction** | `reconstruct_profile`, `reconstruct_w`, `reconstruct_pressure`, `sigma_basis`, `phi`, `phi_int`, `pi3`, `unit_w` |
| **spectral** | `dft_at`, `amplitude_at`, `phase_at`, `spectrum`, `harmonic_amplitudes`, `steady_amplitude`, `celerity` |
| **diagnostics** | `radial_profile`, `harmonic_growth`, `mass_integral`, `airy_Ce`, `airy_kd`, `airy_w_shape`, `airy_p_shape` |
| **plotting** | `plot_field`, `animate_field`, `plot_gauge`, `plot_hovmoller`, `plot_dispersion`, `plot_vertical_profile`, `plot_harmonic_growth`, `plot_radial_decay`, `plot_csv` |

## Ready-made scripts (`examples/`)

```bash
julia --project=. examples/make_wave_animation.jl  [pvd-or-dir]        # η(x,y,t) GIF + snapshot + Hovmöller
julia --project=. examples/plot_dispersion.jl      [csv] [kd_app]      # Cm/Ce(kd) with ±2% band
julia --project=. examples/vertical_profiles.jl    [dir] [x] [y] [kd]  # w(σ), p(σ) vs Airy
julia --project=. examples/conservation_plots.jl   [csv]               # cluster CSV drift/error traces
```

## Vertical-profile reconstruction (w and p from the modes)

Two ways to get the vertical structure at a station:

1. **Stored** — if the run was written with `write_w`/`write_pressure`, read the `w_s<σ>`/`p_s<σ>`
   fields with `sigma_profile` (sampled only at the Nσ σ-nodes).
2. **From the modes** — `reconstruct_profile(sim, x, y; kind, c_bdy, p=1, depth, ω)` rebuilds
   `w(σ)` (`:w`), total `p(σ)` (`:p`) or non‑hydrostatic `p_nh(σ)` (`:pnh`) at **any** σ, purely from
   the stored velocity modes `u{j}x,u{j}y` — so it works even when `w_s`/`p_s` were not written, and
   on a continuous σ‑grid. The math is ported verbatim from the solver's `src/reconstruct.jl`
   (`w = −a·𝖺 + b·𝖻 − c·𝖲`; `p_nh = −ρd²·π·∂ₜ(∇·u)` with the surface‑vanishing moment `πⱼ(σ)=∫_σ¹φⱼ_int`),
   with the σ‑basis (`φⱼ,φⱼ_int,πⱼ`) rebuilt analytically by `sigma_basis` (exact Gauss quadrature, no
   Gridap) and spatial derivatives by central differences on the node grid. Cross‑checked against the
   solver's own `w_s` on the demo run: agreement to ~4–8 % (the FD‑vs‑exact‑FE‑gradient gap), correct
   sinh/cosh shapes, `w(0)=0` and `p_nh(1)=0` exactly.

## Notes

- `regularize!` reconstructs the Cartesian node grid from the (duplicated) VTK point cloud, enabling
  heatmaps, bilinear gauge/transect sampling and grid quadrature. It is called automatically by
  `load_simulation`. Non‑Cartesian meshes fall back to nearest‑node sampling (no heatmaps).
- Amplitudes/phases use the **second half** of the record (post‑transient), matching the solver tests.
- GR runs headless; a `Could not create decoration from factory` line from GR is a benign X11 message.
