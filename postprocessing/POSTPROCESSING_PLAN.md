# `GridapLFEMPost` — postprocessing library: design & plan

## 1. Purpose and scope

The solver writes results as **VTK** (`solution.pvd` + per-step `sol_t_*.vtu`, fields `eta, u1x, u1y,
…, w_s<σ>, p_s<σ>`) and as **CSV** time series (the cluster/convergence tests). This library turns
those raw outputs into Julia data structures that can be *operated on* (sampled, DFT'd, integrated) and
*visualised* (Plots.jl) — closing the loop from "the solver ran" to "here is the validated physics".

It is a **separate, self-contained package** (`postprocessing/`, own `Project.toml`) so the heavy
plotting/VTK dependencies never enter the solver's environment. It has **no dependency on the solver**:
it reads files and computes the little wave theory it needs (Airy relations) itself.

Two questions drive the feature set:
- **What must we *show* to validate the solver?** dispersion curve, vertical `w`/`p` profiles vs Airy,
  harmonic growth over a bar, radial `1/√r` decay, conservation/convergence traces, celerity.
- **What is essential to *observe* the wave propagation?** an animation of `η(x,y,t)`, snapshot
  heatmaps, and space–time (Hovmöller) transects.

---

## 2. Packages to add

| Package | Role | Why this one |
|---------|------|--------------|
| **`ReadVTK.jl`** | read `.vtu`/`.pvd` (points, cells, point-data) into Julia arrays | the standard, maintained VTK *reader* (WriteVTK is write-only) |
| **`Plots.jl`** + **`GR`** | all plotting (heatmaps, lines, animations→GIF) | one API, GR backend needs no system libs, GIF out of the box |
| **`FFTW.jl`** | full spectra, harmonic amplitudes | fast, standard FFT |
| **`Interpolations.jl`** | sample fields at arbitrary points / transects on the regularised grid | bilinear/bicubic on the Cartesian node grid |

Stdlib (no add): `DelimitedFiles` (CSV read/write), `Statistics`, `LinearAlgebra`, `Printf`. Optional
niceties (not required): `ColorSchemes` (wave colormaps), `Measures` (layout), `DataFrames`+`CSV` (only
if richer tabulation is wanted — `DelimitedFiles` suffices).

> Note: `Plots`+`GR` first-precompile is a few minutes; budget for it once per environment.

---

## 3. Data model (structures)

```julia
"One field over the whole run on a fixed (Eulerian) mesh: [n_points × n_times]."
const FieldSeries = Matrix{Float64}

"Regular-grid view of a Cartesian Q_p node cloud, for O(1) reshape/interpolation."
struct GridView
    xs :: Vector{Float64}          # unique sorted node x-coordinates  (length Nx)
    ys :: Vector{Float64}          # unique sorted node y-coordinates  (length Ny)
    idx:: Matrix{Int}              # [Nx × Ny] → row index into `points`/field data
end

"A loaded simulation: fixed mesh + time series of every field."
struct WaveSimulation
    times  :: Vector{Float64}                 # snapshot times            (Nt)
    points :: Matrix{Float64}                 # node coords [n_points × 2]
    fields :: Dict{String,FieldSeries}        # name → [n_points × Nt]
    grid   :: Union{Nothing,GridView}         # set by regularize! if Cartesian
    meta   :: Dict{Symbol,Any}                # :dir, :nt, :bbox, user params
end

"A virtual gauge: a point probe with a precomputed sampling stencil."
struct Gauge
    x :: Float64; y :: Float64
    inode :: Int                              # nearest node (fast path)
    stencil :: Union{Nothing,NTuple{4,Tuple{Int,Float64}}}  # bilinear (grid path)
end

"A σ-level vertical profile reconstructed from the w_s<σ>/p_s<σ> fields at a station."
struct SigmaProfile
    sigma :: Vector{Float64}       # σ levels (parsed from field names)
    value :: Vector{Float64}       # w or p at each level (first-harmonic amplitude)
end

"A tabulated CSV result (cluster/convergence/sweep outputs)."
struct CsvTable
    names :: Vector{String}
    cols  :: Dict{String,Vector{Float64}}
end
```

---

## 4. Module layout

```
postprocessing/
├── Project.toml                 # ReadVTK, Plots, GR, FFTW, Interpolations
├── README.md                    # quick-start + gallery
├── src/GridapLFEMPost.jl        # module entry: deps, includes, exports
│   ├── io.jl          # load_simulation / load_snapshot / regularize! / load_csv
│   ├── probes.jl      # gauge / timeseries / transect / field_at / sigma_profile
│   ├── spectral.jl    # dft_at / amplitude_at / phase_at / spectrum / harmonics / celerity
│   ├── diagnostics.jl # radial_profile / harmonic_growth / mass / energy / airy_*
│   └── plotting.jl    # plot_field / animate_field / plot_gauge / plot_hovmoller /
│                      #   plot_dispersion / plot_vertical_profile / plot_harmonic_growth /
│                      #   plot_radial_decay / plot_csv
└── examples/
    ├── make_wave_animation.jl   # η(x,y,t) → GIF (the flagship visual)
    ├── plot_dispersion.jl       # dispersion_sweep CSV → Cm/Ce(kd) with 2% band
    ├── vertical_profiles.jl     # w(σ), p(σ) vs Airy at a station
    └── conservation_plots.jl    # cluster CSV → mass/energy drift traces
```

---

## 5. API (functions, grouped)

**IO (`io.jl`)**
- `load_simulation(pvd_or_dir; fields=:all, regularize=true) -> WaveSimulation`
- `load_snapshot(vtu) -> (points::Matrix, data::Dict{String,Vector}, time)`
- `regularize!(sim) -> sim` — detect Cartesian node grid → build `GridView`.
- `load_csv(path) -> CsvTable` ; `column(tbl, name)`.

**Probes (`probes.jl`)**
- `gauge(sim, x, y; bilinear=true) -> Gauge`
- `timeseries(sim, field, g::Gauge) -> (t::Vector, v::Vector)`
- `field_at(sim, field, it::Int) -> Vector` and `grid_field(sim, field, it) -> Matrix` (Nx×Ny).
- `transect(sim, field, p0, p1; n=200, it) -> (s::Vector, v::Vector)`
- `sigma_profile(sim, x, y; kind=:w, ω) -> SigmaProfile` (parse `w_s<σ>`/`p_s<σ>`, first-harmonic amp).

**Spectral (`spectral.jl`)**
- `dft_at(t, v, ω) -> Complex`, `amplitude_at(t,v,ω)`, `phase_at(t,v,ω)` (steady window = 2nd half).
- `spectrum(t, v) -> (f::Vector, amp::Vector)` (FFTW; zero-padded, Hann optional).
- `harmonic_amplitudes(t, v, ω; n=3) -> Vector` (a₁…aₙ).
- `steady_amplitude(t, v)`, `envelope(t, v)`.
- `celerity(sim, field, g1::Gauge, g2::Gauge, ω) -> Cm` (phase differencing).

**Diagnostics (`diagnostics.jl`)**
- `radial_profile(sim, field, center, radii; ω) -> (r, amp)` (ring waves).
- `harmonic_growth(sim, y0, ω; xs, n=3) -> (xs, H::Matrix)` (bar/shoal, per-x DFT on a transect).
- `mass(sim, it)`, `energy(sim, it; g, d)` (grid-quadrature integrals).
- `airy_w(σ, kd)`, `airy_p(σ, kd)`, `airy_Ce(k,d;g)`, `airy_kd(ω,d;g)` (theory overlays).

**Plotting (`plotting.jl`)** — all return a `Plots.Plot`; `save=...` writes it.
- `plot_field(sim, field; it, kind=:heatmap, clims, aspect=:equal)` — η(x,y) color map.
- `animate_field(sim, field; fps=15, out="wave.gif", clims=:sym)` — **the propagation animation**.
- `plot_gauge(sim, field, g)` — η(t) at a point.
- `plot_hovmoller(sim, field, p0, p1; n, its=:all)` — transect × time heatmap (celerity = slope).
- `plot_dispersion(tbl; kd_app, band=0.02)` — Cm/Ce vs kd with the 2% band.
- `plot_vertical_profile(prof; airy_kd)` — reconstructed w(σ)/p(σ) vs Airy sinh/cosh.
- `plot_harmonic_growth(xs, H)` — H₁,₂,₃(x).
- `plot_radial_decay(r, amp)` — amp·√r vs r with the 1/√r reference.
- `plot_csv(tbl, x, ys...; kw...)` — generic CSV plotter (conservation/convergence).

---

## 6. Essential validation graphics (the deliverables)

1. **Wave-propagation animation** — `animate_field(sim,"eta")` → GIF: the headline "does it look like
   a wave field" check; reveals reflections, sponge behaviour, dispersion of a packet.
2. **Snapshot heatmap** of `η(x,y)` — crest structure, wavemaker, sponge decay.
3. **Gauge signal** `η(t)` — steady amplitude, transient, cleanliness.
4. **Hovmöller** (transect vs time) — straight crest lines whose slope *is* the celerity; the most
   direct visual of correct propagation speed.
5. **Dispersion curve** `Cm/Ce(kd)` from `dispersion_sweep_M*.csv` with the 2 % band and `kd_app`.
6. **Vertical profiles** `w(σ)`, `p(σ)` vs Airy sinh/cosh (reproduces Yang & Liu §3.2 Figs 6–8).
7. **Harmonic growth** `H₁,₂,₃(x)` over the submerged bar (the §4 shoal benchmark).
8. **Radial decay** `amp·√r` for ring waves (cylindrical spreading).
9. **Conservation / convergence** traces from the cluster CSVs (`dmass_rel`, `rel_err`, order fits).

---

## 7. Implementation phases

- **P1 — IO core.** `load_simulation`/`load_snapshot`/`regularize!`, `load_csv`; verify on a small
  generated dataset (a short `plane_wave` run with `save_every>0`).
- **P2 — Probes + spectral.** gauges, transects, DFT/harmonics, celerity; cross-check the celerity
  against the value the dispersion test measured.
- **P3 — Plotting.** heatmap, animation, gauge, Hovmöller, dispersion, vertical profile; save PNGs/GIF.
- **P4 — Diagnostics + examples.** radial, harmonic growth, CSV plots; the four demo scripts.

**Verification data**: generate once with a tiny run
`setup_and_run(..., save_every=2, write_w=true, write_pressure=true, output_dir="output/pp_demo")`
and drive every function against `output/pp_demo/solution.pvd`.
