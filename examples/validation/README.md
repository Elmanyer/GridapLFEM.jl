# Physical validation examples

Benchmark reproductions from Yang & Liu (2024, *J. Fluid Mech.* **999**, A32) and classical
wave-model validation cases. These are **analysis scripts** (multi-minute to multi-hour runs), not
CI tests — they print measured quantities and light PASS/FAIL where a clean analytical target exists.
The fast, gated correctness tests live in `../../test/` (see `../../ValidationTests.md`).

| Script | Benchmark | What it measures | Paper ref |
|--------|-----------|------------------|-----------|
| `stokes_harmonics.jl` | nonlinear regular wave | bound 2nd-harmonic ratio `a₂/a₁` vs Stokes theory | §4 (Stokes waves) |
| `submerged_bar.jl` | wave over a submerged bar/shoal | harmonic growth `H₁,₂,₃(x)` over the bar (Beji–Battjes/Dingemans) | §4 (submerged shoal) |
| `solitary_wave.jl` | solitary wave | celerity `≈√(g(d+A))` + shape retention | nonlinearity–dispersion balance |
| `ring_spreading.jl` | point-source ring waves | radial symmetry + far-field `1/√r` decay | 2-D isotropy/geometry |
| `bichromatic_sideband.jl` | deep-water two-frequency group | both frequencies present, bounded (scaffold) | §4 (bichromatic / sideband) |
| `spectral_fidelity.jl` | JONSWAP sea via Dirichlet BC generation (WaveSpec.jl) | component-wise amplitude + dispersion transfer through the domain; Hs recovery | spectral fidelity (BC generation) |

## Running

```bash
# from inside GridapBALFEM.jl/
julia --project=. examples/validation/<script>.jl
# or from the parent repository
julia --project=GridapBALFEM.jl GridapBALFEM.jl/examples/validation/<script>.jl
```

The project must be the **`GridapBALFEM.jl` package environment**: the scripts do `using GridapBALFEM`,
which only resolves where the package is available.

All default to a **quick** size; each script's header documents the production size (finer mesh, more
periods) needed for a publication-quality comparison. Key rules (see the root `CLAUDE.md`):

- `p_horizontal ≥ 2` (Q1 zeroes the dispersion term).
- IC-release problems (`solitary_wave.jl`) require `x_wall_bc=true` (closed basin).
- Nonlinear runs need `A_wave` small (`≲ 0.001` scaled) for long-run stability; `stokes_harmonics.jl`
  deliberately uses a larger amplitude to generate a measurable bound harmonic, over a short run.
- Deep-water cases (`bichromatic_sideband.jl`) use P1LFE-3 to keep `kd` inside the applicable band.
- Measurement uses a DFT at the forcing frequency over the second half of the record.

## What still needs the paper's data

`submerged_bar.jl` and `stokes_harmonics.jl` reproduce the *mechanism* (harmonic generation, bound
harmonic); a full quantitative match needs the digitised experimental/reference curves from the
paper overlaid on the printed `H₁,₂,₃(x)` / `a₂/a₁` outputs. `bichromatic_sideband.jl` is a scaffold:
the quantitative envelope-evolution and Benjamin–Feir growth-rate comparisons require much longer
domains/runs and (for sidebands) a seeded-perturbation source.
