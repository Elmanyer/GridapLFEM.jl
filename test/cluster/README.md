# Cluster / distributed validation

Validation runs for the solver **at scale** (multi-node, distributed memory) and **over long
integration times** — the production regime. Where the `test/` suite proves correctness on small,
fast problems, these scripts prove the solver stays correct on **big meshes** run for **many time
steps** on many ranks. The full derivation and the small-scale suite are documented in
`../../building_files/ValidationTests.md` / `.tex`.

Every metric here is a **global reduction** (`sum(∫·dΩ)` reduces across ranks in GridapDistributed) or
the built-in **in-run residual check**, so nothing depends on point gauges (which are sequential-only)
— they work on any core count.

## What is validated, and how

| Script | What it validates | Metric | Gate |
|--------|-------------------|--------|------|
| `cluster_conservation.jl` | long-run **mass conservation** + energy/amplitude **stability** in a closed basin, fully nonlinear | `∫η` drift, energy drift, `‖η‖_{L²}` envelope (all global reductions) | mass drift `< 1e-6`; amplitude bounded |
| `cluster_selfconsistency.jl` | the **full nonlinear time-dependent solver** at scale: an unsteady, finite-amplitude manufactured solution recovered over many CN steps with **all** nonlinear pressure terms active, through the real GMRES+Jacobi+Newton stack | `‖u_n − u*(tₙ)‖ / ‖u*‖` (global reduction) | machine precision (`< 1e-8`) every step |

Both write a **CSV time series** to `output/…` (columns in the header line) for offline plotting and
tabulation, and print a rank-0 summary with an explicit `PASS`/`FAIL` (non-zero exit on failure, so a
SLURM job fails loudly).

### `cluster_conservation.jl` — the invariant test at scale
Closed inviscid basin, initial Gaussian free-surface hump, no wavemaker/sponge. The depth-integrated
continuity weak form conserves `∫η` exactly (test `ψ≡const`); the hydrostatic energy stays bounded
under Crank–Nicolson. A growing `∫η` drift would expose a broken continuity assembly, a leaking wall
BC, or an inconsistent parallel reduction; a growing amplitude would expose an instability. This is
the cheapest, strongest at-scale correctness signal — an **exact** invariant checked over a long run.

### `cluster_selfconsistency.jl` — the nonlinear solver test at scale
An unsteady manufactured solution `u*(x,t)` (degree-2 in space so it is exactly Q2-representable,
oscillating in time, over a curved bed) is marched over many steps. The per-step forcing is the
solver's own residual assembled on `u*`, injected by wrapping the residual
`res_forced = R(u) − R(u*)`; the scheme-consistent θ-point and `u̇` make the recovery **exact**, so
`‖u_n − u*(tₙ)‖` must sit at machine precision at every step. Because the wrapped operator keeps the
same hand Jacobians, the standard distributed Newton/GMRES iterator drives it — so this exercises the
**distributed** nonlinear assembly, the frozen-projection (CG+Jacobi) machinery for the `∇H`/`𝓟`
nonlinear-pressure components, and the GMRES+Jacobi+Newton stack, end-to-end, over a long nonlinear
run. Its sequential twin `../test_selfconsistency.jl` verifies the same on one node.

## The physical benchmarks at scale

The Yang & Liu (2024) §4 physical benchmarks already have cluster-ready, env-configurable scripts in
`../../examples/distributed/`:

- `run_plane_wave_dist.jl` — large flume (dispersion / regular waves),
- `run_ring_wave_dist.jl` — large 2-D ring (cylindrical spreading),
- `run_bathymetry_dist.jl` — submerged bar / shoal (harmonic generation — the flagship §4 case),
- `run_ic_hump_dist.jl` — closed-basin initial-condition release.

Run these at scale for the physical validation. Two at-scale correctness signals come for free:

1. **In-run governing-equation residual check.** All distributed runs enable `check_every` — every N
   steps the θ-scheme discrete residual is *independently reassembled* and its `‖R‖∞` printed; it must
   stay `≤ check_tol` (≈1e-8). A `WARN` here means the parallel solve is not solving the equations. Set
   `LFEM_CHECK_EVERY` (via the driver) and watch the `[check]` lines.
2. **Stability envelope + solver stats.** The `eta_max` time series and per-step Newton/GMRES counts
   (printed by the monitor) show the run stays bounded and the solver converges over the whole run.

The **physical** metrics (dispersion celerity, bound-harmonic amplitudes `H₁,₂,₃(x)`, radial `1/√r`
decay) are extracted **offline** from the VTK snapshots (`LFEM_SAVE_EVERY>0`, `LFEM_WRITE_W`/`_PRESSURE`)
— open `output/…/solution.pvd` in ParaView, or sample the fields with a small serial post-process, and
apply the same DFT-at-forcing-frequency analysis used by the sequential `examples/validation/` scripts.

## Launching

Use Julia's own MPI launcher (`mpiexecjl`); the system `mpiexec` fails with a PMIx mismatch on the dev
machine. `px·py` **must** equal `-n`, and `nx`/`ny` should divide evenly by `px`/`py`.

```bash
# distributed conservation, 8 ranks, 2000×200 mesh, 400 steps
LFEM_PX=8 LFEM_PY=1 LFEM_NX=2000 LFEM_NY=200 LFEM_DT=0.02 LFEM_TFINAL=8.0 \
  ~/.julia/bin/mpiexecjl --project=. -n 8 julia --project=. \
      GridapLFEM.jl/test/cluster/cluster_conservation.jl

# distributed unsteady nonlinear MMS, 8 ranks, 400×200 mesh, 200 steps
LFEM_PX=4 LFEM_PY=2 LFEM_NX=400 LFEM_NY=200 LFEM_NSTEPS=200 LFEM_NLPFULL=1 \
  ~/.julia/bin/mpiexecjl --project=. -n 8 julia --project=. \
      GridapLFEM.jl/test/cluster/cluster_selfconsistency.jl
```

A SLURM template is in `slurm_validation.sh` (edit the account/partition/module lines for your
cluster; it mirrors `../../run/run_snellius.sh`). The **first** full-FEM compile per job is tens of
minutes (JIT) — budget for it, and prefer long runs so compile time is amortised.

## Visualising / tabulating the results

- **CSV** (`output/cluster_conservation/conservation.csv`, `output/cluster_selfconsistency/mms.csv`): load in
  Python/Julia/gnuplot. For conservation, plot `dmass_rel` and `denergy_rel` vs `t` (should be flat
  near machine ε for mass; bounded for energy) and `etaL2` vs `t` (envelope). For MMS, plot `rel_err`
  vs `t` (a flat line at machine precision is the pass signature; any upward trend is a red flag).
- **VTK** (physical benchmarks): `output/…/solution.pvd` in ParaView for animations and spatial
  fields (`eta`, `u1x`, …, reconstructed `w_s<σ>`, `p_s<σ>`); or read back for gauge/DFT extraction.
- **Scaling**: the rank-0 wall-time and `s/step` lines let you tabulate strong/weak scaling across
  core counts (same problem, increasing `-n`).
