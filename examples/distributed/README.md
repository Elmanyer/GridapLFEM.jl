# Distributed algebraic LFE-M example runs (cluster-ready)

MPI-parallel wave-propagation cases for the **algebraic (stacked-layout) 2D LFE-M solver**
(`LFEM_2D/src/LFEModelAlg.jl`), all writing the **full field set** to VTK: surface elevation
`eta`, per-σ-node horizontal velocity components `u<j>x`/`u<j>y`, and the reconstructed
**vertical velocity** `w_s<σ>` and **total pressure** `p_s<σ>` at every vertical σ-node
(`write_w`/`write_pressure` on by default).

Unlike the old solver's distributed examples, these run the **full nonlinear physics through the
single Gridap path** (`setup_and_run_alg_distributed`, hand Jacobians + GMRES+Jacobi+Newton) —
no owned V⊗H loop, no linear-core restriction. New-physics flags are exposed as env vars:
`LFEM_PFULL` (full `P¹L¹+P²L²+P³L³` leading pressure) and `LFEM_NLP68` (nonlinear-pressure
components 6–8).

## Cases

| Script | Physics |
|--------|---------|
| `run_plane_wave_alg_dist.jl` | long-crested plane wave, line-source wavemaker, long flume |
| `run_ring_wave_alg_dist.jl`  | radial ring wave, point-source wavemaker, square basin, 4-side sponge |
| `run_ic_hump_alg_dist.jl`    | Gaussian hump released from rest, **closed basin** (`x_wall_bc=true` — mandatory for IC problems) |
| `run_bathymetry_alg_dist.jl` | shoaling of a plane wave over a smooth submerged bar (variable `d(x)`, slope-pressure package on) |

Each runs at **any** vertical resolution (`LFEM_M`) and any core count. All knobs are
environment variables with sensible defaults — see `_dist_common_alg.jl` (shared) and each
script's header (case-specific).

## Launch

Use Julia's own MPI launcher (system `mpiexec` fails here with a PMIx mismatch). **The MPI rank
count `-n` MUST equal `LFEM_PX · LFEM_PY`**, and keep `LFEM_NX` divisible by `LFEM_PX`,
`LFEM_NY` by `LFEM_PY`.

```bash
# From the project root. Plane wave, M=2, 8 ranks (8×1 grid):
LFEM_M=2 LFEM_PX=8 LFEM_PY=1 \
  ~/.julia/bin/mpiexecjl --project=. -n 8 julia --project=. \
  LFEM_2D/examples/distributed/run_plane_wave_alg_dist.jl

# Ring wave at M=3, 16 ranks (4×4), bigger mesh:
LFEM_M=3 LFEM_PX=4 LFEM_PY=4 LFEM_NX=400 LFEM_NY=400 \
  ~/.julia/bin/mpiexecjl --project=. -n 16 julia --project=. \
  LFEM_2D/examples/distributed/run_ring_wave_alg_dist.jl

# Quick local smoke of the plumbing (2 ranks, tiny mesh, 1 period):
LFEM_PX=2 LFEM_PY=1 LFEM_NX=60 LFEM_NY=4 LFEM_LX=60 LFEM_LY=4 \
LFEM_PERIODS=1 LFEM_SAVE_EVERY=40 LFEM_XWM=12 LFEM_SPONGE=12 \
  ~/.julia/bin/mpiexecjl --project=. -n 2 julia --project=. \
  LFEM_2D/examples/distributed/run_plane_wave_alg_dist.jl
```

> `MPI_Finalize` prints a benign OFI/WiFi-NIC error and exits 143 on the development machine —
> the computation completes correctly beforehand. Julia buffers stdout when redirected to a
> file; the drivers `flush` at every print interval so logs can be followed live.

## On a SLURM cluster

```bash
#!/bin/bash
#SBATCH --job-name=lfem_alg_plane
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --time=08:00:00
#SBATCH --output=lfem_alg_plane_M%a.out
#SBATCH --array=2,3            # job array: one run per M (SLURM_ARRAY_TASK_ID)

cd $SLURM_SUBMIT_DIR
export LFEM_M=$SLURM_ARRAY_TASK_ID
export LFEM_PX=8 LFEM_PY=4     # 8*4 = 32 = --ntasks
export LFEM_NX=4000 LFEM_NY=200
export LFEM_SAVE_EVERY=25
export LFEM_OUTDIR=$SLURM_SUBMIT_DIR/output/plane_alg_M${LFEM_M}

srun julia --project=. LFEM_2D/examples/distributed/run_plane_wave_alg_dist.jl
# (or ~/.julia/bin/mpiexecjl --project=. -n $SLURM_NTASKS julia --project=. <script>)
```

## Validation against the old solver

The sequential algebraic solver is oracle-equivalent to `LFE-M_2D_solver` at machine precision
(`LFEM_2D/test/test_equivalence_alg.jl`); the distributed path is validated by
`LFEM_2D/test/test_basic_alg_distributed.jl` (4 ranks, linear + fully nonlinear, sequential
agreement within GMRES tolerance). For a cross-solver check at scale, run the old
`run_plane_wave_dist.jl` and this `run_plane_wave_alg_dist.jl` with identical env settings and
compare `eta` snapshots/gauge sections in ParaView (identical field naming).
