# Building and using the GridapLFEM system image (Snellius)

A precompiled **system image** (`GridapLFEM_sysimage.so`) bakes the whole compiled
solver stack (Gridap + GridapDistributed + GridapSolvers + the LFE-M residual/Jacobians)
into one file, so every MPI rank **loads** the code instead of JIT-compiling it.

**Why it matters here**

- **No OOM.** Without the sysimage, 32–64 ranks each JIT-compile the full FEM stack
  at once (~4–8 GB/rank) and blow past node memory. With it, ranks share the mmap'd
  image (~1–2 GB, shared) — no compile, no memory spike.
- **Cheaper.** No ~30–45 min per-rank compile billed on every run. Lets you run on the
  normal `rome` partition (budget **L1**, ~196k SBU) instead of scarce `fat_rome`
  (budget **L2**, ~27k SBU).
- **Correct MPI.** Built and launched against the cluster's **system OpenMPI**, not the
  bundled JLL — the mismatch that otherwise crashes runs at launch.

---

## Files in this folder

| File | Role |
|------|------|
| `load_modules_snellius.sh` | `module load` OpenMPI/5.0.3 + Julia, and put the OpenMPI lib on `LD_LIBRARY_PATH`. Sourced by **both** build and run. |
| `set_preferences.jl` | Pin `MPI.jl` to the **system** OpenMPI via `MPIPreferences.use_system_binary()` (writes `LocalPreferences.toml`, forces recompile). |
| `warmup.jl` | Tiny sequential + distributed solves (both regimes) that trace-compile the LFE-M code into the image. |
| `compile.jl` | `create_sysimage(...)` with an **MPI preflight** that refuses to build unless MPI binds system OpenMPI. |
| `compile_snellius.sh` | SLURM job that runs the three build steps end to end (partition `rome`). |

The launcher that **uses** the image:
`../run/dist_small/run_lin_periodic_plane_small_sysimage.sh`.

---

## The one rule (why builds break otherwise)

`MPI.jl` chooses **JLL vs system** at *precompile* time and that choice is **frozen
into the sysimage**. So the MPI binding must be system **in the build process**, and the
**same OpenMPI module + same `--project`** must be used at build and run. Three invariants:

1. **Same OpenMPI module** → `source compile/load_modules_snellius.sh` in build *and* run.
2. **Same `--project`** (`$HOME/GridapLFEM.jl`) → holds the `LocalPreferences.toml` binding.
3. **`-J` the image built with (1)+(2)** active.

A runtime `LocalPreferences.toml` **cannot** fix an image already baked against the JLL —
only what was resolved at build time counts.

---

## Prerequisites (once)

- On the cluster, `~/GridapLFEM.jl` is up to date (`git pull`) — especially
  `compile/set_preferences.jl`, `compile/compile.jl`, `compile/compile_snellius.sh`.
- Build on the **same partition** you run on (`rome`) so the CPU target matches.

---

## Step 1 — verify the MPI binding (do this first, interactively)

This is the decisive gate. Never build until `using MPI` binds the **system** library.

```bash
source ~/GridapLFEM.jl/compile/load_modules_snellius.sh
cd ~/GridapLFEM.jl

# What is currently resolved? (should become "system" after step below)
julia --project=. -e 'using MPIPreferences; @show MPIPreferences.binary'

# (Re)pin to the system OpenMPI — now uses use_system_binary()
julia --project=. compile/set_preferences.jl

# VERIFY: must print the system library, e.g. "Open MPI v5.0.3, ..."
julia --project=. -e 'using MPI; println(MPI.MPI_LIBRARY_VERSION_STRING)'
```

- ✅ `Open MPI v5.0.3, …` → binding correct, proceed.
- ❌ an `~/.julia/artifacts/.../libmpi.so … undefined symbol` error, or
  `MPIPreferences.binary` ≠ `"system"` → **stop**; the module isn't loaded or the
  preference didn't take. Do not build (you'd bake a broken image).

---

## Step 2 — build the sysimage

```bash
cd ~/GridapLFEM.jl/compile
sbatch compile_snellius.sh
```

The job runs three steps: `set_preferences.jl` → `Pkg.precompile()` + **verify MPI is
system (aborts otherwise)** → `mpiexecjl -n 1 julia compile.jl` (warmup traces the code).
Takes ~45–60 min; produces `~/GridapLFEM.jl/GridapLFEM_sysimage.so` (~1 GB). Check:

```bash
ls -lh ~/GridapLFEM.jl/GridapLFEM_sysimage.so
tail -n 30 ~/GridapLFEM.jl/compile/compile_GridapLFEM.*.out   # look for "MPI preflight OK"
```

`compile.jl` will **error out** (not silently succeed) if MPI ever resolves to a JLL, so
a completed build is a correct build.

---

## Step 3 — launch a run against the image

```bash
cd ~/GridapLFEM.jl
sbatch run/dist_small/run_lin_periodic_plane_small_sysimage.sh
```

The launcher sources the same module, passes the same `--project`, and adds
`-J .../GridapLFEM_sysimage.so`. No compile, no OOM, on `rome`/L1.

To make the other cases use the image, add the same two things to their launchers:
`source .../load_modules_snellius.sh` and `-J .../GridapLFEM_sysimage.so`.

---

## When to rebuild

Rebuild only after: a change to `src/*.jl`, a package upgrade, or a change of the OpenMPI
module (then update the `libmpi` path in `set_preferences.jl` first). Editing a run script
or environment variables needs **no** rebuild.

---

## Troubleshooting

**`InitError(mod=:OpenMPI_jll … undefined symbol: opal_single_threaded)` at launch.**
`OpenMPI_jll` (the bundled artifact) got baked into the image and its initializer fires at
startup. There are two independent causes:
1. **The MPI binding wasn't system at build time.** Fix at the source and rebuild:
   ```bash
   source ~/GridapLFEM.jl/compile/load_modules_snellius.sh
   cd ~/GridapLFEM.jl
   julia --project=. compile/set_preferences.jl
   julia --project=. -e 'using MPI; println(MPI.MPI_LIBRARY_VERSION_STRING)'   # must be Open MPI 5.0.3
   ```
   The build's preflight now aborts if this is wrong.
2. **PackageCompiler's transitive-dependency sweep force-loads `OpenMPI_jll`.** Even with
   `binary="system"` (so a plain `using MPI` never loads the JLL), `create_sysimage` with the
   default `include_transitive_dependencies=true` does `using` on every Manifest dep — including
   `OpenMPI_jll` (a dep of `MPI` that is *not* imported under system MPI) — baking its startup
   initializer. `compile.jl` sets **`include_transitive_dependencies=false`** to prevent this.
   Symptom that distinguishes this case: a plain `using MPI` prints `Open MPI 5.0.3` fine, yet
   `strings GridapLFEM_sysimage.so | grep OpenMPI_jll` still shows the JLL. Rebuild after pulling
   the fixed `compile.jl`; the JLL must then be **absent** from that `strings` output.

**Job accepted then vanishes from `squeue`, no `.out`/`.err`.** Not a sysimage issue —
`sacct -j <id> --format=JobID,State,ExitCode,Reason` and check `AdminComment`
(`reason=budget` ⇒ walltime/CPU × rate exceeds the partition's budget; lower `--time`,
use `rome`/L1).

**Memory / billing.** The image is mmap-shared across ranks on a node, so per-rank private
memory is small. On `rome` (2 GB/core) keep `--mem-per-cpu` near 2–4 GB; asking more bills
you for extra cores. Right-size with `sacct -j <id> --format=MaxRSS` after a run.
