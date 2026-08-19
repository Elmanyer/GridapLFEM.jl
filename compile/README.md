# Building and using the GridapBALFEM system image (Snellius)

A precompiled **system image** (`GridapBALFEM_sysimage.so`) bakes the whole compiled
solver stack (Gridap + GridapDistributed + GridapSolvers + the LFE-M residual/Jacobians)
into one file, so every MPI rank **loads** the code instead of JIT-compiling it.

> **This only became true on 2026-08-04.** `GridapBALFEM` used to be loaded with
> `include(src/GridapBALFEM.jl)`, creating a fresh `Main.GridapBALFEM` in every process.
> PackageCompiler retains code belonging to the *packages* it bakes, so the solver's types and —
> far more expensive — every Gridap FEM specialisation keyed on them were **not** in the image and
> were recompiled in each rank on each run. The image removed the library compile but never the
> application compile. `GridapBALFEM` is now a real package and is listed in `create_sysimage`, which
> is what makes the claim above hold. If you ever see ranks in `typeinf`/`optimize` in a backtrace,
> the solver is not in the image — check `compile.jl`'s package preflight.
>
> **Note on the 2026-08-04 OOM kills (exit 137).** They were first attributed to this missing
> application compile. That was wrong: the demonstrated mechanism was a GMRES cache
> over-allocation (139 MB/rank per numerical setup, churned until RSS crossed the 2 GB/core
> limit) — the failing jobs advanced ~140 steady steps before dying, whereas a compile spike
> occurs before step 1. Both defects were real and both are fixed; see `krylov_m` vs `ls_maxiter`
> in the root `CLAUDE.md` §7.

**Every launcher in `run/` and `run/dist_small/` uses the image** — via the shared helper
`run/balfem_env.sh`. Build it once (Steps 1–2), then just `sbatch` the case you want.

**Why it matters here**

- **No OOM.** Without the sysimage, 32–128 ranks each JIT-compile the full FEM stack
  at once (~4–8 GB/rank) and blow past node memory. With it, ranks share the mmap'd
  image (~1–2 GB, shared) — no compile, no memory spike.
- **Cheaper.** No ~30–45 min per-rank compile billed on every run. All jobs now run on the
  normal `rome` partition (budget **L1**, ~196k SBU) at the node-default 2 GB/core, instead of
  scarce `fat_rome` (budget **L2**, ~27k SBU) with inflated memory requests.
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
| `compile_snellius.sh` | SLURM job that runs the build steps end to end (partition `rome`), then stamps the image with a hash of `src/*.jl` for the launchers' staleness check. |

The image is **used** by every launcher in `../run/` and `../run/dist_small/`, through the shared
helper **`../run/balfem_env.sh`** — see "How the launchers use the image" below.

---

## The one rule (why builds break otherwise)

`MPI.jl` chooses **JLL vs system** at *precompile* time and that choice is **frozen
into the sysimage**. So the MPI binding must be system **in the build process**, and the
**same OpenMPI module + same `--project`** must be used at build and run. Three invariants:

1. **Same OpenMPI module** → `source compile/load_modules_snellius.sh` in build *and* run.
2. **Same `--project`** (`$HOME/GridapBALFEM.jl`) → holds the `LocalPreferences.toml` binding.
3. **`-J` the image built with (1)+(2)** active.

A runtime `LocalPreferences.toml` **cannot** fix an image already baked against the JLL —
only what was resolved at build time counts.

---

## Prerequisites (once)

- On the cluster, `~/GridapBALFEM.jl` is up to date (`git pull`) — especially
  `compile/set_preferences.jl`, `compile/compile.jl`, `compile/compile_snellius.sh`.
- Build on the **same partition** you run on (`rome`) so the CPU target matches.

---

## Step 1 — verify the MPI binding (do this first, interactively)

This is the decisive gate. Never build until `using MPI` binds the **system** library.

```bash
source ~/GridapBALFEM.jl/compile/load_modules_snellius.sh
cd ~/GridapBALFEM.jl

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
cd ~/GridapBALFEM.jl/compile
sbatch compile_snellius.sh
```

The job runs three steps: `set_preferences.jl` → `Pkg.precompile()` + **verify MPI is
system (aborts otherwise)** → `mpiexecjl -n 1 julia compile.jl` (warmup traces the code).
Takes ~45–60 min; produces `~/GridapBALFEM.jl/GridapBALFEM_sysimage.so` (~1 GB). Check:

```bash
ls -lh ~/GridapBALFEM.jl/GridapBALFEM_sysimage.so
tail -n 30 ~/GridapBALFEM.jl/compile/compile_GridapBALFEM.*.out   # look for "MPI preflight OK"
```

`compile.jl` will **error out** (not silently succeed) if MPI ever resolves to a JLL, so
a completed build is a correct build.

---

## Step 3 — launch a run against the image

Every launcher already uses the image; just submit the case you want:

```bash
cd ~/GridapBALFEM.jl
sbatch run/dist_small/run_lin_periodic_plane_small.sh   # small-domain case
sbatch run/run_irregularsea.sh                          # production case
```

No compile, no OOM, on `rome`/L1.

---

## How the launchers use the image

All launchers in `run/` and `run/dist_small/` are thin SLURM wrappers around one shared helper,
**`run/balfem_env.sh`**, which holds the three build↔run invariants in a single place. A launcher is
just its `#SBATCH` header, the case's env-var overrides, and one call:

```bash
source $HOME/GridapBALFEM.jl/run/balfem_env.sh    # (1) same modules  (2) resolves the sysimage

export BALFEM_PX=8
export BALFEM_PY=4            # 8*4 = 32 ranks
export BALFEM_REGIME=linear   # case-specific overrides only

balfem_run 32 examples/distributed_small/run_periodic_plane_small.jl
```

`balfem_run <nranks> <script.jl>` (script path relative to the project root) expands to the
`mpiexecjl --project=… -n <nranks> julia --project=… -J<sysimage> <script>` invocation, after
checking that both the image and the script exist — a missing image **fails the job immediately**
with the build command to run, instead of silently falling back to a 45-min-per-rank JIT compile.
It also prints a `[balfem_env]` banner (project, cluster, image, ranks, script) at the top of the
job's `.out`, so what a run actually loaded is on the record.

Helper knobs, exported before sourcing (all optional):

| Variable | Default | Purpose |
|---|---|---|
| `BALFEM_PROJ` | `$HOME/GridapBALFEM.jl` | project root (holds `LocalPreferences.toml`) |
| `BALFEM_CLUSTER` | `snellius` | selects `compile/load_modules_<cluster>.sh` (`blue` for DelftBlue) |
| `BALFEM_SYSIMAGE` | `$BALFEM_PROJ/GridapBALFEM_sysimage.so` | alternate image (e.g. testing a rebuild side by side) |
| `BALFEM_NO_SYSIMAGE` | unset | `=1` drops `-J` and runs the old JIT path — an escape hatch for when the image is stale or mid-rebuild; expect the full compile cost and memory spike back |
| `BALFEM_STRICT_SYSIMAGE` | unset | `=1` turns the "image is stale w.r.t. `src/`" warning into a hard abort (see "When to rebuild") |

To add a case, copy the nearest launcher, adjust the `#SBATCH` header and the exported
`BALFEM_*` overrides, and point `balfem_run` at the parametric script in `examples/distributed_small/`
(or `examples/distributed/`). Nothing sysimage-specific needs repeating.

---

## Partition and memory (post-sysimage)

The launchers request **`rome`** (budget L1, ~196k SBU) rather than the scarce `fat_rome` (L2,
~27k SBU). `fat_rome` was only ever needed because 32–128 ranks JIT-compiling the FEM stack
simultaneously each took ~4–8 GB and OOM'd the node — with the image there is no compile and no
spike.

For the same reason the launchers set **no `--mem-per-cpu`** and take the `rome` node default
(2 GB/core): the image is mmap-shared across the ranks on a node, so per-rank *private* memory is
just the local mesh partition and Krylov vectors. Asking for more than the default does not buy
free memory — SLURM bills the extra as additional cores per rank, which is exactly the cost the
sysimage was introduced to avoid. Only raise it if a genuinely large case is killed for memory,
and confirm with `sacct -j <id> --format=MaxRSS` first.

(DelftBlue's `run_blue.sh` keeps `--mem-per-cpu=3900M`, which *is* that cluster's per-core default.)

---

## When to rebuild

Rebuild only after: a change to `src/*.jl`, a package upgrade, or a change of the OpenMPI
module (then update the `libmpi` path in `set_preferences.jl` first). Editing a run script
or environment variables needs **no** rebuild.

**Staleness is detected, not prevented.** Every launcher runs
`balfem_check_sysimage_freshness` (in `run/balfem_env.sh`) before starting the ranks, and **warns** if
the image no longer matches `src/`:

```
[balfem_env] WARNING: the system image appears STALE.
[balfem_env]   checked: content stamp
[balfem_env]   reason : src/*.jl content changed since the build (baked ff68b3a48579…, now ad42e9e9c1ca…)
[balfem_env]   The image bakes a compiled copy of src/*.jl, so this job would
[balfem_env]   run the OLD solver code — not your current sources.
```

Two mechanisms, strongest first:

1. **Content stamp** (used whenever it exists). `compile_snellius.sh` calls
   `balfem_write_sysimage_stamp` after a successful build, writing
   `GridapBALFEM_sysimage.so.src.sha256` — a hash of every `src/*.jl`. At launch the hash is
   recomputed and compared, so **only a real source change trips it**: a `touch`, a re-clone, or a
   `git checkout` that restores identical content does not. Edits, added files and deleted files are
   all caught.
2. **mtime fallback** (no stamp — an image built before this check existed). Warns if any `src/*.jl`
   is newer than the image. Coarser: it can cry wolf after anything that rewrites mtimes without
   changing content. Rebuild once to get a stamp and the exact check.

The check **warns and continues** — a stale image still runs, and only you know whether the change
mattered. Set **`BALFEM_STRICT_SYSIMAGE=1`** to abort instead; worth doing for long production jobs,
where discovering the staleness afterwards costs the entire run. When the image is current the
banner simply reads `[balfem_env] freshness: image matches src/ (content stamp)`.

So: after touching `src/*.jl`, rebuild before submitting; to run edited sources immediately without
rebuilding, launch with `BALFEM_NO_SYSIMAGE=1` (JIT path — correct but slow; the freshness check is
skipped there, since no image is used).

---

## Troubleshooting

**`InitError(mod=:OpenMPI_jll … undefined symbol: opal_single_threaded)` at launch.**
`OpenMPI_jll` (the bundled artifact) got baked into the image and its initializer fires at
startup. There are two independent causes:
1. **The MPI binding wasn't system at build time.** Fix at the source and rebuild:
   ```bash
   source ~/GridapBALFEM.jl/compile/load_modules_snellius.sh
   cd ~/GridapBALFEM.jl
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
   `strings GridapBALFEM_sysimage.so | grep OpenMPI_jll` still shows the JLL. Rebuild after pulling
   the fixed `compile.jl`; the JLL must then be **absent** from that `strings` output.

**Job accepted then vanishes from `squeue`, no `.out`/`.err`.** Not a sysimage issue —
`sacct -j <id> --format=JobID,State,ExitCode,Reason` and check `AdminComment`
(`reason=budget` ⇒ walltime/CPU × rate exceeds the partition's budget; lower `--time`,
use `rome`/L1).

**Memory / billing.** See "Partition and memory" above: take the node default, don't add
`--mem-per-cpu`, and right-size with `sacct -j <id> --format=MaxRSS` after a run.

**`[balfem_env] ERROR: sysimage not found`.** The job stops before launching because
`GridapBALFEM_sysimage.so` is missing from the project root — build it (Step 2), or set
`BALFEM_NO_SYSIMAGE=1` to run the JIT path meanwhile. This is deliberate: the alternative is a
silent 45-min-per-rank compile that then OOMs.

**Run behaves like an older version of the solver.** A stale image — the job's `.out` will carry the
`[balfem_env] WARNING: the system image appears STALE` block. Rebuild, or use `BALFEM_NO_SYSIMAGE=1`.
See "When to rebuild".

**Freshness says stale right after a `git pull`/re-clone that changed nothing in `src/`.** You are on
the coarse mtime fallback because the image predates the content stamp. Rebuild once (the build now
writes `GridapBALFEM_sysimage.so.src.sha256`) and the check becomes content-based and exact.
