# `run/local/` — running GridapLFEM on this workstation

The cluster loop is too slow to debug against: the four archived runs in `output/` burned 8 h,
44 h, 58 h and 72 h before failing. These launchers move the loop onto the local machine, where a
case finishes in **minutes** and can be watched in real time.

Everything here is sized for **6 cores**.

---

## Entry points

| | what it is |
|---|---|
| `lfem_local.sh` | the shared helper — resolves the project, caps the rank count, launches. `lfem_local_run <script.jl>` (sequential, **keeps point gauges**) and `lfem_local_mpi <n> <script.jl>` (MPI, `n ≤ 6`). Source it; don't run it. |
| `run_1d_<case>.sh` (7) | quasi-1D flume cases → `examples/local_1d/run_flume_1d.jl` |
| `run_2d_<case>.sh` (8) | small 2-D cases → `examples/local_2d/run_small_2d.jl` |
| `../../test/local/run_local_tests.sh` | the validation *gates* (sponge, relaxation zone, boundary modes, rest state) — pass/fail, not observation |

```bash
run/local/run_1d_lin_inner_flat.sh              # one 1-D case, sequential, ~8 min
run/local/run_2d_nl_full_plane_flat.sh          # one 2-D case, 6 MPI ranks
julia --project=. examples/inspect_run.jl output/local_2d/*   # judge the results
```

---

## The two things to know before reading a result

**1. Sequential runs have gauges; MPI runs do not.** The distributed driver evaluates no point
gauges — a point search would need inter-rank communication (`src/timeloop_dist.jl:21-22`). So:

* the **1-D** cases run **sequentially** by default (`LFEM_MPI=0`) and return `diags` with
  `gauge_vals`, which is what phase- and amplitude-based analysis needs;
* the **2-D** cases run under **MPI** (`3×2 = 6` ranks) and are judged entirely from
  `<output_dir>/diagnostics.csv` — the machine-readable step log.

`LFEM_MPI` flips either one, at the cost of the corresponding capability.

**2. The ~3.5 min JIT compile is paid once per process.** A case that runs for 4 minutes spends
nearly half its wall time compiling. Running several cases inside one Julia process amortises it;
running them as separate launchers does not. That is why the *tests* group several runs per file.

---

## What to look at

`diagnostics.csv` is written every `LFEM_DIAG_EVERY` steps and carries the columns that the
archived cluster failures needed and did not have:

| column | question it answers |
|---|---|
| `eta_max`, `x_at_max` | how big, and **where** — a maximum pinned at a domain edge is the boundary mode, not a wave |
| `eta_max_int`, `eta_max_damped` | interior vs sponge/relaxation zone. `damped/interior > 1` means the maximum is *inside* the absorber |
| `u_max` | with `eta_max`, the kinematic ratio: a real wave gives `ω/tanh(kd)`; the η-dominated mode gives far less |
| `mass`, `mass_drift`, `energy` | the cheapest early warning of an instability |
| `lin_min`, `lin_max`, `lin_sat` | GMRES iteration range and whether it hit its cap. `lin_sat=1` means the solve was **truncated** and Newton is getting poor steps — the defect that hid for months |
| `nl_iters`, `nl_stages` | `nl_iters` is summed over the SDIRK stages; divide by `nl_stages` for the per-stage count |
| `rss_mb`, `rss_peak_mb` | per-rank memory. Flat is fine; a monotone climb is the OOM in progress |

`examples/inspect_run.jl` turns that file into a verdict; it is stdlib-only, so it also runs
against a cluster output directory. For fields, spectra and plots use
`postprocessing/GridapLFEMPost`.

---

## Configuring a case

Both scripts take their whole configuration from `LFEM_*` environment variables, so a launcher is
just the overrides that make it that case. The full list is in each script's header and in
`examples/distributed/_dist_common.jl`; the ones you change most often:

```bash
LFEM_WAVE_GEN     1-D: inner|bc|sea          2-D: line|point|bc|sea
LFEM_REGIME       linear | nonlinear
LFEM_NL_PRESSURE  none | native | full
LFEM_FLAT_BED     1 = flat bed | 0 = submerged bar
LFEM_AWAVE        wave amplitude [m]         LFEM_PERIODS  duration in wave periods
LFEM_NX/LFEM_NY   mesh                       LFEM_DIAG_EVERY  diagnostics sampling
```

Two constraints the scripts enforce and will error on rather than let you discover later:
`LFEM_NX % LFEM_PX == 0` (and the same in y), and `LFEM_NY ≥ 3` whenever the lateral BC is
`:periodic` — Gridap requires at least 3 elements in a periodic direction.

---

## Not the cluster

`run/lfem_env.sh` (the cluster helper) loads modules and resolves the prebuilt system image;
none of that exists locally, so `lfem_local.sh` deliberately shares no code with it. The cluster
counterparts of these cases are `run/dist_small/run_1d_*.sh` (the same 1-D script at 32 ranks and
200 m) and the pre-existing 20-case suite in `run/dist_small/`; each 2-D launcher names its
cluster sibling in its header.

**After editing `src/*.jl`, rebuild the cluster system image** (`compile/compile_snellius.sh`)
before submitting anything — the image bakes a compiled copy of the solver, and a stale one runs
old code. The launchers warn; `LFEM_STRICT_SYSIMAGE=1` makes it an abort.
