# PROMPT — Plan the local-validation & runtime-diagnostics work package

> **What this file is.** A self-contained brief for the next agent. Read it in full, then
> produce **one planning document** (see §Deliverable). It is a *planning* task: you write the
> plan, you do **not** implement it.

---

## 0. Role and operating rules

You are working in `GridapLFEM.jl/` — the production 2-D LFE-M non-hydrostatic wave solver
(stacked `[η,𝖴x,𝖴y]` layout, serial + distributed). Before planning anything:

1. **Read `CLAUDE.md` (this folder) and `README.md` in full.** They are current and accurate as
   of commit `051befa`; §8 "Solver conventions and key rules" and the "Current Implementation
   Stage" section are the two most load-bearing parts for this work package.
2. **Verify against the code, not the prose.** Every claim you put in the plan about what the
   solver currently does must be backed by a `file.jl:line` reference you actually opened. The
   relevant files are `src/monitor.jl`, `src/timeloop.jl`, `src/timeloop_dist.jl`,
   `src/utilities.jl`, `src/utilities_dist.jl`, `src/horizontal.jl`, `src/waveinput.jl`,
   `run/lfem_env.sh`, `examples/distributed_small/*.jl`, `run/dist_small/*.sh`.
3. **Julia is invoked through `julia-mcp`**, never `julia` directly.
4. **Do not modify solver physics.** The residual, Jacobians and vertical tensors are validated
   (21 test files, oracle equivalence ≤7e-15). Nothing in this work package should touch
   `problem.jl`, `nlpressure.jl`, `vertical.jl`, or `tensors.jl`. If a task appears to require
   it, **stop and report** rather than planning a workaround (global operating rule: design
   errors in validated code are reported, not patched).
5. **Nothing destructive.** No deletion of `output/`, no rewriting of existing validated tests.
   New files only, except for the specific, named edits the plan justifies.

---

## 1. Deliverable

**One file: `building_files/LOCAL_VALIDATION_PLAN.md`** (repo convention: plans live in
`building_files/`, alongside `LFEM_runs.md`, `SPONGE_WAVEGEN_PLAN.md`,
`PACKAGE_MIGRATION_PLAN.md`).

It must contain, in this order:

- **§0 Executive summary** — 10 lines max: what the work package achieves and why now.
- **§1 Findings of record** — the established facts this plan builds on (see §3 below), each
  with its evidence (file, log line, commit).
- **§2–§6** — one section per task, in the execution order 1→5 of §4 below.
- **§7 Dependency graph and execution order** — an explicit ordering with the reasons; call out
  which items *must* precede others (in particular: task 2's instrumentation should land before
  the task 3–5 runs so the new diagnostics are actually exercised by them).
- **§8 Risks, open questions and decision points** — anything you could not resolve from the
  code, phrased as a question with your recommended answer.
- **§9 Definition of done** — a checklist that can be ticked off.

**Every planned deliverable file must be specified with:** target path, purpose, the exact
configuration it runs (physics controls, geometry, mesh, ranks, `dt`, `T_final`), its expected
wall-clock runtime on 6 cores, and its **quantitative** pass criterion. "Check it looks right"
is not a pass criterion.

**Do not write code in the plan** beyond short illustrative snippets (≤10 lines) where a
signature or a formula is clearer than prose.

---

## 2. Context: why this work package exists

Every cluster run on record predates at least one of three 2026-08 fixes — the surface-damping
sponge (`95f5ec6`), the GMRES configuration fix (`f7fe62d`), and the `mu_max` 8→40 raise — so
the archived `output/` is **not a baseline**. The cluster feedback loop is also far too slow to
debug against: the runs on record burned 8 h, 44 h and 72 h of wall time before failing, and one
128-rank directional-sea job spent 58 h to reach t=1.82 s.

The strategic goal of this work package is therefore to **move the validation loop onto the local
machine (6 cores, minutes-to-an-hour per run)** so the solver's internal machinery — sponge,
relaxation zone, boundary behaviour, wave generation — can be checked in real time against
quantitative criteria, and to **make the runtime logs diagnostic enough** that when a cluster run
does fail, the log alone says why.

---

## 3. Findings of record (established — do not re-derive, but do cite)

**F1 — The 2 GB/core memory experiment failed.** A small-domain simulation was launched on the
`rome` partition at the node-default 2 GB/core (i.e. *without* `--mem-per-cpu=4G`) and was killed
for lack of memory, with the same error as the previous crashes. **Consequences:**
- The launchers **keep** the `--mem-per-cpu=4G` header for now. Any plan step that assumes it can
  be dropped is wrong.
- The `CLAUDE.md` / `README.md` claims that the memory request "only ever covered the compile
  spike" and can be dropped once the sysimage is confirmed are now **falsified or incomplete** and
  must be corrected as part of the plan.
- The open question "Is the sysimage actually removing the per-rank compile?" is now *partly*
  answered: either it is not, or the memory is being consumed somewhere other than compilation.
  The plan must design an experiment that distinguishes these.

**F2 — What the archived `.out` files actually show.** Four logs exist under `output/*/`:

| log | config | outcome |
|---|---|---|
| `small_plane_linear_none_flat_A0.001_T2.0_M2_bis` | linear, none, flat, A=0.001, 32 ranks | `eta_max` grew monotonically 1e-3 → **4.4e+1 m**, Newton still reporting `[conv]` throughout; non-convergence only at step 617, after 8 h |
| `small_irregular_nonlinear_full_flat_Hs0.2_M2` | nonlinear/full/flat, Hs=0.2, 32 ranks | NaN at t=13.8 after **44 h**; NL 21.4 it/step average |
| `small_irregular_nonlinear_full_bar_Hs0.2_M2` | nonlinear/full/bar, Hs=0.2, 32 ranks | non-convergence at t=14.28 after **72 h**; NL 21.2 it/step |
| `directional_sea_dist_M3` | 128 ranks, 1200×600 | non-converged at step 90 after **58 h**; 2311 s/step |

All four show **`gmres=100` on every single step** — the pinned-at-the-cap signature of the
pre-`f7fe62d` bug — while the banner advertised `max iters = 2000`, a cap that never existed.
These are pre-fix runs; they are evidence about *diagnostics*, not about the current solver.

**F3 — The diagnostic failures those logs demonstrate** (this is the raw material for task 2):
- A linear, small-amplitude run grew its surface elevation by **4 orders of magnitude** without a
  single warning. The only guard is `eta_max > 1e4` (`timeloop.jl:220`, `timeloop_dist.jl:260`) —
  an absolute threshold, blind to a run whose incident amplitude is 1e-3 m.
- The linear solver sat at its iteration cap for 600+ steps and nothing said so. `SolverMonitor`
  records `lin_iters` (`monitor.jl:94-96`) but never compares it to the cap.
- The banner reported configured *kwargs*, not the values actually installed in the solver
  object — which is exactly how the 2000-vs-100 discrepancy hid for months.
- Nothing logs memory. Given F1, this is the single most valuable missing datum.
- `max|η|` is reported as a bare global maximum, with no location and no interior/sponge split —
  yet the post-hoc diagnosis of the open-boundary spurious mode (CLAUDE.md §8) was done exactly
  by binning `max|η|` by `x` and comparing `|u|/|η|` against `ω/kd`.

**F4 — The distributed driver has no point gauges.** `setup_and_run_distributed` returns
`(t, eta_max)` only; gauges are sequential-only (`setup_and_run`, kwarg `gauges`). A
"distributed-gauge utility" is already an open item in `CLAUDE.md`. This directly constrains
tasks 3–5: any local run on 6 MPI ranks currently has *no* way to probe η at a station, which is
what every one of the requested validation tests needs.

---

## 4. The five tasks

Plan them in this order. For each, the plan section must state: objective, what already exists
(with `file:line`), the concrete steps, the files to be created/modified, the acceptance
criterion, and the estimated effort.

### Task 1 — Record the memory finding and design the memory diagnosis

Reflect F1 in the repository and turn the ambiguity into an experiment.

Plan must cover:
- **Documentation correction.** Which statements in `CLAUDE.md` and `README.md` about
  `--mem-per-cpu`, the `rome` partition sizing, and the sysimage/compile hypothesis are now
  wrong, and what they should say instead. Keep the launchers' 4 GB header; explain in the header
  comment *why* it is no longer provisional.
- **A memory-attribution experiment.** Design the smallest set of cluster runs that discriminates
  between the candidate consumers, and say what measurement decides each:
  1. per-rank JIT compilation (i.e. the sysimage is not being used, is stale, or does not contain
     the specialisations) — check the freshness stamp
     (`lfem_check_sysimage_freshness` in `run/lfem_env.sh`), whether `-J` was actually on the
     command line, and *when* in the run the RSS peak occurs (before step 1 ⇒ compile);
  2. GMRES cache allocation — was the failing run built from a commit at or after `f7fe62d`, and
     what does `krylov_m` × (vectors + dense Hessenberg per rank) predict for this configuration?
  3. steady growth across steps (a leak: Gridap caches, `nl_pressure=:full` frozen-projection
     factorisations, VTK/reconstruction buffers) — RSS rising monotonically with step index;
  4. baseline assembly/matrix footprint — is 2 GB/core simply not enough for this problem size?
- **The evidence to collect**: `sacct -j <id> --format=MaxRSS,MaxRSSTask,Elapsed`, `seff`, the
  step index at which the kill occurred, and the rank that peaked. State explicitly which of
  these the user must run (they have cluster access; you do not).
- **A decision rule**: what result would justify dropping back to 2 GB/core, and what result
  would instead escalate to a real memory-reduction task.

### Task 2 — Runtime instrumentation: assess, prioritise, and plan

Assess what additional runtime information would make both successful and failed runs
diagnosable, then plan the implementation.

**Do this in three explicit steps in the plan:**

**(a) Inventory what is already logged.** Read `src/monitor.jl` end to end plus the reporting
paths in both time loops. Produce a table: quantity → where it is computed → where it is printed
→ its known limitation. Nothing already implemented may appear as a proposal.

**(b) Assess the candidate additions.** Below is the candidate list distilled from F3 — treat it
as a starting point to **evaluate, extend and prune**, not as a specification. For each, judge
diagnostic value against cost (compute, code churn, distributed collectives, log volume), and
give a verdict: *implement now / implement later / reject*, with the reason.

*Solver-health candidates*
- Linear-solver saturation: flag when GMRES reaches `maxiter` or fails its `rtol`; report
  min/mean/max over the Newton iterations and RK stages of a step, not just the last solve.
- Newton convergence quality: per-iteration residual ratio / stagnation detection, and a
  distinction between "converged on atol" and "converged on rtol".
- Per-stage breakdown under RK: `nl_iters` currently *sums* over SDIRK stages
  (`monitor.jl:109-116`), so a printed "NL 24 it" is 2 stages of 12 — make this unambiguous.
- A banner that reports the values **read back from the constructed solver objects**, so a
  mis-passed argument like the `ls_maxiter`/`krylov_m` bug is visible on line 1 of the log.
- CG iteration counts for the `nl_pressure=:full` frozen-projection solves (`nlpressure.jl:228`).
- Split the per-step wall time into Jacobian assembly / linear solve / other.

*Physical-health candidates*
- Memory: per-rank RSS and the global max, sampled every N steps (Linux: `/proc/self/statm`; or
  `Sys.maxrss()`), plus Julia GC pressure. Given F1 this is likely the top priority.
- Location of `max|η|` (x,y), and a split of `max|η|` into **interior vs sponge/boundary zone** —
  the exact quantity that diagnoses the open-boundary mode.
- `|u|/|η|` at the location of the maximum, compared against the expected `ω/(kd)` for a genuine
  wave — the discriminator documented in CLAUDE.md §8.
- Global invariants per step: total mass `∫η`, total energy, and their drift — cheap collective
  reductions, and the earliest possible warning of instability.
- A **relative** divergence guard: abort when `max|η|` exceeds a multiple of the expected/incident
  amplitude (or grows with a positive e-folding rate over a wave period), instead of the absolute
  `1e4` threshold that let a 4-order-of-magnitude growth run for 8 hours.
- Abort-on-non-finite with a final VTK dump of the state that produced it.

*Operational candidates*
- Checkpoint/restart (a 44 h run that NaNs at t=13.8 currently loses everything).
- A machine-readable per-step log (CSV/JSON alongside the human log) so postprocessing and the
  suite in tasks 3–5 can assert on the run's history without parsing `.out` files. Consider
  whether `GridapLFEMPost` should consume it.

**(c) Plan the implementation.** For everything marked *implement now*: which file, which
function, how the sequential and distributed paths stay identical in layout (the shared
formatting helpers in `monitor.jl:209-321` are the mechanism), which quantities need a collective
reduction and how they are made rank-0-safe, the cost per step at 32–128 ranks, the new driver
kwargs and their defaults (must be **off or cheap by default** — no regression for the validated
tests), the new `LFEM_*` env vars and their propagation through `examples/distributed*/`, and how
each addition is itself tested.

### Task 3 — Local validation tests for the solver's internal machinery

Design a suite of tests that run **locally on ≤6 cores in minutes**, targeting the machinery that
the failed runs implicate. At minimum, the three the user named:

1. **Sponge test.** Drive a wave into a sponge layer; verify the envelope decays *as the damping
   function predicts* (derive the expected spatial decay from the μ profile and the group
   velocity — `make_sponge` is quadratic, `utilities.jl:91-100`) and that the reflection back into
   the domain is below a stated threshold. Note that the sponge now damps **both** η and velocity
   (`+∫ μ q η`); the test must exercise that, since a velocity-only sponge is structurally blind
   to the boundary mode.
2. **Relaxation-zone test with BC wave generation.** With `wave_bc` + `relax_bc`, verify the
   amplitude *inside* the relaxation zone matches the prescribed inflow state, and that outgoing/
   reflected energy is absorbed. State the tolerance and how the incident field is computed
   (`waveinput.jl` gives the exact component table).
3. **Boundary spurious-mode test.** Monitor the solution *at* the domain boundaries over a
   multi-period run and assert no spurious growth: use the documented signature — amplitude at the
   last boundary node versus its inward neighbours, spatial disconnection from the incident field,
   e-folding growth rate, and `|u|/|η| ≈ 0.4` (mode) versus `≈ ω/kd` (genuine wave). Cover both
   `x_wall_bc=false` (open/sponged) and the wall/periodic lateral conditions.

Also consider — and justify including or excluding — a wave-generation amplitude/phase transfer
check, a rest-state test (an undisturbed surface must stay at rest to machine precision), and a
reflection coefficient measured with the **Goda–Suzuki** decomposition already implemented for
`test_bc_spectrum.jl`.

Constraints the plan must resolve:
- **Reuse, don't reinvent.** `test_bc_spectrum.jl` (Goda–Suzuki), `test_bc_generation.jl`, and
  `postprocessing/GridapLFEMPost` (`probes`, `spectral`, `diagnostics`) already contain most of
  the measurement machinery. State what is reused and what is genuinely new.
- **The gauge problem (F4).** These tests are gauge-based. Decide per test whether it runs
  sequentially (gauges available, single core, possibly fast enough) or distributed on 6 ranks
  (needs the distributed-gauge utility to be built first — in which case that becomes an explicit
  prerequisite item in the plan, with its design sketched).
- **Runtime budget.** State a per-test target (suggest ≤5 min each, ≤30 min for the suite) and the
  mesh/`dt`/`T_final` that meet it. Remember the ~3.5 s package load and that Q2 + `Nσ` fields are
  not cheap.
- **Placement and convention.** Where do these live — `test/local/`? extend `test/`? — and how are
  they invoked (a single runner script? added to an existing suite?). Match the existing style:
  standalone scripts using `Test`, `using GridapLFEM`, printed PASS counts.
- **1D and/or small 2D.** The user asked for 1D-horizontal tests "and maybe small 2D". Decide
  which of the three tests are genuinely 1D-adequate and which need 2D, and say why.

### Task 4 — 1D-horizontal simulation files (local + cluster)

Produce runnable 1D-horizontal cases with launchers for both environments.

**Resolve this design question first, and make it §8's headline decision point:** the solver is
structurally 2-D (`CartesianDiscreteModel` on a rectangle, `horizontal.jl:28-38`; `Ex`/`Ey`
throughout the residual). "1D horizontal" therefore means one of:

- **(A) Quasi-1D flume** — a narrow domain with `y_wall_bc=:periodic` (or `:wall`) and very few
  cells in y. Zero solver changes. Check what the minimum viable `ny` is for Q2 with
  `isperiodic=(false,true)` and whether `ny=1` degenerates.
- **(B) True 1-D horizontal support** — a genuine 1-D mesh and a `𝖴x`-only field set. This touches
  `horizontal.jl`, the residual's `Ey` terms, the FE spaces, the BC tags and the reconstruction —
  i.e. validated code. Per operating rule 4, if you conclude (B) is required, **stop and report**
  rather than planning it unilaterally.

Recommend one (A is the expected answer) with the reasoning, and note the physics caveat: a
quasi-1D run is not identical to a 1-D model — say what differs (lateral DOFs, y-periodic mode
content, cost).

Then plan:
- The case set — which physics configurations (`regime` × `nl_pressure` × `flat_bed`) and which
  wave-generation mechanisms (`:inner_res` line source vs `:bc_gen` boundary) are worth covering,
  and why each earns its slot.
- Local variants (6 cores) and cluster variants (more cores) of each, following the existing
  parametric-script + thin-launcher pattern (`examples/distributed_small/*.jl` +
  `run/dist_small/*.sh`, env-configured via `get!`/`genv_*` so the banner and solver stay
  consistent). Cluster launchers go through `run/lfem_env.sh` and `lfem_run <n> <script>`, and
  must satisfy `px·py = lfem_run ranks = #SBATCH --ntasks`.
- A **local launcher convention** — there is none yet. `run/lfem_env.sh` is cluster-only (modules,
  sysimage, SLURM). Decide whether to add a `run/local/` with a small `mpiexecjl -n 6` wrapper,
  and specify it.
- Sizing: for 6 ranks, what partition `(px,py)` and mesh keep the cells-per-rank sane and the run
  in the minutes range.

### Task 5 — Small 2-D local simulation files

Same pattern, small 2-D domains, runnable on 6 cores, spanning several configurations.

Plan must give:
- **The configuration matrix**, with a one-line justification per case — what each case is
  *for* diagnostically. Candidate axes: `regime` (linear/nonlinear), `nl_pressure`
  (none/native/full), `flat_bed` (flat / submerged bar), wave generation (`:inner_res` line,
  `:inner_res` point/ring, `:bc_gen` regular, `:bc_gen` irregular/directional sea), lateral BC
  (wall/open/periodic), and IC release. Prefer a small, well-argued matrix over an exhaustive one.
- The relationship to the existing 20-case `run/dist_small/` suite: these local cases should be
  *scaled-down siblings* of those, so a local result is predictive of the cluster one. Say
  explicitly which local case corresponds to which cluster case.
- Mesh/`dt`/`T_final`/`save_every` sized for a **real-time-ish** loop on 6 cores, plus the
  expected output volume per case (VTK with `write_w`/`write_pressure` is not small).
- What is monitored and asserted for each — tie this back to the task-2 instrumentation so these
  runs are the first consumers of it.
- How results are inspected: which `GridapLFEMPost` entry points, and whether a small
  "inspect this run" script should ship alongside.

---

## 5. Cross-cutting constraints the plan must respect

- **Local hardware: 6 cores.** MPI runs use `~/.julia/bin/mpiexecjl --project=. -n <=6`; the
  system `mpiexec` fails with a PMIx mismatch. `nx`/`ny` must divide by `px`/`py`.
  `MPI_Finalize` prints a benign OFI error and exits 143 — any runner script must not treat that
  as failure.
- **`p_horizontal ≥ 2`** — Q1 zeroes the dispersion term and silently turns the model into
  shallow-water.
- **Solid-wall Dirichlet BCs must include the corner tags**; IC-release problems need
  `x_wall_bc=true`.
- **Amplitude guidance**: `A_wave ≤ 0.001` for long fully nonlinear runs; the small-domain suite
  deliberately runs harder cases (A=0.1, Hs=0.2) — say which regime each planned case is in.
- **Sponge width must cover the longest component** (`kd_min` ⇒ `λ_max`), not the peak period.
- **Julia buffers stdout to files** — output appears only at exit unless flushed; any runner
  script must account for this.
- **Sysimage staleness**: editing `src/*.jl` (task 2 does) invalidates the cluster image. The
  plan must include the rebuild step for any cluster-side item, and note
  `LFEM_STRICT_SYSIMAGE=1` / `LFEM_NO_SYSIMAGE=1`.
- **Non-regression**: the existing 21-file test suite must still pass unchanged after task 2. Say
  how that is verified and how long it takes.
- **Docs**: `CLAUDE.md`, `README.md` and `building_files/LFEM_runs.md` are kept current by
  convention. Each task section must name the doc updates it entails.
- **`.gitignore`**: new `output/` subtrees and any generated artefacts.

---

## 6. Anti-goals

Do not plan: solver physics changes; a rewrite of the existing test suite; long cluster
production runs; the sheared-current extension; new boundary-generation sides; or postprocessing
features not required by tasks 3–5. Do not silently widen or narrow the five tasks — if you
believe one is misframed, say so in §8 and plan it as asked anyway.

---

## 7. Acceptance criteria for the plan itself

The plan is done when:

1. `building_files/LOCAL_VALIDATION_PLAN.md` exists with sections §0–§9 as specified.
2. Every "what the code currently does" statement carries a `file.jl:line` citation.
3. Task 2 contains the three-part inventory → assessment → implementation structure, and every
   candidate carries an explicit *implement now / later / reject* verdict with a reason.
4. Every planned file has path, purpose, configuration, expected 6-core runtime, and a
   quantitative pass criterion.
5. The 1D-horizontal approach (A vs B) is decided and justified, with the physics caveats stated.
6. The distributed-gauge constraint (F4) is resolved for every gauge-dependent test.
7. The execution order and dependencies are explicit, and the plan states what the user must do
   themselves (cluster submissions, `sacct`/`seff` queries).
8. Open questions are collected in §8 with a recommended answer each — not scattered through the
   document.
