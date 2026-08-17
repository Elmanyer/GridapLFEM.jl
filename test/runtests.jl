# ==============================================================
#  runtests.jl — the sequential suite runner
#
#  WHY THIS EXISTS, AND WHY IT DOES NOT USE EXIT CODES.
#  On 2026-08-16 the whole sequential suite was run for the first time in weeks
#  and FOUR tests were found silently broken — they had been *documented* as
#  passing rather than *re-measured* — plus one that errored before executing a
#  single gate. There was no batch runner, so nothing noticed.
#
#  The trap that made it worse: test_dispersion_nonlinear.jl was wrapped in
#  `if abspath(PROGRAM_FILE) == @__FILE__`, so under `include()` it printed three
#  header lines and returned cleanly having executed NOTHING — and was recorded
#  as passing, twice. (That guard has since been removed.)
#
#  Hence the rule this runner is built on:
#
#      A CLEAN EXIT CODE IS NOT EVIDENCE THAT A TEST RAN.
#
#  Every file is therefore judged on its GATE OUTPUT — the "PASS"/"FAIL" lines it
#  prints — and a file that produces NO gate lines at all is reported as BLANK
#  and counted as a failure, however cleanly it exited. A test that can complete
#  without asserting anything is a test that will eventually be believed without
#  running.
#
#  Each file runs in its OWN subprocess, so a hard error in one cannot abort the
#  batch and no test can leak state into the next.
#
#  USAGE
#    julia --project=. test/runtests.jl              # default tier: fast+medium
#    LFEM_TESTS=fast  julia --project=. test/runtests.jl
#    LFEM_TESTS=all   julia --project=. test/runtests.jl   # incl. the slow MMS studies
#    LFEM_TESTS=<comma-separated file names>         # an explicit subset
#
#  NOT INCLUDED: the MPI tests (they need mpiexecjl and a rank count) and
#  test/local/ (its own runner, run_local_tests.sh). Both are listed at the end
#  with the exact command, so their absence is visible rather than assumed.
# ==============================================================

using Printf

const HERE    = @__DIR__
const PROJECT = dirname(HERE)
const JULIA   = joinpath(Sys.BINDIR, "julia" * (Sys.iswindows() ? ".exe" : ""))
const LOG_DIR = get(ENV, "LFEM_TEST_LOGS",
                    joinpath(PROJECT, "output", "test_logs"))
mkpath(LOG_DIR)

#  tier: :fast (seconds)  :medium (minutes)  :slow (tens of minutes)
#  :retired files are run but NEVER counted as failures — see test_equivalence.jl.
const SUITE = [
    # ---- verification: the analytic MMS + the Jacobian gate ------------------
    ("test_mms_forcing.jl",              :fast,   "MMS forcing gates, linear"),
    ("test_mms_forcing_nonlinear.jl",    :fast,   "MMS forcing gates, nonlinear"),
    ("test_linear_newton_gate.jl",       :medium, "linear ⇒ 1 Newton iter/stage (sloping bed)"),
    ("test_jacobians_ad.jl",             :slow,   "hand ∂R/∂u, ∂R/∂u̇ vs AD, all 8 models"),
    ("test_mms_convergence.jl",          :slow,   "order of accuracy, Model 1"),
    ("test_mms_convergence_nonlinear.jl",:slow,   "order of accuracy, Models 3–4"),
    # ---- base ----------------------------------------------------------------
    ("test_vertical.jl",                 :fast,   "vertical tensor identities"),
    ("test_primitives.jl",               :fast,   "tensor index order / contraction semantics"),
    ("test_basic.jl",                    :medium, "smoke, linear + fully nonlinear"),
    ("test_dispersion.jl",               :medium, "phase speed vs linear theory, kd=3"),
    ("test_nlpressure.jl",               :medium, "nonlinear-pressure identities + dynamics"),
    ("test_sloshing.jl",                 :medium, "standing-wave period"),
    ("test_conservation.jl",             :medium, "mass conservation, closed basin"),
    # ---- physics / validation -------------------------------------------------
    ("test_dispersion_curve.jl",         :fast,   "Cm/Ce(kd) sweep, applicable kd"),
    ("test_selfconsistency.jl",          :medium, "SUPPORT ONLY: Jacobians/time-stepping consistent"),
    ("test_convergence.jl",              :medium, "Richardson temporal order"),
    ("test_vertical_profile.jl",         :medium, "reconstructed w(σ) vs Airy sinh"),
    ("test_energy.jl",                   :medium, "non-dissipativity (pins :theta)"),
    ("test_dispersion_nonlinear.jl",     :medium, "full-NL ⇒ Airy at vanishing amplitude"),
    ("test_shallow_water.jl",            :medium, "kd→0 ⇒ √(gd)"),
    # ---- boundary wave generation ---------------------------------------------
    ("test_waveinput.jl",                :fast,   "Dirichlet generation data + WaveSpec converter"),
    ("test_bc_generation.jl",            :medium, "generated regular wave, end to end"),
    ("test_bc_spectrum.jl",              :medium, "3-component sea, Goda–Suzuki"),
    # ---- retired ---------------------------------------------------------------
    ("test_equivalence.jl",              :retired,"oracle predates R_P — provenance only"),
]

const MPI_TESTS = [
    ("test_basic_distributed.jl",        4, "linear + nonlinear vs sequential refs"),
    ("test_nlpressure_distributed.jl",   4, "full nonlinear pressure vs sequential"),
    ("test_bc_generation_distributed.jl",4, "Dirichlet generation vs sequential"),
]

# --- tier selection ----------------------------------------------------------
sel = get(ENV, "LFEM_TESTS", "default")
selected =
    if sel == "fast"
        [t for t in SUITE if t[2] === :fast]
    elseif sel == "all"
        SUITE
    elseif sel == "default"
        [t for t in SUITE if t[2] in (:fast, :medium, :retired)]
    else
        names = strip.(split(sel, ","))
        got = [t for t in SUITE if t[1] in names]
        unknown = setdiff(names, [t[1] for t in SUITE])
        isempty(unknown) || error("runtests: unknown test file(s): $(join(unknown, ", "))")
        got
    end

println("=" ^ 78)
println("  GridapLFEM sequential test suite — $(length(selected)) file(s), tier '$sel'")
println("  Judged on GATE OUTPUT (PASS/FAIL lines), never on exit status alone.")
println("  Logs: $LOG_DIR")
println("=" ^ 78)

results = NamedTuple[]
t_start = time()

for (i, (file, tier, what)) in enumerate(selected)
    path = joinpath(HERE, file)
    @printf("\n[%2d/%2d] %-38s (%s)\n        %s\n", i, length(selected), file, tier, what)
    flush(stdout)

    #  Own subprocess: a hard error cannot abort the batch, and no test can leak
    #  state (globals, Gridap caches, ENV) into the next.
    #
    #  Output goes to a FILE, not an in-memory buffer: with a non-waited process
    #  the async copy into an IOBuffer is not guaranteed to have finished when
    #  wait() returns, so the verdict could be computed from a truncated log —
    #  and a truncated log looks exactly like a test that printed no gates. The
    #  file also survives the run, which is what you want when something fails.
    log = joinpath(LOG_DIR, replace(file, ".jl" => ".log"))
    t0  = time()
    proc = run(pipeline(Cmd(`$JULIA --project=$PROJECT $path`);
                        stdout = log, stderr = log), wait = false)
    wait(proc)
    dt  = time() - t0
    txt = read(log, String)

    #  THE GATE COUNT IS THE VERDICT. Both spellings are in use across the suite:
    #  "  PASS  name" lines, and test_linear_newton_gate's "n/m checks passed".
    npass = count(m -> true, eachmatch(r"^\s*PASS\b"m, txt))
    nfail = count(m -> true, eachmatch(r"^\s*FAIL\b"m, txt))
    if npass == 0 && nfail == 0
        m = match(r"(\d+)/(\d+) checks passed", txt)
        if m !== nothing
            npass = parse(Int, m.captures[1])
            nfail = parse(Int, m.captures[2]) - npass
        end
    end

    status =
        tier === :retired            ? :retired :
        (npass == 0 && nfail == 0)   ? :blank   :   # ran nothing — treat as broken
        nfail > 0                    ? :fail    :
        !success(proc)               ? :error   :   # gates passed but the file threw
        :pass

    push!(results, (file=file, tier=tier, status=status, npass=npass, nfail=nfail,
                    secs=dt, code=proc.exitcode, output=txt))

    tag = status === :pass  ? "PASS"  : status === :retired ? "RET " :
          status === :blank ? "BLANK" : status === :error   ? "ERR " : "FAIL"
    @printf("        → %-5s  %d PASS / %d FAIL   %.0fs  (exit %d)\n",
            tag, npass, nfail, dt, proc.exitcode)
    flush(stdout)

    #  Print the tail of anything that did not pass — the whole point of a batch
    #  run is not having to re-run a failure to find out what it said.
    if status in (:fail, :blank, :error)
        lines = split(rstrip(txt), '\n')
        println("        ┌─ last ", min(25, length(lines)), " lines of ", log)
        for l in lines[max(1, end-24):end]
            println("        │ ", l)
        end
        println("        └────────────────────────────────────────")
    end
end

# --- summary -----------------------------------------------------------------
println()
println("=" ^ 78)
@printf("  SUMMARY — %d file(s) in %.1f min\n", length(results), (time()-t_start)/60)
println("=" ^ 78)
@printf("  %-38s %-7s %6s %6s %8s\n", "file", "status", "pass", "fail", "secs")
for r in results
    @printf("  %-38s %-7s %6d %6d %8.0f\n",
            r.file, uppercase(String(r.status)), r.npass, r.nfail, r.secs)
end

nbad = count(r -> r.status in (:fail, :blank, :error), results)
nret = count(r -> r.status === :retired, results)
println("-" ^ 78)
if nret > 0
    println("  $nret retired file(s) run for provenance and NOT counted — see their headers.")
end
if any(r -> r.status === :blank, results)
    println("  ⚠ BLANK = the file produced no gate output at all. That is a FAILURE:")
    println("    it means the file ran nothing, regardless of its exit code.")
end

println()
println("  Not covered by this runner:")
for (f, n, what) in MPI_TESTS
    println("    ~/.julia/bin/mpiexecjl --project=. -n $n julia --project=. test/$f")
    println("        ($what)")
end
println("    bash test/local/run_local_tests.sh      (the local machinery suite)")
println("    LFEM_TESTS=all                          (adds the slow MMS rate studies)")

nbad == 0 ? println("\n  ALL SELECTED TESTS PASSED.") :
            error("runtests: $nbad file(s) failed, blank or errored — see the summary above.")
