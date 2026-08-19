# ============================================================================
#  balfem_env.sh — shared launcher environment for every GridapBALFEM SLURM script
#
#  Sourced (not executed) by every script in run/ and run/dist_small/. It does
#  the three things that are identical across all cases, and that MUST agree
#  with how the sysimage was built:
#
#    (1) load the SAME cluster modules used at build time (OpenMPI + Julia),
#    (2) resolve + verify the prebuilt sysimage GridapBALFEM_sysimage.so,
#    (3) expose `balfem_run <nranks> <script.jl>` which launches the run with
#        the SAME --project (holds LocalPreferences.toml, i.e. the system-MPI
#        binding) and -J<sysimage>.
#
#  WHY THE SYSIMAGE. Without -J, every rank JIT-compiles the full Gridap FEM
#  stack independently: ~30-45 min of wall time per run, and a ~4-8 GB/rank
#  memory spike that OOM'd whole nodes at 32-128 ranks (which is why these jobs
#  used to need fat_rome). With -J the ranks mmap one shared, already-compiled
#  image: no compile, no spike, and the jobs fit the cheap rome/L1 budget at the
#  node-default 2 GB/core — do NOT add --mem-per-cpu to get memory back, that
#  only makes SLURM bill extra cores per rank.
#
#  Build the image once (same partition + same OpenMPI module as the runs):
#      cd $HOME/GridapBALFEM.jl/compile && sbatch compile_snellius.sh
#  Rebuild it after any change to src/*.jl or a package/OpenMPI upgrade;
#  editing a launcher or its env vars needs no rebuild. Full walkthrough and
#  troubleshooting: compile/README.md.
#
#  USAGE in a launcher
#      source $HOME/GridapBALFEM.jl/run/balfem_env.sh
#      export BALFEM_PX=8; export BALFEM_PY=4          # 8*4 = 32 ranks
#      export BALFEM_REGIME=linear                   # case-specific overrides
#      balfem_run 32 examples/distributed_small/run_periodic_plane_small.jl
#
#  KNOBS (export before sourcing, or in the submit environment)
#      BALFEM_PROJ         project root            (default $HOME/GridapBALFEM.jl)
#      BALFEM_CLUSTER      snellius | blue         (default snellius)
#      BALFEM_SYSIMAGE     path to the .so         (default $BALFEM_PROJ/GridapBALFEM_sysimage.so)
#      BALFEM_NO_SYSIMAGE  1 = run without -J      (fallback while the image is
#                                                 stale/rebuilding; the ranks
#                                                 then JIT-compile — expect the
#                                                 old cost and memory spike)
#      BALFEM_STRICT_SYSIMAGE  1 = abort instead of warning when the image is
#                                stale w.r.t. src/ (see "Sysimage freshness")
# ============================================================================

set -euo pipefail

: "${BALFEM_PROJ:=$HOME/GridapBALFEM.jl}"
: "${BALFEM_CLUSTER:=snellius}"
: "${BALFEM_SYSIMAGE:=$BALFEM_PROJ/GridapBALFEM_sysimage.so}"

# (1) the SAME OpenMPI module the sysimage was built against — a mismatch here
#     is what produces the launch-time MPI symbol errors.
source "$BALFEM_PROJ/compile/load_modules_${BALFEM_CLUSTER}.sh"

# ============================================================================
#  Sysimage freshness — is the image older than the solver sources?
#
#  The image bakes a COMPILED COPY of src/*.jl. Nothing in Julia, SLURM or the
#  launcher notices that the sources changed afterwards: the job simply runs the
#  OLD solver, produces plausible-looking output, and the staleness is only
#  discovered when the results disagree with the code you think you ran. These
#  helpers make that visible before the ranks start.
#
#  Two checks, strongest first:
#    (a) CONTENT STAMP — the build writes <sysimage>.src.sha256, a hash of every
#        src/*.jl. At run time the hash is recomputed and compared, so only a
#        real source change trips it: a `touch`, a re-clone, or a git checkout
#        that restores identical content does NOT cry wolf.
#    (b) MTIME fallback — used when no stamp exists (i.e. an image built before
#        this check was added). Any src/*.jl newer than the image trips it, so it
#        can report false positives after operations that rewrite mtimes without
#        changing content. Rebuild once to get a stamp and the precise check.
#
#  The result is a WARNING, not an error: a stale image still runs, and only you
#  know whether the change mattered. Set BALFEM_STRICT_SYSIMAGE=1 to make it fatal
#  instead — recommended for long production jobs, where finding out afterwards
#  costs the entire run.
# ============================================================================

# Path of the stamp file that accompanies a given sysimage.
balfem_stamp_path() { echo "${1}.src.sha256"; }

# Hash of all src/*.jl (content + relative path, order-independent).
# Relative paths keep the hash stable across differing project roots.
balfem_src_hash() {
    local srcdir="$1"
    ( cd "$srcdir" && find . -name '*.jl' -type f -print0 \
        | LC_ALL=C sort -z | xargs -0 -r sha256sum ) | sha256sum | awk '{print $1}'
}

# Record the current sources against a freshly built image.
# Called by compile/compile_*.sh right after create_sysimage succeeds.
balfem_write_sysimage_stamp() {
    local sysimg="${1:-$BALFEM_SYSIMAGE}"
    local srcdir="${2:-$BALFEM_PROJ/src}"
    local stamp; stamp="$(balfem_stamp_path "$sysimg")"

    [ -f "$sysimg" ] || { echo "[balfem_env] cannot stamp: no image at $sysimg" >&2; return 1; }
    [ -d "$srcdir" ] || { echo "[balfem_env] cannot stamp: no sources at $srcdir" >&2; return 1; }
    command -v sha256sum >/dev/null 2>&1 || {
        echo "[balfem_env] cannot stamp: sha256sum unavailable (mtime check will be used)" >&2
        return 0
    }

    balfem_src_hash "$srcdir" > "$stamp"
    echo "[balfem_env] source stamp written: $stamp ($(cut -c1-12 < "$stamp")…)"
}

# Warn (or abort) if the image no longer matches src/. Safe to call standalone.
balfem_check_sysimage_freshness() {
    local sysimg="${1:-$BALFEM_SYSIMAGE}"
    local srcdir="${2:-$BALFEM_PROJ/src}"
    [ -f "$sysimg" ] || return 0
    [ -d "$srcdir" ] || return 0

    local stamp; stamp="$(balfem_stamp_path "$sysimg")"
    local stale="" reason="" method=""

    if [ -f "$stamp" ] && command -v sha256sum >/dev/null 2>&1; then
        method="content stamp"
        local now built
        now="$(balfem_src_hash "$srcdir")"      || return 0
        built="$(cat "$stamp" 2>/dev/null)"   || return 0
        if [ "$now" != "$built" ]; then
            stale=1
            reason="src/*.jl content changed since the build (baked ${built:0:12}…, now ${now:0:12}…)"
        fi
    else
        method="mtime (no stamp — rebuild once for the exact check)"
        local newer
        newer="$(find "$srcdir" -name '*.jl' -type f -newer "$sysimg" -print 2>/dev/null | head -n 5)" || true
        if [ -n "$newer" ]; then
            stale=1
            reason="these src files are NEWER than the image:
$(echo "$newer" | sed 's|^|[balfem_env]             |')"
        fi
    fi

    if [ -z "$stale" ]; then
        echo "[balfem_env] freshness: image matches src/ ($method)"
        return 0
    fi

    echo "[balfem_env] ---------------------------------------------------------------" >&2
    echo "[balfem_env] WARNING: the system image appears STALE." >&2
    echo "[balfem_env]   image  : $sysimg" >&2
    echo "[balfem_env]   checked: $method" >&2
    echo "[balfem_env]   reason : $reason" >&2
    echo "[balfem_env]   The image bakes a compiled copy of src/*.jl, so this job would" >&2
    echo "[balfem_env]   run the OLD solver code — not your current sources." >&2
    echo "[balfem_env]   Rebuild: cd $BALFEM_PROJ/compile && sbatch compile_${BALFEM_CLUSTER}.sh" >&2
    echo "[balfem_env]   Or run the current sources via JIT: BALFEM_NO_SYSIMAGE=1" >&2
    echo "[balfem_env] ---------------------------------------------------------------" >&2

    if [ "${BALFEM_STRICT_SYSIMAGE:-0}" = "1" ]; then
        echo "[balfem_env] BALFEM_STRICT_SYSIMAGE=1 — refusing to launch a stale image." >&2
        exit 1
    fi
    return 0
}

# ----------------------------------------------------------------------------
#  balfem_run <nranks> <julia-script> [script args...]
#    <julia-script> is relative to $BALFEM_PROJ (absolute paths also accepted).
# ----------------------------------------------------------------------------
balfem_run() {
    local nranks="$1"; shift
    local script="$1"; shift
    [[ "$script" == /* ]] || script="$BALFEM_PROJ/$script"

    [ -f "$script" ] || {
        echo "[balfem_env] ERROR: run script not found: $script" >&2
        exit 1
    }

    # (2) resolve the sysimage
    local -a jflag=()
    local simg_msg
    if [ "${BALFEM_NO_SYSIMAGE:-0}" = "1" ]; then
        simg_msg="<disabled: BALFEM_NO_SYSIMAGE=1 — ranks will JIT-compile>"
    else
        [ -f "$BALFEM_SYSIMAGE" ] || {
            echo "[balfem_env] ERROR: sysimage not found: $BALFEM_SYSIMAGE" >&2
            echo "[balfem_env]   build it once:  cd $BALFEM_PROJ/compile && sbatch compile_${BALFEM_CLUSTER}.sh" >&2
            echo "[balfem_env]   or set BALFEM_NO_SYSIMAGE=1 to run without it (slow, memory-hungry)." >&2
            exit 1
        }
        jflag=(-J"$BALFEM_SYSIMAGE")
        simg_msg="$BALFEM_SYSIMAGE"
    fi

    echo "[balfem_env] project  : $BALFEM_PROJ"
    echo "[balfem_env] cluster  : $BALFEM_CLUSTER"
    echo "[balfem_env] sysimage : $simg_msg"
    echo "[balfem_env] ranks    : $nranks"
    echo "[balfem_env] script   : $script"

    # Is the image still in sync with src/? (warns; fatal under BALFEM_STRICT_SYSIMAGE=1)
    if [ "${BALFEM_NO_SYSIMAGE:-0}" != "1" ]; then
        balfem_check_sysimage_freshness "$BALFEM_SYSIMAGE" "$BALFEM_PROJ/src"
    fi

    # (3) --project must be the project holding LocalPreferences.toml AND the
    #     GridapBALFEM package itself. It is given to mpiexecjl only: mpiexecjl
    #     strips every --project=* from the forwarded arguments and re-exports the
    #     choice to the ranks as JULIA_PROJECT, so a second --project after
    #     `julia` is silently dropped — passing it was misleading, not effective.
    #     -J adds the image built with (1)+(2) active.
    mpiexecjl --project="$BALFEM_PROJ" -n "$nranks" \
        julia ${jflag[@]+"${jflag[@]}"} "$script" "$@"
}
