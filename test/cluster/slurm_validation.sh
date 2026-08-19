#!/bin/bash
# ==============================================================
#  slurm_validation.sh — SLURM template for the cluster VALIDATION runs
#  (mirror of ../../run/run_snellius.sh; edit for your cluster).
#
#  Submit one validation at a time by setting VALIDATION=, or duplicate the
#  final block. px*py MUST equal the total task count (-n).
# ==============================================================
#SBATCH --job-name="BALFEM-validate"
#SBATCH --partition=rome
#SBATCH --time=24:00:00
#SBATCH --nodes=1                     # increase for multi-node; keep ntasks = nodes*ntasks-per-node
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --output=BALFEM-validate.%j.out
#SBATCH --error=BALFEM-validate.%j.err

# --- environment (edit paths/modules for your machine) --------------------------
ROOT=$HOME/GridapBALFEM.jl
source $ROOT/compile/load_modules_snellius.sh    # or your cluster's module setup

NP=$SLURM_NTASKS                                  # total ranks (= -n)
export BALFEM_PX=${BALFEM_PX:-$NP}                    # process grid; PX*PY must equal NP
export BALFEM_PY=${BALFEM_PY:-1}

# --- which validation to run ----------------------------------------------------
VALIDATION=${VALIDATION:-conservation}            # conservation | mms

case "$VALIDATION" in
  conservation)
    # long-run mass/energy conservation on a big closed basin
    export BALFEM_NX=${BALFEM_NX:-2000}  BALFEM_NY=${BALFEM_NY:-200}
    export BALFEM_LX=${BALFEM_LX:-200.0} BALFEM_LY=${BALFEM_LY:-20.0}
    export BALFEM_DT=${BALFEM_DT:-0.02}  BALFEM_TFINAL=${BALFEM_TFINAL:-40.0}
    export BALFEM_A=${BALFEM_A:-0.01}    BALFEM_ADVECTION=${BALFEM_ADVECTION:-1}
    export BALFEM_OUTDIR=${BALFEM_OUTDIR:-$ROOT/output/cluster_conservation}
    SCRIPT=$ROOT/test/cluster/cluster_conservation.jl ;;
  mms)
    # unsteady nonlinear manufactured-solution recovery at scale (all 𝓝 comps)
    export BALFEM_NX=${BALFEM_NX:-400}   BALFEM_NY=${BALFEM_NY:-200}
    export BALFEM_NSTEPS=${BALFEM_NSTEPS:-200} BALFEM_DT=${BALFEM_DT:-0.05}
    export BALFEM_NLPFULL=${BALFEM_NLPFULL:-1}  BALFEM_AMP=${BALFEM_AMP:-1.0}
    export BALFEM_OUTDIR=${BALFEM_OUTDIR:-$ROOT/output/cluster_selfconsistency}
    SCRIPT=$ROOT/test/cluster/cluster_selfconsistency.jl ;;
  *) echo "unknown VALIDATION=$VALIDATION (use conservation|mms)"; exit 1 ;;
esac

echo "### BALFEM validation: $VALIDATION | -n $NP | grid ${BALFEM_PX}x${BALFEM_PY}"
mpiexecjl -n $NP julia --project=$ROOT $SCRIPT
