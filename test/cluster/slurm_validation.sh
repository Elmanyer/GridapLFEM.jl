#!/bin/bash
# ==============================================================
#  slurm_validation.sh — SLURM template for the cluster VALIDATION runs
#  (mirror of ../../run/run_snellius.sh; edit for your cluster).
#
#  Submit one validation at a time by setting VALIDATION=, or duplicate the
#  final block. px*py MUST equal the total task count (-n).
# ==============================================================
#SBATCH --job-name="LFEM-validate"
#SBATCH --partition=rome
#SBATCH --time=24:00:00
#SBATCH --nodes=1                     # increase for multi-node; keep ntasks = nodes*ntasks-per-node
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --output=LFEM-validate.%j.out
#SBATCH --error=LFEM-validate.%j.err

# --- environment (edit paths/modules for your machine) --------------------------
ROOT=$HOME/GridapLFEM.jl
source $ROOT/compile/load_modules_snellius.sh    # or your cluster's module setup

NP=$SLURM_NTASKS                                  # total ranks (= -n)
export LFEM_PX=${LFEM_PX:-$NP}                    # process grid; PX*PY must equal NP
export LFEM_PY=${LFEM_PY:-1}

# --- which validation to run ----------------------------------------------------
VALIDATION=${VALIDATION:-conservation}            # conservation | mms

case "$VALIDATION" in
  conservation)
    # long-run mass/energy conservation on a big closed basin
    export LFEM_NX=${LFEM_NX:-2000}  LFEM_NY=${LFEM_NY:-200}
    export LFEM_LX=${LFEM_LX:-200.0} LFEM_LY=${LFEM_LY:-20.0}
    export LFEM_DT=${LFEM_DT:-0.02}  LFEM_TFINAL=${LFEM_TFINAL:-40.0}
    export LFEM_A=${LFEM_A:-0.01}    LFEM_ADVECTION=${LFEM_ADVECTION:-1}
    export LFEM_OUTDIR=${LFEM_OUTDIR:-$ROOT/output/cluster_conservation}
    SCRIPT=$ROOT/test/cluster/cluster_conservation.jl ;;
  mms)
    # unsteady nonlinear manufactured-solution recovery at scale (all 𝓝 comps)
    export LFEM_NX=${LFEM_NX:-400}   LFEM_NY=${LFEM_NY:-200}
    export LFEM_NSTEPS=${LFEM_NSTEPS:-200} LFEM_DT=${LFEM_DT:-0.05}
    export LFEM_NLPFULL=${LFEM_NLPFULL:-1}  LFEM_AMP=${LFEM_AMP:-1.0}
    export LFEM_OUTDIR=${LFEM_OUTDIR:-$ROOT/output/cluster_mms}
    SCRIPT=$ROOT/test/cluster/cluster_mms.jl ;;
  *) echo "unknown VALIDATION=$VALIDATION (use conservation|mms)"; exit 1 ;;
esac

echo "### LFEM validation: $VALIDATION | -n $NP | grid ${LFEM_PX}x${LFEM_PY}"
mpiexecjl -n $NP julia --project=$ROOT $SCRIPT
