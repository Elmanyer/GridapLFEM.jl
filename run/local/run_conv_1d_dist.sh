#!/bin/bash
# Suite B — quasi-1D, DISTRIBUTED (4 ranks). GMRES+Jacobi at ls_rtol=1e-13 so the
# linear solve cannot cap the spatial error; verify with run_conv_tolerance_gate.sh.
set -uo pipefail; cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source run/local/lfem_local.sh; mkdir -p output/local/logs
export LFEM_CONV_PU="${PU:-2,3,4}" LFEM_CONV_DOMAIN=d1 LFEM_CONV_MODE="${MODE:-both}"
export LFEM_CONV_LEVELS="${LEVELS:-4}" LFEM_CONV_NX0="${NX0:-16}" LFEM_CONV_DIST=1
export LFEM_CONV_PX="${PX:-4}" LFEM_CONV_PY="${PY:-1}"
export LFEM_CONV_LSRTOL="${LSRTOL:-1e-13}" LFEM_CONV_NLTOL="${NLTOL:-1e-13}"
export LFEM_CONV_DT="${DT:-1e-5}" LFEM_CONV_NSTEPS="${NSTEPS:-100}"
export LFEM_CONV_OUT=output/local/mms_conv
lfem_local_mpi "${RANKS:-4}" examples/local_mms/run_mms_matrix.jl 2>&1 | tee output/local/logs/conv_1d_dist.log
