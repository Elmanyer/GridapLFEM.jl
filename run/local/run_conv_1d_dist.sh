#!/bin/bash
# Suite B — quasi-1D, DISTRIBUTED (4 ranks). GMRES+Jacobi at ls_rtol=1e-13 so the
# linear solve cannot cap the spatial error; verify with run_conv_tolerance_gate.sh.
set -uo pipefail; cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source run/local/balfem_local.sh; mkdir -p output/local/logs
export BALFEM_CONV_PU="${PU:-2,3,4}" BALFEM_CONV_DOMAIN=d1 BALFEM_CONV_MODE="${MODE:-both}"
export BALFEM_CONV_LEVELS="${LEVELS:-4}" BALFEM_CONV_NX0="${NX0:-16}" BALFEM_CONV_DIST=1
export BALFEM_CONV_PX="${PX:-4}" BALFEM_CONV_PY="${PY:-1}"
export BALFEM_CONV_LSRTOL="${LSRTOL:-1e-13}" BALFEM_CONV_NLTOL="${NLTOL:-1e-13}"
export BALFEM_CONV_DT="${DT:-1e-5}" BALFEM_CONV_NSTEPS="${NSTEPS:-100}"
export BALFEM_CONV_OUT=output/local/mms_conv
balfem_local_mpi "${RANKS:-4}" examples/local_mms/run_mms_matrix.jl 2>&1 | tee output/local/logs/conv_1d_dist.log
