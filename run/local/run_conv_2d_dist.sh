#!/bin/bash
# Suite D — 2-D, DISTRIBUTED (12 ranks). The most expensive suite; run LAST.
set -uo pipefail; cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source run/local/balfem_local.sh; mkdir -p output/local/logs
export BALFEM_CONV_PU="${PU:-2,3}" BALFEM_CONV_DOMAIN=d2 BALFEM_CONV_MODE="${MODE:-static}"
export BALFEM_CONV_LEVELS="${LEVELS:-3}" BALFEM_CONV_NX0="${NX0:-6}" BALFEM_CONV_NY0="${NY0:-4}"
export BALFEM_CONV_DIST=1 BALFEM_CONV_PX="${PX:-4}" BALFEM_CONV_PY="${PY:-3}"
export BALFEM_CONV_LSRTOL="${LSRTOL:-1e-13}" BALFEM_CONV_NLTOL="${NLTOL:-1e-13}"
export BALFEM_CONV_DT="${DT:-1e-5}" BALFEM_CONV_NSTEPS="${NSTEPS:-100}"
export BALFEM_CONV_OUT=output/local/mms_conv
balfem_local_mpi "${RANKS:-12}" examples/local_mms/run_mms_matrix.jl 2>&1 | tee output/local/logs/conv_2d_dist.log
