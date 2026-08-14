#!/bin/bash
# Suite D — 2-D, DISTRIBUTED (12 ranks). The most expensive suite; run LAST.
set -uo pipefail; cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source run/local/lfem_local.sh; mkdir -p output/local/logs
export LFEM_CONV_PU="${PU:-2,3}" LFEM_CONV_DOMAIN=d2 LFEM_CONV_MODE="${MODE:-static}"
export LFEM_CONV_LEVELS="${LEVELS:-3}" LFEM_CONV_NX0="${NX0:-6}" LFEM_CONV_NY0="${NY0:-4}"
export LFEM_CONV_DIST=1 LFEM_CONV_PX="${PX:-4}" LFEM_CONV_PY="${PY:-3}"
export LFEM_CONV_LSRTOL="${LSRTOL:-1e-13}" LFEM_CONV_NLTOL="${NLTOL:-1e-13}"
export LFEM_CONV_DT="${DT:-1e-5}" LFEM_CONV_NSTEPS="${NSTEPS:-100}"
export LFEM_CONV_OUT=output/local/mms_conv
lfem_local_mpi "${RANKS:-12}" examples/local_mms/run_mms_matrix.jl 2>&1 | tee output/local/logs/conv_2d_dist.log
