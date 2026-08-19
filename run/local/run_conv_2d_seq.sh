#!/bin/bash
# Suite C — 2-D, SEQUENTIAL. Q4 fine level dominates the cost; keep NX0 modest.
set -uo pipefail; cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source run/local/balfem_local.sh; mkdir -p output/local/logs
export BALFEM_CONV_PU="${PU:-2,3,4}" BALFEM_CONV_DOMAIN=d2 BALFEM_CONV_MODE="${MODE:-static}"
export BALFEM_CONV_LEVELS="${LEVELS:-3}" BALFEM_CONV_NX0="${NX0:-6}" BALFEM_CONV_NY0="${NY0:-4}"
export BALFEM_CONV_DIST=0 BALFEM_CONV_DT="${DT:-1e-5}" BALFEM_CONV_NSTEPS="${NSTEPS:-100}"
export BALFEM_CONV_OUT=output/local/mms_conv
balfem_local_run examples/local_mms/run_mms_matrix.jl 2>&1 | tee output/local/logs/conv_2d_seq.log
