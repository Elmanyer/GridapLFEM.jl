#!/bin/bash
# Suite C — 2-D, SEQUENTIAL. Q4 fine level dominates the cost; keep NX0 modest.
set -uo pipefail; cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source run/local/lfem_local.sh; mkdir -p output/local/logs
export LFEM_CONV_PU="${PU:-2,3,4}" LFEM_CONV_DOMAIN=d2 LFEM_CONV_MODE="${MODE:-static}"
export LFEM_CONV_LEVELS="${LEVELS:-3}" LFEM_CONV_NX0="${NX0:-6}" LFEM_CONV_NY0="${NY0:-4}"
export LFEM_CONV_DIST=0 LFEM_CONV_DT="${DT:-1e-5}" LFEM_CONV_NSTEPS="${NSTEPS:-100}"
export LFEM_CONV_OUT=output/local/mms_conv
lfem_local_run examples/local_mms/run_mms_matrix.jl 2>&1 | tee output/local/logs/conv_2d_seq.log
