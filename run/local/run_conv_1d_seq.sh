#!/bin/bash
# Suite A — quasi-1D, SEQUENTIAL (direct LU ⇒ no linear tolerance can cap the error).
# Runs FIRST: if p+1 does not hold here, the Q3/Q2 result was a coincidence and
# nothing else in the campaign is worth running. Plan: MMS_CONVERGENCE_CAMPAIGN.md
set -uo pipefail; cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source run/local/balfem_local.sh; mkdir -p output/local/logs
export BALFEM_CONV_PU="${PU:-2,3,4}" BALFEM_CONV_DOMAIN=d1 BALFEM_CONV_MODE="${MODE:-both}"
export BALFEM_CONV_LEVELS="${LEVELS:-4}" BALFEM_CONV_NX0="${NX0:-16}" BALFEM_CONV_DIST=0
export BALFEM_CONV_DT="${DT:-1e-5}" BALFEM_CONV_NSTEPS="${NSTEPS:-100}"
export BALFEM_CONV_OUT=output/local/mms_conv
balfem_local_run examples/local_mms/run_mms_matrix.jl 2>&1 | tee output/local/logs/conv_1d_seq.log
