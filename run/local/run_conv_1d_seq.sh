#!/bin/bash
# Suite A — quasi-1D, SEQUENTIAL (direct LU ⇒ no linear tolerance can cap the error).
# Runs FIRST: if p+1 does not hold here, the Q3/Q2 result was a coincidence and
# nothing else in the campaign is worth running. Plan: MMS_CONVERGENCE_CAMPAIGN.md
set -uo pipefail; cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source run/local/lfem_local.sh; mkdir -p output/local/logs
export LFEM_CONV_PU="${PU:-2,3,4}" LFEM_CONV_DOMAIN=d1 LFEM_CONV_MODE="${MODE:-both}"
export LFEM_CONV_LEVELS="${LEVELS:-4}" LFEM_CONV_NX0="${NX0:-16}" LFEM_CONV_DIST=0
export LFEM_CONV_DT="${DT:-1e-5}" LFEM_CONV_NSTEPS="${NSTEPS:-100}"
export LFEM_CONV_OUT=output/local/mms_conv
lfem_local_run examples/local_mms/run_mms_matrix.jl 2>&1 | tee output/local/logs/conv_1d_seq.log
