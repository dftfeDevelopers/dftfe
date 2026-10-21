#!/bin/bash
## ---------------------------------------------------------------------
##
## Copyright (c) 2017-2025 The Regents of the University of Michigan and DFT-FE
## authors.
##
## This file is part of the DFT-FE code.
##
## The DFT-FE code is free software; you can use it, redistribute
## it, and/or modify it under the terms of the GNU Lesser General
## Public License as published by the Free Software Foundation; either
## version 2.1 of the License, or (at your option) any later version.
## The full text of the license can be found in the file LICENSE at
## the top level of the DFT-FE distribution.
##
## ---------------------------------------------------------------------
#
# Runs one DFT-FE regression test in the current directory and compares its
# output with the reference (see cmake/dftfeTests.cmake):
#
#   runTest.sh <test name> <perl> <numdiff> <reference> -- <command ...>
#
# The command's stdout and stderr go to "output", which normalize.pl
# normalizes before numdiff compares it with the reference under absolute
# tolerance 1e-6 and relative tolerance 1e-8. A failed run leaves
# "failing_output", a failed comparison "failing_diff".
#
# Every thread pool runs one thread per MPI rank, so that a test on N ranks
# occupies N cores; DFTFE_TEST_NUM_THREADS overrides the count.
#

set -u
TEST_NAME="$1"
PERL="$2"
NUMDIFF="$3"
REFERENCE="$4"
shift 4
if [ "$1" != "--" ]; then
  echo "${TEST_NAME}: usage: runTest.sh <test> <perl> <numdiff> <reference> -- <command>"
  exit 2
fi
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NUM_THREADS="${DFTFE_TEST_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${NUM_THREADS}"
export DFTFE_NUM_THREADS="${NUM_THREADS}"
export DEAL_II_NUM_THREADS="${NUM_THREADS}"
export BLIS_NUM_THREADS="${NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${NUM_THREADS}"
export MKL_NUM_THREADS="${NUM_THREADS}"

# Allow more ranks than cores, and leave the ranks unbound, so that tests
# started concurrently by "ctest -j" do not all pin to the same cores.
export OMPI_MCA_rmaps_base_oversubscribe=1
export PRTE_MCA_rmaps_default_mapping_policy=:oversubscribe
export OMPI_MCA_hwloc_base_binding_policy=none
export PRTE_MCA_hwloc_default_binding_policy=none

rm -f output failing_output diff failing_diff

"$@" > output 2>&1
RETURN_VALUE=$?
if [ ${RETURN_VALUE} -ne 0 ]; then
  mv output failing_output
  echo "${TEST_NAME}: RUN failed. ------ Return code ${RETURN_VALUE}"
  echo "${TEST_NAME}: RUN failed. ------ Output: $(pwd)/failing_output"
  echo "${TEST_NAME}: RUN failed. ------ Last 40 lines:"
  tail -n 40 failing_output
  exit 1
fi

"${PERL}" -pi "${SCRIPT_DIR}/normalize.pl" output

"${NUMDIFF}" -a 1e-6 -r 1e-8 -s ' \t\r\n=,:;<>[](){}^' \
  "${REFERENCE}" output > diff
if [ $? -eq 0 ]; then
  echo "${TEST_NAME}: PASSED"
  exit 0
fi

mv diff failing_diff
echo "${TEST_NAME}: DIFF failed. ------ Reference: ${REFERENCE}"
echo "${TEST_NAME}: DIFF failed. ------ Result:    $(pwd)/output"
echo "${TEST_NAME}: DIFF failed. ------ Diff:      $(pwd)/failing_diff"
echo "${TEST_NAME}: DIFF failed. ------ First 20 lines of numdiff output:"
head -n 20 failing_diff
exit 1
