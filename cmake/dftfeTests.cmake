# ---------------------------------------------------------------------
#
# Copyright (c) 2017-2025 The Regents of the University of Michigan and DFT-FE
# authors.
#
# This file is part of the DFT-FE code.
#
# The DFT-FE code is free software; you can use it, redistribute
# it, and/or modify it under the terms of the GNU Lesser General
# Public License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
# The full text of the license can be found in the file LICENSE at
# the top level of the DFT-FE distribution.
#
# ---------------------------------------------------------------------
#
# CTest registration of the DFT-FE regression tests.
#
# A test directory holds, per test <name>,
#   <name>.prm.in              the input; @SOURCE_DIR@ expands to the test
#                              source directory
#   <name>.mpirun=<N>.output   the reference output of a run on N MPI ranks;
#                              one such file per rank count to test
#
# dftfe_pickup_tests() registers one ctest per reference output, named
# <directory>/<name>.mpirun=<N>.<build type>, run in
# <build dir>/<name>.<build type>/mpirun=<N>. tests/scripts/runTest.sh runs
# the executable, normalizes the output with tests/scripts/normalize.pl and
# compares it with numdiff. Every ctest invocation runs the executable again.
#
# Options:
#   TEST_TIME_LIMIT      timeout of a test in seconds
#   TEST_MPI_RANK_LIMIT  when positive, tests on more ranks are not registered
#   TEST_PICKUP_REGEX    when set, only tests whose <directory>/<name> matches
#                        it are registered
#   DFTFE_TEST_MPIEXEC, DFTFE_TEST_MPIEXEC_NUMPROC_FLAG,
#   DFTFE_TEST_MPIEXEC_PREFLAGS, DFTFE_TEST_MPIEXEC_POSTFLAGS
#                        the MPI launcher, e.g. -DDFTFE_TEST_MPIEXEC=srun;
#                        by default the one deal.II was configured with
#   NUMDIFF_DIR          install prefix of numdiff (CMake or environment
#                        variable), if not on CMAKE_PREFIX_PATH or PATH
#

find_program(NUMDIFF_EXECUTABLE
  NAMES numdiff
  HINTS ${NUMDIFF_DIR} ENV NUMDIFF_DIR
  PATH_SUFFIXES bin
  )
if(NOT NUMDIFF_EXECUTABLE)
  message(FATAL_ERROR
    "WITH_TESTING requires numdiff. Add its install directory to "
    "CMAKE_PREFIX_PATH or set NUMDIFF_DIR.")
endif()
# numdiff builds without working tolerances exist; compare two files that
# agree only within a relative tolerance
set(_numdiff_check_dir ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY})
file(WRITE ${_numdiff_check_dir}/numdiff-1.txt "0.99999999998\n2.0\n1.0\n")
file(WRITE ${_numdiff_check_dir}/numdiff-2.txt "1.00000000001\n2.0\n1.0\n")
execute_process(
  COMMAND ${NUMDIFF_EXECUTABLE} -r 1.0e-8 --
          ${_numdiff_check_dir}/numdiff-1.txt ${_numdiff_check_dir}/numdiff-2.txt
  TIMEOUT 4
  OUTPUT_VARIABLE _numdiff_check_output
  ERROR_VARIABLE _numdiff_check_output
  RESULT_VARIABLE _numdiff_check_status
  )
if(NOT "${_numdiff_check_status}" STREQUAL "0")
  message(FATAL_ERROR
    "${NUMDIFF_EXECUTABLE} does not apply a relative tolerance:\n"
    "${_numdiff_check_output}")
endif()

find_program(DFTFE_TEST_BASH NAMES bash REQUIRED)
find_program(DFTFE_TEST_PERL NAMES perl REQUIRED)

# The MPI launcher: by default the one deal.II was configured with, else the
# one CMake's FindMPI finds; any of the four can be set on the command line.
if(DEAL_II_MPIEXEC AND NOT "${DEAL_II_MPIEXEC}" STREQUAL "MPIEXEC_EXECUTABLE-NOTFOUND")
  set(_mpiexec ${DEAL_II_MPIEXEC})
  set(_numproc_flag ${DEAL_II_MPIEXEC_NUMPROC_FLAG})
  set(_preflags ${DEAL_II_MPIEXEC_PREFLAGS})
  set(_postflags ${DEAL_II_MPIEXEC_POSTFLAGS})
else()
  find_package(MPI REQUIRED)
  set(_mpiexec ${MPIEXEC_EXECUTABLE})
  set(_numproc_flag ${MPIEXEC_NUMPROC_FLAG})
  set(_preflags ${MPIEXEC_PREFLAGS})
  set(_postflags ${MPIEXEC_POSTFLAGS})
endif()
set(DFTFE_TEST_MPIEXEC "${_mpiexec}" CACHE STRING
  "MPI launcher of the regression tests")
set(DFTFE_TEST_MPIEXEC_NUMPROC_FLAG "${_numproc_flag}" CACHE STRING
  "Flag of DFTFE_TEST_MPIEXEC that precedes the number of ranks")
set(DFTFE_TEST_MPIEXEC_PREFLAGS "${_preflags}" CACHE STRING
  "Flags of DFTFE_TEST_MPIEXEC before the executable")
set(DFTFE_TEST_MPIEXEC_POSTFLAGS "${_postflags}" CACHE STRING
  "Flags of DFTFE_TEST_MPIEXEC after the executable")
if("${DFTFE_TEST_MPIEXEC}" STREQUAL "" OR
   "${DFTFE_TEST_MPIEXEC}" STREQUAL "MPIEXEC_EXECUTABLE-NOTFOUND")
  message(FATAL_ERROR
    "WITH_TESTING requires an MPI launcher; set DFTFE_TEST_MPIEXEC.")
endif()
# flags given as one string, e.g. "--ppn 12 --cpu-bind depth", become
# separate arguments
foreach(_flags NUMPROC_FLAG PREFLAGS POSTFLAGS)
  string(REPLACE ";" " " _string "${DFTFE_TEST_MPIEXEC_${_flags}}")
  separate_arguments(_DFTFE_TEST_MPIEXEC_${_flags} UNIX_COMMAND "${_string}")
endforeach()
message(STATUS "Regression tests run with: ${DFTFE_TEST_MPIEXEC} "
  "${DFTFE_TEST_MPIEXEC_NUMPROC_FLAG} <N> ${DFTFE_TEST_MPIEXEC_PREFLAGS} "
  "dftfe ${DFTFE_TEST_MPIEXEC_POSTFLAGS} <input>")

if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  set(DFTFE_TEST_BUILD_SUFFIX "debug")
else()
  set(DFTFE_TEST_BUILD_SUFFIX "release")
endif()

set(DFTFE_TEST_SCRIPT_DIR ${CMAKE_SOURCE_DIR}/tests/scripts)

# Builds the executable before any test runs, so that a plain "ctest" after
# a source change tests the changed code.
add_test(NAME test_dependency/${TARGET}.executable
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target ${TARGET}
  )
set_tests_properties(test_dependency/${TARGET}.executable PROPERTIES
  FIXTURES_SETUP ${TARGET}.executable
  LABELS test_dependency
  )

function(dftfe_pickup_tests)
  get_filename_component(_category ${CMAKE_CURRENT_SOURCE_DIR} NAME)
  file(GLOB _references "${CMAKE_CURRENT_SOURCE_DIR}/*.output")
  set(_n_tests 0)
  foreach(_reference ${_references})
    get_filename_component(_file ${_reference} NAME)
    if(NOT _file MATCHES "^([^.]+)\\.mpirun=([0-9]+)\\.output$")
      message(FATAL_ERROR
        "${_reference}: reference outputs are named <name>.mpirun=<N>.output")
    endif()
    set(_name ${CMAKE_MATCH_1})
    set(_n_ranks ${CMAKE_MATCH_2})

    if(NOT "${TEST_PICKUP_REGEX}" STREQUAL "" AND
       NOT "${_category}/${_name}" MATCHES "${TEST_PICKUP_REGEX}")
      continue()
    endif()
    if(TEST_MPI_RANK_LIMIT GREATER 0 AND _n_ranks GREATER TEST_MPI_RANK_LIMIT)
      continue()
    endif()

    set(_input_template ${CMAKE_CURRENT_SOURCE_DIR}/${_name}.prm.in)
    if(NOT EXISTS ${_input_template})
      message(FATAL_ERROR
        "${_reference}: no input ${_input_template}")
    endif()
    set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
    set(_input ${CMAKE_CURRENT_BINARY_DIR}/${_name}.prm)
    configure_file(${_input_template} ${_input} @ONLY)

    set(_test ${_category}/${_name}.mpirun=${_n_ranks}.${DFTFE_TEST_BUILD_SUFFIX})
    set(_directory
      ${CMAKE_CURRENT_BINARY_DIR}/${_name}.${DFTFE_TEST_BUILD_SUFFIX}/mpirun=${_n_ranks})
    file(MAKE_DIRECTORY ${_directory})

    add_test(NAME ${_test}
      COMMAND ${DFTFE_TEST_BASH} ${DFTFE_TEST_SCRIPT_DIR}/runTest.sh
              ${_test} ${DFTFE_TEST_PERL} ${NUMDIFF_EXECUTABLE} ${_reference}
              --
              ${DFTFE_TEST_MPIEXEC} ${_DFTFE_TEST_MPIEXEC_NUMPROC_FLAG} ${_n_ranks}
              ${_DFTFE_TEST_MPIEXEC_PREFLAGS} $<TARGET_FILE:${TARGET}>
              ${_DFTFE_TEST_MPIEXEC_POSTFLAGS} ${_input}
      WORKING_DIRECTORY ${_directory}
      )
    set_tests_properties(${_test} PROPERTIES
      PROCESSORS ${_n_ranks}
      TIMEOUT ${TEST_TIME_LIMIT}
      LABELS ${_category}
      FIXTURES_REQUIRED ${TARGET}.executable
      )
    math(EXPR _n_tests "${_n_tests} + 1")
  endforeach()
  message(STATUS "Test category ${_category}: ${_n_tests} tests")
endfunction()
