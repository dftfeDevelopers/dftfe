##
#  CMakeList file for DFT-FE
##

# Set the name of the project and target:
SET(TARGET "main")
SET(TARGETLIB "dftfe")

INCLUDE_DIRECTORIES(
 ${CMAKE_SOURCE_DIR}/include
)

# Declare all source files the target consists of:
SET(TARGET_SRC
  ./src/dft/dft.cc
  ./src/poisson/poisson.cc
  ./src/eigen/eigen.cc
  ./src/force/force.cc
  ./src/force/eshelbyTensor/eshelbyTensor.cc
  ./src/force/eshelbyTensor/eshelbyTensorSpinPolarized.cc
  ./src/triangulation/meshMovement/meshMovement.cc
  ./src/triangulation/meshMovement/meshMovementGaussian.cc
  ./src/triangulation/meshMovement/meshMovementAffineTransform.cc
  ./src/triangulation/triangulationManager/triangulationManager.cc
  ./src/symmetry/initGroupSymmetry.cc
  ./src/geoOpt/geoOptIon.cc
  ./src/geoOpt/geoOptCell.cc
  ./src/solvers/solverFunction.cc
  ./src/solvers/nonLinearSolver.cc
  ./src/solvers/cgSolvers/cgPRPNonLinearSolver.cc
  ./utils/fileReaders.cc
  ./utils/dftParameters.cc
  ./utils/constraintMatrixInfo.cc
  ./utils/dftUtils.cc
  ./utils/vectorTools/interpolateFieldsFromPreviousMesh.cc
  ./utils/vectorTools/vectorTools.cc
  # You can specify additional files here!
  )

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

FIND_PACKAGE(deal.II 8.4.0 QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

#
# Are all dependencies fullfilled?
#
IF(NOT DEAL_II_WITH_LAPACK OR
   NOT DEAL_II_WITH_PETSC OR
   NOT DEAL_II_WITH_SLEPC)
  MESSAGE(FATAL_ERROR "
Error! The deal.II library found at ${DEAL_II_PATH} was not configured with
    DEAL_II_WITH_LAPACK = ON
    DEAL_II_WITH_PETSC = ON
    DEAL_II_WITH_SLEPC = ON
which is required."
    )
ENDIF()


DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT(${TARGET})

#
# RPATHS: https://cmake.org/Wiki/CMake_RPATH_handling#Mac_OS_X_and_the_RPATH
#
IF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  ## cmake  --help-policy CMP0042
  set(CMAKE_MACOSX_RPATH ON)
ENDIF()

# use, i.e. don't skip the full RPATH for the build tree
SET(CMAKE_SKIP_BUILD_RPATH  FALSE)

# when building, don't use the install RPATH already
# (but later on when installing)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# the RPATH to be used when installing, but only if it's not a system directory
LIST(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
IF("${isSystemDir}" STREQUAL "-1")
   SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
ENDIF("${isSystemDir}" STREQUAL "-1")

# make sure RPATHs are used on macOS
SET(CMAKE_INSTALL_NAME_DIR "${CMAKE_INSTALL_PREFIX}/lib")

# FIXME: fix warnings, remove flags
SET(WARNINGSLIST  "-w -Wno-sign-compare -Wno-ignored-qualifiers -Wno-deprecated-declarations -Wno-vla-extension -Wno-reorder -Wno-sometimes-uninitialized -Wno-undefined-var-template -Wno-ignored-qualifiers")
SET(DEAL_II_CXX_FLAGS  "${DEAL_II_CXX_FLAGS} ${WARNINGSLIST}" )

#
#Set periodic BC flag
#
IF (DEAL_II_PETSC_WITH_COMPLEX)
   ADD_DEFINITIONS(-DENABLE_PERIODIC_BC)
   MESSAGE(STATUS "Periodic")
ELSE()
   ADD_DEFINITIONS(-DDISABLE_PERIODIC_BC)
   MESSAGE(STATUS "Non-periodic")
ENDIF()

ADD_LIBRARY(${TARGETLIB} SHARED
    ${TARGET_SRC}
  )

ADD_EXECUTABLE(${TARGET}
    src/main.cc
  )

DEAL_II_SETUP_TARGET(${TARGETLIB})
DEAL_II_SETUP_TARGET(${TARGET})

TARGET_LINK_LIBRARIES(${TARGET}
    ${TARGETLIB}
  )

#
# REQUIRED DEPENDENCIES:
#

#
# Alglib. Set with -DALGLIB_DIR=/path/to/prefix
#
IF (DEFINED ALGLIB_DIR)
  FIND_LIBRARY(ALGLIB_LIBRARY
    NAMES Alglib alglib
    HINTS ${ALGLIB_DIR}/lib ${ALGLIB_DIR}
    NO_DEFAULT_PATH
  )
  IF(ALGLIB_LIBRARY STREQUAL "ALGLIB_LIBRARY-NOTFOUND")
      MESSAGE(FATAL_ERROR "-- Alglib was not found in ${ALGLIB_DIR}")
  ENDIF()
  MESSAGE("-- Use Alglib from ${ALGLIB_DIR}")
  TARGET_LINK_LIBRARIES(${TARGETLIB}
	  "${ALGLIB_LIBRARY}"
  )
  INCLUDE_DIRECTORIES ("${ALGLIB_DIR}")
  INCLUDE_DIRECTORIES ("${ALGLIB_DIR}/include")
ELSE()
  MESSAGE(FATAL_ERROR "-- Provide path to AlgLIB: -DALGLIB_DIR=/path/to/prefix")
ENDIF (DEFINED ALGLIB_DIR)

#
# LIBXC. Set with -DLIBXC_DIR=/path/to/prefix
#
IF (DEFINED LIBXC_DIR)
  FIND_LIBRARY(LIBXC_LIBRARY
    NAMES xc
    HINTS ${LIBXC_DIR}/lib
    NO_DEFAULT_PATH
  )
  IF(LIBXC_LIBRARY STREQUAL "LIBXC_LIBRARY-NOTFOUND")
      MESSAGE(FATAL_ERROR "-- LibXC was not found in ${LIBXC_DIR}")
  ENDIF()
  MESSAGE("-- Use LibXC from ${LIBXC_DIR}")
  TARGET_LINK_LIBRARIES(${TARGETLIB}
	    "${LIBXC_LIBRARY}"
  )
  INCLUDE_DIRECTORIES ("${LIBXC_DIR}/include")
ELSE()
  MESSAGE(FATAL_ERROR "-- Provide path to prefix of LibXC: -DLIBXC_DIR=/path/to/prefix")
ENDIF (DEFINED LIBXC_DIR)


#
# SPG. Set with -DSPGLIB_DIR=/path/to/prefix
#
IF (DEFINED SPGLIB_DIR)
  FIND_LIBRARY(SPGLIB_LIBRARY
    NAMES symspg
    HINTS ${SPGLIB_DIR}/lib
    NO_DEFAULT_PATH
  )
  IF(SPGLIB_LIBRARY STREQUAL "SPGLIB_LIBRARY-NOTFOUND")
      MESSAGE(FATAL_ERROR "-- SPGLib was not found in ${SPGLIB_DIR}")
  ENDIF()
  MESSAGE("-- Use SPGLib from ${SPGLIB_DIR}")
  TARGET_LINK_LIBRARIES(${TARGETLIB}
	    "${SPGLIB_LIBRARY}"
  )
  INCLUDE_DIRECTORIES ("${SPGLIB_DIR}/include")
ELSE()
  MESSAGE(FATAL_ERROR "-- Provide path to prefix of SPGLIB: -DSPGLIB_DIR=/path/to/prefix")
ENDIF (DEFINED SPGLIB_DIR)

ADD_DEFINITIONS(-DDFT_PATH="${CMAKE_CURRENT_SOURCE_DIR}")

# Enable testing and descent into tests/ subdirectory:
# Make sure numdiff is in path before configuring, otherwise due to roundoff
# errors many tests may fail.
# Test will run in Debug or Release mode based on the way library is compiled.
ENABLE_TESTING()
INCLUDE(CTest)
# default 600sec is not enough for some tests, increase:
SET (TEST_TIME_LIMIT 3600)
#
IF (DEAL_II_PETSC_WITH_COMPLEX)
   ADD_SUBDIRECTORY(tests/dft/pseudopotential/periodic)
ELSE()
   ADD_SUBDIRECTORY(tests/dft/pseudopotential/nonPeriodic)
ENDIF()
