##
#  CMakeList file for DFT-FE
##

# Set the name of the project and target:
SET(TARGET "dftfe")
SET(TARGETLIB "dftfem")
IF (WITH_GPU)
   SET(DFTFECUDALIB "dftfecuda")
ENDIF()

INCLUDE_DIRECTORIES(
 ${CMAKE_SOURCE_DIR}/include
)

# Declare all source files the target consists of:
SET(TARGET_SRC
  ./src/dft/dft.cc
  ./src/dft/vselfBinsManager.cc
  ./src/dft/energyCalculator.cc
  ./src/dft/densityCalculator.cc
  ./src/poisson/poissonSolverProblem.cc
  ./src/helmholtz/kerkerSolverProblem.cc
  ./src/dftOperator/kohnShamDFTOperator.cc
  ./src/dftOperator/operator.cc
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
  ./src/solvers/nonlinearSolverProblem.cc
  ./src/solvers/nonLinearSolver.cc
  ./src/solvers/linearSolver.cc
  ./src/solvers/cgSolvers/cgPRPNonLinearSolver.cc
  ./src/solvers/cgSolvers/cg_descent.cc
  ./src/solvers/eigenSolvers/chebyshevOrthogonalizedSubspaceIterationSolver.cc
  ./src/solvers/eigenSolver.cc
  ./src/linAlg/linearAlgebraOperationsOpt.cc
  ./src/linAlg/linearAlgebraOperationsInternal.cc
  ./src/linAlg/elpaScalaManager.cc
  ./src/solvers/dealiiLinearSolver.cc
  ./src/solvers/dealiiLinearSolverProblem.cc
  ./src/md/molecularDynamics.cc
  ./utils/fileReaders.cc
  ./utils/dftParameters.cc
  ./utils/constraintMatrixInfo.cc
  ./utils/dftUtils.cc
  ./utils/vectorTools/interpolateFieldsFromPreviousMesh.cc
  ./utils/vectorTools/vectorUtilities.cc
  ./utils/pseudoConverter.cc
  ./pseudoConverters/upfToxml.cc
  ./utils/PeriodicTable.cc
  ./utils/xmlTodftfeParser.cc
  # You can specify additional files here!
  )

IF (WITH_GPU)
  CMAKE_MINIMUM_REQUIRED(VERSION 3.9)
ELSE()
  CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12)
ENDIF()

FIND_PACKAGE(deal.II 9.1.0 REQUIRED
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
# DEAL_II_WITH_64BIT_INDICES is a requirement because of
# the choice of the parallel storage data-structure for
# wavefunctions in DFT-FE. The maximum global index of this
# parallel storage data-structure is equal to
# total number of dofs times total number of wavefunctions,
# which can cross the 32 bit integer limits even for a moderate
# problem size.
IF(NOT DEAL_II_WITH_LAPACK OR
   NOT DEAL_II_WITH_P4EST OR
   NOT DEAL_II_WITH_64BIT_INDICES OR
   NOT DEAL_II_WITH_MPI OR
   NOT DEAL_II_WITH_SCALAPACK)
  MESSAGE(FATAL_ERROR "
Error! The deal.II library found at ${DEAL_II_PATH} was not configured with
    DEAL_II_WITH_LAPACK = ON
    DEAL_II_WITH_P4EST = ON
    DEAL_II_WITH_64BIT_INDICES = ON
    DEAL_II_WITH_MPI = ON
    DEAL_II_WITH_SCALAPACK = ON
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
#Set use complex/use real flag
#
IF (WITH_COMPLEX)
   ADD_DEFINITIONS(-DUSE_COMPLEX)
   MESSAGE(STATUS "Using complex")
ELSE()
   ADD_DEFINITIONS(-DUSE_REAL)
   MESSAGE(STATUS "Using real")
ENDIF()

#
#Set use petsc and slepc flag
#
IF (DEAL_II_WITH_PETSC AND DEAL_II_WITH_SLEPC)
   ADD_DEFINITIONS(-DUSE_PETSC)
   MESSAGE(STATUS "Using petsc and slepc")
ENDIF()

#
#Set use intel mkl flag
#
IF (WITH_INTEL_MKL)
   ADD_DEFINITIONS(-DWITH_MKL)
   MESSAGE(STATUS "With intel mkl")
ENDIF()

#
#Set GPU compilation flag
#
IF (WITH_GPU)
  ADD_DEFINITIONS(-DDFTFE_WITH_GPU)
  MESSAGE(STATUS "DFTFE with GPU")

  IF(NOT DEAL_II_WITH_CUDA OR
     NOT DEAL_II_MPI_WITH_CUDA_SUPPORT)
     MESSAGE(FATAL_ERROR "
     Error! The deal.II library found at ${DEAL_II_PATH} was not configured with
     DEAL_II_WITH_CUDA=ON
     DEAL_II_MPI_WITH_CUDA_SUPPORT=ON
      which is required for WITH_GPU=ON"
     )
  ENDIF()

  SET(CUDA_SRC
   ./src/setupGPU.cu
   ./src/dft/densityCalculatorCUDA.cu
   ./src/dftOperator/operatorCUDA.cu
   ./src/dftOperator/kohnShamDFTOperatorCUDA.cu
   ./src/dftOperator/kohnShamDFTOperatorCUDAKernels.cu
   ./src/dft/solveVselfInBinsCUDA.cu
   ./src/solvers/eigenSolvers/chebyshevOrthogonalizedSubspaceIterationSolverCUDA.cu
   ./src/linAlg/linearAlgebraOperationsInternalCUDA.cu
   ./src/linAlg/linearAlgebraOperationsCUDA.cu
   ./src/linAlg/pseudoGSCUDA.cu
   ./src/linAlg/rayleighRitzCUDA.cu
   ./src/force/forceCUDA.cu
   ./utils/constraintMatrixInfoCUDA.cu
   ./utils/vectorTools/vectorUtilitiesCUDA.cu
   )

  find_package(CUDA REQUIRED)
  set(CUDA_PROPAGATE_HOST_FLAGS ON)
  list( APPEND CUDA_NVCC_FLAGS -arch=sm_70; -std=c++11; -expt-relaxed-constexpr)
  CUDA_ADD_LIBRARY(${DFTFECUDALIB} SHARED ${CUDA_SRC})
ENDIF()


ADD_LIBRARY(${TARGETLIB} SHARED
  ${TARGET_SRC}
)

ADD_EXECUTABLE(${TARGET}   src/main.cc)

DEAL_II_SETUP_TARGET(${TARGETLIB})
DEAL_II_SETUP_TARGET(${TARGET})

IF (WITH_GPU)
 DEAL_II_SETUP_TARGET(${DFTFECUDALIB})
 TARGET_LINK_LIBRARIES(${TARGET}
    ${TARGETLIB} ${DFTFECUDALIB}
  )
ELSE()
 TARGET_LINK_LIBRARIES(${TARGET}
    ${TARGETLIB}
  )
ENDIF()

#
# REQUIRED DEPENDENCIES:
#

IF (WITH_GPU)
  include_directories(${CUDA_INCLUDE_DIRS})
  SET(ALL_CUDA_LIBS ${CUDA_LIBRARIES} ${CUDA_cusparse_LIBRARY} ${CUDA_cublas_LIBRARY})
  message(STATUS "CUDA_LIBRARIES: ${CUDA_INCLUDE_DIRS} ${ALL_CUDA_LIBS}")
  TARGET_LINK_LIBRARIES(${TARGET}
            "${ALL_CUDA_LIBS}")
ENDIF()

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
  TARGET_LINK_LIBRARIES(${TARGET}
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
  TARGET_LINK_LIBRARIES(${TARGET}
	    "${LIBXC_LIBRARY}"
  )
  INCLUDE_DIRECTORIES ("${LIBXC_DIR}/include")
ELSE()
  MESSAGE(FATAL_ERROR "-- Provide path to prefix of LibXC: -DLIBXC_DIR=/path/to/prefix")
ENDIF (DEFINED LIBXC_DIR)

#
# XML. Set with -DXML_LIB_DIR and -DXML_INCLUDE_DIR
#
IF (DEFINED XML_INCLUDE_DIR)
  FIND_LIBRARY(XML_LIBRARY
    NAMES xml2
    HINTS ${XML_LIB_DIR}
    NO_DEFAULT_PATH
  )
  IF(XML_LIBRARY STREQUAL "XML_LIBRARY-NOTFOUND")
      MESSAGE(FATAL_ERROR "-- XML was not found in ${XML_LIB_DIR}")
  ENDIF()
  MESSAGE("-- Use Xml from ${XML_LIB_DIR}")
  TARGET_LINK_LIBRARIES(${TARGET}
	    "${XML_LIBRARY}"
  )
  INCLUDE_DIRECTORIES ("${XML_INCLUDE_DIR}")
ELSE()
  MESSAGE(FATAL_ERROR "-- Provide path of XML: -DXML_LIB_DIR and -DXML_INCLUDE_DIR")
ENDIF (DEFINED XML_INCLUDE_DIR)



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
  TARGET_LINK_LIBRARIES(${TARGET}
	    "${SPGLIB_LIBRARY}"
  )
  INCLUDE_DIRECTORIES ("${SPGLIB_DIR}/include")
ELSE()
  MESSAGE(FATAL_ERROR "-- Provide path to prefix of SPGLIB: -DSPGLIB_DIR=/path/to/prefix")
ENDIF (DEFINED SPGLIB_DIR)

#
# ELPA. Set with -DELPA_LIB_DIR and -DELPA_INCLUDE_DIR
#
IF (WITH_ELPA)
   ADD_DEFINITIONS(-DDFTFE_WITH_ELPA)

   IF (DEFINED ELPA_INCLUDE_DIR)
      FIND_LIBRARY(ELPA_LIBRARY
	NAMES elpa elpa_openmp
	HINTS ${ELPA_LIB_DIR}
	NO_DEFAULT_PATH
      )
      IF(ELPA_LIBRARY STREQUAL "ELPA_LIBRARY-NOTFOUND")
	  MESSAGE(FATAL_ERROR "-- ELPA was not found in ${ELPA_LIB_DIR}")
      ENDIF()
      MESSAGE("-- Use ELPA from ${ELPA_LIB_DIR}")
      TARGET_LINK_LIBRARIES(${TARGET}
		"${ELPA_LIBRARY}"
      )
      INCLUDE_DIRECTORIES ("${ELPA_INCLUDE_DIR}")
   ELSE()
      MESSAGE(FATAL_ERROR "-- Provide path of ELPA: -DELPA_LIB_DIR and -DELPA_INCLUDE_DIR")
   ENDIF (DEFINED ELPA_INCLUDE_DIR)
ENDIF()

#
# Custom "debug" and "release" make targets:
#
ADD_CUSTOM_TARGET(debug
COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ${CMAKE_SOURCE_DIR}
COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
COMMENT "Switch CMAKE_BUILD_TYPE to Debug"
)

ADD_CUSTOM_TARGET(release
COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
COMMENT "Switch CMAKE_BUILD_TYPE to Release"
)


#
# FIXME: that will not play well with installation
#
ADD_DEFINITIONS(-DDFT_PATH="${CMAKE_CURRENT_SOURCE_DIR}")

# Enable testing and descent into tests/ subdirectory:
# Make sure numdiff is in path before configuring, otherwise due to roundoff
# errors many tests may fail.
# Test will run in Debug or Release mode based on the way library is compiled.
IF (WITH_TESTING)
  ENABLE_TESTING()
  INCLUDE(CTest)
  # default 600sec is not enough for some tests, increase:
  SET (TEST_TIME_LIMIT 3600)
  #
  IF (WITH_COMPLEX)
     ADD_SUBDIRECTORY(tests/dft/pseudopotential/complex)
     ADD_SUBDIRECTORY(tests/dft/allElectron/complex)
  ELSE()
     ADD_SUBDIRECTORY(tests/dft/pseudopotential/real)
     ADD_SUBDIRECTORY(tests/dft/allElectron/real)
  ENDIF()
ENDIF()


#
# installation target.
# use with -DCMAKE_INSTALL_PREFIX=/path/to/
#
SET(TARGETLIBEXPORT "femdftTargets")
# 1. Install defined library target
install (TARGETS ${TARGETLIB} EXPORT ${TARGETLIBEXPORT}
         LIBRARY DESTINATION lib)
# 2. Install the binary
install (TARGETS ${TARGET}
         RUNTIME DESTINATION bin)
# 3. Install content of include dir
install (DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
         DESTINATION include)
