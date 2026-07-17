##
# CMakeLists.txt for DFT-FE
#
# DFT-FE is built as an installable shared library (libdftfeReal.so or
# libdftfeComplex.so) plus a thin executable wrapper (dftfe).
#
# Downstream projects consume the library via:
#   find_package(dftfe REQUIRED HINTS <prefix>/lib/cmake/dftfe)
#   target_link_libraries(myapp PRIVATE dftfe::dftfeReal)
#
# --- Feature flag convention ---
#
# Build-time feature flags (USE_REAL, DFTFE_WITH_DEVICE, etc.) are propagated
# through config.h, not through INTERFACE_COMPILE_DEFINITIONS. The pattern is:
#
#   1. Set a CMake variable when the feature is active (done in this file):
#        set(DFTFE_WITH_MY_FEATURE 1)
#
#   2. Add a matching #cmakedefine to include/dftfe/config.h.in:
#        #cmakedefine DFTFE_WITH_MY_FEATURE
#
#   3. Include config.h in any header/source that tests the flag:
#        #include <dftfe/config.h>
#        #ifdef DFTFE_WITH_MY_FEATURE ...
#
# configure_file() (called via dftfeGenerateConfig()) reads the CMake variables
# and writes the corresponding #define / #undef lines into config.h at build
# time. The installed config.h gives consumers the same flag values without
# requiring any -D flags on their compiler command line.
#
# The sole exception is DFTFE_PATH (the runtime data directory), whose value
# differs between the build tree and the install tree. It is expressed as a
# generator expression and kept as a PRIVATE target_compile_definition.
##

# @author Bikash Kanungo

cmake_minimum_required(VERSION 3.17)

# ── Version ───────────────────────────────────────────────────────────────────
set(DFTFE_VERSION "1.0-pre")

set(CMAKE_CXX_STANDARD 17
    CACHE STRING "Minimum required C++ standard (default is 17)")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 64-bit integer workaround for Intel SYCL (must precede project())
IF (USE_64BIT_INT)
  IF ("${GPU_VENDOR}" STREQUAL "intel")
    add_compile_options("-fno-sycl-id-queries-fit-in-int")
  ENDIF()
ENDIF()

# ── Project ───────────────────────────────────────────────────────────────────
project(dftfe VERSION 1.0 LANGUAGES CXX)

# Enable GPU languages after project() so the host compiler is already known.
IF (WITH_GPU)
    IF ("${GPU_LANG}" STREQUAL "cuda")
      enable_language(CUDA)
    ELSEIF ("${GPU_LANG}" STREQUAL "hip")
      enable_language(HIP)
    ELSEIF ("${GPU_LANG}" STREQUAL "sycl")
      # SYCL uses icpx as the host compiler; no separate language keyword needed.
    ELSE ()
      MESSAGE(FATAL_ERROR "\n"
    "*** Could not find valid GPU_LANG=cuda/hip/sycl ***\n\n"
    )
    ENDIF()
ENDIF()

# Suppress all warnings (legacy behaviour; keeps existing build output clean)
add_definitions(-w)

# ── Target names ─────────────────────────────────────────────────────────────
SET(TARGET "dftfe_exe")
IF (WITH_COMPLEX)
  SET(TARGETLIB "dftfeComplex")
ELSE()
  SET(TARGETLIB "dftfeReal")
ENDIF()

OPTION(BUILD_SHARED_LIBS "Build using shared libraries" ON)

# ── Module path ──────────────────────────────────────────────────────────────
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

# ── RPATH handling ────────────────────────────────────────────────────────────
IF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set(CMAKE_MACOSX_RPATH ON)
ENDIF()
SET(CMAKE_SKIP_BUILD_RPATH  FALSE)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
LIST(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
IF("${isSystemDir}" STREQUAL "-1")
   SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
ENDIF("${isSystemDir}" STREQUAL "-1")
SET(CMAKE_INSTALL_NAME_DIR "${CMAKE_INSTALL_PREFIX}/lib")

# ── Git information ───────────────────────────────────────────────────────────
# git_info.h is generated into the build tree (not the source tree) so that
# the source directory is not modified during configuration.
execute_process(
    COMMAND git rev-parse HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_COMMIT
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
    COMMAND git rev-parse --abbrev-ref HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_BRANCH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
configure_file(
    ${CMAKE_SOURCE_DIR}/include/dftfe/git_info.h.in
    ${CMAKE_BINARY_DIR}/include/dftfe/git_info.h
    @ONLY
)

# ── Executable wrapper ────────────────────────────────────────────────────────
ADD_EXECUTABLE(${TARGET} src/main.cc)
set_property(TARGET "dftfe_exe" PROPERTY OUTPUT_NAME "dftfe")

# ── Source lists ──────────────────────────────────────────────────────────────
SET(TARGET_SRC
  ./src/dftfeWrapper.cc
  ./src/dft/dft.cc
  ./src/dft/vselfBinsManager.cc
  ./src/dft/energyCalculator.cc
  ./src/dft/energyDensity.cc
  ./src/dft/densityCalculator.cc
  ./src/dft/computeAuxProjectedDensityMatrixFromPSI.cc
  ./src/dft/densityFirstOrderResponseCalculator.cc
  ./src/dft/applyMultipoleDirichletBC.cc
  ./src/dft/expConfiningPotential.cpp
  ./src/excManager/excDensityLDAClass.cpp
  ./src/excManager/excDensityGGAClass.cpp
  ./src/excManager/excDensityLLMGGAClass.cpp
  ./src/excManager/excTauMGGAClass.cpp
  ./src/excManager/excManager.cpp
  ./src/excManager/AuxDensityMatrixFE.cpp
  ./src/excManager/ExcDFTPlusU.cpp
  ./src/excManager/AtomicBasisData.cpp
  ./src/excManager/NNGGA.cc
  ./src/excManager/NNLDA.cc
  ./src/excManager/NNLLMGGA.cpp
  ./src/excManager/AtomicBasisData.cpp
  ./src/excManager/SlaterBasis.cpp
  ./src/excManager/GaussianBasis.cpp
  ./src/excManager/hubbardClass.cpp
  ./src/excManager/exchangeCorrelationFunctionalEvaluatorCPU.cc
  ./src/poisson/poissonSolverProblem.cc
  ./src/poisson/MultiVectorPoissonLinearSolverProblem.cpp
  ./src/helmholtz/kerkerSolverProblem.cc
  ./src/helmholtz/ldosSolverProblem.cc
  ./src/dftOperator/KohnShamDFTBaseOperator.cc
  ./src/dftOperator/KohnShamDFTStandardEigenOperator.cc
  ./src/dftOperator/veffPrimeForLRDM.cc
  ./src/triangulation/meshMovement/meshMovement.cc
  ./src/triangulation/meshMovement/meshMovementGaussian.cc
  ./src/triangulation/meshMovement/meshMovementAffineTransform.cc
  ./src/triangulation/triangulationManager/triangulationManager.cc
  ./src/symmetry/groupSymmetry.cc
  ./src/geoOpt/geoOptIon.cc
  ./src/geoOpt/geoOptCell.cc
  ./src/matrixFree/MatrixFree.cc
  ./src/solvers/nonlinearSolverProblem.cc
  ./src/solvers/nonLinearSolver.cc
  ./src/solvers/linearSolver.cc
  ./src/solvers/cgSolvers/cgPRPNonLinearSolver.cc
  ./src/solvers/cgSolvers/BFGSNonLinearSolver.cc
  ./src/solvers/cgSolvers/LBFGSNonLinearSolver.cc
  ./src/solvers/eigenSolvers/chebyshevOrthogonalizedSubspaceIterationSolver.cc
  ./src/solvers/eigenSolver.cc
  ./src/linAlg/linearAlgebraOperationsOpt.cc
  ./src/linAlg/linearAlgebraOperationsCPU.cc
  ./src/linAlg/linearAlgebraOperationsInternal.cc
  ./src/linAlg/elpaScalaManager.cc
  ./src/linAlg/scalapackWrapper.cc
  ./src/linAlg/process_grid.cc
  ./src/linAlg/mixingClass.cc
  ./src/solvers/dealiiLinearSolver.cc
  ./src/solvers/dealiiLinearSolverProblem.cc
  ./src/solvers/MultiVectorCGSolver.cpp
  ./src/solvers/MultiVectorMinResSolver.cpp
  ./src/md/molecularDynamicsClass.cc
  ./src/neb/nudgedElasticBandClass.cc
  ./src/geoOpt/geometryOptimizationClass.cc
  ./utils/fileReaders.cc
  ./utils/dftParameters.cc
  ./utils/runParameters.cc
  ./utils/constraintMatrixInfo.cc
  ./utils/dftUtils.cc
  ./utils/vectorTools/vectorUtilities.cc
  ./utils/pseudoConverter.cc
  ./utils/Exceptions.cc
  ./utils/MPIRequestersNBX.cc
  ./utils/MPICommunicatorP2P.cc
  ./utils/MPICommunicatorP2PKernels.cc
  ./utils/MemoryManager.cc
  ./utils/BLASWrapperHost.cc
  ./utils/MPIWriteOnFile.cpp
  ./utils/QuadDataCompositeWrite.cpp
  ./utils/PeriodicTable.cc
  ./utils/FiniteDifference.cc
  ./utils/StringOperations.cpp
  ./utils/SphericalFunctionUtil.cpp
  ./src/dft/dftd.cc
  ./src/mdi/MDIEngine.cpp
  ./src/mdi/libraryMDI.cpp
  ./src/dft/atomicRho.cc
  ./src/dft/charge.cc
  ./src/dft/density.cc
  ./src/dft/dos.cc
  ./src/dft/femUtilityFunctions.cc
  ./src/dft/fermiEnergy.cc
  ./src/dft/generateImageCharges.cc
  ./src/dft/initBoundaryConditions.cc
  ./src/dft/initCoreRho.cc
  ./src/dft/initElectronicFields.cc
  ./src/dft/initPseudoLocal.cc
  ./src/dft/initRho.cc
  ./src/dft/initUnmovedTriangulation.cc
  ./src/dft/kohnShamEigenSolve.cc
  ./src/dft/localizationLength.cc
  ./src/dft/moveAtoms.cc
  ./src/dft/moveMeshToAtoms.cc
  ./src/dft/kerker.cc
  ./src/dft/ldos.cc
  ./src/dft/pRefinedDoFHandler.cc
  ./src/dft/psiInitialGuess.cc
  ./src/dft/publicMethods.cc
  ./src/dft/restart.cc
  ./src/dft/solveNSCF.cc
  ./src/dft/solveBands.cc
  ./src/dft/lowrankApproxScfDielectricMatrixInv.cc
  ./src/dft/lowrankApproxScfDielectricMatrixInvSpinPolarized.cc
  ./src/dft/computeOutputDensityDirectionalDerivative.cc
  ./src/force/configurationalForce.cc
  ./utils/FEBasisOperationsKernelsInternalHost.cc
  ./utils/FEBasisOperations.cc
  ./utils/FEBasisOperationsKernels.cc
  ./src/atom/AtomicCenteredNonLocalOperator.cc
  ./src/atom/AtomPseudoWavefunctions.cc
  ./src/atom/AtomCenteredSphericalFunctionContainer.cc
  ./src/atom/AtomCenteredSphericalFunctionBase.cc
  ./src/atom/AtomCenteredSphericalFunctionSinc.cc
  ./src/atom/AtomCenteredSphericalFunctionBessel.cc
  ./src/atom/AtomCenteredSphericalFunctionGaussian.cc
  ./src/atom/AtomCenteredSphericalFunctionSpline.cc
  ./src/atom/AtomCenteredSphericalFunctionValenceDensitySpline.cc
  ./src/atom/AtomCenteredSphericalFunctionCoreDensitySpline.cc
  ./src/atom/AtomCenteredSphericalFunctionLocalPotentialSpline.cc
  ./src/atom/AtomCenteredSphericalFunctionProjectorSpline.cc
  ./src/atom/AtomCenteredPseudoWavefunctionSpline.cpp
  ./src/TransferBetweenMeshes/InterpolateFromCellToLocalPoints.cpp
  ./src/TransferBetweenMeshes/TransferBetweenMeshesIncompatiblePartitioning.cpp
  ./src/TransferBetweenMeshes/InterpolateCellWiseDataToPoints.cpp
  ./utils/NodalData.cpp
  ./utils/functionalTests/testMultiVectorCGForPoisson.cpp
  ./utils/functionalTests/testDataTransfer.cpp
  ./src/pseudo/oncv/oncvClass.cc
  ./src/pseudo/oncv/atomCenteredPostProcessing.cc)

IF ("${GPU_LANG}" STREQUAL "cuda")
SET(DEVICE_SRC
  ./utils/MemoryTransferKernelsDevice.cc
  ./utils/DeviceKernelsGeneric.cc
  ./utils/DeviceDirectCCLWrapper.cc
  ./src/dft/densityCalculatorDeviceKernels.cc
  ./src/dft/densityFirstOrderResponseCalculatorDeviceKernels.cc
  ./src/dft/solveVselfInBinsDeviceKernels.cc
  ./src/linAlg/linearAlgebraOperationsDeviceKernels.cc
  ./utils/constraintMatrixInfoDeviceKernels.cc
  ./utils/DeviceAPICalls.cu.cc
  ./utils/BLASWrapperDevice.cc
  ./utils/MPICommunicatorP2PKernelsDevice.cc
  ./utils/compressionWrapperDevice.cc
  ./src/solvers/linearSolverCGDeviceKernels.cc
  ./src/matrixFree/MatrixFreeDevice.cc
  ./utils/FEBasisOperationsKernelsInternalDevice.cc
  ./src/atom/AtomicCenteredNonLocalOperatorKernelsDevice.cc
  ./src/force/configurationalForceKernels.cc
  ./src/dftOperator/KohnShamHamiltonianOperatorDeviceKernels.cc
  ./src/excManager/excManagerDeviceKernels.cc
  ./src/excManager/exchangeCorrelationFunctionalEvaluatorDevice.cc
  )
ELSEIF ("${GPU_LANG}" STREQUAL "hip")
SET(DEVICE_SRC
  ./utils/MemoryTransferKernelsDevice.cc
  ./utils/DeviceKernelsGeneric.cc
  ./utils/DeviceDirectCCLWrapper.cc
  ./src/dft/densityCalculatorDeviceKernels.cc
  ./src/dft/densityFirstOrderResponseCalculatorDeviceKernels.cc
  ./src/dft/solveVselfInBinsDeviceKernels.cc
  ./src/linAlg/linearAlgebraOperationsDeviceKernels.cc
  ./utils/constraintMatrixInfoDeviceKernels.cc
  ./utils/DeviceAPICalls.hip.cc
  ./utils/BLASWrapperDevice.cc
  ./utils/MPICommunicatorP2PKernelsDevice.cc
  ./utils/compressionWrapperDevice.cc
  ./src/solvers/linearSolverCGDeviceKernels.cc
  ./src/matrixFree/MatrixFreeDevice.cc
  ./utils/FEBasisOperationsKernelsInternalDevice.cc
  ./src/atom/AtomicCenteredNonLocalOperatorKernelsDevice.cc
  ./src/dftOperator/KohnShamHamiltonianOperatorDeviceKernels.cc
  ./src/excManager/excManagerDeviceKernels.cc
  ./src/excManager/exchangeCorrelationFunctionalEvaluatorDevice.cc
  ./src/force/configurationalForceKernels.cc
  )
ELSEIF ("${GPU_LANG}" STREQUAL "sycl")
SET(DEVICE_SRC
  ./utils/MemoryTransferKernelsDevice.cc
  ./utils/DeviceKernelsGeneric.cc
  ./utils/DeviceDirectCCLWrapper.cc
  ./src/dft/densityCalculatorDeviceKernels.cc
  ./src/dft/densityFirstOrderResponseCalculatorDeviceKernels.cc
  ./src/dft/solveVselfInBinsDeviceKernels.cc
  ./src/linAlg/linearAlgebraOperationsDeviceKernels.cc
  ./utils/constraintMatrixInfoDeviceKernels.cc
  ./utils/DeviceAPICalls.sycl.cc
  ./utils/BLASWrapperDevice.cc
  ./utils/MPICommunicatorP2PKernelsDevice.cc
  ./utils/compressionWrapperDevice.cc
  ./src/solvers/linearSolverCGDeviceKernels.cc
  ./src/matrixFree/MatrixFreeDevice.cc
  ./utils/FEBasisOperationsKernelsInternalDevice.cc
  ./src/atom/AtomicCenteredNonLocalOperatorKernelsDevice.cc
  ./src/dftOperator/KohnShamHamiltonianOperatorDeviceKernels.cc
  ./src/excManager/excManagerDeviceKernels.cc
  ./src/excManager/exchangeCorrelationFunctionalEvaluatorDevice.cc
  ./src/force/configurationalForceKernels.cc
  )
ENDIF()

IF (WITH_GPU)
    IF ("${GPU_LANG}" STREQUAL "cuda")
       set_source_files_properties(${DEVICE_SRC} PROPERTIES LANGUAGE CUDA)
    ELSEIF ("${GPU_LANG}" STREQUAL "hip")
       set_source_files_properties(${DEVICE_SRC} PROPERTIES LANGUAGE HIP)
    ELSEIF ("${GPU_LANG}" STREQUAL "sycl")
      set_source_files_properties(${DEVICE_SRC} PROPERTIES COMPILE_FLAGS ${CMAKE_SYCL_FLAGS})
    ENDIF()
ENDIF()

IF (WITH_GPU)
  LIST(APPEND TARGET_SRC
  ./src/linAlg/linearAlgebraOperationsDevice.cc
  ./utils/constraintMatrixInfoDevice.cc
  ./src/dft/solveVselfInBinsDevice.cc
  ./src/linAlg/pseudoGSDevice.cc
  ./src/linAlg/rayleighRitzDevice.cc
  ./src/poisson/poissonSolverProblemDevice.cc
  ./src/helmholtz/kerkerSolverProblemDevice.cc
  ./src/helmholtz/ldosSolverProblemDevice.cc
  ./src/solvers/eigenSolvers/chebyshevOrthogonalizedSubspaceIterationSolverDevice.cc
  ./src/solvers/linearSolverDevice.cc
  ./src/solvers/linearSolverProblemDevice.cc
  ./src/solvers/linearSolverCGDevice.cc
  )
  LIST(APPEND TARGET_SRC ${DEVICE_SRC})
ENDIF()

# ── Library target ───────────────────────────────────────────────────────────
ADD_LIBRARY(${TARGETLIB} ${TARGET_SRC})

# ── Include directories ───────────────────────────────────────────────────────
# Generator expressions keep paths portable across machines:
#   BUILD_INTERFACE   — used when building within this source tree
#   INSTALL_INTERFACE — used by consumers after installation (relative to prefix)
# The build dir is also needed because config.h and git_info.h are generated there.
target_include_directories(${TARGETLIB} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>)

# ── Feature flags ────────────────────────────────────────────────────────────
# Set CMake variables that configure_file() reads when generating config.h.
# Flags are propagated to consumers via #include <dftfe/config.h>, not via
# INTERFACE_COMPILE_DEFINITIONS, so no target_compile_definitions calls are
# needed here. The sole exception is DFTFE_PATH, which is a generator-
# expression value and must stay as a PRIVATE target compile definition.

# Real vs. complex arithmetic
IF (WITH_COMPLEX)
  set(USE_COMPLEX 1)
  MESSAGE(STATUS "DFT-FE: complex arithmetic")
ELSE()
  set(USE_REAL 1)
  MESSAGE(STATUS "DFT-FE: real arithmetic")
ENDIF()

IF (HIGHERQUAD_PSP)
  set(DFTFE_WITH_HIGHERQUAD_PSP 1)
ENDIF()

IF (USE_64BIT_INT)
  set(DFTFE_WITH_64BIT_INT 1)
ENDIF()

IF (WITH_CUSTOMIZED_DEALII)
  set(DFTFE_WITH_CUSTOMIZED_DEALII 1)
  MESSAGE(STATUS "DFT-FE: customised deal.II")
ENDIF()

OPTION(DISABLE_ASSERT "Disable runtime assertions" OFF)
IF (DISABLE_ASSERT)
  set(DFTFE_DISABLE_ASSERT 1)
ENDIF()

# Runtime data path: PRIVATE because it is an internal implementation detail.
# BUILD_INTERFACE uses the source tree; INSTALL_INTERFACE uses the install prefix.
target_compile_definitions(${TARGETLIB} PRIVATE
    $<BUILD_INTERFACE:DFTFE_PATH="${CMAKE_SOURCE_DIR}">
    $<INSTALL_INTERFACE:DFTFE_PATH="${CMAKE_INSTALL_PREFIX}/share/dftfe">)

# ── MPI ──────────────────────────────────────────────────────────────────────
if(ENABLE_MPI)
  find_package(MPI REQUIRED)
  target_include_directories(${TARGETLIB} PUBLIC ${MPI_CXX_INCLUDE_PATH})
  target_link_libraries(${TARGETLIB} PUBLIC ${MPI_CXX_LIBRARIES})
endif()

# ── GPU ───────────────────────────────────────────────────────────────────────
IF (WITH_GPU)
  set(DFTFE_WITH_DEVICE 1)

  IF ("${GPU_VENDOR}" STREQUAL "nvidia")
    set(DFTFE_WITH_DEVICE_NVIDIA 1)
  ELSEIF ("${GPU_VENDOR}" STREQUAL "amd")
    set(DFTFE_WITH_DEVICE_AMD 1)
  ELSEIF ("${GPU_VENDOR}" STREQUAL "intel")
    set(DFTFE_WITH_DEVICE_INTEL 1)
    set(DFTFE_WITH_DEVICE_MKL 1)
  ENDIF()

  IF ("${GPU_LANG}" STREQUAL "cuda")
    MESSAGE(STATUS "DFT-FE: GPU language = CUDA")
    set(DFTFE_WITH_DEVICE_LANG_CUDA 1)
    find_package(CUDAToolkit REQUIRED)
    # CUDA toolkit includes are PUBLIC because CUDA types appear in device headers
    target_include_directories(${TARGETLIB} PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(${TARGETLIB} PUBLIC CUDA::cudart)
    target_link_libraries(${TARGETLIB} PUBLIC CUDA::cublas)
  ELSEIF ("${GPU_LANG}" STREQUAL "hip")
    MESSAGE(STATUS "DFT-FE: GPU language = HIP")
    set(DFTFE_WITH_DEVICE_LANG_HIP 1)
    IF ("${GPU_VENDOR}" STREQUAL "amd")
      set(__HIP_PLATFORM_AMD__ 1)
    ELSEIF ("${GPU_VENDOR}" STREQUAL "nvidia")
      set(__HIP_PLATFORM_NVIDIA__ 1)
    ENDIF()
    find_package(hipblas)
    target_link_libraries(${TARGETLIB} PUBLIC roc::hipblas)
  ELSEIF ("${GPU_LANG}" STREQUAL "sycl")
    MESSAGE(STATUS "DFT-FE: GPU language = SYCL")
    set(DFTFE_WITH_DEVICE_LANG_SYCL 1)
    find_program(MPI_BASE_CXX_COMPILER "icpx")
    get_filename_component(_REALPATH_SYCL_COMPILER "${MPI_BASE_CXX_COMPILER}" REALPATH)
    get_filename_component(SYCL_BIN_DIR "${_REALPATH_SYCL_COMPILER}" DIRECTORY)
    get_filename_component(SYCL_PACKAGE_DIR "${SYCL_BIN_DIR}" DIRECTORY CACHE)
    find_file(SYCL_INCLUDE_DIR
    NAMES
    include
    HINTS
    ${SYCL_PACKAGE_DIR}
    NO_DEFAULT_PATH)
    # SYCL dirs are PRIVATE: internal compilation detail, not exposed in public headers
    target_include_directories(${TARGETLIB} PUBLIC ${SYCL_INCLUDE_DIR})
    target_include_directories(${TARGETLIB} PUBLIC ${SYCL_INCLUDE_DIR}/sycl)
    set(MKL_INTERFACE lp64)
    set(ENABLE_SCALAPACK ON)
    set(MKL_MPI mpich)
    find_PACKAGE(IntelSYCL REQUIRED)
    find_package(MKL REQUIRED)
    target_link_libraries(${TARGETLIB} PUBLIC MKL::MKL)
    get_target_property(MKL_SYCL_INCLUDE_DIRS MKL::MKL_SYCL INTERFACE_INCLUDE_DIRECTORIES)
    target_include_directories(${TARGETLIB} PUBLIC ${MKL_SYCL_INCLUDE_DIRS})
    get_target_property(mkl_sycl_includes MKL::MKL_SYCL INTERFACE_INCLUDE_DIRECTORIES)
    target_include_directories(${TARGETLIB} PUBLIC ${mkl_sycl_includes})
    get_target_property(mkl_sycl_libs MKL::MKL_SYCL INTERFACE_LINK_LIBRARIES)
    target_link_libraries(${TARGETLIB} PUBLIC ${mkl_sycl_libs})
    set_target_properties(${TARGETLIB} PROPERTIES LINK_FLAGS ${CMAKE_SYCL_FLAGS})
  ENDIF()

  IF (WITH_GPU_AWARE_MPI)
    set(DFTFE_WITH_DEVICE_AWARE_MPI 1)
    MESSAGE(STATUS "DFT-FE: GPU-aware MPI")
  ENDIF()
ENDIF()

# ── deal.II ──────────────────────────────────────────────────────────────────
FIND_PACKAGE(deal.II 9.5.1 REQUIRED
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()
IF(NOT DEAL_II_WITH_LAPACK OR
   NOT DEAL_II_WITH_P4EST OR
   NOT DEAL_II_WITH_64BIT_INDICES OR
   NOT DEAL_II_WITH_MPI)
  MESSAGE(FATAL_ERROR "
Error! The deal.II library found at ${DEAL_II_PATH} was not configured with
    DEAL_II_WITH_LAPACK = ON
    DEAL_II_WITH_P4EST = ON
    DEAL_II_WITH_64BIT_INDICES = ON
    DEAL_II_WITH_MPI = ON
which is required."
    )
ENDIF()

# deal.II is PUBLIC: its types appear in dftfe's public headers.
# Include directories are carried by the imported target (dealii::dealii_release
# / dealii::dealii_debug) via INTERFACE_INCLUDE_DIRECTORIES, so no explicit
# TARGET_INCLUDE_DIRECTORIES call is needed. DEAL_II_INCLUDE_RELDIR is a
# relative path and must not be used directly — it resolves to the source tree.
IF("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    TARGET_LINK_LIBRARIES (${TARGETLIB} PUBLIC dealii::dealii_release)
ELSE()
   TARGET_LINK_LIBRARIES (${TARGETLIB} PUBLIC dealii::dealii_debug)
ENDIF()

# PETSc/SLEPc (detected through deal.II configuration)
IF (DEAL_II_WITH_PETSC AND DEAL_II_WITH_SLEPC)
  set(USE_PETSC 1)
  MESSAGE(STATUS "DFT-FE: PETSc and SLEPc (via deal.II)")
ENDIF()

# ── AlgLib ───────────────────────────────────────────────────────────────────
# PRIVATE: AlgLib types are not exposed in dftfe's public headers.
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
  TARGET_LINK_LIBRARIES(${TARGETLIB} PUBLIC "${ALGLIB_LIBRARY}")
  TARGET_INCLUDE_DIRECTORIES(${TARGETLIB} PUBLIC "${ALGLIB_DIR}")
  TARGET_INCLUDE_DIRECTORIES(${TARGETLIB} PUBLIC "${ALGLIB_DIR}/include")
ELSE()
  MESSAGE(FATAL_ERROR "-- Provide path to AlgLIB: -DALGLIB_DIR=/path/to/prefix")
ENDIF (DEFINED ALGLIB_DIR)

# ── LibXC ────────────────────────────────────────────────────────────────────
# PRIVATE: LibXC types are not exposed in dftfe's public headers.
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
  TARGET_LINK_LIBRARIES(${TARGETLIB} PUBLIC "${LIBXC_LIBRARY}")
  TARGET_INCLUDE_DIRECTORIES(${TARGETLIB} PUBLIC "${LIBXC_DIR}/include")
ELSE()
  MESSAGE(FATAL_ERROR "-- Provide path to prefix of LibXC: -DLIBXC_DIR=/path/to/prefix")
ENDIF (DEFINED LIBXC_DIR)

# ── LibTorch (optional) ───────────────────────────────────────────────────────
IF (WITH_TORCH)
  FIND_PACKAGE(Torch PATHS ${TORCH_DIR} REQUIRED)
  target_link_libraries(${TARGETLIB} PUBLIC ${TORCH_LIBRARIES})
  set(DFTFE_WITH_TORCH 1)
ENDIF()

# ── LibXML2 ──────────────────────────────────────────────────────────────────
# PRIVATE: XML types are not exposed in dftfe's public headers.
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
  TARGET_LINK_LIBRARIES(${TARGETLIB} PRIVATE "${XML_LIBRARY}")
  TARGET_INCLUDE_DIRECTORIES(${TARGETLIB} PRIVATE "${XML_INCLUDE_DIR}")
ELSE()
  MESSAGE(FATAL_ERROR "-- Provide path of XML: -DXML_LIB_DIR and -DXML_INCLUDE_DIR")
ENDIF (DEFINED XML_INCLUDE_DIR)

# ── SPGLib ───────────────────────────────────────────────────────────────────
# PRIVATE: SPGLib types are not exposed in dftfe's public headers.
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
  TARGET_LINK_LIBRARIES(${TARGETLIB} PRIVATE "${SPGLIB_LIBRARY}")
  TARGET_INCLUDE_DIRECTORIES(${TARGETLIB} PRIVATE "${SPGLIB_DIR}/include")
ELSE()
  MESSAGE(FATAL_ERROR "-- Provide path to prefix of SPGLIB: -DSPGLIB_DIR=/path/to/prefix")
ENDIF (DEFINED SPGLIB_DIR)

# ── DFTD3 (optional) ──────────────────────────────────────────────────────────
find_package(DFTD3)
IF (DFTD3_FOUND)
  MESSAGE(STATUS "DFT-FE: DFTD3 enabled")
  set(DFTFE_WITH_DFTD3 1)
  target_link_libraries(${TARGETLIB} PUBLIC DFTD3)
ENDIF()

# ── DFTD4 (optional) ──────────────────────────────────────────────────────────
find_package(DFTD4)
IF (DFTD4_FOUND)
  MESSAGE(STATUS "DFT-FE: DFTD4 enabled")
  set(DFTFE_WITH_DFTD4 1)
  target_link_libraries(${TARGETLIB} PUBLIC DFTD4)
ENDIF()

# ── MDI (optional) ────────────────────────────────────────────────────────────
# PRIVATE: MDI types are not exposed in dftfe's public headers.
IF (WITH_MDI)
  find_library(MDI_LIBRARY
      NAMES mdi
      PATHS ${MDI_PATH}/lib64/mdi
      NO_DEFAULT_PATH
    )
  MESSAGE("-- Use MDI library from ${MDI_PATH}/lib64/mdi")
  IF(MDI_LIBRARY STREQUAL "MDI_LIBRARY-NOTFOUND")
    MESSAGE("-- MDI was not found in ${MDI_PATH}")
  ELSE()
    set(DFTFE_WITH_MDI 1)
    TARGET_LINK_LIBRARIES(${TARGETLIB} PRIVATE "${MDI_LIBRARY}")
    TARGET_INCLUDE_DIRECTORIES(${TARGETLIB} PRIVATE "${MDI_PATH}/include/mdi")
  ENDIF()
ENDIF()

# ── ELPA ──────────────────────────────────────────────────────────────────────
# PUBLIC: ELPA types may appear in dftfe public interfaces.
find_package(ELPA 2023 REQUIRED)
MESSAGE(STATUS "DFT-FE: ELPA enabled")
target_link_libraries(${TARGETLIB} PUBLIC ELPA::ELPA)

# ── NCCL (CUDA collective communication, optional) ────────────────────────────
IF (WITH_GPU AND "${GPU_LANG}" STREQUAL "cuda")
 IF (WITH_DCCL)
   find_package(NCCL REQUIRED)
   IF (NCCL_FOUND)
      MESSAGE(STATUS "DFT-FE: NCCL enabled")
      set(DFTFE_WITH_CUDA_NCCL 1)
      find_package(MPI REQUIRED)
      target_link_libraries(${TARGETLIB} PUBLIC NCCL MPI::MPI_CXX)
   ENDIF()
 ENDIF()
ENDIF()

# ── RCCL (HIP collective communication, optional) ─────────────────────────────
IF (WITH_GPU AND "${GPU_LANG}" STREQUAL "hip")
 IF (WITH_DCCL)
   find_package(RCCL REQUIRED)
   IF (RCCL_FOUND)
      MESSAGE(STATUS "DFT-FE: RCCL enabled")
      set(DFTFE_WITH_HIP_RCCL 1)
      find_package(MPI REQUIRED)
      target_link_libraries(${TARGETLIB} PUBLIC RCCL MPI::MPI_CXX)
   ENDIF()
 ENDIF()
ENDIF()

# ── Executable links against the library ─────────────────────────────────────
TARGET_LINK_LIBRARIES(${TARGET} PUBLIC ${TARGETLIB})

# ── Generate config.h ─────────────────────────────────────────────────────────
# dftfeGenerateConfig() owns the full lifecycle of config.h: it processes
# include/dftfe/config.h.in into ${CMAKE_BINARY_DIR}/include/dftfe/config.h
# and registers the install rule that copies it to <prefix>/include/dftfe/.
# See cmake/dftfeGenerateConfig.cmake for the feature-flag convention.
include(dftfeGenerateConfig)
dftfeGenerateConfig()

# ── Package config files ──────────────────────────────────────────────────────
# These files allow downstream projects to use find_package(dftfe).
include(CMakePackageConfigHelpers)
configure_package_config_file(
    "${CMAKE_SOURCE_DIR}/cmake/dftfeConfig.cmake.in"
    "${CMAKE_BINARY_DIR}/cmake/dftfeConfig.cmake"
    INSTALL_DESTINATION "lib/cmake/dftfe")
write_basic_package_version_file(
    "${CMAKE_BINARY_DIR}/cmake/dftfeConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"   # must be numeric; DFTFE_VERSION may have a suffix
    COMPATIBILITY SameMajorVersion)

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

# ── Testing ───────────────────────────────────────────────────────────────────
IF (WITH_TESTING)
  ENABLE_TESTING()
  INCLUDE(CTest)
  SET (TEST_TIME_LIMIT 5000)
  IF (WITH_COMPLEX)
     ADD_SUBDIRECTORY(tests/dft/pseudopotential/complex)
     ADD_SUBDIRECTORY(tests/dft/allElectron/complex)
     IF (DFTD3_FOUND)
      ADD_SUBDIRECTORY(tests/dft/pseudopotential/complex/d3)
     ENDIF()
     IF (DFTD4_FOUND)
      ADD_SUBDIRECTORY(tests/dft/pseudopotential/complex/d4)
     ENDIF()
  ELSE()
     ADD_SUBDIRECTORY(tests/dft/pseudopotential/real)
     ADD_SUBDIRECTORY(tests/dft/allElectron/real)
     IF (DFTD3_FOUND)
      ADD_SUBDIRECTORY(tests/dft/pseudopotential/real/d3)
     ENDIF()
     IF (DFTD4_FOUND)
      ADD_SUBDIRECTORY(tests/dft/pseudopotential/real/d4)
     ENDIF()
  ENDIF()
ENDIF()

# ── Documentation (optional) ─────────────────────────────────────────────────
option(BUILD_DOCS "Build documentation (requires Doxygen and Sphinx)" OFF)
if(BUILD_DOCS)
  find_package(Doxygen)
  find_package(Sphinx)
  if(${Doxygen_FOUND} AND ${Sphinx_FOUND})
    add_subdirectory("docs")
  else()
    message(WARNING "Doxygen and/or Sphinx not found -- unable to build documentation.")
  endif()
endif()

# ── Installation ──────────────────────────────────────────────────────────────

# Library with export set so downstream projects can import the target
install(TARGETS ${TARGETLIB}
        EXPORT dftfeTargets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

# Executable
install(TARGETS ${TARGET}
        RUNTIME DESTINATION bin)

# Public headers: install include/dftfe/* → <prefix>/include/dftfe/
# Exclude .h.in template files (consumers only need the generated outputs)
install(DIRECTORY "${CMAKE_SOURCE_DIR}/include/dftfe"
        DESTINATION include
        PATTERN "*.in" EXCLUDE)

# Runtime data files (pseudopotential tables, etc.)
install(DIRECTORY "${CMAKE_SOURCE_DIR}/data/"
        DESTINATION share/dftfe/data)

# Helper files (e.g. parameterFile.prm) referenced as DFTFE_PATH/helpers/
install(DIRECTORY "${CMAKE_SOURCE_DIR}/helpers/"
        DESTINATION share/dftfe/helpers)

# Exported CMake targets file (defines dftfe::dftfeReal / dftfe::dftfeComplex)
install(EXPORT dftfeTargets
        FILE dftfeTargets.cmake
        NAMESPACE dftfe::
        DESTINATION lib/cmake/dftfe)

# Package config and version files (entry point for find_package)
install(FILES
        "${CMAKE_BINARY_DIR}/cmake/dftfeConfig.cmake"
        "${CMAKE_BINARY_DIR}/cmake/dftfeConfigVersion.cmake"
        DESTINATION lib/cmake/dftfe)

# Bundled Find modules referenced by dftfeConfig.cmake
install(FILES "${CMAKE_SOURCE_DIR}/cmake/FindELPA.cmake"
        DESTINATION lib/cmake/dftfe)
