##
#  CMakeList file for DFT-FE
##
CMAKE_MINIMUM_REQUIRED(VERSION 3.17)
# Set the name of the project and target:
SET(TARGET "dftfe_exe")
SET(TARGETLIB "dftfe")
SET(TARGET_VERSION "1.0-pre") # used in docs/Doxyfile.in

add_definitions(-w)

#FIXME: find a better way to replace O3 with O2 in CMAKE_CXX_FLAGS_RELEASE
IF("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
  add_compile_options("-O2")
ENDIF()

PROJECT(${TARGET} LANGUAGES CXX)
IF (WITH_GPU)
    IF ("${GPU_LANG}" STREQUAL "cuda")
      enable_language(CUDA)
    ELSEIF ("${GPU_LANG}" STREQUAL "hip")
      enable_language(HIP)
    ELSE ()
      MESSAGE(FATAL_ERROR "\n"
    "*** Could not find valid GPU_LANG=cuda/hip ***\n\n"
    )
    ENDIF()
ENDIF()


ADD_EXECUTABLE(${TARGET}   src/main.cc)
set_property(TARGET "dftfe_exe" PROPERTY OUTPUT_NAME "dftfe")

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)


if(ENABLE_MPI)
  find_package(MPI REQUIRED)
  include_directories(${MPI_CXX_INCLUDE_PATH})
  set(DFTFE_MPI_LIBRARIES ${MPI_CXX_LIBRARIES})
endif()


# Declare all source files the target consists of:
SET(TARGET_SRC
  ./src/dftfeWrapper.cc
  ./src/dft/dft.cc
  ./src/dft/vselfBinsManager.cc
  ./src/dft/energyCalculator.cc
  ./src/dft/densityCalculator.cc
  ./src/dft/densityFirstOrderResponseCalculator.cc
  ./src/excManager/excDensityBaseClass.cpp
  ./src/excManager/excDensityLDAClass.cpp
  ./src/excManager/excWavefunctionBaseClass.cpp
  ./src/excManager/excDensityGGAClass.cpp
  ./src/excManager/excManager.cpp
  ./src/excManager/excWavefunctionNoneClass.cpp
  ./src/excManager/NNGGA.cc
  ./src/excManager/NNLDA.cc
  ./src/poisson/poissonSolverProblem.cc
  ./src/helmholtz/kerkerSolverProblem.cc
  ./src/dftOperator/kohnShamDFTOperator.cc
  ./src/dftOperator/operator.cc
  ./src/force/force.cc
  ./src/force/forceWfcContractions.cc  
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
  ./src/solvers/cgSolvers/BFGSNonLinearSolver.cc
  ./src/solvers/cgSolvers/LBFGSNonLinearSolver.cc
  ./src/solvers/eigenSolvers/chebyshevOrthogonalizedSubspaceIterationSolver.cc
  ./src/solvers/eigenSolver.cc
  ./src/linAlg/linearAlgebraOperationsOpt.cc
  ./src/linAlg/linearAlgebraOperationsInternal.cc
  ./src/linAlg/elpaScalaManager.cc
  ./src/linAlg/scalapackWrapper.cc
  ./src/linAlg/process_grid.cc
  ./src/linAlg/mixingClass.cc
  ./src/solvers/dealiiLinearSolver.cc
  ./src/solvers/dealiiLinearSolverProblem.cc
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
  ./pseudoConverters/upfToxml.cc
  ./utils/PeriodicTable.cc
  ./utils/xmlTodftfeParser.cc
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
  ./src/dft/initPseudo-OV.cc
  ./src/dft/initPseudoLocal.cc
  ./src/dft/initRho.cc
  ./src/dft/initUnmovedTriangulation.cc
  ./src/dft/kohnShamEigenSolve.cc
  ./src/dft/localizationLength.cc
  ./src/dft/moveAtoms.cc
  ./src/dft/moveMeshToAtoms.cc
  ./src/dft/kerker.cc
  ./src/dft/pRefinedDoFHandler.cc
  ./src/dft/psiInitialGuess.cc
  ./src/dft/publicMethods.cc
  ./src/dft/restart.cc
  ./src/dft/solveNSCF.cc
  ./src/dft/lowrankApproxScfDielectricMatrixInv.cc
  ./src/dft/lowrankApproxScfDielectricMatrixInvSpinPolarized.cc
  ./src/dft/computeOutputDensityDirectionalDerivative.cc
  ./src/force/configurationalForceCompute/FNonlinearCoreCorrectionGammaAtomsElementalContribution.cc
  ./src/force/configurationalForceCompute/FPSPLocalGammaAtomsElementalContribution.cc
  ./src/force/configurationalForceCompute/FSmearedChargesGammaAtomsElementalContribution.cc
  ./src/force/configurationalForceCompute/FnlGammaElementalContribution.cc
  ./src/force/configurationalForceCompute/accumulateForceContributionGammaAtomsFloating.cc
  ./src/force/configurationalForceCompute/computeFloatingAtomsForces.cc
  ./src/force/configurationalForceCompute/configurationalForceEEshelbyFPSPFnlLinFE.cc
  ./src/force/configurationalForceCompute/configurationalForceEselfLinFE.cc
  ./src/force/configurationalForceCompute/gaussianGeneratorConfForceOpt.cc
  ./src/force/configurationalStressCompute/ENonlinearCoreCorrectionContribution.cc
  ./src/force/configurationalStressCompute/EnlStressContribution.cc
  ./src/force/configurationalStressCompute/EPSPStressContribution.cc
  ./src/force/configurationalStressCompute/ESmearedStressContribution.cc
  ./src/force/configurationalStressCompute/computeStressEEshelbyEPSPEnlEk.cc
  ./src/force/configurationalStressCompute/computeStressEself.cc
  ./src/force/configurationalStressCompute/stress.cc
  ./src/force/createBinObjectsForce.cc
  ./src/force/locateAtomCoreNodesForce.cc  
  ./utils/FEBasisOperationsKernelsInternalHost.cc
  ./utils/FEBasisOperations.cc
  ./utils/FEBasisOperationsKernels.cc)

IF ("${GPU_LANG}" STREQUAL "cuda")

SET(DEVICE_SRC
  ./utils/MemoryTransferKernelsDevice.cc
  ./utils/DeviceKernelsGeneric.cc
  ./utils/DeviceDirectCCLWrapper.cc
  ./src/dft/densityCalculatorDeviceKernels.cc
  ./src/dft/densityFirstOrderResponseCalculatorDeviceKernels.cc
  ./src/dftOperator/operatorDevice.cc
  ./src/dftOperator/kohnShamDFTOperatorDevice.cc
  ./src/dft/solveVselfInBinsDevice.cc
  ./src/solvers/eigenSolvers/chebyshevOrthogonalizedSubspaceIterationSolverDevice.cc
  ./src/linAlg/linearAlgebraOperationsDevice.cc
  ./src/linAlg/pseudoGSDevice.cc
  ./src/linAlg/rayleighRitzDevice.cc
  ./src/force/forceWfcContractionsDevice.cc
  ./utils/constraintMatrixInfoDevice.cc
  ./utils/DeviceAPICalls.cu.cc
  ./utils/DeviceBlasWrapper.cu.cc
  ./utils/BLASWrapperDevice.cu.cc
  ./utils/MPICommunicatorP2PKernelsDevice.cc
  ./src/solvers/linearSolverDevice.cc
  ./src/solvers/linearSolverCGDevice.cc
  ./src/solvers/linearSolverProblemDevice.cc
  ./src/poisson/poissonSolverProblemDevice.cc
  ./src/helmholtz/kerkerSolverProblemDevice.cc
  ./utils/FEBasisOperationsKernelsInternalDevice.cc
  )

ELSEIF ("${GPU_LANG}" STREQUAL "hip")

SET(DEVICE_SRC
  ./utils/MemoryTransferKernelsDevice.cc
  ./utils/DeviceKernelsGeneric.cc
  ./utils/DeviceDirectCCLWrapper.cc
  ./src/dft/densityCalculatorDeviceKernels.cc
  ./src/dft/densityFirstOrderResponseCalculatorDeviceKernels.cc
  ./src/dftOperator/operatorDevice.cc
  ./src/dftOperator/kohnShamDFTOperatorDevice.cc
  ./src/dft/solveVselfInBinsDevice.cc
  ./src/solvers/eigenSolvers/chebyshevOrthogonalizedSubspaceIterationSolverDevice.cc
  ./src/linAlg/linearAlgebraOperationsDevice.cc
  ./src/linAlg/pseudoGSDevice.cc
  ./src/linAlg/rayleighRitzDevice.cc
  ./src/force/forceWfcContractionsDevice.cc
  ./utils/constraintMatrixInfoDevice.cc
  ./utils/DeviceAPICalls.hip.cc
  ./utils/DeviceBlasWrapper.hip.cc
  ./utils/BLASWrapperDevice.hip.cc
  ./utils/MPICommunicatorP2PKernelsDevice.cc
  ./src/solvers/linearSolverDevice.cc
  ./src/solvers/linearSolverCGDevice.cc
  ./src/solvers/linearSolverProblemDevice.cc
  ./src/poisson/poissonSolverProblemDevice.cc
  ./src/helmholtz/kerkerSolverProblemDevice.cc
  ./utils/FEBasisOperationsKernelsInternalDevice.cc
  )

ENDIF()


IF (WITH_GPU)
    IF ("${GPU_LANG}" STREQUAL "cuda")
       set_source_files_properties(${DEVICE_SRC} PROPERTIES LANGUAGE CUDA)
    ELSEIF ("${GPU_LANG}" STREQUAL "hip")
       set_source_files_properties(${DEVICE_SRC} PROPERTIES LANGUAGE HIP)
       IF ("${GPU_VENDOR}" STREQUAL "amd")
          ADD_DEFINITIONS(-D__HIP_PLATFORM_AMD__)
       ELSEIF ("${GPU_VENDOR}" STREQUAL "nvidia")
          ADD_DEFINITIONS(-D__HIP_PLATFORM_NVIDIA__)
       ENDIF()
    ENDIF()
ENDIF()
IF (WITH_GPU)
  LIST(APPEND TARGET_SRC ${DEVICE_SRC})
ENDIF()


ADD_LIBRARY(${TARGETLIB} SHARED ${TARGET_SRC})

#IF("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
#
#target_compile_options(${TARGETLIB}
#    PUBLIC
#        -O2)
#ENDIF()

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

IF (MINIMAL_COMPILE)
  ADD_DEFINITIONS(-DDFTFE_MINIMAL_COMPILE)
ENDIF()

IF (HIGHERQUAD_PSP)
  ADD_DEFINITIONS(-DDFTFE_WITH_HIGHERQUAD_PSP)
ENDIF()

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
#Set GPU compilation flag
#
IF (WITH_GPU)
  ADD_DEFINITIONS(-DDFTFE_WITH_DEVICE)
  
  IF ("${GPU_VENDOR}" STREQUAL "nvidia")
    ADD_DEFINITIONS(-DDFTFE_WITH_DEVICE_NVIDIA)
  ELSEIF ("${GPU_VENDOR}" STREQUAL "amd")
    ADD_DEFINITIONS(-DDFTFE_WITH_DEVICE_AMD)
  ENDIF()

  IF ("${GPU_LANG}" STREQUAL "cuda")
    MESSAGE(STATUS "DFTFE with GPU LANG CUDA")
    ADD_DEFINITIONS(-DDFTFE_WITH_DEVICE_LANG_CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_include_directories(${TARGETLIB} PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(${TARGETLIB} PUBLIC CUDA::cudart)
    target_link_libraries(${TARGETLIB} PUBLIC CUDA::cublas)
  ELSEIF ("${GPU_LANG}" STREQUAL "hip")
    MESSAGE(STATUS "DFTFE with GPU LANG HIP")
    ADD_DEFINITIONS(-DDFTFE_WITH_DEVICE_LANG_HIP)
    find_package(hipblas)
    target_link_libraries(${TARGETLIB} PUBLIC roc::hipblas)
  ENDIF()
  IF (WITH_GPU_AWARE_MPI)
    ADD_DEFINITIONS(-DDFTFE_WITH_DEVICE_AWARE_MPI)
    MESSAGE(STATUS "DFTFE with GPU Aware MPI")
  ENDIF()
ENDIF()

TARGET_LINK_LIBRARIES(${TARGETLIB} PUBLIC ${DFTFE_MPI_LIBRARIES})

#
#Dependencies
#

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
# Are all deal.II dependencies fullfilled?
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


#DEAL_II_INITIALIZE_CACHED_VARIABLES()


TARGET_INCLUDE_DIRECTORIES(${TARGETLIB} PUBLIC ${DEAL_II_INCLUDE_DIRS})
IF("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    TARGET_LINK_LIBRARIES (${TARGETLIB} PUBLIC ${DEAL_II_LIBRARIES_RELEASE})
ELSE()
   TARGET_LINK_LIBRARIES (${TARGETLIB} PUBLIC ${DEAL_II_LIBRARIES_DEBUG})
ENDIF()

IF (WITH_CUSTOMIZED_DEALII)
ADD_DEFINITIONS(-DDFTFE_WITH_CUSTOMIZED_DEALII)
MESSAGE(STATUS "Using customized dealii")
ENDIF()


#
#Set use petsc and slepc flag
#
IF (DEAL_II_WITH_PETSC AND DEAL_II_WITH_SLEPC)
   ADD_DEFINITIONS(-DUSE_PETSC)
   MESSAGE(STATUS "Using petsc and slepc")
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
  TARGET_LINK_LIBRARIES(${TARGETLIB} PUBLIC "${ALGLIB_LIBRARY}"
  )
  TARGET_INCLUDE_DIRECTORIES (${TARGETLIB} PUBLIC "${ALGLIB_DIR}")
  TARGET_INCLUDE_DIRECTORIES (${TARGETLIB} PUBLIC "${ALGLIB_DIR}/include")
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
      PUBLIC "${LIBXC_LIBRARY}"
  )
  TARGET_INCLUDE_DIRECTORIES (${TARGETLIB} PUBLIC "${LIBXC_DIR}/include")
ELSE()
  MESSAGE(FATAL_ERROR "-- Provide path to prefix of LibXC: -DLIBXC_DIR=/path/to/prefix")
ENDIF (DEFINED LIBXC_DIR)

IF (WITH_TORCH)
#
# TORCH. Set with -DTORCH_DIR=/path/to/prefix
#
FIND_PACKAGE(Torch PATHS "/scratch/vikramg_root/vikramg/shared_data/conda_envs/pytorch_cpu/lib/python3.9/site-packages/torch/share/cmake/Torch/" Required) 
target_link_libraries(${TARGETLIB} PUBLIC ${TORCH_LIBRARIES}) 
ADD_DEFINITIONS(-DDFTFE_WITH_TORCH)
ENDIF()

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
  TARGET_LINK_LIBRARIES(${TARGETLIB} PUBLIC
      "${XML_LIBRARY}"
  )
  TARGET_INCLUDE_DIRECTORIES (${TARGETLIB} PUBLIC "${XML_INCLUDE_DIR}")
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
  TARGET_LINK_LIBRARIES(${TARGETLIB} PUBLIC
      "${SPGLIB_LIBRARY}"
  )
  TARGET_INCLUDE_DIRECTORIES (${TARGETLIB} PUBLIC "${SPGLIB_DIR}/include")
ELSE()
  MESSAGE(FATAL_ERROR "-- Provide path to prefix of SPGLIB: -DSPGLIB_DIR=/path/to/prefix")
ENDIF (DEFINED SPGLIB_DIR)

#
# DFTD3. Pass path to -DCMAKE_PREFIX_PATH
#
find_package(DFTD3)
IF (DFTD3_FOUND)
  MESSAGE(STATUS "Will Link With DFTD3 ${S-DFTD3}")
  ADD_DEFINITIONS(-DDFTFE_WITH_DFTD3)
  target_link_libraries(${TARGETLIB} PUBLIC DFTD3)
ENDIF()


#
# DFTD4. Pass path to -DCMAKE_PREFIX_PATH
#
find_package(DFTD4)
IF (DFTD4_FOUND)
  MESSAGE(STATUS "Will Link With DFTD4 ${DFTD4}")
  ADD_DEFINITIONS(-DDFTFE_WITH_DFTD4)
  target_link_libraries(${TARGETLIB} PUBLIC DFTD4)
ENDIF()


#
# MDI.
#
IF (WITH_MDI)
  find_library(MDI_LIBRARY
      NAMES mdi
      PATHS ${MDI_PATH}/lib64/mdi
      NO_DEFAULT_PATH
    )
  MESSAGE("-- Use MDI library from ${MDI_PATH}/lib64/mdi")
  TARGET_LINK_LIBRARIES(${TARGETLIB} PUBLIC
        "${MDI_LIBRARY}"
    )
  TARGET_INCLUDE_DIRECTORIES (${TARGETLIB} PUBLIC "${MDI_PATH}/include/mdi")
  IF(MDI_LIBRARY STREQUAL "MDI_LIBRARY-NOTFOUND")
    MESSAGE("-- MDI was not found in ${MDI_PATH}")
  ELSE()
    ADD_DEFINITIONS(-DDFTFE_WITH_MDI)
  ENDIF()
ENDIF()


#
# ELPA. Pass path to -DCMAKE_PREFIX_PATH
#
find_package(ELPA 2022 REQUIRED)
MESSAGE(STATUS "Will Link With ELPA")
target_link_libraries(${TARGETLIB} PUBLIC ELPA::ELPA)

#
# NCCL. Pass path to -DCMAKE_PREFIX_PATH
#
IF (WITH_GPU AND "${GPU_LANG}" STREQUAL "cuda")
 IF (WITH_DCCL)
   find_package(NCCL REQUIRED)
   IF (NCCL_FOUND)
      MESSAGE(STATUS "Will Link With NCCL")
      ADD_DEFINITIONS(-DDFTFE_WITH_CUDA_NCCL)
      find_package(MPI REQUIRED)
      target_link_libraries(${TARGETLIB} PUBLIC NCCL MPI::MPI_CXX)
   ENDIF()
 ENDIF()
ENDIF()

#
# RCCL. Pass path to -DCMAKE_PREFIX_PATH
#
IF (WITH_GPU AND "${GPU_LANG}" STREQUAL "hip")
 IF (WITH_DCCL)
   find_package(RCCL REQUIRED)
   IF (RCCL_FOUND)
      MESSAGE(STATUS "Will Link With RCCL")
      ADD_DEFINITIONS(-DDFTFE_WITH_HIP_RCCL)
      find_package(MPI REQUIRED)
      target_link_libraries(${TARGETLIB} PUBLIC RCCL MPI::MPI_CXX)
   ENDIF()
 ENDIF()
ENDIF()


TARGET_INCLUDE_DIRECTORIES(${TARGETLIB} PUBLIC ${CMAKE_SOURCE_DIR}/include)
TARGET_LINK_LIBRARIES(${TARGET} PUBLIC ${TARGETLIB})


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
ADD_DEFINITIONS(-DDFTFE_PATH="${CMAKE_CURRENT_SOURCE_DIR}")

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

# Build documentation
option(BUILD_DOCS "Build documentation (requires doxygen and sphinx)" OFF)
if(BUILD_DOCS)
  find_package(Doxygen)
  find_package(Sphinx)
  if(${Doxygen_FOUND} AND ${Sphinx_FOUND})
    add_subdirectory("docs")
  else()
    message(WARNING "Doxygen and/or Sphinx not found -- unable to build documentation.")
  endif()
endif()

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
