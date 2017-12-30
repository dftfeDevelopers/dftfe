##
#  CMake script for DFT-FE 
##

##Note: Set ALGLIB, LIBXC, PETSC and SLEPC paths below. 

##set ALGLIB paths
SET(ALGLIB_INCLUDE "/nfs/mcfs_comp/home/rudraa/software/alglib/cpp/src/") #path to interpolation.h
SET(ALGLIB_LIB "/nfs/mcfs_comp/home/rudraa/software/alglib/cpp/src/libAlglib.so") #path to libAlglib.so
##Note: On some platforms, you may have to set DYLD_LIBRARY_PATH(OSX)/LD_LIBRARY_PATH to include path ALGLIB_LIB. For example:
##export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/Algib/.so/file

##set LIBXC paths
SET(LIBXC_INCLUDE "/nfs/mcfs_comp/home/rudraa/software/libxc/libxc-2.2.0/installDir/include") #path to xc.h
SET(LIBXC_LIB "/nfs/mcfs_comp/home/rudraa/software/libxc/libxc-2.2.0/installDir/lib/libxc.a") #path to libxc.a

##set PETSc Real paths
SET(PETSC_REAL_INCLUDE "/home/vikramg/DFT-FE-softwares/softwareCentos/petsc/intel_petsc3.7.5_double_elemental/include")
SET(PETSC_REAL_LIB "/home/vikramg/DFT-FE-softwares/softwareCentos/petsc/intel_petsc3.7.5_double_elemental/lib/libpetsc.so")

##set SLEPc Real paths
SET(SLEPC_REAL_INCLUDE "/home/vikramg/DFT-FE-softwares/softwareCentos/slepc/intel_slepc3.7.3_double_elemental/include")
SET(SLEPC_REAL_LIB "/home/vikramg/DFT-FE-softwares/softwareCentos/slepc/intel_slepc3.7.3_double_elemental/lib/libslepc.so") 

##set PETSc Complex paths
SET(PETSC_COMPLEX_INCLUDE "/home/vikramg/DFT-FE-softwares/softwareCentos/petsc/intel_petsc3.7.5_complex/include")
SET(PETSC_COMPLEX_LIB "/home/vikramg/DFT-FE-softwares/softwareCentos/petsc/intel_petsc3.7.5_complex/lib/libpetsc.so")

##set SLEPc Complex paths
SET(SLEPC_COMPLEX_INCLUDE "/home/vikramg/DFT-FE-softwares/softwareCentos/slepc/intel_slepc3.7.3_complex/include")
SET(SLEPC_COMPLEX_LIB "/home/vikramg/DFT-FE-softwares/softwareCentos/slepc/intel_slepc3.7.3_complex/lib/libslepc.so") 

#
# Usually, you will not need to modify anything beyond this point...
#

# Set the name of the project and target:
SET(TARGET "main")

# Declare all source files the target consists of:
SET(TARGET_SRC
  ./src/${TARGET}.cc
  ./src/dft/dft.cc
  ./src/poisson/poisson.cc
  ./src/eigen/eigen.cc
  ./src/force/force.cc
  ./src/force/eshelbyTensor/eshelbyTensor.cc
  ./src/meshMovement/meshMovement.cc
  ./src/meshMovement/meshMovementGaussian.cc
  ./utils/fileReaders.cc
  ./utils/dftParameters.cc
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
IF(NOT DEAL_II_WITH_LAPACK)
  MESSAGE(FATAL_ERROR "
Error! The deal.II library found at ${DEAL_II_PATH} was not configured with
    DEAL_II_WITH_LAPACK = ON
which is required."
    )
ENDIF()


DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT(${TARGET})

SET(WARNINGSLIST  "-w -Wno-sign-compare -Wno-ignored-qualifiers -Wno-deprecated-declarations -Wno-vla-extension -Wno-reorder -Wno-sometimes-uninitialized -Wno-undefined-var-template -Wno-ignored-qualifiers")
SET(DEAL_II_CXX_FLAGS  "${DEAL_II_CXX_FLAGS} ${WARNINGSLIST}" )

#Set periodic BC flag
IF (DEFINED _ENABLE_PERIODIC)
   ADD_DEFINITIONS(-DENABLE_PERIODIC_BC)
ELSE()
   ADD_DEFINITIONS(-DDISABLE_PERIODIC_BC)
ENDIF()

ADD_EXECUTABLE(${TARGET} ${TARGET_SRC})
DEAL_II_SETUP_TARGET(${TARGET})

IF (DEFINED _ENABLE_PERIODIC) 
MESSAGE(STATUS "==== Periodic ====")
INCLUDE_DIRECTORIES(${ALGLIB_INCLUDE} ${LIBXC_INCLUDE} ${PETSC_COMPLEX_INCLUDE} ${SLEPC_COMPLEX_INCLUDE})
TARGET_LINK_LIBRARIES(${TARGET} ${LIBXC_LIB} ${ALGLIB_LIB} ${PETSC_COMPLEX_LIB} ${SLEPC_COMPLEX_LIB})
set(ignoreMe "${_ENABLE_PERIODIC}")
ELSE()
MESSAGE(STATUS "==== Non Periodic ====")
INCLUDE_DIRECTORIES(${ALGLIB_INCLUDE} ${LIBXC_INCLUDE} ${PETSC_REAL_INCLUDE} ${SLEPC_REAL_INCLUDE})
TARGET_LINK_LIBRARIES(${TARGET} ${LIBXC_LIB} ${ALGLIB_LIB} ${PETSC_REAL_LIB} ${SLEPC_REAL_LIB})
ENDIF()

#DEAL_II_INVOKE_AUTOPILOT()

