##
#  CMake script for DFT-FE 
##

#Note: Set the ALGLIB and LIBXC paths below. 

#set ALGLIB paths
SET(ALGLIB_INCLUDE "/Users/rudraa/workspace/software/alglib/cpp/src/") #path to interpolation.h
SET(ALGLIB_LIB "/Users/rudraa/workspace/software/alglib/cpp/src/libAlglib.so") #path to libAlglib.so
#Note: On some platforms, you may have to set DYLD_LIBRARY_PATH(OSX)/LD_LIBRARY_PATH to include path ALGLIB_LIB. For example:
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/Algib/.so/file

#set LIBXC paths
SET(LIBXC_INCLUDE "/Users/rudraa/workspace/software/libxc/libxc-3.0.0/installDIR/include") #path to xc.h
SET(LIBXC_LIB "/Users/rudraa/workspace/software/libxc/libxc-3.0.0/installDIR/lib/libxc.a") #path to libxc.a


#
# Usually, you will not need to modify anything beyond this point...
#

# Set the name of the project and target:
SET(TARGET "main")

# Declare all source files the target consists of:
SET(TARGET_SRC
  ./src/${TARGET}.cc
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

SET(SIGN_COMPARISON_WARNING  "-Wno-sign-compare -Wno-ignored-qualifiers -Wno-deprecated-declarations -Wno-vla-extension -Wno-reorder -Wno-sometimes-uninitialized -Wno-undefined-var-template")
SET(DEAL_II_CXX_FLAGS  "${DEAL_II_CXX_FLAGS} ${SIGN_COMPARISON_WARNING}" )

INCLUDE_DIRECTORIES(${ALGLIB_INCLUDE} ${LIBXC_INCLUDE})

ADD_EXECUTABLE(${TARGET} ${TARGET_SRC})
DEAL_II_SETUP_TARGET(${TARGET})
TARGET_LINK_LIBRARIES(${TARGET} ${LIBXC_LIB} ${ALGLIB_LIB})

#DEAL_II_INVOKE_AUTOPILOT()

