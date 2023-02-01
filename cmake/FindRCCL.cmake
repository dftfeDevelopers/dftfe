# Find the rccl libraries
# from https://github.com/xuhdev/pytorch/blob/a3b4accf014e18bf84f58d3018854435cbc3d55b/cmake/Modules/FindRCCL.cmake
#
# The following variables are optionally searched for defaults
#  RCCL_ROOT: Base directory where all RCCL components are found
#  RCCL_INCLUDE_DIR: Directory where RCCL header is found
#  RCCL_LIB_DIR: Directory where RCCL library is found
#
# The following are set after configuration is done:
#  RCCL_FOUND
#  RCCL_INCLUDE_DIRS
#  RCCL_LIBRARIES
#
# The path hints include CUDA_TOOLKIT_ROOT_DIR seeing as some folks
# install RCCL in the same location as the CUDA toolkit.
# See https://github.com/caffe2/caffe2/issues/1601

set(RCCL_INCLUDE_DIR $ENV{RCCL_INCLUDE_DIR} CACHE PATH "Folder contains NVIDIA RCCL headers")
set(RCCL_LIB_DIR $ENV{RCCL_LIB_DIR} CACHE PATH "Folder contains NVIDIA RCCL libraries")
set(RCCL_VERSION $ENV{RCCL_VERSION} CACHE STRING "Version of RCCL to build with")

list(APPEND RCCL_ROOT ${RCCL_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR})
# Compatible layer for CMake <3.12. RCCL_ROOT will be accounted in for searching paths and libraries for CMake >=3.12.
list(APPEND CMAKE_PREFIX_PATH ${RCCL_ROOT})

find_path(RCCL_INCLUDE_DIRS
  NAMES rccl.h
  HINTS ${RCCL_INCLUDE_DIR})

if (USE_STATIC_RCCL)
  MESSAGE(STATUS "USE_STATIC_RCCL is set. Linking with static RCCL library.")
  SET(RCCL_LIBNAME "rccl_static")
  if (RCCL_VERSION)  # Prefer the versioned library if a specific RCCL version is specified
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".a.${RCCL_VERSION}" ${CMAKE_FIND_LIBRARY_SUFFIXES})
  endif()
else()
  SET(RCCL_LIBNAME "rccl")
  if (RCCL_VERSION)  # Prefer the versioned library if a specific RCCL version is specified
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".so.${RCCL_VERSION}" ${CMAKE_FIND_LIBRARY_SUFFIXES})
  endif()
endif()

find_library(RCCL_LIBRARIES
  NAMES ${RCCL_LIBNAME}
  HINTS ${RCCL_LIB_DIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(RCCL DEFAULT_MSG RCCL_INCLUDE_DIRS RCCL_LIBRARIES)

if(RCCL_FOUND)
  set (RCCL_HEADER_FILE "${RCCL_INCLUDE_DIRS}/rccl.h")
  message (STATUS "Determining RCCL version from the header file: ${RCCL_HEADER_FILE}")
  file (STRINGS ${RCCL_HEADER_FILE} RCCL_MAJOR_VERSION_DEFINED
        REGEX "^[ \t]*#define[ \t]+RCCL_MAJOR[ \t]+[0-9]+.*$" LIMIT_COUNT 1)
  if (RCCL_MAJOR_VERSION_DEFINED)
    string (REGEX REPLACE "^[ \t]*#define[ \t]+RCCL_MAJOR[ \t]+" ""
            RCCL_MAJOR_VERSION ${RCCL_MAJOR_VERSION_DEFINED})
    message (STATUS "RCCL_MAJOR_VERSION: ${RCCL_MAJOR_VERSION}")
  endif ()
  message(STATUS "Found RCCL (include: ${RCCL_INCLUDE_DIRS}, library: ${RCCL_LIBRARIES})")
  # Create a new-style imported target (RCCL)
  if (USE_STATIC_RCCL)
      add_library(RCCL STATIC IMPORTED)
  else()
      add_library(RCCL SHARED IMPORTED)
  endif ()
  set_property(TARGET RCCL PROPERTY
               IMPORTED_LOCATION ${RCCL_LIBRARIES})
  set_property(TARGET RCCL PROPERTY
               LANGUAGE CUDA)
  target_include_directories(RCCL INTERFACE ${RCCL_INCLUDE_DIRS})

  mark_as_advanced(RCCL_ROOT_DIR RCCL_INCLUDE_DIRS RCCL_LIBRARIES)
endif()
