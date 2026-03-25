# FindONECCL.cmake
#
# ONECCL. Pass path to -DCMAKE_PREFIX_PATH (or set ONECCL_ROOT_DIR / ONECCL_INCLUDE_DIR / ONECCL_LIB_DIR).
#
# Optionally searched for defaults:
#   ONECCL_ROOT_DIR  : Base directory where all ONECCL components are found
#   ONECCL_INCLUDE_DIR: Directory where ONECCL headers are found
#   ONECCL_LIB_DIR   : Directory where ONECCL library is found
#   ONECCL_VERSION   : Version suffix preference (e.g., 2021.12.0)
#
# Sets after configuration:
#   ONECCL_FOUND
#   ONECCL_INCLUDE_DIRS
#   ONECCL_LIBRARIES
#
# Creates imported target:
#   ONECCL   (IMPORTED)
#
# Notes:
# - oneCCL headers are typically at: <prefix>/include/oneapi/ccl.hpp (or ccl.hpp)
# - oneCCL library is typically: lib/libccl.so (or libccl.a)
# - Unlike NCCL, there is no "ccl_static" library name; static is still "ccl" with .a
#

# Cache from env for convenience
set(ONECCL_INCLUDE_DIR $ENV{ONECCL_INCLUDE_DIR} CACHE PATH "Folder contains Intel oneCCL headers")
set(ONECCL_LIB_DIR     $ENV{ONECCL_LIB_DIR}     CACHE PATH "Folder contains Intel oneCCL libraries")
set(ONECCL_VERSION     $ENV{ONECCL_VERSION}     CACHE STRING "Version of oneCCL to build with")
# Root hint (like NCCL_ROOT_DIR)
set(ONECCL_ROOT_DIR    $ENV{ONECCL_ROOT_DIR}    CACHE PATH "Root folder of Intel oneCCL installation")

# Compatible layer for CMake <3.12.
# ONECCL_ROOT will be accounted for in searching paths and libraries for CMake >=3.12.
list(APPEND ONECCL_ROOT ${ONECCL_ROOT_DIR})
list(APPEND CMAKE_PREFIX_PATH ${ONECCL_ROOT})

# ---- Headers ----
# Prefer modern header path oneapi/ccl.hpp; fall back to ccl.hpp if needed.
find_path(ONECCL_INCLUDE_DIRS
  NAMES oneapi/ccl.hpp ccl.hpp
  HINTS ${ONECCL_INCLUDE_DIR})

# ---- Library name & suffix preference ----
# oneCCL library name is "ccl" for both shared and static variants.
if (USE_STATIC_ONECCL)
  message(STATUS "USE_STATIC_ONECCL is set. Linking with static oneCCL library.")
  set(ONECCL_LIBNAME "ccl")
  if (ONECCL_VERSION)  # Prefer the versioned library if a specific oneCCL version is specified
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".a.${ONECCL_VERSION}" ${CMAKE_FIND_LIBRARY_SUFFIXES})
  endif()
else()
  set(ONECCL_LIBNAME "ccl")
  if (ONECCL_VERSION)  # Prefer the versioned library if a specific oneCCL version is specified
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".so.${ONECCL_VERSION}" ${CMAKE_FIND_LIBRARY_SUFFIXES})
  endif()
endif()

# ---- Library ----
find_library(ONECCL_LIBRARIES
  NAMES ${ONECCL_LIBNAME}
  HINTS ${ONECCL_LIB_DIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ONECCL DEFAULT_MSG ONECCL_INCLUDE_DIRS ONECCL_LIBRARIES)

if(ONECCL_FOUND)
  # Try to report version (best-effort) from header macros if present
  set(ONECCL_HEADER_FILE "${ONECCL_INCLUDE_DIRS}/oneapi/ccl.hpp")
  if (NOT EXISTS "${ONECCL_HEADER_FILE}")
    set(ONECCL_HEADER_FILE "${ONECCL_INCLUDE_DIRS}/ccl.hpp")
  endif()

  if (EXISTS "${ONECCL_HEADER_FILE}")
    message(STATUS "Determining oneCCL version from: ${ONECCL_HEADER_FILE}")
    file(STRINGS ${ONECCL_HEADER_FILE} ONECCL_MAJOR_VERSION_DEFINED
         REGEX "^[ \t]*#define[ \t]+CCL_MAJOR_VERSION[ \t]+[0-9]+.*$" LIMIT_COUNT 1)
    if (ONECCL_MAJOR_VERSION_DEFINED)
      string(REGEX REPLACE ".*CCL_MAJOR_VERSION[ \t]+([0-9]+).*" "\\1"
             ONECCL_MAJOR_VERSION "${ONECCL_MAJOR_VERSION_DEFINED}")
      message(STATUS "ONECCL_MAJOR_VERSION: ${ONECCL_MAJOR_VERSION}")
    endif()
  endif()

  message(STATUS "Found ONECCL (include: ${ONECCL_INCLUDE_DIRS}, library: ${ONECCL_LIBRARIES})")

  # Create a new-style imported target (ONECCL)
  if (USE_STATIC_ONECCL)
    add_library(ONECCL STATIC IMPORTED)
  else()
    add_library(ONECCL SHARED IMPORTED)
  endif()

  set_property(TARGET ONECCL PROPERTY
               IMPORTED_LOCATION ${ONECCL_LIBRARIES})
  # oneCCL is a C++ library; expose headers to consumers
  target_include_directories(ONECCL INTERFACE ${ONECCL_INCLUDE_DIRS})

  mark_as_advanced(ONECCL_ROOT_DIR ONECCL_INCLUDE_DIRS ONECCL_LIBRARIES)
endif()
