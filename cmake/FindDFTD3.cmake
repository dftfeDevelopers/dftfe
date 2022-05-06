find_package(PkgConfig)
set(PKG_CONFIG_USE_CMAKE_PREFIX_PATH TRUE)
pkg_search_module(PC_DFTD3 s-dftd3)
pkg_search_module(PC_MCTC mctc-lib)
if(DFTD3_FIND_REQUIRED AND NOT PC_DFTD3_FOUND)
    MESSAGE(FATAL_ERROR "Unable to find DFTD3 and/or its dependencies. Try adding dir containing lib64/pkgconfig/s-dftd3.pc to -DCMAKE_PREFIX_PATH")
endif()

find_path(DFTD3_INCLUDE_DIR
    NAMES dftd3.h s-dftd3.h
    PATHS ${PC_DFTD3_INCLUDE_DIRS}
)

find_library(DFTD3_LIBRARIES
    NAMES ${PC_DFTD3_LIBRARIES}
    PATHS ${PC_DFTD3_LIBRARY_DIRS}
    DOC "dftd3 libraries list"
)
find_library(MCTC_LIBRARIES
    NAMES ${PC_MCTC_LIBRARIES}
    PATHS ${PC_MCTC_LIBRARY_DIRS}
    DOC "mctc libraries list"
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(DFTD3
    DEFAULT_MSG DFTD3_INCLUDE_DIR DFTD3_LIBRARIES
)

MESSAGE(STATUS "Using DFTD3_INCLUDE_DIR = ${DFTD3_INCLUDE_DIR}")
MESSAGE(STATUS "Using DFTD3_LIBRARIES = ${DFTD3_LIBRARIES};${MCTC_LIBRARIES}")

if(DFTD3_FOUND)
    add_library(DFTD3 INTERFACE IMPORTED)
    set_target_properties(DFTD3 PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${DFTD3_INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES "${DFTD3_LIBRARIES};${MCTC_LIBRARIES};-lgfortran"
)
endif()

