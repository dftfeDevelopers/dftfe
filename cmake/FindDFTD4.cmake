find_package(PkgConfig)
set(PKG_CONFIG_USE_CMAKE_PREFIX_PATH TRUE)
pkg_search_module(PC_DFTD4 dftd4)
pkg_search_module(PC_MCTC mctc-lib)
pkg_search_module(PC_MULTICHARGE multicharge)
if(DFTD4_FIND_REQUIRED AND NOT PC_DFTD4_FOUND)
    MESSAGE(FATAL_ERROR "Unable to find DFTD4 and/or its dependencies. Try adding dir containing lib64/pkgconfig/dftd4.pc to -DCMAKE_PREFIX_PATH")
endif()

find_path(DFTD4_INCLUDE_DIR
    NAMES dftd4.h
    PATHS ${PC_DFTD4_INCLUDE_DIRS}
)

find_library(DFTD4_LIBRARIES
    NAMES ${PC_DFTD4_LIBRARIES}
    PATHS ${PC_DFTD4_LIBRARY_DIRS}
    DOC "dftd4 libraries list"
)
find_library(MCTC_LIBRARIES
    NAMES ${PC_MCTC_LIBRARIES}
    PATHS ${PC_MCTC_LIBRARY_DIRS}
    DOC "mctc libraries list"
)
find_library(MULTICHARGE_LIBRARIES
    NAMES ${PC_MULTICHARGE_LIBRARIES}
    PATHS ${PC_MULTICHARGE_LIBRARY_DIRS}
    DOC "multicharge libraries list"
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(DFTD4
    DEFAULT_MSG DFTD4_INCLUDE_DIR DFTD4_LIBRARIES
)

MESSAGE(STATUS "Using DFTD4_INCLUDE_DIR = ${DFTD4_INCLUDE_DIR}")
MESSAGE(STATUS "Using DFTD4_LIBRARIES = ${DFTD4_LIBRARIES} ${MCTC_LIBRARIES} ${MULTICHARGE_LIBRARIES}")

if(DFTD4_FOUND)
    add_library(DFTD4 INTERFACE IMPORTED)
    set_target_properties(DFTD4 PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${DFTD4_INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES "${DFTD4_LIBRARIES};${MCTC_LIBRARIES};${MULTICHARGE_LIBRARIES};-lgfortran"
)
endif()

