# FindELPA.cmake
#
# Finds the ELPA library
#
# This will define the following variables
#
#    ELPA_FOUND
#    ELPA_INCLUDE_DIRS
#    ELPA_LIBRARIES
#
# and the following imported targets
#
#     ELPA::ELPA
#
# Author: David M. Rogers <predictivestatmech@gmail.com>

find_package(PkgConfig)
# elpa-2020.05.001.pc
set(PKG_CONFIG_USE_CMAKE_PREFIX_PATH TRUE)
foreach(pkg elpa_openmp elpa) # prioritize elpa_openmp
    foreach(ver 2020.05.001 2020.11.001 2021.05.001)
        pkg_search_module(PC_ELPA ${pkg}-${ver})
        if(PC_ELPA_FOUND)
            break()
        endif()
    endforeach()
    if(PC_ELPA_FOUND)
        break()
    endif()
endforeach()

if(ELPA_FIND_REQUIRED AND NOT PC_ELPA_FOUND)
    MESSAGE(FATAL_ERROR "Unable to find ELPA. Try adding dir containing lib/pkgconfig/elpa-2020.11.001.pc to -DCMAKE_PREFIX_PATH")
endif()

find_path(ELPA_INCLUDE_DIR
    NAMES elpa/elpa.h elpa/elpa_constants.h
    PATHS ${PC_ELPA_INCLUDE_DIRS}
)
find_library(ELPA_LIBRARIES
    #    NAMES elpa elpa_openmp
    NAMES ${PC_ELPA_LIBRARIES}
    PATHS ${PC_ELPA_LIBRARY_DIRS}
    DOC "elpa libraries list"
)
MESSAGE(STATUS "Using ELPA_INCLUDE_DIR = ${ELPA_INCLUDE_DIR}")
MESSAGE(STATUS "Using ELPA_LIBRARIES = ${ELPA_LIBRARIES}")
set(ELPA_VERSION ${PC_ELPA_VERSION})

mark_as_advanced(ELPA_FOUND ELPA_INCLUDE_DIR ELPA_VERSION)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ELPA
    REQUIRED_VARS ELPA_INCLUDE_DIR ELPA_LIBRARIES
    VERSION_VAR ELPA_VERSION
)

if(ELPA_FOUND)
    set(ELPA_INCLUDE_DIRS ${ELPA_INCLUDE_DIR})
endif()

if(ELPA_FOUND AND NOT TARGET ELPA::ELPA)
    add_library(ELPA::ELPA INTERFACE IMPORTED)
    set_target_properties(ELPA::ELPA PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${ELPA_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES "${ELPA_LIBRARIES}"
        #INTERFACE_COMPILE_FEATURES c_std_99
    )
endif()
