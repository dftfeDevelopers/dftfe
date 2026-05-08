# cmake/dftfeGenerateConfig.cmake
#
# Provides dftfeGenerateConfig(), which processes include/dftfe/config.h.in
# into ${CMAKE_BINARY_DIR}/include/dftfe/config.h using configure_file.
#
# --- How feature flags work ---
#
# DFT-FE propagates build-time feature flags through config.h rather than
# through target_compile_definitions(PUBLIC). The pattern for every flag is:
#
#   1. CMakeLists.txt sets a CMake variable when the feature is active:
#
#        if(WITH_MY_FEATURE)
#          set(DFTFE_WITH_MY_FEATURE 1)
#        endif()
#
#   2. include/dftfe/config.h.in contains a matching #cmakedefine:
#
#        #cmakedefine DFTFE_WITH_MY_FEATURE
#
#      configure_file() writes "#define DFTFE_WITH_MY_FEATURE" when the CMake
#      variable is set and truthy, or "/* #undef DFTFE_WITH_MY_FEATURE */"
#      otherwise.
#
#   3. Any dftfe header or source that tests the flag includes config.h first:
#
#        #include <dftfe/config.h>
#        #ifdef DFTFE_WITH_MY_FEATURE
#          ...
#        #endif
#
# Because config.h is installed alongside the other public headers, downstream
# consumers get the same flag values simply by including <dftfe/config.h> —
# no -D flags need to appear on the consumer's compiler command line.
#
# The sole exception is DFTFE_PATH (the runtime data directory), whose value
# differs between the build tree and the install tree. It is expressed as a
# generator expression and must stay as a PRIVATE target_compile_definition.
#
# Call dftfeGenerateConfig() after all set() calls for feature flags have been
# made (i.e. after all find_package / find_library blocks), so that
# configure_file sees the final values of every variable.

function(dftfeGenerateConfig)
  configure_file(
    "${CMAKE_SOURCE_DIR}/include/dftfe/config.h.in"
    "${CMAKE_BINARY_DIR}/include/dftfe/config.h"
    @ONLY
  )
  message(STATUS "DFT-FE: generated ${CMAKE_BINARY_DIR}/include/dftfe/config.h")

  # Install config.h alongside the other public headers. This lives here rather
  # than in the top-level install() block because dftfeGenerateConfig owns the
  # full lifecycle of config.h: template, generation, and installation.
  install(FILES "${CMAKE_BINARY_DIR}/include/dftfe/config.h"
          DESTINATION include/dftfe)
endfunction()
