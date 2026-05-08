# cmake/dftfeGenerateConfig.cmake
#
# Provides dftfeGenerateConfig(), which processes include/dftfe/config.h.in
# into ${CMAKE_BINARY_DIR}/include/dftfe/config.h using configure_file.
#
# Call dftfeGenerateConfig() from CMakeLists.txt after all build options have
# been resolved, so that the correct set of #cmakedefine flags is written.

function(dftfeGenerateConfig)
  configure_file(
    "${CMAKE_SOURCE_DIR}/include/dftfe/config.h.in"
    "${CMAKE_BINARY_DIR}/include/dftfe/config.h"
    @ONLY
  )
  message(STATUS "DFT-FE: generated ${CMAKE_BINARY_DIR}/include/dftfe/config.h")
endfunction()
