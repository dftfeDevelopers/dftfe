# Making and Using DFT-FE as a Library

## Overview

DFT-FE can be built as an installable shared library (`libdftfeReal.so` or
`libdftfeComplex.so`) that downstream projects consume via CMake's
`find_package(dftfe)`.

The changes made to support this are:

- Public headers are installed under `<prefix>/include/dftfe/` so they are
  included as `#include <dftfe/foo.h>`, avoiding namespace clashes with other
  libraries.
- A generated `config.h` (from `include/dftfe/config.h.in`) encodes all
  build-time feature flags. It is installed alongside the other headers so
  consumers can test feature availability without guessing compile flags.
- `CMakeLists.txt` uses modern CMake target-based semantics: generator
  expressions for include paths so build-tree paths are never baked into the
  exported targets, `set()` variables feeding `config.h` for all feature flags,
  and `PRIVATE`/`PUBLIC` linkage based on whether third-party types appear in
  the public headers.
- `cmake/dftfeConfig.cmake.in` and the associated version file enable
  `find_package(dftfe)` in downstream projects without any manual path
  configuration beyond specifying the install prefix.

---

## Prerequisites

| Dependency    | Minimum version | How to specify              |
|---------------|-----------------|-----------------------------|
| CMake         | 3.17            | —                           |
| C++17 compiler| —               | GCC 9+, Clang 10+, icpx    |
| MPI           | —               | OpenMPI or MPICH            |
| deal.II       | 9.5.1           | `-DDEAL_II_DIR=`            |
| ELPA          | 2023            | via `CMAKE_PREFIX_PATH`     |
| AlgLib        | 3.x             | `-DALGLIB_DIR=`             |
| LibXC         | 6.x             | `-DLIBXC_DIR=`              |
| SPGLib        | —               | `-DSPGLIB_DIR=`             |
| LibXML2       | —               | `-DXML_LIB_DIR=` `-DXML_INCLUDE_DIR=` |

deal.II must be compiled with: `LAPACK`, `p4est`, `MPI`, and `64BIT_INDICES`.

### Optional dependencies

| Dependency    | CMake flags                                              |
|---------------|----------------------------------------------------------|
| CUDA (NVIDIA) | `-DWITH_GPU=ON -DGPU_LANG=cuda -DGPU_VENDOR=nvidia`     |
| HIP (ROCm)    | `-DWITH_GPU=ON -DGPU_LANG=hip -DGPU_VENDOR=amd`         |
| SYCL (Intel)  | `-DWITH_GPU=ON -DGPU_LANG=sycl -DGPU_VENDOR=intel`      |
| LibTorch      | `-DWITH_TORCH=ON -DTORCH_DIR=/path/to/torch`            |
| DFTD3         | `-DCMAKE_PREFIX_PATH=/path/to/dftd3`                    |
| DFTD4         | `-DCMAKE_PREFIX_PATH=/path/to/dftd4`                    |
| MDI           | `-DWITH_MDI=ON -DMDI_PATH=/path/to/mdi`                 |

---

## Configure, Build, and Install

Use an out-of-source build:

```bash
mkdir build && cd build

cmake /path/to/dftfe \
  -DCMAKE_INSTALL_PREFIX=/opt/dftfe \
  -DCMAKE_BUILD_TYPE=Release \
  -DDEAL_II_DIR=/path/to/deal.II \
  -DALGLIB_DIR=/path/to/alglib \
  -DLIBXC_DIR=/path/to/libxc \
  -DSPGLIB_DIR=/path/to/spglib \
  -DXML_LIB_DIR=/path/to/libxml2/lib \
  -DXML_INCLUDE_DIR=/path/to/libxml2/include/libxml2 \
  -DENABLE_MPI=ON
  # Optional: add -DWITH_COMPLEX=ON for the complex-arithmetic variant

make -j$(nproc)
make install
```

---

## Install Tree Layout

After `make install` with `CMAKE_INSTALL_PREFIX=/opt/dftfe`:

```
/opt/dftfe/
├── bin/
│   └── dftfe                          ← standalone DFT-FE executable
├── lib/
│   ├── libdftfeReal.so                ← shared library (or libdftfeComplex.so)
│   └── cmake/dftfe/
│       ├── dftfeConfig.cmake          ← find_package() entry point
│       ├── dftfeConfigVersion.cmake   ← version compatibility checking
│       ├── dftfeTargets.cmake         ← exported target definitions
│       └── FindELPA.cmake             ← bundled Find module
├── include/
│   └── dftfe/
│       ├── config.h                   ← generated feature-flag header
│       ├── dftfeWrapper.h             ← primary public API header
│       ├── dft.h
│       ├── ...                        ← all other public headers
│       └── XCfunctionalDefs/          ← XC functional inline definitions
└── share/
    └── dftfe/
        └── data/                      ← runtime data (pseudopotentials, etc.)
```

---

## Using DFT-FE in a Downstream Project

### CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.17)
project(myapp LANGUAGES CXX)

find_package(dftfe REQUIRED
  HINTS /opt/dftfe/lib/cmake/dftfe)

add_executable(myapp main.cpp)

# A single target_link_libraries call propagates:
#   - /opt/dftfe/include on the include path (gives access to dftfe/config.h)
#   - transitive links to deal.II, ELPA, MPI, ...
# Feature flags (USE_REAL, DFTFE_WITH_DEVICE, etc.) are not injected as
# -D defines; include <dftfe/config.h> to access them.
target_link_libraries(myapp PRIVATE dftfe::dftfeReal)
# or dftfe::dftfeComplex if built with -DWITH_COMPLEX=ON
```

### main.cpp

```cpp
#include <dftfe/dftfeWrapper.h>

int main(int argc, char *argv[])
{
    dftfe::dftfeWrapper wrapper(/* ... */);
    // ...
}
```

---

## Developer Code of Conduct

### Including DFT-FE headers

Always use the namespaced angle-bracket form so the code works both when
building DFT-FE itself and when installed as a library:

```cpp
// CORRECT
#include <dftfe/dft.h>
#include <dftfe/TypeConfig.h>

// WRONG — breaks the installed library and conflicts with flat installs
#include <dft.h>
#include "TypeConfig.h"
```

### Build-time feature flags

Feature flags are not present on the compiler command line unless explicitly
propagated. Always query them through `config.h`:

```cpp
// CORRECT
#include <dftfe/config.h>

#ifdef DFTFE_WITH_DEVICE
  // GPU code path
#endif

// WRONG — config.h might not be included, so the flag is undefined;
//          behaviour depends on whether the translation unit happened to
//          include another dftfe header that pulls in config.h first.
#ifdef DFTFE_WITH_DEVICE
```

### CMake: feature flags

Feature flags are propagated through `config.h`, not through
`INTERFACE_COMPILE_DEFINITIONS`. The pattern is:

1. In `CMakeLists.txt`, set a CMake variable when the feature is enabled:
   ```cmake
   # CORRECT — configure_file() reads this variable and writes #define into config.h
   if(WITH_MY_FEATURE)
     set(DFTFE_WITH_MY_FEATURE 1)
   endif()
   ```

2. In `include/dftfe/config.h.in`, add the matching `#cmakedefine`:
   ```c
   #cmakedefine DFTFE_WITH_MY_FEATURE
   ```

3. In any header or source that tests the flag, include `config.h` first:
   ```cpp
   #include <dftfe/config.h>
   #ifdef DFTFE_WITH_MY_FEATURE
     // ...
   #endif
   ```

Never use `add_definitions` or `target_compile_definitions(PUBLIC)` for feature
flags — the former is directory-scoped and never exported, the latter is
redundant once `config.h` is the single source of truth.

The one exception is `DFTFE_PATH`: it is a path whose value differs between the
build tree and the install tree, so it must remain a PRIVATE target compile
definition using generator expressions and cannot live in `config.h`.

### CMake: linkage (PUBLIC vs PRIVATE)

Use `PRIVATE` for dependencies whose headers do not appear in DFT-FE's public
headers; use `PUBLIC` for those that do:

```cmake
# deal.II types appear in public headers → PUBLIC
target_include_directories(dftfeReal PUBLIC ${DEAL_II_INCLUDE_RELDIR})
target_link_libraries(dftfeReal PUBLIC dealii::dealii_release)

# AlgLib and LibXC types are implementation details → PRIVATE
target_include_directories(dftfeReal PRIVATE "${ALGLIB_DIR}/include")
target_link_libraries(dftfeReal PRIVATE "${ALGLIB_LIBRARY}")
```

`PRIVATE` dependencies do not appear in `dftfeTargets.cmake`, so consumers do
not need to have them installed.

### CMake: portable paths

Never hardcode build-tree paths. Use generator expressions so the installed
target file contains correct paths on every machine:

```cmake
target_include_directories(dftfeReal PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>   # used while building
    $<INSTALL_INTERFACE:include>)                     # used after installation
```

`$<INSTALL_INTERFACE:include>` expands to `<prefix>/include` relative to the
install prefix, so it works on any machine regardless of where DFT-FE is
installed.

### Adding a new header

1. Place it in `include/dftfe/`.
2. Include other dftfe headers as `#include <dftfe/foo.h>`.
3. If it tests any build-time flag, add `#include <dftfe/config.h>` before
   those tests.
4. If it introduces a new optional dependency:
   - Add `#cmakedefine DFTFE_WITH_NEW_DEP` to `include/dftfe/config.h.in`.
   - In `CMakeLists.txt`, add the `find_package` / `find_library` call, then
     `set(DFTFE_WITH_NEW_DEP 1)` when the dependency is found, then
     `target_link_libraries`.
   - Decide on `PUBLIC` vs `PRIVATE` linkage based on whether the dependency's
     types appear in dftfe's public headers.
