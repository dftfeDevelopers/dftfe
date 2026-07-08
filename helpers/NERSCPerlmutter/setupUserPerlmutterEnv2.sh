#!/bin/bash
# script to setup and build DFT-FE.
#
# Variant of setupUserPerlmutter.sh for the personal/env2-style dependency
# install under /pscratch (as opposed to the shared m4067 group install).
# Same withGPU toggle as the local-machine setupDFTFE.sh: flip withGPU=OFF
# below to produce a genuine CPU-only build for the CPU ctest suite, and
# withGPU=ON for the GPU-dependent testsGPU suite. Each variant needs its
# own build directory (see BUILD_ROOT below) since re-running cmake with a
# different WITH_GPU value in the same tree only reconfigures, it does not
# separate the two binaries.

set -e
set -o pipefail

if [ -s CMakeLists.txt ]; then
    echo "This script must be run from the build directory!"
    exit 1
fi

# Path to project source
SRC=$(cd "$(dirname "$0")/../.." && pwd) # dftfe repo root (this script lives at helpers/NERSCPerlmutter/)

########################################################################
#Provide paths below for external libraries, compiler options and flags,
# and optimization flag

ENV2="/pscratch/sd/r/rezgar/install_DFTFE/env2"

#Paths for required external libraries (single dealii install serves both
#real and complex builds)
dealiiDir="$ENV2/dealii"
alglibDir="$ENV2/lib/alglib"
libxcDir="$ENV2"
spglibDir="$ENV2"
xmlIncludeDir="/usr/include/libxml2"
xmlLibDir="/usr/lib64"
ELPA_PATH="$ENV2"

#Paths for optional external libraries
DCCL_PATH="$NCCL_DIR"
mdiPath=""

#Toggle GPU compilation - set to OFF for a genuine CPU-only build.
#Can be overridden without editing this file, e.g.:
#   withGPU=OFF ./setupUserPerlmutterEnv2.sh
withGPU=${withGPU:-ON}
gpuLang="cuda"     # Use "cuda"/"hip"
gpuVendor="nvidia" # Use "nvidia/amd"
withGPUAwareMPI=OFF #Please use this option with care
                   #Only use if the machine supports
                   #device aware MPI and is profiled
                   #to be fast

#Option to link to DCCL library (Only for GPU compilation)
withDCCL=OFF
withMDI=OFF

#Compiler options and flags
cxx_compiler=CC  #sets DCMAKE_CXX_COMPILER
cxx_flags="-march=znver3 -fPIC" #sets DCMAKE_CXX_FLAGS
cxx_flagsRelease="-fPIC -target-accel=nvidia80 -I$MPICH_DIR/include" #sets DCMAKE_CXX_FLAGS_RELEASE
device_flags="-I$MPICH_DIR/include -arch=sm_80" # set DCMAKE_CXX_CUDA/HIP_FLAGS
                           #(only applicable for withGPU=ON)
device_architectures="80" # set DCMAKE_CXX_CUDA/HIP_ARCHITECTURES
                           #(only applicable for withGPU=ON)

#Option to compile with default or higher order quadrature for storing pseudopotential data
#ON is recommended for MD simulations with hard pseudopotentials
withHigherQuadPSP=ON

# build type: "Release" or "Debug"
build_type=Release

#Enable ctest + numdiff for the CPU regression suite
testing=ON
numdiffExecutable="$ENV2/numdiff_install/bin/numdiff"

useInt64=ON
###########################################################################
#Usually, no changes are needed below this line
#
out=`echo "$build_type" | tr '[:upper:]' '[:lower:]'`
if [[ "$withGPU" == "ON" ]]; then
  out="${out}GPU"
else
  out="${out}CPU"
fi

function cmake_real() {
  mkdir -p real && cd real
  if [ "$withGPU" = "ON" ] && [ "$gpuLang" = "cuda" ]; then
    cmake -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_COMPILER=$cxx_compiler\
    -DCMAKE_CXX_FLAGS="$cxx_flags"\
    -DCMAKE_CXX_FLAGS_RELEASE="$cxx_flagsRelease" \
    -DCMAKE_BUILD_TYPE=$build_type -DDEAL_II_DIR=$dealiiDir \
    -DALGLIB_DIR=$alglibDir -DLIBXC_DIR=$libxcDir \
    -DSPGLIB_DIR=$spglibDir -DXML_LIB_DIR=$xmlLibDir \
    -DXML_INCLUDE_DIR=$xmlIncludeDir\
    -DWITH_MDI=$withMDI -DMDI_PATH=$mdiPath \
    -DWITH_DCCL=$withDCCL -DCMAKE_PREFIX_PATH="$ELPA_PATH;$DCCL_PATH"\
    -DWITH_COMPLEX=OFF -DWITH_GPU=$withGPU -DGPU_LANG=$gpuLang -DGPU_VENDOR=$gpuVendor -DWITH_GPU_AWARE_MPI=$withGPUAwareMPI -DCMAKE_CUDA_FLAGS="$device_flags" -DCMAKE_CUDA_ARCHITECTURES="$device_architectures"\
    -DCMAKE_SHARED_LINKER_FLAGS="-L$MPICH_DIR/lib -lmpich"\
    -DWITH_TESTING=$testing -DNUMDIFF_EXECUTABLE=$numdiffExecutable \
    -DHIGHERQUAD_PSP=$withHigherQuadPSP -DUSE_64BIT_INT=$useInt64 $1
  else
    cmake -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_COMPILER=$cxx_compiler\
    -DCMAKE_CXX_FLAGS="$cxx_flags"\
    -DCMAKE_CXX_FLAGS_RELEASE="$cxx_flagsRelease" \
    -DCMAKE_BUILD_TYPE=$build_type -DDEAL_II_DIR=$dealiiDir \
    -DALGLIB_DIR=$alglibDir -DLIBXC_DIR=$libxcDir \
    -DSPGLIB_DIR=$spglibDir -DXML_LIB_DIR=$xmlLibDir \
    -DXML_INCLUDE_DIR=$xmlIncludeDir\
    -DWITH_MDI=$withMDI -DMDI_PATH=$mdiPath \
    -DWITH_DCCL=OFF -DCMAKE_PREFIX_PATH="$ELPA_PATH"\
    -DWITH_COMPLEX=OFF -DWITH_GPU=OFF \
    -DWITH_TESTING=$testing -DNUMDIFF_EXECUTABLE=$numdiffExecutable \
    -DHIGHERQUAD_PSP=$withHigherQuadPSP -DUSE_64BIT_INT=$useInt64 $1
  fi
  cd ..
}

function cmake_cplx() {
  mkdir -p complex && cd complex
  if [ "$withGPU" = "ON" ] && [ "$gpuLang" = "cuda" ]; then
    cmake -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_COMPILER=$cxx_compiler\
    -DCMAKE_CXX_FLAGS="$cxx_flags"\
    -DCMAKE_CXX_FLAGS_RELEASE="$cxx_flagsRelease" \
    -DCMAKE_BUILD_TYPE=$build_type -DDEAL_II_DIR=$dealiiDir \
    -DALGLIB_DIR=$alglibDir -DLIBXC_DIR=$libxcDir \
    -DSPGLIB_DIR=$spglibDir -DXML_LIB_DIR=$xmlLibDir \
    -DXML_INCLUDE_DIR=$xmlIncludeDir\
    -DWITH_MDI=$withMDI -DMDI_PATH=$mdiPath \
    -DWITH_DCCL=$withDCCL -DCMAKE_PREFIX_PATH="$ELPA_PATH;$DCCL_PATH"\
    -DWITH_COMPLEX=ON -DWITH_GPU=$withGPU -DGPU_LANG=$gpuLang -DGPU_VENDOR=$gpuVendor -DWITH_GPU_AWARE_MPI=$withGPUAwareMPI -DCMAKE_CUDA_FLAGS="$device_flags" -DCMAKE_CUDA_ARCHITECTURES="$device_architectures"\
    -DCMAKE_SHARED_LINKER_FLAGS="-L$MPICH_DIR/lib -lmpich"\
    -DWITH_TESTING=$testing -DNUMDIFF_EXECUTABLE=$numdiffExecutable \
    -DHIGHERQUAD_PSP=$withHigherQuadPSP -DUSE_64BIT_INT=$useInt64 $1
  else
    cmake -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_COMPILER=$cxx_compiler\
    -DCMAKE_CXX_FLAGS="$cxx_flags"\
    -DCMAKE_CXX_FLAGS_RELEASE="$cxx_flagsRelease" \
    -DCMAKE_BUILD_TYPE=$build_type -DDEAL_II_DIR=$dealiiDir \
    -DALGLIB_DIR=$alglibDir -DLIBXC_DIR=$libxcDir \
    -DSPGLIB_DIR=$spglibDir -DXML_LIB_DIR=$xmlLibDir \
    -DXML_INCLUDE_DIR=$xmlIncludeDir\
    -DWITH_MDI=$withMDI -DMDI_PATH=$mdiPath \
    -DWITH_DCCL=OFF -DCMAKE_PREFIX_PATH="$ELPA_PATH"\
    -DWITH_COMPLEX=ON -DWITH_GPU=OFF \
    -DWITH_TESTING=$testing -DNUMDIFF_EXECUTABLE=$numdiffExecutable \
    -DHIGHERQUAD_PSP=$withHigherQuadPSP -DUSE_64BIT_INT=$useInt64 $1
  fi
  cd ..
}

RCol='\e[0m'
Blu='\e[0;34m';
if [ -d "$out" ]; then # build directory exists
    echo -e "${Blu}$out directory already present${RCol}"
else
    mkdir -p "$out"
    echo -e "${Blu}Creating $out ${RCol}"
fi

cd $out

echo -e "${Blu}Building Real executable ($out) in $build_type mode...${RCol}"
cmake_real "$SRC" && (cd real && make -j8)
cd ..

echo -e "${Blu}Building Complex executable ($out) in $build_type mode...${RCol}"
cmake_cplx "$SRC" && (cd complex && make -j8)
cd ..

echo -e "${Blu}Build complete.${RCol}"
