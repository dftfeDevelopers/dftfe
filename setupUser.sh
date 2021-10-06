#!/bin/bash
# script to setup and build DFT-FE.

set -e
set -o pipefail

if [ -s CMakeLists.txt ]; then
    echo "This script must be run from the build directory!"
    exit 1
fi

# Path to project source
SRC=`dirname $0` # location of source directory

########################################################################
#Provide paths below for external libraries, compiler options and flags,
# and optimization flag

#Paths for required external libraries
dealiiDir="/global/project/projectdirs/m1759/dsambit/softwaresDFTFE/dealiiDevCustomized/installGcc8.3CUDA11.1.1Mklscalapack"
alglibDir="/global/project/projectdirs/m1759/dsambit/softwaresDFTFE/alglib/cpp/src"
libxcDir="/global/project/projectdirs/m1759/dsambit/softwaresDFTFE/libxc/installGcc8.3.0Libxc5.1.3"
spglibDir="/global/project/projectdirs/m1759/dsambit/softwaresDFTFE/spglib/installGcc8.3.0"
xmlIncludeDir="/usr/include/libxml2"
xmlLibDir="/usr/lib64"
ELPA_PATH="/global/project/projectdirs/m1759/dsambit/softwaresDFTFE/elpa/install-elpa-2021.05.002-cuda"

#Paths for optional external libraries
NCCL_PATH="/global/project/projectdirs/m1759/dsambit/softwaresDFTFE/nccl/build"

#Toggle GPU compilation
withGPU=ON

#Option to link to NCCL library (Only for GPU compilation)
withNCCL=ON

#Compiler options and flags
cxx_compiler=mpic++
cxx_flagsRelease="-O2 -fPIC"
cuda_flags="-arch=sm_70" #only applicable for withGPU=ON

#Option to compile with default or higher order quadrature for storing pseudopotential data
#ON is recommended for MD simulations with hard pseudopotentials
withHigherQuadPSP=OFF

# build type: "Release" or "Debug"
build_type=Release

testing=OFF
minimal_compile=ON
###########################################################################
#Usually, no changes are needed below this line
#

#if [[ x"$build_type" == x"Release" ]]; then
#  c_flags="$c_flagsRelease"
#  cxx_flags="$c_flagsRelease"
#else
#fi
out=`echo "$build_type" | tr '[:upper:]' '[:lower:]'`

function cmake_real() {
  mkdir -p real && cd real
  cmake -DCMAKE_CXX_COMPILER=$cxx_compiler \
	-DCMAKE_CXX_FLAGS_RELEASE="$cxx_flagsRelease" \
	-DCMAKE_BUILD_TYPE=$build_type -DDEAL_II_DIR=$dealiiDir \
	-DALGLIB_DIR=$alglibDir -DLIBXC_DIR=$libxcDir \
	-DSPGLIB_DIR=$spglibDir -DXML_LIB_DIR=$xmlLibDir \
	-DXML_INCLUDE_DIR=$xmlIncludeDir\
	-DWITH_NCCL=$withNCCL -DCMAKE_PREFIX_PATH="$ELPA_PATH;$NCCL_PATH"\
	-DWITH_COMPLEX=OFF -DWITH_GPU=$withGPU -DCMAKE_CUDA_FLAGS="$cuda_flags"\
	-DWITH_TESTING=$testing -DMINIMAL_COMPILE=$minimal_compile\
	-DHIGHERQUAD_PSP=$withHigherQuadPSP $1
}

function cmake_cplx() {
  mkdir -p complex && cd complex
  cmake -DCMAKE_CXX_COMPILER=$cxx_compiler \
	-DCMAKE_CXX_FLAGS_RELEASE="$cxx_flagsRelease" \
	-DCMAKE_BUILD_TYPE=$build_type -DDEAL_II_DIR=$dealiiDir \
	-DALGLIB_DIR=$alglibDir -DLIBXC_DIR=$libxcDir \
	-DSPGLIB_DIR=$spglibDir -DXML_LIB_DIR=$xmlLibDir \
	-DXML_INCLUDE_DIR=$xmlIncludeDir \
  -DWITH_NCCL=$withNCCL -DCMAKE_PREFIX_PATH="$ELPA_PATH;$NCCL_PATH" \
	-DWITH_COMPLEX=ON -DWITH_GPU=$withGPU -DCMAKE_CUDA_FLAGS="-arch=sm_70"\
	-DWITH_TESTING=$testing -DMINIMAL_COMPILE=$minimal_compile \
  -DHIGHERQUAD_PSP=$withHigherQuadPSP\
	  $1
}

RCol='\e[0m'
Blu='\e[0;34m';
if [ -d "$out" ]; then # build directory exists
    echo -e "${Blu}$out directory already present${RCol}"
else
    rm -rf "$out"
    echo -e "${Blu}Creating $out ${RCol}"
    mkdir -p "$out"
fi

cd $out

echo -e "${Blu}Building Real executable in $build_type mode...${RCol}"
cmake_real "$SRC" && make -j8
cd ..

echo -e "${Blu}Building Complex executable in $build_type mode...${RCol}"
cmake_cplx "$SRC" && make -j8
cd ..

echo -e "${Blu}Build complete.${RCol}"
