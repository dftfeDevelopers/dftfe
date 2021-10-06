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

#Paths for external libraries
dealiiDir="/home/vikramg/DFT-FE-softwares/dealiiDevCustomized/install_gcc8.2.0_openmpi4.0.6_cuda11.4.0_gpu_10.0-pre"
alglibDir="/home/vikramg/DFT-FE-softwares/alglibGCC/cpp/src"
libxcDir="/home/vikramg/DFT-FE-softwares/libxc/install_libxc5.1.5"
spglibDir="/home/vikramg/DFT-FE-softwares/spglibGCC"
xmlIncludeDir="/usr/include/libxml2"
xmlLibDir="/usr/lib64"
ELPA_PATH="/home/vikramg/DFT-FE-softwares/elpa/elpa2021.05.002_gcc8.2.0_openmpi4.0.6_install"


#Toggle GPU compilation
withGPU=ON

#Compiler options and flags
cxx_compiler=mpicxx
cxx_flagsRelease="-O2 -fPIC"

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
	-DCMAKE_PREFIX_PATH="$ELPA_PATH" \
	-DWITH_COMPLEX=OFF -DWITH_GPU=$withGPU -DCMAKE_CUDA_FLAGS="-arch=sm_70"\
	-DWITH_TESTING=$testing -DMINIMAL_COMPILE=$minimal_compile\
  -DCMAKE_PREFIX_PATH="$ELPA_PATH" \
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
  -DWITH_COMPLEX=ON -DWITH_GPU=$withGPU -DCMAKE_CUDA_FLAGS="-arch=sm_70"\
  -DWITH_TESTING=$testing -DMINIMAL_COMPILE=$minimal_compile\
  -DCMAKE_PREFIX_PATH="$ELPA_PATH" \
  -DHIGHERQUAD_PSP=$withHigherQuadPSP $1
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
cmake_real "$SRC" && make -j4
cd ..

echo -e "${Blu}Building Complex executable in $build_type mode...${RCol}"
cmake_cplx "$SRC" && make -j4
cd ..

echo -e "${Blu}Build complete.${RCol}"
