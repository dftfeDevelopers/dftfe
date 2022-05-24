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
dealiiDir="/global/common/software/m3916/softwareDFTFE/dealii/install"
alglibDir="/global/common/software/m3916/softwareDFTFE/alglib/alglib-cpp/src"
libxcDir="/global/common/software/m3916/softwareDFTFE/libxc/install"
spglibDir="/global/common/software/m3916/softwareDFTFE/spglib/install"
xmlIncludeDir="/global/common/software/m3916/softwareDFTFE/libxml2/install/include/libxml2"
xmlLibDir="/global/common/software/m3916/softwareDFTFE/libxml2/install/lib"
ELPA_PATH="/global/common/software/m3916/softwareDFTFE/elpa/install_elpa-2021.05.002_gcc11"

#Paths for optional external libraries
NCCL_PATH="$NCCL_DIR"

#Toggle GPU compilation
withGPU=ON

#Option to link to NCCL library (Only for GPU compilation)
withNCCL=ON

#Compiler options and flags
cxx_compiler=CC
cxx_flagsRelease="-O2 -fPIC -target-accel=nvidia80"
cuda_flags="-I$MPICH_DIR/include -L$MPICH_DIR/lib -lmpich -arch=sm_80" #only applicable for withGPU=ON

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
  cmake -DCMAKE_INSTALL_PREFIX:PATH=/global/common/software/m3916/softwareDFTFE/dftfe/installReal \
        -DCMAKE_CXX_COMPILER=$cxx_compiler \
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
  cmake -DCMAKE_INSTALL_PREFIX:PATH=/global/common/software/m3916/softwareDFTFE/dftfe/installReal \
        -DCMAKE_CXX_COMPILER=$cxx_compiler \
	-DCMAKE_CXX_FLAGS_RELEASE="$cxx_flagsRelease" \
	-DCMAKE_BUILD_TYPE=$build_type -DDEAL_II_DIR=$dealiiDir \
	-DALGLIB_DIR=$alglibDir -DLIBXC_DIR=$libxcDir \
	-DSPGLIB_DIR=$spglibDir -DXML_LIB_DIR=$xmlLibDir \
	-DXML_INCLUDE_DIR=$xmlIncludeDir \
  -DWITH_NCCL=$withNCCL -DCMAKE_PREFIX_PATH="$ELPA_PATH;$NCCL_PATH" \
	-DWITH_COMPLEX=ON -DWITH_GPU=$withGPU -DCMAKE_CUDA_FLAGS="$cuda_flags"\
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
cmake_real "$SRC" && make -j8 && make install
cd ..

echo -e "${Blu}Building Complex executable in $build_type mode...${RCol}"
cmake_cplx "$SRC" && make -j8 && make install
cd ..

echo -e "${Blu}Build complete.${RCol}"
