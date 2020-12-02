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

PROJ=/ccs/proj/eng110

. $PROJ/setup-env.sh
. $PROJ/venvs/summit/bin/activate # for building docs

########################################################################
#Provide paths below for external libraries, compiler options and flags,
# and optimization flag

#Paths for external libraries
dealiiDir="$PROJ/software/dealiiDevelop/installDealii"
alglibDir="$PROJ/software/alglib/cpp/src"
libxcDir="$PROJ/software/libxc/installGcc6.4.0"
spglibDir="$PROJ/software/spglib/installGcc6.4.0"
xmlIncludeDir="/usr/include/libxml2"
xmlLibDir="/usr/lib64"
ELPA_PATH="$PROJ/software/elpaSMPI20200121/installElpa2020Gcc6.4.0CUDA10"
NCCL_PATH="$PROJ/software/nccl/ncclDevelopBranch/build"


#If you have installed dealii by linking with intel mkl library set underlying flag to "ON",
#otherwise set it to "OFF"
withIntelMkl=OFF

#Toggle GPU compilation
withGPU=ON

#Option to link to NCCL library
withNCCL=ON

#Compiler options and flags
c_compiler=mpicc
cxx_compiler=mpicxx
c_flagsRelease="-O2 -fPIC -fopenmp"
cxx_flagsRelease="-O2 -fPIC -fopenmp"

#Option to link to ELPA
withELPA=ON

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
  cmake -DCMAKE_C_COMPILER=$c_compiler -DCMAKE_CXX_COMPILER=$cxx_compiler \
	-DCMAKE_CXX_FLAGS_RELEASE="$cxx_flagsRelease" \
	-DCMAKE_C_FLAGS_RELEASE="$c_flagsRelease" \
	-DCMAKE_BUILD_TYPE=$build_type -DDEAL_II_DIR=$dealiiDir \
	-DALGLIB_DIR=$alglibDir -DLIBXC_DIR=$libxcDir \
	-DSPGLIB_DIR=$spglibDir -DXML_LIB_DIR=$xmlLibDir \
	-DXML_INCLUDE_DIR=$xmlIncludeDir -DWITH_INTEL_MKL=$withIntelMkl \
  -DWITH_ELPA=$withELPA -DWITH_NCCL=$withNCCL -DCMAKE_PREFIX_PATH="$ELPA_PATH;$NCCL_PATH" \
	-DWITH_COMPLEX=OFF -DWITH_GPU=$withGPU \
	-DWITH_TESTING=$testing -DMINIMAL_COMPILE=$minimal_compile \
  -DHIGHERQUAD_PSP=$withHigherQuadPSP $1
}

function cmake_cplx() {
  mkdir -p complex && cd complex
  cmake -DCMAKE_C_COMPILER=$c_compiler -DCMAKE_CXX_COMPILER=$cxx_compiler \
	-DCMAKE_CXX_FLAGS_RELEASE="$cxx_flagsRelease" \
	-DCMAKE_C_FLAGS_RELEASE="$c_flagsRelease" \
	-DCMAKE_BUILD_TYPE=$build_type -DDEAL_II_DIR=$dealiiDir \
	-DALGLIB_DIR=$alglibDir -DLIBXC_DIR=$libxcDir \
	-DSPGLIB_DIR=$spglibDir -DXML_LIB_DIR=$xmlLibDir \
	-DXML_INCLUDE_DIR=$xmlIncludeDir -DWITH_INTEL_MKL=$withIntelMkl \
	-DWITH_COMPLEX=ON -DWITH_TESTING=$testing -DMINIMAL_COMPILE=$minimal_compile\
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
cmake_real "$SRC" && make -j8
cd ..

echo -e "${Blu}Building Complex executable in $build_type mode...${RCol}"
cmake_cplx "$SRC" && make -j8
cd ..

echo -e "${Blu}Build complete.${RCol}"
