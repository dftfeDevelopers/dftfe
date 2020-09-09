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
dealiiDir="$PROJ/software/dealiiCustomizedSMPI20200121/installDealiiCustomizedCUDA10.1.243Gcc6.4.0"
alglibDir="$PROJ/software/alglib/cpp/src"
libxcDir="$PROJ/software/libxc/installGcc6.4.0"
spglibDir="$PROJ/software/spglib/installGcc6.4.0"
xmlIncludeDir="/usr/include/libxml2"
xmlLibDir="/usr/lib64"
#elpaIncludeDir="$PROJ/software/elpaSMPI20200121/installElpa2020Gcc6.4.0CUDA10/include/elpa-2020.05.001.rc1"
#elpaLibDir="$PROJ/software/elpaSMPI20200121/installElpa2020Gcc6.4.0CUDA10/lib"
#elpaIncludeDir="$PROJ/Rogers_Tests/elpaSMPI202005/include/elpa_openmp-2020.05.001.rc1"
#elpaLibDir="$PROJ/Rogers_Tests/elpaSMPI202005/lib"
PREFIX_PATH="$PROJ/software/elpaSMPI20200121/installElpa2020Gcc6.4.0CUDA10"

#If you have installed dealii by linking with intel mkl library set underlying flag to "ON",
#otherwise set it to "OFF"
withIntelMkl=OFF

#Toggle GPU compilation
withGPU=ON

#Compiler options and flags
c_compiler=mpicc
cxx_compiler=mpicxx
c_flagsRelease="-O2 -fPIC -fopenmp"
cxx_flagsRelease="-O2 -fPIC -fopenmp"

#Option to link to ELPA
withELPA=ON

# build type: "Release" or "Debug"
build_type=Release
testing=OFF

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
	-DWITH_ELPA=$withELPA -DCMAKE_PREFIX_PATH="$PREFIX_PATH" \
	-DWITH_COMPLEX=OFF -DWITH_GPU=$withGPU \
	-DWITH_TESTING=$testing \
	  $1
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
	-DWITH_COMPLEX=ON -DWITH_TESTING=$testing \
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
