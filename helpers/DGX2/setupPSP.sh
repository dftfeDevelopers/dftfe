#!/bin/bash
# script to setup and build DFT-FE.

set -e
set -o pipefail

########################################################################
#Provide paths below for external libraries, compiler options and flags,
# and optimization flag

#Paths for external libraries
dealiiDir="/home/nv-tb-03/software/dealii/installMpich"
alglibDir="/home/nv-tb-03/software/alglib/cpp/src"
libxcDir="/home/nv-tb-03/software/libxc/libxc-4.3.4/install"
spglibDir="/home/nv-tb-03/software/spglib/install"
xmlIncludeDir="/usr/include/libxml2"
xmlLibDir="/usr/lib/x86_64-linux-gnu"

# Path to project source (should be script run directory)
SRC="$PWD"

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
withELPA=OFF

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
out=`echo "build/$build_type" | tr '[:upper:]' '[:lower:]'`

function cmake_real() {
  mkdir -p real && cd real
  cmake -DCMAKE_C_COMPILER=$c_compiler -DCMAKE_CXX_COMPILER=$cxx_compiler \
	-DCMAKE_CXX_FLAGS_RELEASE="$cxx_flagsRelease" \
	-DCMAKE_C_FLAGS_RELEASE="$c_flagsRelease" \
	-DCMAKE_BUILD_TYPE=$build_type -DDEAL_II_DIR=$dealiiDir \
	-DALGLIB_DIR=$alglibDir -DLIBXC_DIR=$libxcDir \
	-DSPGLIB_DIR=$spglibDir -DXML_LIB_DIR=$xmlLibDir \
	-DXML_INCLUDE_DIR=$xmlIncludeDir -DWITH_INTEL_MKL=$withIntelMkl \
	-DWITH_ELPA=$withELPA -DELPA_LIB_DIR=$elpaLibDir \
	-DELPA_INCLUDE_DIR=$elpaIncludeDir \
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
if [ -d "$out" ]; then
    echo -e "${Blu}$out directory already present${RCol}"
    # Control will enter here if build directory exists.
else
    rm -rf "$out"
    echo -e "${Blu}Creating $out ${RCol}"
    mkdir -p "$out"
fi

wd="$PWD"
cd "$wd/$out"
echo -e "${Blu}Building Real executable in $build_type mode...${RCol}"
cmake_real "$SRC" && make -j4

cd "$wd/$out"
echo -e "${Blu}Building Complex executable in $build_type mode...${RCol}"
cmake_cplx "$SRC" && make -j4

echo -e "${Blu}Build complete.${RCol}"
