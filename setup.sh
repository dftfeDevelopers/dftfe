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
dealiiPetscRealDir="/home/vikramg/DFT-FE-softwares/dealiinew/intel18.0.5_dealiicustom_real"
dealiiPetscComplexDir="/home/vikramg/DFT-FE-softwares/dealiinew/intel18.0.5_dealiicustom_complex"
alglibDir="/home/vikramg/DFT-FE-softwares/alglib/cpp/src"
libxcDir="/home/vikramg/DFT-FE-softwares/libxc/intel2018_libxc_4.3.4"
spglibDir="/home/vikramg/DFT-FE-softwares/spglib"
xmlIncludeDir="/usr/include/libxml2"
xmlLibDir="/usr/lib64"


#If you have installed dealii by linking with intel mkl library set underlying flag to "ON",
#otherwise set it to "OFF"
withIntelMkl=ON

#Compiler options and flags
c_compiler=mpicc
cxx_compiler=mpicxx
c_flagsRelease="-O3 -fPIC -fopenmp -xCORE-AVX512 -qopt-zmm-usage=high"
cxx_flagsRelease="-O3 -fPIC -fopenmp -xCORE-AVX512 -qopt-zmm-usage=high"

#Option to link to ELPA
withELPA=OFF

#Optmization flag: 1 for optimized mode and 0 for debug mode compilation
build_type=Release

testing=ON
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
	-DCMAKE_BUILD_TYPE=$build_type -DDEAL_II_DIR=$dealiiPetscRealDir \
	-DALGLIB_DIR=$alglibDir -DLIBXC_DIR=$libxcDir \
	-DSPGLIB_DIR=$spglibDir -DXML_LIB_DIR=$xmlLibDir \
	-DXML_INCLUDE_DIR=$xmlIncludeDir -DWITH_INTEL_MKL=$withIntelMkl \
	-DWITH_ELPA=$withELPA -DCMAKE_PREFIX_PATH="$PREFIX_PATH" \
	-DWITH_COMPLEX=OFF -DWITH_GPU=$withGPU \
	-DWITH_TESTING=$testing -DMINIMAL_COMPILE=$minimal_compile\
	  $1
}

function cmake_cplx() {
  mkdir -p complex && cd complex
  cmake -DCMAKE_C_COMPILER=$c_compiler -DCMAKE_CXX_COMPILER=$cxx_compiler \
	-DCMAKE_CXX_FLAGS_RELEASE="$cxx_flagsRelease" \
	-DCMAKE_C_FLAGS_RELEASE="$c_flagsRelease" \
	-DCMAKE_BUILD_TYPE=$build_type -DDEAL_II_DIR=$dealiiPetscComplexDir \
	-DALGLIB_DIR=$alglibDir -DLIBXC_DIR=$libxcDir \
	-DSPGLIB_DIR=$spglibDir -DXML_LIB_DIR=$xmlLibDir \
	-DXML_INCLUDE_DIR=$xmlIncludeDir -DWITH_INTEL_MKL=$withIntelMkl \
	-DWITH_COMPLEX=ON -DWITH_TESTING=$testing -DMINIMAL_COMPILE=$minimal_compile\
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
cmake_real "$SRC" && make -j4
cd ..

echo -e "${Blu}Building Complex executable in $build_type mode...${RCol}"
cmake_cplx "$SRC" && make -j4
cd ..

echo -e "${Blu}Build complete.${RCol}"
