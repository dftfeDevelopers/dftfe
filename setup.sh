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
dealiiPetscRealDir="/home/vikramg/DFT-FE-softwares/dealiiDevCustomized/install_gcc8.2.0_openmpi3.1.4_real_cpu2"
dealiiPetscComplexDir="/home/vikramg/DFT-FE-softwares/dealiiDevCustomized/install_gcc8.2.0_openmpi3.1.4_complex_cpu2"
alglibDir="/home/vikramg/DFT-FE-softwares/alglibGCC/cpp/src"
libxcDir="/home/vikramg/DFT-FE-softwares/libxc/gcc8.2.0_libxc_4.3.4"
spglibDir="/home/vikramg/DFT-FE-softwares/spglibGCC"
xmlIncludeDir="/usr/include/libxml2"
xmlLibDir="/usr/lib64"
ELPA_PATH="/home/vikramg/DFT-FE-softwares/elpa/elpa2020.05.001_gcc8.2.0_openmpi3.1.4_install_noopenmp2"


#Compiler options and flags
cxx_compiler=mpicxx
cxx_flagsRelease="-O0 -g -fPIC -fopenmp"

#Option to link to ELPA
withELPA=ON

#Option to compile with default or higher order quadrature for storing pseudopotential data
#ON is recommended for MD simulations with hard pseudopotentials
withHigherQuadPSP=OFF

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
  cmake -DCMAKE_CXX_COMPILER=$cxx_compiler \
	-DCMAKE_CXX_FLAGS_RELEASE="$cxx_flagsRelease" \
	-DCMAKE_BUILD_TYPE=$build_type -DDEAL_II_DIR=$dealiiPetscRealDir \
	-DALGLIB_DIR=$alglibDir -DLIBXC_DIR=$libxcDir \
	-DSPGLIB_DIR=$spglibDir -DXML_LIB_DIR=$xmlLibDir \
	-DXML_INCLUDE_DIR=$xmlIncludeDir \
	-DWITH_ELPA=$withELPA -DCMAKE_PREFIX_PATH="$ELPA_PATH" \
	-DWITH_COMPLEX=OFF -DWITH_GPU=$withGPU \
	-DWITH_TESTING=$testing -DMINIMAL_COMPILE=$minimal_compile \
  -DHIGHERQUAD_PSP=$withHigherQuadPSP\
	  $1
}

function cmake_cplx() {
  mkdir -p complex && cd complex
  cmake -DCMAKE_CXX_COMPILER=$cxx_compiler \
	-DCMAKE_CXX_FLAGS_RELEASE="$cxx_flagsRelease" \
	-DCMAKE_BUILD_TYPE=$build_type -DDEAL_II_DIR=$dealiiPetscComplexDir \
	-DALGLIB_DIR=$alglibDir -DLIBXC_DIR=$libxcDir \
	-DSPGLIB_DIR=$spglibDir -DXML_LIB_DIR=$xmlLibDir \
	-DXML_INCLUDE_DIR=$xmlIncludeDir \
	-DWITH_ELPA=$withELPA -DCMAKE_PREFIX_PATH="$ELPA_PATH" \
	-DWITH_COMPLEX=ON \
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
cmake_real "$SRC" && make -j4
cd ..

echo -e "${Blu}Building Complex executable in $build_type mode...${RCol}"
cmake_cplx "$SRC" && make -j4
cd ..

echo -e "${Blu}Build complete.${RCol}"
