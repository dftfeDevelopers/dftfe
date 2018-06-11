#!/bin/bash
set -e
set -o pipefail
#script to setup and build DFT-FE
#Provide paths for external libraries and optimization flag (0 for Debug, 1 for Release)
#dealiiPetscRealDir="/project/projectdirs/m2360/softwaresDFTFE/dealii/intel18_petscDouble"
#dealiiPetscComplexDir="/project/projectdirs/m2360/softwaresDFTFE/dealii/intel18_petscComplex"
dealiiPetscRealDir="/project/projectdirs/m2360/softwaresDFTFE/dealiiKNL/intel18_petscReal_64Bit_scalapack"
dealiiPetscComplexDir="/project/projectdirs/m2360/softwaresDFTFE/dealiiKNL/intel18_petscComplex_64Bit_scalapack"
alglibDir="/project/projectdirs/m2360/softwaresDFTFE/alglib/cpp/src"
libxcDir="/project/projectdirs/m2360/softwaresDFTFE/libxc/install_libxc2.2.0_intel18"
spglibDir="/project/projectdirs/m2360/softwaresDFTFE/spglib"
optimizedMode=1
#
#Usually, no changes are needed below this line
#
RCol='\e[0m'
Blu='\e[0;34m';
if [ $optimizedMode == 1 ]; then
  if [ -d "build/release" ]; then
    echo -e "${Blu}build/release directory already present${RCol}"
    # Control will enter here if build directory exists.
    cd build
    cd release
    echo -e "${Blu}Building Non-Periodic executable in Optimized (Release) mode...${RCol}"
    mkdir -p nonPeriodic && cd nonPeriodic && cmake -DCMAKE_C_COMPILER=cc -DCMAKE_CXX_COMPILER=CC -DCMAKE_CXX_FLAGS_RELEASE="-O3" -DCMAKE_BUILD_TYPE=Release -DDEAL_II_DIR=$dealiiPetscRealDir -DALGLIB_DIR=$alglibDir -DLIBXC_DIR=$libxcDir -DSPGLIB_DIR=$spglibDir ../../../. && make && cd ..
    echo -e "${Blu}Building Periodic executable in Optimized (Release) mode...${RCol}"
    mkdir -p periodic && cd periodic && cmake -DCMAKE_C_COMPILER=cc -DCMAKE_CXX_COMPILER=CC -DCMAKE_CXX_FLAGS_RELEASE="-O3" -DCMAKE_BUILD_TYPE=Release -DDEAL_II_DIR=$dealiiPetscComplexDir -DALGLIB_DIR=$alglibDir -DLIBXC_DIR=$libxcDir -DSPGLIB_DIR=$spglibDir  ../../../. && make && cd ../..
  else
    rm -rf build/release
    echo -e "${Blu}Creating build directory...${RCol}"
    mkdir -p build && cd build
    mkdir -p release && cd release
    echo -e "${Blu}Building Non-Periodic executable in Optimized (Release) mode...${RCol}"
    mkdir -p nonPeriodic && cd nonPeriodic && cmake -DCMAKE_C_COMPILER=cc -DCMAKE_CXX_COMPILER=CC -DCMAKE_CXX_FLAGS_RELEASE="-O3" -DCMAKE_BUILD_TYPE=Release -DDEAL_II_DIR=$dealiiPetscRealDir -DALGLIB_DIR=$alglibDir -DLIBXC_DIR=$libxcDir -DSPGLIB_DIR=$spglibDir ../../../. && make && cd ..
    echo -e "${Blu}Building Periodic executable in Optimized (Release) mode...${RCol}"
    mkdir -p periodic && cd periodic && cmake -DCMAKE_C_COMPILER=cc -DCMAKE_CXX_COMPILER=CC -DCMAKE_CXX_FLAGS_RELEASE="-O3" -DCMAKE_BUILD_TYPE=Release -DDEAL_II_DIR=$dealiiPetscComplexDir -DALGLIB_DIR=$alglibDir -DLIBXC_DIR=$libxcDir -DSPGLIB_DIR=$spglibDir ../../../. && make && cd ../..
  fi
else
  if [ -d "build/debug" ]; then
    echo -e "${Blu}build/debug directory already present${RCol}"
    # Control will enter here if build directory exists.
    cd build
    cd debug
    echo -e "${Blu}Building Non-Periodic executable in Debug mode...${RCol}"
    mkdir -p nonPeriodic && cd nonPeriodic && cmake -DCMAKE_C_COMPILER=cc -DCMAKE_CXX_COMPILER=CC -DCMAKE_BUILD_TYPE=Debug -DDEAL_II_DIR=$dealiiPetscRealDir -DALGLIB_DIR=$alglibDir -DLIBXC_DIR=$libxcDir -DSPGLIB_DIR=$spglibDir ../../../. && make && cd ..
    echo -e "${Blu}Building Periodic executable in Debug mode...${RCol}"
    mkdir -p periodic && cd periodic && cmake -DCMAKE_C_COMPILER=cc -DCMAKE_CXX_COMPILER=CC -DCMAKE_BUILD_TYPE=Debug -DDEAL_II_DIR=$dealiiPetscComplexDir -DALGLIB_DIR=$alglibDir -DLIBXC_DIR=$libxcDir -DSPGLIB_DIR=$spglibDir  ../../../. && make && cd ../..
  else
    rm -rf build/debug
    echo -e "${Blu}Creating build directory...${RCol}"
    mkdir -p build && cd build
    mkdir -p debug && cd debug
    echo -e "${Blu}Building Non-Periodic executable in Debug mode...${RCol}"
    mkdir -p nonPeriodic && cd nonPeriodic && cmake -DCMAKE_C_COMPILER=cc -DCMAKE_CXX_COMPILER=CC -DCMAKE_BUILD_TYPE=Debug -DDEAL_II_DIR=$dealiiPetscRealDir -DALGLIB_DIR=$alglibDir -DLIBXC_DIR=$libxcDir -DSPGLIB_DIR=$spglibDir ../../../. && make && cd ..
    echo -e "${Blu}Building Periodic executable in Debug mode...${RCol}"
    mkdir -p periodic && cd periodic && cmake -DCMAKE_C_COMPILER=cc -DCMAKE_CXX_COMPILER=CC -DCMAKE_BUILD_TYPE=Debug -DDEAL_II_DIR=$dealiiPetscComplexDir -DALGLIB_DIR=$alglibDir -DLIBXC_DIR=$libxcDir -DSPGLIB_DIR=$spglibDir  ../../../. && make && cd ../..
  fi
fi
echo -e "${Blu}Build complete.${RCol}"
