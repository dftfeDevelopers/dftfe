#!/bin/bash
set -e
set -o pipefail
#script to setup and build DFT-FE
#Provide paths for external libraries and optimization flag (0 for Debug, 1 for Release)
dealiiPetscRealDir="/home/vikramg/DFT-FE-softwares/softwareCentos/dealiiDev/intel_18.0.1_petscReal_noavx"
dealiiPetscComplexDir="/home/vikramg/DFT-FE-softwares/softwareCentos/dealiiDev/intel_18.0.1_petscComplex_noavx"
#dealiiPetscRealDir="/home/vikramg/DFT-FE-softwares/softwareCentos/dealii8.5.1/intel_18.0.1_petscDouble_noavx"
#dealiiPetscComplexDir="/home/vikramg/DFT-FE-softwares/softwareCentos/dealii8.5.1/intel_18.0.1_petscComplex_noavx"
alglibDir="/nfs/mcfs_comp/home/rudraa/software/alglib/cpp/src"
libxcDir="/home/vikramg/DFT-FE-softwares/softwareCentos/libxcNew/install_intel18"
spglibDir="/home/vikramg/DFT-FE-softwares/softwareCentos/spglib"
optimizedMode=0
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
    mkdir -p nonPeriodic && cd nonPeriodic && cmake -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_CXX_FLAGS_RELEASE="-O3" -DCMAKE_BUILD_TYPE=Release -DDEAL_II_DIR=$dealiiPetscRealDir -DALGLIB_DIR=$alglibDir -DLIBXC_DIR=$libxcDir -DSPGLIB_DIR=$spglibDir ../../../. && make && cd ..
    echo -e "${Blu}Building Periodic executable in Optimized (Release) mode...${RCol}"
    mkdir -p periodic && cd periodic && cmake -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_CXX_FLAGS_RELEASE="-O3" -DCMAKE_BUILD_TYPE=Release -DDEAL_II_DIR=$dealiiPetscComplexDir -DALGLIB_DIR=$alglibDir -DLIBXC_DIR=$libxcDir -DSPGLIB_DIR=$spglibDir  ../../../. && make && cd ../..
  else
    rm -rf build/release
    echo -e "${Blu}Creating build directory...${RCol}"
    mkdir -p build && cd build
    mkdir -p release && cd release
    echo -e "${Blu}Building Non-Periodic executable in Optimized (Release) mode...${RCol}"
    mkdir -p nonPeriodic && cd nonPeriodic && cmake -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_CXX_FLAGS_RELEASE="-O3" -DCMAKE_BUILD_TYPE=Release -DDEAL_II_DIR=$dealiiPetscRealDir -DALGLIB_DIR=$alglibDir -DLIBXC_DIR=$libxcDir -DSPGLIB_DIR=$spglibDir ../../../. && make && cd ..
    echo -e "${Blu}Building Periodic executable in Optimized (Release) mode...${RCol}"
    mkdir -p periodic && cd periodic && cmake -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_CXX_FLAGS_RELEASE="-O3" -DCMAKE_BUILD_TYPE=Release -DDEAL_II_DIR=$dealiiPetscComplexDir -DALGLIB_DIR=$alglibDir -DLIBXC_DIR=$libxcDir -DSPGLIB_DIR=$spglibDir ../../../. && make && cd ../..
  fi
else
  if [ -d "build/debug" ]; then
    echo -e "${Blu}build/debug directory already present${RCol}"
    # Control will enter here if build directory exists.
    cd build
    cd debug
    echo -e "${Blu}Building Non-Periodic executable in Debug mode...${RCol}"
    mkdir -p nonPeriodic && cd nonPeriodic && cmake -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_BUILD_TYPE=Debug -DDEAL_II_DIR=$dealiiPetscRealDir -DALGLIB_DIR=$alglibDir -DLIBXC_DIR=$libxcDir -DSPGLIB_DIR=$spglibDir ../../../. && make && cd ..
    echo -e "${Blu}Building Periodic executable in Debug mode...${RCol}"
    mkdir -p periodic && cd periodic && cmake -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_BUILD_TYPE=Debug -DDEAL_II_DIR=$dealiiPetscComplexDir -DALGLIB_DIR=$alglibDir -DLIBXC_DIR=$libxcDir -DSPGLIB_DIR=$spglibDir  ../../../. && make && cd ../..
  else
    rm -rf build/debug
    echo -e "${Blu}Creating build directory...${RCol}"
    mkdir -p build && cd build
    mkdir -p debug && cd debug
    echo -e "${Blu}Building Non-Periodic executable in Debug mode...${RCol}"
    mkdir -p nonPeriodic && cd nonPeriodic && cmake -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_BUILD_TYPE=Debug -DDEAL_II_DIR=$dealiiPetscRealDir -DALGLIB_DIR=$alglibDir -DLIBXC_DIR=$libxcDir -DSPGLIB_DIR=$spglibDir ../../../. && make && cd ..
    echo -e "${Blu}Building Periodic executable in Debug mode...${RCol}"
    mkdir -p periodic && cd periodic && cmake -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_BUILD_TYPE=Debug -DDEAL_II_DIR=$dealiiPetscComplexDir -DALGLIB_DIR=$alglibDir -DLIBXC_DIR=$libxcDir -DSPGLIB_DIR=$spglibDir  ../../../. && make && cd ../..
  fi
fi
echo -e "${Blu}Build complete.${RCol}"
