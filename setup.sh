#!/bin/bash
set -e
set -o pipefail
#script to setup and build DFT-FE 
#set CMAKE path
cmake=/usr/bin/cmake
#
#Usually, no changes are needed below this line
#
RCol='\e[0m'
Blu='\e[0;34m';
optimizedMode=0
if [ $optimizedMode == 1 ]; then
  if [ -d "build/release" ]; then
    echo -e "${Blu}build/release directory already present${RCol}"
    # Control will enter here if build directory exists.
    cd build
    cd release
    echo -e "${Blu}Building Non-Periodic executable in Optimized (Release) mode...${RCol}"
    mkdir -p nonPeriodic && cd nonPeriodic && $cmake -DCMAKE_BUILD_TYPE=Release ../../../. && make && cd ..
    #echo -e "${Blu}Building Periodic executable in Optimized (Release) mode...${RCol}"
    #mkdir -p periodic && cd periodic && $cmake -DCMAKE_BUILD_TYPE=Release -D_ENABLE_PERIODIC=TRUE ../../../. && make && cd ../..
  else
    rm -rf build
    echo -e "${Blu}Creating build directory...${RCol}"
    mkdir -p build && cd build
    mkdir -p release && cd release
    echo -e "${Blu}Building Non-Periodic executable in Optimized (Release) mode...${RCol}"
    mkdir -p nonPeriodic && cd nonPeriodic && $cmake -DCMAKE_BUILD_TYPE=Release ../../../. && make && cd ..
    #echo -e "${Blu}Building Periodic executable in Optimized (Release) mode...${RCol}"
    #mkdir -p periodic && cd periodic && $cmake -DCMAKE_BUILD_TYPE=Release -D_ENABLE_PERIODIC=TRUE ../../../. && make && cd ../..
  fi
else
  if [ -d "build/debug" ]; then
    echo -e "${Blu}build/debug directory already present${RCol}"
    # Control will enter here if build directory exists.
    cd build
    cd debug
    echo -e "${Blu}Building Non-Periodic executable in Debug mode...${RCol}"
    mkdir -p nonPeriodic && cd nonPeriodic && $cmake -DCMAKE_BUILD_TYPE=Debug ../../../. && make && cd ..
    #echo -e "${Blu}Building Periodic executable in Debug mode...${RCol}"
    #mkdir -p periodic && cd periodic && $cmake -DCMAKE_BUILD_TYPE=Debug -D_ENABLE_PERIODIC=TRUE ../../../. && make && cd ../..
  else
    rm -rf build
    echo -e "${Blu}Creating build directory...${RCol}"
    mkdir -p build && cd build
    mkdir -p debug && cd debug
    echo -e "${Blu}Building Non-Periodic executable in Debug mode...${RCol}"
    mkdir -p nonPeriodic && cd nonPeriodic && $cmake -DCMAKE_BUILD_TYPE=Debug ../../../. && make && cd ..
    #echo -e "${Blu}Building Periodic executable in Debug mode...${RCol}"
    #mkdir -p periodic && cd periodic && $cmake -DCMAKE_BUILD_TYPE=Debug -D_ENABLE_PERIODIC=TRUE ../../../. && make && cd ../..
  fi
fi
echo -e "${Blu}Build complete.${RCol}"
