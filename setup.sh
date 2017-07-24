#!/bin/bash
set -e
set -o pipefail
#script to setup and build DFT-FE 
#set CMAKE path
cmake=/sw/arcts/centos7/cmake/3.5.2/bin/cmake
#
#Usually, no changes are needed below this line
#
RCol='\e[0m'
Blu='\e[0;34m';
rm -rf build
echo -e "${Blu}Creating build directory...${RCol}"
mkdir -p build && cd build
#mkdir -p debug && cd debug
#echo -e "${Blu}Building Non-Periodic executable in Debug mode...${RCol}"
#mkdir -p nonPeriodic && cd nonPeriodic && $cmake -DCMAKE_BUILD_TYPE=Debug ../../../. && make && cd ..
#echo -e "${Blu}Building Periodic executable in Debug mode...${RCol}"
#mkdir -p periodic && cd periodic && $cmake -DCMAKE_BUILD_TYPE=Debug -D_ENABLE_PERIODIC=TRUE ../../../. && make && cd ../..
mkdir -p release && cd release 
echo -e "${Blu}Building Non-Periodic executable in Optimized (Release) mode...${RCol}"
mkdir -p nonPeriodic && cd nonPeriodic && $cmake -DCMAKE_BUILD_TYPE=Release ../../../. && make && cd ..
echo -e "${Blu}Building Periodic executable in Optimized (Release) mode...${RCol}"
mkdir -p periodic && cd periodic && $cmake -DCMAKE_BUILD_TYPE=Release -D_ENABLE_PERIODIC=TRUE ../../../. && make && cd ../..
echo -e "${Blu}Build complete.${RCol}"
