#!/bin/bash
cmake=/Applications/CMake.app/Contents/bin/cmake
mkdir -p build && cd build
mkdir -p debug && cd debug && $cmake -DCMAKE_BUILD_TYPE=Debug ../../. && make && cd ..
mkdir -p release && cd release && $cmake -DCMAKE_BUILD_TYPE=Release ../../. && make && cd ..
