#!/bin/bash
cmake=/sw/arcts/centos7/cmake/3.5.2/bin/cmake
rm -rf build
mkdir -p build && cd build
mkdir -p debug && cd debug && $cmake -DCMAKE_BUILD_TYPE=Debug ../../. && make && cd ..
mkdir -p release && cd release && $cmake -DCMAKE_BUILD_TYPE=Release ../../. && make && cd ..
