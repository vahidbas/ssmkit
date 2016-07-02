#!/bin/bash
set -e # Exit with nonzero exit code if anything fails

# check if the build directory exists
if [ ! -d "build" ]; then
  mkdir build
fi

cd build
# force cmake not to look for C/C++ compiler
cmake \
  -DCMAKE_C_COMPILER="echo" \
  -DCMAKE_CXX_COMPILER="echo" \
  -DCMAKE_C_COMPILER_WORKS=1 \
  -DCMAKE_CXX_COMPILER_WORKS=1 \
  ..
# bulid doc
make doc
