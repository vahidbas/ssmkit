#!/bin/bash
set -e # Exit with nonzero exit code if anything fails

if [ ! -d "build" ]; then
  mkdir build
fi

cd build
cmake \
  -DCMAKE_C_COMPILER="echo" \
  -DCMAKE_CXX_COMPILER="echo" \
  -DCMAKE_C_COMPILER_WORKS=1 \
  -DCMAKE_CXX_COMPILER_WORKS=1 \
  ..

make doc
