#!/bin/bash
set -ex

build_dir=$1

cd $build_dir
touch tests/test01_elementwise_add.cc
touch tests/test02_matmul.cc
touch tests/test02_matmul_tile.cc

# run the test_mains and generate c code
make -j6
set +e
ctest -V
set -e

make -j6
ctest -V

