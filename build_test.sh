#!/bin/bash
set -ex

build_dir=$1

cd $build_dir
touch tests/test01_elementwise_add.cc
touch tests/test01_elementwise_add.h
touch tests/test02_matmul_tile.h
touch tests/test02_matmul.cc
touch tests/test02_matmul_tile.cc
touch tests/test02_matmul.h
touch tests/test02_matmul_block.cc
touch tests/test02_matmul_block.h
touch tests/test02_matmul_vectorize.h
touch tests/test02_matmul_vectorize.cc
touch tests/test02_matmul_loop_permutation.cc
touch tests/test02_matmul_loop_permutation.h
touch tests/test02_matmul_array_packing.cc
touch tests/test02_matmul_array_packing.h

make test01_elementwise_add_main -j6
make test02_matmul_main -j6

ctest -R test01_elementwise_add_main
ctest -R test02_matmul_main

# run the test_mains and generate c code
make -j6
ctest -V
