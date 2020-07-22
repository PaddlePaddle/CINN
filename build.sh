#!/bin/bash
set -ex

workspace=$PWD
build_dir=$workspace/build

JOBS=8


function check_style {
    export PATH=/usr/bin:$PATH
    #pre-commit install
    clang-format --version

    if ! pre-commit run -a ; then
        git diff
        exit 1
    fi
}

function prepare {
    mkdir -p $build_dir
    cd $build_dir
    mkdir -p tests
    mkdir -p cinn/backends

    touch tests/test01_elementwise_add.cc
    touch tests/test01_elementwise_add_compute_at.cc
    touch tests/test02_matmul.cc
    touch tests/test02_matmul_tile.cc
    touch tests/test02_matmul_block.cc
    touch tests/test02_matmul_vectorize.cc
    touch tests/test02_matmul_loop_permutation.cc
    touch tests/test02_matmul_array_packing.cc
    touch tests/test02_matmul_split.cc
    touch tests/test02_matmul_varient_shape.cc
    touch tests/test02_matmul_varient_shape_tile.cc
    touch tests/test02_matmul_array_packing_dynamic_shape.cc
    touch tests/test02_matmul_call.cc
    touch tests/test01_elementwise_add_vectorize.cc
    touch cinn/backends/generated_module1.cc
    touch cinn/backends/generated1.cu
    touch tests/test03_convolution.cc
}

function prepare_llvm {
    cd $workspace
    clang++ -mavx2 -masm=intel -S -emit-llvm cinn/runtime/cinn_runtime.cc -I$PWD
    cd -

    export runtime_include_dir=$workspace/cinn/runtime
}

function cmake_ {
    prepare
    mkdir -p $build_dir
    cp $workspace/cmake/config.cmake $build_dir
    echo "set(ISL_HOME /usr/local)" >> $build_dir/config.cmake
    echo "set(WITH_CUDA OFF)" >> $build_dir/config.cmake
    echo "set(WITH_MKL_CBLAS ON)" >> $build_dir/config.cmake
    cd $build_dir
    cmake ..
}

function build {
    cd $build_dir
    make test01_elementwise_add_main -j $JOBS
    make test02_matmul_main -j $JOBS
    make test03_conv_main -j $JOBS

    ctest -R test01_elementwise_add_main
    ctest -R test02_matmul_main
    ctest -R test03_conv_main

    make -j $JOBS
}

function run_test {
    cd $build_dir
    ctest -V
}

function CI {
    mkdir -p $build_dir
    cd $build_dir

    prepare_llvm
    cmake_
    build
    run_test
}


function main {
    # Parse command line.
    for i in "$@"; do
        case $i in
            check_style)
                check_style
                shift
                ;;
            cmake)
                cmake_
                shift
                ;;
            build)
                build
                shift
                ;;
            test)
                run_test
                shift
                ;;
            ci)
                CI
                shift
                ;;
            prepare_llvm)
                prepare_llvm
                ;;
        esac
    done
}

main $@
