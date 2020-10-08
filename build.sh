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

function prepare_model {
    cd $build_dir/thirds
    wget http://paddle-inference-dist.bj.bcebos.com/CINN/ResNet18.tar
    tar -xvf ResNet18.tar
    wget http://paddle-inference-dist.bj.bcebos.com/CINN/MobileNetV2.tar
    tar -xvf MobileNetV2.tar
    wget http://paddle-inference-dist.bj.bcebos.com/CINN/EfficientNet.tar
    tar -xvf EfficientNet.tar
    python $workspace/python/tests/fake_model/naive_mul.py
    python $workspace/python/tests/fake_model/naive_multi_fc.py
    python $workspace/python/tests/fake_model/resnet_model.py
}

function build {
    cd $build_dir

    # build gtest first, it tends to broke the CI
    make extern_gtest

    make test01_elementwise_add_main -j $JOBS
    make test02_matmul_main -j $JOBS
    make test03_conv_main -j $JOBS
    make test_codegen_c -j $JOBS

    ctest -R test01_elementwise_add_main
    ctest -R test02_matmul_main
    ctest -R test03_conv_main
    ctest -R test_codegen_c

    make -j $JOBS
}

function run_test {
    cd $build_dir
    ctest --parallel 10 -V
}

function CI {
    mkdir -p $build_dir
    cd $build_dir

    prepare_llvm
    cmake_
    build
    prepare_model
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
            prepare_model)
                prepare_model
                shift
                ;;
            prepare_llvm)
                prepare_llvm
                ;;
        esac
    done
}

main $@
