#!/bin/bash
set -ex

workspace=$PWD
build_dir_name=${cinn_build:-build}
build_dir=$workspace/${build_dir_name}

JOBS=8

cuda_config=OFF

function gpu_on {
    cuda_config=ON
}

function check_style {
    export PATH=/usr/bin:$PATH
    clang-format --version

    if ! pre-commit run -a ; then
        git diff
        exit 1
    fi
}

function prepare {
    mkdir -p $build_dir
    cd $build_dir

    python3 -m pip install sphinx sphinx_gallery recommonmark exhale scipy --trusted-host mirrors.aliyun.com
    apt install doxygen -y

    mkdir -p tests
    mkdir -p cinn/backends
}

function prepare_llvm {
    cd $workspace
    clang++ -mavx2 -masm=intel -S -emit-llvm cinn/runtime/cinn_runtime.cc -I$PWD
    cd -

    export runtime_include_dir=$workspace/cinn/runtime/cuda

    export PATH=${LLVM11_DIR}/bin:$PATH
}

function make_doc {
    cd $build_dir
    rm -f $workspace/python/cinn/core_api.so
    ln -s $build_dir/cinn/pybind/core_api.so $workspace/python/cinn/
    cd $workspace/docs
    mkdir -p docs/source/cpp
    cat $workspace/tutorials/matmul.cc | python $workspace/tools/gen_c++_tutorial.py  > $workspace/docs/source/matmul.md
    make html
}

function cmake_ {
    prepare
    mkdir -p $build_dir
    cp $workspace/cmake/config.cmake $build_dir
    echo "set(ISL_HOME /usr/local)" >> $build_dir/config.cmake
    # To enable Cuda backend, set(WITH_CUDA ON)
    echo "set(WITH_CUDA $cuda_config)" >> $build_dir/config.cmake
    echo "set(WITH_MKL_CBLAS ON)" >> $build_dir/config.cmake
    cd $build_dir
    cmake .. -DLLVM_DIR=${LLVM11_DIR}/lib/cmake/llvm -DMLIR_DIR=${LLVM11_DIR}/lib/cmake/mlir

    make GEN_LLVM_RUNTIME_IR_HEADER
    # make the code generated compilable
    sed -i 's/0git/0/g' $build_dir/cinn/backends/llvm/cinn_runtime_llvm_ir.h
}

function prepare_model {
    cd $build_dir/thirds
    if [[ ! -f "ResNet18.tar" ]]; then
        wget http://paddle-inference-dist.bj.bcebos.com/CINN/ResNet18.tar
        tar -xvf ResNet18.tar
    fi
    if [[ ! -f "MobileNetV2.tar" ]]; then
        wget http://paddle-inference-dist.bj.bcebos.com/CINN/MobileNetV2.tar
        tar -xvf MobileNetV2.tar
    fi
    if [[ ! -f "EfficientNet.tar" ]]; then
        wget http://paddle-inference-dist.bj.bcebos.com/CINN/EfficientNet.tar
        tar -xvf EfficientNet.tar
    fi
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
    if [[ $cuda_config == "ON" ]]; then
        make test_codegen_cuda_dev -j $JOBS
        ctest -R test_codegen_cuda_dev
    fi

    ctest -R test01_elementwise_add_main
    ctest -R test02_matmul_main
    ctest -R test03_conv_main
    ctest -R "test_codegen_c$"

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

    make_doc
}


function main {
    # Parse command line.
    for i in "$@"; do
        case $i in
            gpu_on)
                gpu_on
                shift
                ;;
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
            make_doc)
                make_doc
                shift
                ;;
            prepare_llvm)
                prepare_llvm
                ;;
        esac
    done
}

main $@
