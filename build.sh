#!/bin/bash
set -ex

workspace=$PWD
build_dir_name=${cinn_build:-build}
build_dir=$workspace/${build_dir_name}

JOBS=8

cuda_config=OFF
cudnn_config=OFF

function gpu_on {
    cuda_config=ON
    cudnn_config=ON
}

function cudnn_off {
    cudnn_config=OFF
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

    python3 -m pip install sphinx==3.3.1 sphinx_gallery==0.8.1 recommonmark==0.6.0 exhale scipy breathe==4.24.0 --trusted-host mirrors.aliyun.com
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
    cd $workspace/tutorials
    if [[ -f "ResNet18.tar" ]]; then
        echo "model file for tutorials already downloaded."
    elif [[ -f "$build_dir/thirds/ResNet18.tar" ]]; then
        rm -rf $workspace/tutorials/ResNet18
        ln -s $build_dir/thirds/ResNet18 $workspace/tutorials/ResNet18
    else
        wget http://paddle-inference-dist.bj.bcebos.com/CINN/ResNet18.tar
        tar -xvf ResNet18.tar
    fi
    if [[ $cuda_config == "ON" ]]; then
        mkdir is_cuda
    fi

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
    echo "set(WITH_CUDNN $cudnn_config)" >> $build_dir/config.cmake
    echo "set(WITH_MKL_CBLAS ON)" >> $build_dir/config.cmake
    cd $build_dir
    cmake .. -DLLVM11_DIR=${LLVM11_DIR} -DLLVM_DIR=${LLVM11_DIR}/lib/cmake/llvm -DMLIR_DIR=${LLVM11_DIR}/lib/cmake/mlir -DPUBLISH_LIBS=ON

    make GEN_LLVM_RUNTIME_IR_HEADER
    # make the code generated compilable
    sed -i 's/0git/0/g' $build_dir/cinn/backends/llvm/cinn_runtime_llvm_ir.h
}

function _download_and_untar {
    local tar_file=$1
    if [[ ! -f $tar_file ]]; then
        wget https://paddle-inference-dist.bj.bcebos.com/CINN/$tar_file
        tar -xvf $tar_file
    fi
}

function prepare_model {
    cd $build_dir/thirds

    _download_and_untar ResNet18.tar
    _download_and_untar MobileNetV2.tar
    _download_and_untar EfficientNet.tar
    _download_and_untar MobilenetV1.tar
    _download_and_untar ResNet50.tar
    _download_and_untar SqueezeNet.tar

    mkdir -p $build_dir/paddle
    cd $build_dir/paddle
    if [[ ! -f "libexternal_kernels.so.tgz" ]]; then
        wget https://github.com/T8T9/files/raw/main/libexternal_kernels.so.tgz
    fi
    tar -zxvf libexternal_kernels.so.tgz
    if [[ ! -f "paddle_1.8_fc_model.tgz" ]]; then
        wget https://github.com/T8T9/files/raw/main/paddle_1.8_fc_model.tgz
    fi
    tar -zxvf paddle_1.8_fc_model.tgz
    if [[ ! -f "mkldnn.tgz" ]]; then
        wget https://github.com/T8T9/files/raw/main/mkldnn.tgz
    fi
    tar -zxvf mkldnn.tgz
    export LD_LIBRARY_PATH=$build_dir/paddle/mkldnn:$build_dir/thirds/install/mklml/lib:$LD_LIBRARY_PATH
    cd -
    python3 $workspace/python/tests/fake_model/naive_mul.py
    python3 $workspace/python/tests/fake_model/naive_multi_fc.py
    python3 $workspace/python/tests/fake_model/resnet_model.py
}

function build {
    cd $build_dir

    # build gtest first, it tends to broke the CI
    make extern_gtest

    if [[ $cuda_config == "ON" ]]; then
        make test_codegen_cuda_dev -j $JOBS
        ctest -R test_codegen_cuda_dev -V
    fi

    make test01_elementwise_add_main -j $JOBS
    make test02_matmul_main -j $JOBS
    make test03_conv_main -j $JOBS
    make test_codegen_c -j $JOBS

    ctest -R test01_elementwise_add_main
    ctest -R test02_matmul_main
    ctest -R test03_conv_main
    ctest -R "test_codegen_c$"

    make -j $JOBS
}

function run_demo {
    cd $build_dir/dist
    export LD_LIBRARY_PATH=$build_dir/dist/cinn/lib:$LD_LIBRARY_PATH
    bash build_demo.sh
    ./demo
    rm ./demo
    cd -
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
    run_demo
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
            cudnn_off)
                cudnn_off
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
