#!/usr/bin/env bash

# Copyright (c) 2021 CINN Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -ex
workspace=$PWD
build_dir_name=${cinn_build:-build}
build_dir=$workspace/${build_dir_name}

#export LLVM11_DIR=${workspace}/THIRDS/usr

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

OLD_HTTP_PROXY=$http_proxy
OLD_HTTPS_PROXY=$https_proxy
function proxy_off {
  unset http_proxy
  unset https_proxy
}
function proxy_on {
  export http_proxy=$OLD_HTTP_PROXY
  export https_proxy=$OLD_HTTPS_PROXY
}

function prepare_ci {
  cd $workspace
  proxy_off
  if [[ $(command -v python) == $build_dir/ci-env/bin/python ]]; then
    return
  elif [[ -e $build_dir/ci-env/bin/activate ]]; then
    source $build_dir/ci-env/bin/activate
    return
  fi

  apt update
  echo "the current user EUID=$EUID: $(whoami)"
  if ! command -v doxygen &> /dev/null; then
    apt install -y doxygen
  fi

  if ! command -v python3.8-config &> /dev/null; then
    apt install -y python3.8-dev
  fi

  if ! python3.8 -m venv $build_dir/ci-env &> /dev/null; then
    apt install -y python3.8-venv
    python3.8 -m venv $build_dir/ci-env
  fi
  source $build_dir/ci-env/bin/activate
  pip install -U pip --trusted-host mirrors.aliyun.com --index-url https://mirrors.aliyun.com/pypi/simple/
  pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
  pip config set global.trusted-host mirrors.aliyun.com
  pip install pre-commit
  pip install clang-format==9.0
  pip install sphinx==3.3.1 sphinx_gallery==0.8.1 recommonmark==0.6.0 exhale scipy breathe==4.24.0 matplotlib
  pip install paddlepaddle-gpu==2.1.2.post101 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
}

function make_doc {
    proxy_off
    cd $workspace/tutorials
    if [[ -f "ResNet18.tar.gz" ]]; then
        echo "model file for tutorials already downloaded."
    elif [[ -f "$build_dir/thirds/ResNet18.tar.gz" ]]; then
        rm -rf $workspace/tutorials/ResNet18
        ln -s $build_dir/thirds/ResNet18 $workspace/tutorials/ResNet18
    else
        wget http://paddle-inference-dist.bj.bcebos.com/CINN/ResNet18.tar.gz
        tar -zxvf ResNet18.tar.gz
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
    proxy_off
    mkdir -p $build_dir
    cp $workspace/cmake/config.cmake $build_dir
    # To enable Cuda backend, set(WITH_CUDA ON)
    echo "set(WITH_CUDA $cuda_config)" >> $build_dir/config.cmake
    echo "set(WITH_CUDNN $cudnn_config)" >> $build_dir/config.cmake
    echo "set(WITH_MKL_CBLAS ON)" >> $build_dir/config.cmake
    cd $build_dir
    cmake .. -DPUBLISH_LIBS=ON -DWITH_TESTING=ON
}

function _download_and_untar {
    proxy_off
    local tar_file=$1
    if [[ ! -f $tar_file ]]; then
        wget https://paddle-inference-dist.bj.bcebos.com/CINN/$tar_file
        tar -zxvf $tar_file
    fi
}

function prepare_model {
    proxy_off
    cd $build_dir/thirds

    _download_and_untar ResNet18.tar.gz
    _download_and_untar MobileNetV2.tar.gz
    _download_and_untar EfficientNet.tar.gz
    _download_and_untar MobilenetV1.tar.gz
    _download_and_untar ResNet50.tar.gz
    _download_and_untar SqueezeNet.tar.gz

    proxy_on
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
    cd $build_dir/thirds
    python3 $workspace/python/tests/fake_model/naive_mul.py
    python3 $workspace/python/tests/fake_model/naive_multi_fc.py
    python3 $workspace/python/tests/fake_model/resnet_model.py
}

function codestyle_check {
    proxy_on
    cd $workspace
    pre-commit run -a
    if ! git diff-index --quiet HEAD --; then

        echo "Code is dirty, please run 'pre-commit run -a' to reformat the code changes"
        exit -1
    fi
}

function build {
    proxy_on
    cd $build_dir

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
    export LD_LIBRARY_PATH=$build_dir/paddle/mkldnn:$build_dir/thirds/install/mklml/lib:$LD_LIBRARY_PATH
    ctest --parallel 10 -V
}

function CI {
    mkdir -p $build_dir
    cd $build_dir
    export runtime_include_dir=$workspace/cinn/runtime/cuda

    prepare_ci
    codestyle_check

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
                codestyle_check
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
        esac
    done
}

main $@
