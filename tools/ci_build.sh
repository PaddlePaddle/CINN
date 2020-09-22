#!/bin/bash
set -ex

readonly workspace=$PWD

function install_isl {
    if [ ! -d isl ]; then
        git clone https://github.com/inducer/isl.git isl
    fi

    cd isl
    git checkout a72ac2e
    ./autogen.sh

    find /usr -name "SourceLocation.h"

    ./configure --with-clang=system
    make -j
    sudo make install
}

function compile_cinn {
    cd $workspace
    cmake .
    make -j
}

function run_test {
    ctest -V
}

install_isl

#compile_cinn

#run_test
