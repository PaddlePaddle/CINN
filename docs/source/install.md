# Build from source code

## Dependencies

- gcc-8
- g++-8
- isl 0.22

### Install isl


```sh
git clone https://github.com/Meinersbur/isl.git
git reset --hard isl-0.22
cd isl
./configure --with-clang=system
make -j
make install
```

### compile

```sh
cd CINN
cp cmake/config.cmake <build_dir>/
```

Modify the `config.cmake`, change the `ISL_HOME` to the path isl installed.


### Install LLVM and MLIR
To use the latest version of MLIR, the latest llvm-project should be compiled and installed.

The git commit is `f9dc2b7079350d0fed3bb3775f496b90483c9e42`

*download llvm source code*

```sh
git clone https://github.com/llvm/llvm-project.git
```
The git of the llvm-project is huge and git cloning in China is quite slow, use a http proxy if necessary.

*compile and install to local directory*

```sh
cd llvm-project
mkdir build

cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_BUILD_EXAMPLES=OFF \
  -DLLVM_TARGETS_TO_BUILD="X86" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_ZLIB=OFF \
  -DLLVM_ENABLE_RTTI=ON \
  -DCMAKE_INSTALL_PREFIX=$PWD/../install/llvmorg-9e42
  
ninja install -j8
```

*add binary executables to environment variables*

```sh
export PATH="$PWD/../install/llvmorg-9e42/bin:$PATH"
```

*check the llvm version*

```sh
llvm-config --version

# should get 12.0.0git
```
