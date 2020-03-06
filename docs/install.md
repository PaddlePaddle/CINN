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
