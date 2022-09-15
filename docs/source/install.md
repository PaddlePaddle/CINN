# Build from source code

## Dependencies
CINN is build and tested on Ubuntu-18.04 with GCC 8.2.0, third party libraries are provided for that environment and will be downloaded automatically. Other compatible environments should work, but we cann't gurantee it. Currently, CINN is under very active development, we provide Docker environment for you to have a quick experience. If you have any problem building CINN in your own environment, please try using Docker. More portability will be added to CINN in the future.

Docker image we used to build and test CINN: `registry.baidubce.com/paddlepaddle/paddle:latest-dev-cuda11.2-cudnn8-gcc82`.

## Build without Docker
Build without Docker is not recommended for now. Third party dependencies are downloaded automatically by cmake, some libraries will be compiled, and others are static prebuilt. If you indeed have interest to build CINN in your own environment, you can use the content in `Build using Docker` section as a reference. 

## Build using Docker

Checkout CINN source code from git. 

```bash
$ git clone --depth 1 https://github.com/PaddlePaddle/CINN.git
```

Download docker image from registry.

```bash
$ docker pull registry.baidubce.com/paddlepaddle/paddle:latest-dev-cuda11.2-cudnn8-gcc82
```

Start the container.

```bash
$ docker run --gpus=all -it -v $PWD/CINN:/CINN /registry.baidubce.com/paddlepaddle/paddle:latest-dev-cuda11.2-cudnn8-gcc82 bin/bash
```

Build CINN in the created container.

```bash
# create a build directory
$ mkdir /CINN/build && cd /CINN/build
# use cmake to generate Makefile and download dependencies, use flags to toggle on/off CUDA and CUDNN support
# e.g. 1) build with CUDA & CUDNN support
#   cmake .. -DWITH_CUDA=ON -DWITH_CUDNN=On
# e.g. 2) build without CUDA & CUDNN support(CPU only, default)
#   cmake .. -DWITH_CUDA=OFF -DWITH_CUDNN=OFF
$ cmake ..  -DWITH_CUDA=ON -DWITH_CUDNN=ON
# build CINN
$ make 
```

`build/python/dist/cinn-xxxxx.whl` is the generated python wheel package, the real file name will differ given by the build options, python version, build environments, and git tag.

```bash
$ pip install build/python/dist/cinn.*.whl
```

A demo using CINN's computation API.
```python
import numpy as np
from cinn.frontend import *
from cinn import Target
from cinn.framework import *
from cinn import runtime
from cinn import ir
from cinn import lang
from cinn.common import *

target = DefaultHostTarget()
#target = DefaultNVGPUTarget()

builder = NetBuilder("test_basic")
a = builder.create_input(Float(32), (1, 24, 56, 56), "A")
b = builder.create_input(Float(32), (1, 24, 56, 56), "B")
c = builder.add(a, b)
d = builder.create_input(Float(32), (144, 24, 1, 1), "D")
e = builder.conv(c, d)

computation = Computation.build_and_compile(target, builder)

A_data = np.random.random([1, 24, 56, 56]).astype("float32")
B_data = np.random.random([1, 24, 56, 56]).astype("float32")
D_data = np.random.random([144, 24, 1, 1]).astype("float32")

computation.get_tensor("A").from_numpy(A_data, target)
computation.get_tensor("B").from_numpy(B_data, target)
computation.get_tensor("D").from_numpy(D_data, target)

computation.execute()

e_tensor = computation.get_tensor(str(e))
edata_cinn = e_tensor.numpy(target)
print(edata_cinn)
```

run the demo.
```
$ python demo.py
```
`target = DefaultHostTarget()` indicates CINN to use CPU for computing, well `target = DefaultNVGPUTarget()` uses GPU. 
