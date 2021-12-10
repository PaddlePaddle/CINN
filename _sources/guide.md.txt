# Install CINN using docker

### Step 1. Start a docker container

Start a docker container based on upstream image.

`nvidia-docker run --name $CONTAINER_NAME -it --net=host registry.baidubce.com/paddlepaddle/paddle:2.2.0-gpu-cuda11.2-cudnn8 /bin/bash`

If you are using the latest version of docker, try:

`docker run --gpus all --name $CONTAINER_NAME -it --net=host registry.baidubce.com/paddlepaddle/paddle:2.2.0-gpu-cuda11.2-cudnn8 /bin/bash`

And notice that if your cuda version is not 11.2, replace the docker image to the corresponding paddle image with identical cuda version [here](https://registry.hub.docker.com/r/paddlepaddle/paddle).

### Step 2. Clone Source Code

After entering the container, clone the source code from github.

`git clone https://github.com/PaddlePaddle/CINN.git`

### Step 3. Build CINN and do ci test

Build CINN and do ci test to verify correctness.

`cd CINN`

There are 5 kinds of ci test:

1. Test on CPU(X86) backends: `bash ./build.sh ci`
2. Test on CPU(X86) backends without mklcblas: `bash ./build.sh mklcblas_off ci`
3. Test on CPU(X86) backends without mkldnn: `bash ./build.sh mkldnn_off ci`
4. Test on NVGPU(cuda) backends with CUDNN library: `bash ./build.sh gpu_on ci`
5. Test on NVGPU(cuda) backends without CUDNN library: `bash ./build.sh gpu_on cudnn_off ci`
