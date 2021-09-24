# CINN INSTAllATION GUIDANCE

### Step 1. Clone Source Code

Clone CINN from github.

`git clone https://github.com/PaddlePaddle/CINN.git`

### Step 2. Build Docker Image

Build docker image based on the given dockerfile in ./tools/docker/Dockerfile.

`cd ./CINN/tools/docker`

`sudo docker build -t cinn_image:v1 .`

### Step 3. Start a docker container

Start a docker container and mount folder ./CINN into it.

Go back to the path where you clone CINN.

`sudo nvidia-docker run -it --net=host -v $PWD/CINN:/WorkSpace/CINN --name=your_docker_name cinn_image:v1`

### Step 4. Prepare dependencies

After enter the container, run ./CINN/tools/ci_build.sh

`./CINN/tools/ci_build.sh`

### Step 5. Build CINN and do ci test

Build CINN and do ci test to verify correctness.

`cd CINN`

There are 3 kinds of ci test:

1. Test on CPU(X86) backends:  `./build.sh ci`
2. Test on NVGPU(cuda) backends with CUDNN library: `./build.sh gpu_on ci`
3. Test on NVGPU(cuda) backends without CUDNN library: `./build.sh gpu_on cudnn_off ci`
