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
"""
Training ResNet50 using Paddle compiled with CINN
=====================

**Note:** Docker execution environment is required, and you should use the docker image 
``registry.baidubce.com/paddlepaddle/paddle:latest-dev-cuda11.2-cudnn8-gcc82``
to create a container for trying the next steps in this tutorial. You can use the following
command to create a required container:

.. code-block:: bash

    # Docker version 20.10.11, build dea9396
    docker run --gpus all --name cinn_train_test \\
        --shm-size="8g" --net=host -v $PWD:/work \\
        -it registry.baidubce.com/paddlepaddle/paddle:latest-dev-cuda11.2-cudnn8-gcc82 /bin/bash

All the code below should be executed in the ``cinn_train_test`` container.

This is a beginner-friendly tutorial on how to train models using Paddle compiled with CINN.
This tutorial assumes that you have installed Paddle compiled with CINN. Otherwise, please
enable the ``-DWITH_CINN`` compilation option to recompile Paddle and reinstall it. To avoid
the tedious compilation process, you can also use the following command to install the 
pre-compiled ``.whl`` package.

.. code-block:: bash

    wget https://paddle-inference-dist.bj.bcebos.com/CINN_release/paddlepaddle_gpu-0.0.0-cp36-cp36m-linux_x86_64.whl
    pip3.6 install paddlepaddle_gpu-0.0.0-cp36-cp36m-linux_x86_64.whl
    export LD_LIBRARY_PATH=/usr/local/lib/python3.6/dist-packages/paddle/libs/:$LD_LIBRARY_PATH
    # Please use python3.6 to execute the following python codes.
"""

import os
# Paddle compiled with CINN only supports the single GPU training now.
# CUDA_VISIBLE_DEVICES should be set before paddle imported.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import paddle
# sphinx_gallery_thumbnail_path = './paddlepaddle.png'

##################################################################
#
# Enable static execution mode
# ---------------------------------------------
#
# Currently, we only support static graphs, so call ``paddle.enable_static()``  in advance.
#
paddle.enable_static()

##################################################################
#
# Enable training with CINN in Paddle
# ---------------------------------------------
#
# To train models with CINN, you need to set ``FLAGS_use_cinn`` to true.
#
# When training models with CINN, some Paddle operators will be replaced by CINN primitives.
# You can use the flag ``FLAGS_allow_cinn_ops`` to specify Paddle operators replaced by CINN.
#
# The fellowing operators are supported in CINN now.
# ``batch_norm,batch_norm_grad,conv2d,conv2d_grad, elementwise_add,elementwise_add_grad,relu,relu_grad,sum``
#
allow_ops = "batch_norm;batch_norm_grad;conv2d;conv2d_grad;elementwise_add;elementwise_add_grad;relu;relu_grad;sum"
try:
    paddle.set_flags({
        'FLAGS_use_cinn': True,
        'FLAGS_allow_cinn_ops': allow_ops
    })
except ValueError:
    # If the used PaddlePaddle is not compiled with CINN, just skip and
    # the following steps will not train with CINN.
    pass

##################################################################
#
# Select Device On Multi-GPU System
# -----------------------------------------------
# **Note:** At present, Paddle compiled with CINN only supports the single GPU.
# If you train models with CINN on a multi-GPU system, you should specify a device
# by setting ``CUDA_VISIBLE_DEVICES=GPU_ID`` in the system environment.
#
# Then you can specify the device id by using ``paddle.CUDAPlace(device_id))`` to get the device context.
# The sample code is shown below:
#
place = paddle.CUDAPlace(0)

##################################################################
#
# Build the model by using Paddle API
# ---------------------------------------------
#
# This example shows how to train ``ResNet50`` by using Paddle compiled with CINN.
# You can find more about Paddle APIs from this `website <https://www.paddlepaddle.org.cn/documentation/docs/en/api/index_en.html>`_.
# We set the batch size to 32 and input shape to [32, 3, 224, 224].
#
batch_size = 32
startup_program = paddle.static.Program()
main_program = paddle.static.Program()
with paddle.static.program_guard(main_program, startup_program):
    image = paddle.static.data(
        name='image', shape=[-1, 3, 224, 224], dtype='float32')
    label = paddle.static.data(name='label', shape=[-1], dtype='int64')

    model = paddle.vision.models.resnet50()
    prediction = model(image)
    loss = paddle.nn.functional.cross_entropy(input=prediction, label=label)
    loss = paddle.mean(loss)

    adam = paddle.optimizer.Adam(learning_rate=0.0125)
    adam.minimize(loss)

##################################################################
#
# Generate random fake data as input
# ----------------------------------------------
#
# Before running, you can load or generate some data as the feeding of a model.
# Here, we generate some fake input data by NumPy replacing the real data.
#
loop_num = 10
feed = []
for _ in range(loop_num):
    fake_input = {'image': np.random.randint(0, 256, size=[batch_size, 3, 224, 224]).astype('float32'), \
                 'label': np.random.randint(0, 1000, size=[batch_size]).astype('int64')}
    feed.append(fake_input)

##################################################################
#
# Executing program and print result
# -----------------------------
#
# Then we create an executor to train the model.
# You can learn more about Paddle from `paddlepaddle.org.cn <https://www.paddlepaddle.org.cn/>`_.
#
exe = paddle.static.Executor(place)

compiled_prog = paddle.static.CompiledProgram(main_program).with_data_parallel(
    loss_name=loss.name)
scope = paddle.static.Scope()

with paddle.static.scope_guard(scope):
    exe.run(startup_program)
    for step in range(loop_num):
        loss_v = exe.run(
            compiled_prog,
            feed=feed[step],
            fetch_list=[loss],
            return_numpy=True)
        print("Train step: {} loss: {}".format(step, loss_v[0][0]))
