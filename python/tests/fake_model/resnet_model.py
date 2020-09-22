import numpy
import sys, os
import numpy as np
import paddle.fluid as fluid
from paddle.fluid.backward import append_backward

resnet_input = fluid.layers.data(
    name="resnet_input",
    append_batch_size=False,
    shape=[1, 3, 224, 224],
    dtype='float32')
label = fluid.layers.data(
    name="label",
    append_batch_size=False,
    shape=[1, 64, 56, 56],
    dtype='float32')

temp1 = fluid.layers.conv2d(
    input=resnet_input, num_filters=64, filter_size=7, padding=3, stride=2)
temp2 = fluid.layers.batch_norm(input=temp1)
temp3 = fluid.layers.relu(temp2)
temp4 = fluid.layers.pool2d(
    input=temp3,
    pool_size=[3, 3],
    pool_type='max',
    pool_stride=[2, 2],
    pool_padding=[1, 1],
    global_pooling=False,
    ceil_mode=False,
    exclusive=True,
    data_format="NCHW")
temp5 = fluid.layers.conv2d(
    input=temp4, num_filters=64, filter_size=1, padding=0, stride=1)
temp6 = fluid.layers.batch_norm(input=temp5)
temp7 = fluid.layers.relu(temp6)

cost = fluid.layers.square_error_cost(temp7, label)
avg_cost = fluid.layers.mean(cost)

optimizer = fluid.optimizer.SGD(learning_rate=0.001)
optimizer.minimize(avg_cost)

cpu = fluid.core.CPUPlace()
exe = fluid.Executor(cpu)

exe.run(fluid.default_startup_program())

fluid.io.save_inference_model("./resnet_model", [resnet_input.name], [temp7],
                              exe)
print('res', temp7.name)
