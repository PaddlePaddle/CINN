import numpy
import sys, os
import numpy as np
import paddle.fluid as fluid
from paddle.fluid.backward import append_backward

resnet_input = fluid.layers.data(
    name="resnet_input",
    append_batch_size=False,
    shape=[2, 24, 56, 56],
    dtype='float32')
label = fluid.layers.data(
    name="label",
    append_batch_size=False,
    shape=[2, 144, 56, 56],
    dtype='float32')
d = fluid.layers.relu6(resnet_input)
f = fluid.layers.conv2d(
    input=d, num_filters=144, filter_size=1, stride=1, padding=0, dilation=1)
g = fluid.layers.scale(f, scale=2.0, bias=0.5)
temp6 = fluid.layers.softmax(g, axis=1)
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
