import numpy
import sys, os
import numpy as np
import paddle.fluid as fluid
from paddle.fluid.backward import append_backward

resnet_input = fluid.layers.data(
    name="resnet_input",
    append_batch_size=False,
    shape=[1, 32, 112, 112],
    dtype='float32')
label = fluid.layers.data(
    name="label",
    append_batch_size=False,
    shape=[1, 32, 112, 112],
    dtype='float32')

param = fluid.initializer.NumpyArrayInitializer(
    np.random.random([32, 1, 3, 3]).astype("float32"))
temp1 = fluid.layers.conv2d(
    input=resnet_input,
    num_filters=32,
    filter_size=3,
    padding=1,
    stride=1,
    groups=32,
    param_attr=param,
    use_cudnn=False)
temp3 = fluid.layers.relu6(temp1)
temp7 = fluid.layers.relu(temp3)

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
