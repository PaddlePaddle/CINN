import numpy
import sys, os
import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.backward import append_backward

size = 30
paddle.enable_static()

a = fluid.layers.data(name="A", shape=[-1, size], dtype='float32')
label = fluid.layers.data(name="label", shape=[size], dtype='float32')

a1 = fluid.layers.fc(
    input=a, size=size, act="relu", bias_attr=None, num_flatten_dims=1)

cost = fluid.layers.square_error_cost(a1, label)
avg_cost = fluid.layers.mean(cost)

optimizer = fluid.optimizer.SGD(learning_rate=0.001)
optimizer.minimize(avg_cost)

cpu = fluid.core.CPUPlace()
loss = exe = fluid.Executor(cpu)

exe.run(fluid.default_startup_program())

fluid.io.save_inference_model("./naive_mul_model", [a.name], [a1], exe)
print('res', a1.name)
