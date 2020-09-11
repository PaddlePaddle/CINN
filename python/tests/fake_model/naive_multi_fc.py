"""
A fake model with multiple FC layers to test CINN on a more complex model.
"""
import numpy
import sys, os
import numpy as np
import paddle.fluid as fluid
from paddle.fluid.backward import append_backward

size = 6
num_layers = 3

a = fluid.layers.data(name="a", shape=[-1, size], dtype='float32')
label = fluid.layers.data(name="label", shape=[size], dtype='float32')

fc_out = fluid.layers.fc(
    input=a,
    size=size,
    act="relu",
    bias_attr=fluid.ParamAttr(name="fc_bias"),
    num_flatten_dims=1)

for i in range(num_layers - 1):
    fc_out = fluid.layers.fc(
        input=fc_out,
        size=size,
        act="relu",
        bias_attr=fluid.ParamAttr(name="fc_bias"),
        num_flatten_dims=1)

cost = fluid.layers.square_error_cost(fc_out, label)
avg_cost = fluid.layers.mean(cost)

optimizer = fluid.optimizer.SGD(learning_rate=0.001)
optimizer.minimize(avg_cost)

cpu = fluid.core.CPUPlace()
loss = exe = fluid.Executor(cpu)

exe.run(fluid.default_startup_program())

fluid.io.save_inference_model("./multi_fc_model", [a.name], [fc_out], exe)
print('res', fc_out.name)
