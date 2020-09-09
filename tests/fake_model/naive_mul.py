import numpy
import sys, os
import numpy as np
import paddle.fluid as fluid
from paddle.fluid.backward import append_backward

a = fluid.layers.data(name="a", shape=[-1, 10], dtype='float32')
label = fluid.layers.data(name="label", shape=[10], dtype='float32')

a1 = fluid.layers.fc(
    input=a,
    size=10,
    act="relu",
    bias_attr=fluid.ParamAttr(name="fc_bias"),
    num_flatten_dims=1)

cost = fluid.layers.square_error_cost(a1, label)
avg_cost = fluid.layers.mean(cost)

optimizer = fluid.optimizer.SGD(learning_rate=0.001)
optimizer.minimize(avg_cost)

cpu = fluid.core.CPUPlace()
loss = exe = fluid.Executor(cpu)

exe.run(fluid.default_startup_program())

fluid.io.save_inference_model("./model2", [a.name], [a1], exe)
print('res', a1.name)
