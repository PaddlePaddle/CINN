import numpy
import sys, os
import numpy as np
import paddle
import paddle.fluid as fluid

#For paddlepaddle version >=2.0rc, we need to set paddle.enable_static()
paddle.enable_static()

a = fluid.data(name="A", shape=[512, 512], dtype='float32')
b = fluid.data(name="B", shape=[512, 512], dtype='float32')

label = fluid.layers.data(name="label", shape=[512, 512], dtype='float32')

a1 = fluid.layers.mul(a, b)

#cost = fluid.layers.square_error_cost(a1, label)
#avg_cost = fluid.layers.mean(cost)

#optimizer = fluid.optimizer.SGD(learning_rate=0.001)
#optimizer.minimize(avg_cost)

cpu = fluid.core.CPUPlace()
loss = exe = fluid.Executor(cpu)

exe.run(fluid.default_startup_program())

fluid.io.save_inference_model("./elementwise_add_model", [a.name, b.name],
                              [a1], exe)
print('input and output names are: ', a.name, b.name, a1.name)
