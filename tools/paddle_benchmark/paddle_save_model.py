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

cpu = fluid.core.CPUPlace()
loss = exe = fluid.Executor(cpu)

exe.run(fluid.default_startup_program())

fluid.io.save_inference_model("./elementwise_add_model", [a.name, b.name],
                              [a1], exe)
print('input and output names are: ', a.name, b.name, a1.name)
