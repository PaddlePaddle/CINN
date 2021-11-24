#!/usr/bin/env python3

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

import unittest
from cinn import Target
from cinn.frontend import *
from cinn.common import *
import numpy as np


class OpFusionTest(unittest.TestCase):
    def _get_target(self):
        return DefaultNVGPUTarget() if is_compiled_with_cuda(
        ) else DefaultHostTarget()

    def build_add_relu_net(self, backward=True):
        builder = NetBuilder("add_relu")

        x0 = builder.create_input(Float(32), [32, 32], "x0")
        x1 = builder.create_input(Float(32), [32, 32], "x1")
        y0 = builder.elementwise_add(x0, x1, axis=-1)
        y1 = builder.relu(y0)
        # Insert an op that cannot be fused.
        y2 = builder.reduce_sum(y0, dim=[0, 1])

        inputs = [x0, x1]
        feeds = [
            np.random.random([32, 32]).astype("float32"),
            np.random.random([32, 32]).astype("float32"),
        ]

        if backward:
            # Insert an op that cannot be fused.
            fake_loss = builder.reduce_sum(y1, dim=[0, 1])

            dy1 = builder.create_input(Float(32), [32, 32], "dout")
            dy0 = builder.relu_grad(dy1, y1)
            dx0, dx1 = builder.elementwise_add_grad(dy0, x0, x1, axis=-1)

            inputs.append(dy1)
            feeds.append(np.random.random([32, 32]).astype("float32"))
            outputs = [dx0, dx1]
        else:
            outputs = [y0, y1]

        return builder, inputs, feeds, outputs

    def build_add_add_net(self, backward=True):
        builder = NetBuilder("add_add")

        x0 = builder.create_input(Float(32), [32, 32], "x0")
        x1 = builder.create_input(Float(32), [32], "x1")
        x2 = builder.create_input(Float(32), [32, 32], "x2")
        y0 = builder.elementwise_add(x0, x1, axis=-1)
        y1 = builder.elementwise_add(y0, x2, axis=-1)

        inputs = [x0, x1, x2]
        feeds = [
            np.random.random([32, 32]).astype("float32"),
            np.random.random([32]).astype("float32"),
            np.random.random([32, 32]).astype("float32"),
        ]

        if backward:
            # Insert an op that cannot be fused.
            fake_loss = builder.reduce_sum(y1, dim=[0, 1])

            dy1 = builder.create_input(Float(32), [32, 32], "dout")
            dy0, dx2 = builder.elementwise_add_grad(dy1, y0, x2, axis=-1)
            dx0, dx1 = builder.elementwise_add_grad(dy0, x0, x1, axis=-1)

            inputs.append(dy1)
            feeds.append(np.random.random([32, 32]).astype("float32"))
            outputs = [dx0, dx1, dx2]
        else:
            outputs = [y1]

        return builder, inputs, feeds, outputs

    def build_reduce_reduce_net(self, backward=True):
        builder = NetBuilder("reduce_reduce")

        x = builder.create_input(Float(32), [32, 32, 32], "x")
        y0 = builder.reduce_sum(x, dim=[1])
        square_x = builder.elementwise_mul(x, builder.identity(x))
        y1 = builder.reduce_sum(square_x, dim=[1])

        inputs = [x]
        feeds = [np.random.random([32, 32, 32]).astype("float32")]
        outputs = [y0, y1]

        return builder, inputs, feeds, outputs

    def test_fusion(self):
        func_name = "build_reduce_reduce_net"
        func = getattr(self, func_name)
        builder, inputs, feeds, outputs = func(backward=False)

        target = self._get_target()
        prog = builder.build()
        prog.apply_pass(target, ["Decomposer"])
        for i in range(prog.size()):
            print(prog[i])
        results = prog.build_and_get_output(target, inputs, feeds, outputs)


if __name__ == "__main__":
    unittest.main()
