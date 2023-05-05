#!/usr/bin/env python3

# Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

# Copyright (c) 2022 CINN Authors. All Rights Reserved.
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
import numpy as np
from op_test import OpTest, OpTestTool
import paddle
import cinn
from cinn.frontend import *
from cinn.common import *
import logging
import os
from itertools import product

logging.basicConfig(level=os.environ.get('LOG_LEVEL', 'INFO').upper())
logger = logging.getLogger(name="gather")


@OpTestTool.skip_if(not is_compiled_with_cuda(),
                    "x86 test will be skipped due to timeout.")
class TestGatherOp(OpTest):
    def setUp(self):
        self.data = []
        self.init_case()

    def init_case(self):
        self.inputs = [{"x": [3, 4, 3], "index": [4], "axis": 1}]
        self.dtypes = ["float32"]

    def build_paddle_program(self, target):
        for inputs, dtype in product(self.inputs, self.dtypes):
            axis = inputs["axis"]
            x_shape = inputs["x"]
            index_shape = inputs["index"]
            # Paddle does not support negative axis values.
            axis = axis if axis >= 0 else len(x_shape) + axis
            x = np.random.randn(*x_shape).astype(dtype)
            index = np.random.randint(0, x_shape[axis],
                                      index_shape).astype("int32")
            self.data.append([x, index])
            x = paddle.to_tensor(x, stop_gradient=True)
            index = paddle.to_tensor(index, stop_gradient=True)
            out = paddle.gather(x, index, axis)
            logger.debug(" -- The output of Paddle:\n{}".format(out))
            self.paddle_outputs.append(out)

    def build_cinn_program(self, target):
        for i, (inputs, dtype) in enumerate(product(self.inputs, self.dtypes)):
            axis = inputs["axis"]
            builder = NetBuilder("gather")
            x = builder.create_input(
                self.nptype2cinntype(dtype), inputs["x"], "x")
            index = builder.create_input(Int(32), inputs["index"], "index")
            out = builder.gather(x, index, axis=axis)
            prog = builder.build()
            res = self.get_cinn_output(prog, target, [x, index], self.data[i],
                                       [out])
            logger.debug(" -- The output of CINN:\n{}".format(res))
            self.cinn_outputs.extend(res)

    def test_check_results(self):
        self.check_outputs_and_grads(all_equal=True)


class TestGatherOpAll(TestGatherOp):
    def init_case(self):
        self.inputs = [
            {
                "x": [128],
                "index": [64],
                "axis": 0
            },
            {
                "x": [16, 32],
                "index": [32],
                "axis": 1
            },
            {
                "x": [8, 16, 32],
                "index": [16],
                "axis": -3
            },
            {
                "x": [8, 16, 32],
                "index": [8],
                "axis": -2
            },
            {
                "x": [8, 16, 32],
                "index": [8],
                "axis": -1
            },
            {
                "x": [8, 16, 32],
                "index": [4],
                "axis": 2
            },
            {
                "x": [16, 8, 4, 2, 1],
                "index": [4],
                "axis": 2
            },
        ]
        self.dtypes = [
            "float32", "float64", "int8", "int16", "int32", "int64", "uint8"
        ]


if __name__ == "__main__":
    unittest.main()
