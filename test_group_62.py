#!/usr/bin/env python3

# Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

# Please set "export PYTHONPATH=${CINN_ROOT}/build/python:${PYTHONPATH}" first

import unittest
import numpy as np
from cinn.frontend import NetBuilder
from cinn.common import DefaultNVGPUTarget


def random(shape, dtype="float32", low=0.0, high=1.0):
    assert bool(shape), "Shape should not empty!"
    assert -1 not in shape, "Shape should not -1!"
    if dtype in ["float16", "float32", "float64"]:
        return np.random.uniform(low, high, shape).astype(dtype)
    elif dtype == "bool":
        return np.random.choice(a=[False, True], size=shape).astype(dtype)
    elif dtype in ["int8", "uint8", "int32", "int64"]:
        return np.random.randint(low, high, shape).astype(dtype)
    else:
        raise Exception("Not supported yet.")


class TestGroup(unittest.TestCase):
    def test_group(self):
        builder = NetBuilder("group_test")

        var_1545 = builder.create_input(
            type="float32", shape=[128, 12, 128, 128], id_hint="var_1545")
        eager_in_tmp_2 = builder.create_input(
            type="float32", shape=[128, 1, 1, 128], id_hint="eager_in_tmp_2")

        var_3713 = builder.broadcast_to(
            eager_in_tmp_2,
            broadcast_axes=[0, 1, 2, 3],
            out_shape=[128, 12, 128, 128])
        var_1547 = builder.elementwise_add(var_1545, var_3713, axis=-1)
        var_1549 = builder.reduce_max(var_1547, dim=[3], keep_dim=True)
        var_5993 = builder.broadcast_to(
            var_1549,
            broadcast_axes=[0, 1, 2, 3],
            out_shape=[128, 12, 128, 128])
        var_1551 = builder.subtract(var_1547, var_5993, axis=-1)
        var_1553 = builder.exp(var_1551)

        feed_list = [var_1545, eager_in_tmp_2]
        fetch_list = [var_1553]

        prog = builder.build()

        feed_data = [
            random(shape=var.shape(), dtype=var.type()) for var in feed_list
        ]
        result = prog.build_and_get_output(DefaultNVGPUTarget(), feed_list,
                                           feed_data, fetch_list)

        # result = [res.numpy(DefaultNVGPUTarget()) for res in result]
        # for i in range(len(result)):
        #   info_str = fetch_list[i].name()
        #   info_str += ", shape=" + str(result[i].shape)
        #   info_str += ", dtype=" + str(result[i].dtype) + ":\n"
        #   info_str += str(result[i])
        #   print(info_str)


if __name__ == "__main__":
    import os
    PID = os.getpid()
    print('Program pid:', PID)
    print('Pause here to enter DBG')
    # input("read")
    unittest.main()
