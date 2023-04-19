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
from op_test import OpTest


class TestGroup(unittest.TestCase):
    def test_group(self):
        builder = NetBuilder("dropout_test")

        eager_tmp_0 = builder.create_input(
            type="float16", shape=[128, 12, 128, 128], id_hint="eager_tmp_0")

        fill_constant_1__tmp_0 = builder.fill_constant(
            dtype="float32",
            force_cpu=False,
            shape=[1],
            value=0.10000000149011612)
        var_3 = builder.custom_call(
            dtype="float32",
            max=1.00000000,
            min=0.00000000,
            original_op="uniform_random",
            seed=0,
            shape=[128, 12, 128, 128])
        var_5 = builder.greater_equal(var_3, fill_constant_1__tmp_0, axis=-1)
        var_11 = builder.cast(var_5, dtype="uint8")
        var_7 = builder.cast(var_5, dtype="float16")
        var_9 = builder.elementwise_mul(eager_tmp_0, var_7, axis=-1)
        var_15 = builder.scale(
            var_9, bias=0.00000000, bias_after_scale=True, scale=1.11111116)

        feed_list = [eager_tmp_0]
        fetch_list = [var_15, var_11]

        prog = builder.build()

        feed_data = [
            OpTest.random(shape=var.shape(), dtype=var.type())
            for var in feed_list
        ]
        result = prog.build_and_get_output(DefaultNVGPUTarget(), feed_list,
                                           feed_data, fetch_list)

        result = [res.numpy(DefaultNVGPUTarget()) for res in result]
        for i in range(len(result)):
            info_str = fetch_list[i].name()
            info_str += ", shape=" + str(result[i].shape)
            info_str += ", dtype=" + str(result[i].dtype) + ":\n"
            info_str += str(result[i])
            print(info_str)


if __name__ == "__main__":
    unittest.main()
