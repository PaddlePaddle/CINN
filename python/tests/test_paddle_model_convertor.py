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

import sys
import unittest
import numpy as np
from ops.op_test import OpTest, OpTestTool
import paddle
import cinn
from cinn.frontend import *
from cinn.common import *

paddle.enable_static()

# first save paddle model like:
# ```
# import paddle
# paddle.enable_static()

# x = paddle.static.data(name='x', shape=[10, 12, 128, 128], dtype='float32')
# y = paddle.static.data(name='y', shape=[10, 12, 128, 128], dtype='float32')
# prediction = paddle.stack([x, y], 1)

# place = paddle.CUDAPlace(0)

# exe = paddle.static.Executor(place)
# exe.run(paddle.static.default_startup_program())
# prog = paddle.static.default_main_program()

# paddle.fluid.io.save_inference_model("./stack", [x.name, y.name], [prediction], exe, prog)
# ```
# Second load and run model like:
# ```
# python test_paddle_model.py "./stack" ON
# ```

# The second argument(ON/OFF): wether to use GPU
enable_gpu = sys.argv.pop()
# The first argument(PATH_TO_MODEL): the inference model directory path
model_dir = sys.argv.pop()


class TestPaddleModel(OpTest):
    def setUp(self):
        if enable_gpu == "ON":
            self.target = DefaultNVGPUTarget()
            self.place = paddle.CUDAPlace(0)
        else:
            self.target = DefaultHostTarget()
            self.place = paddle.CPUPlace()

        self.model_dir = model_dir

        self.load_paddle_program()
        self.init_case()

    @staticmethod
    def dtype_paddle2str(dtype):
        if dtype == paddle.float32:
            return "float32"
        elif dtype == paddle.int32:
            return "int32"
        elif dtype == paddle.int64:
            return "int64"
        elif dtype == paddle.bool:
            return "bool"
        else:
            raise Exception("CINN only support float/int/bool! But here " +
                            str(dtype))

    @staticmethod
    def eliminate_unkown_shape(shape):
        return [1 if dim == -1 else dim for dim in shape]

    def init_case(self):
        self.feed_data = dict()
        for i in range(len(self.feed_names)):
            # check no repeat variable
            self.assertNotIn(
                self.feed_names[i],
                self.feed_data,
                msg="Repeat feed name: " + self.feed_names[i])

            dtype = self.dtype_paddle2str(self.feed_dtypes[i])
            # random int type data should not limited to [0, 1]
            high = 1 if dtype != "int32" else self.feed_shapes[i][0]

            # the paddle's feed list need dict not list
            self.feed_data[self.feed_names[i]] = self.random(
                self.eliminate_unkown_shape(self.feed_shapes[i]),
                dtype,
                high=high)

    def load_paddle_program(self):
        self.exe = paddle.static.Executor(self.place)

        [self.inference_program, self.feed_names,
         self.fetch_targets] = paddle.fluid.io.load_inference_model(
             dirname=self.model_dir, executor=self.exe)

        self.feed_shapes = []
        self.feed_dtypes = []
        for var in self.inference_program.list_vars():
            if var.name in self.feed_names:
                self.feed_shapes.append(var.shape)
                self.feed_dtypes.append(var.dtype)

        self.assertEqual(
            len(self.feed_names),
            len(self.feed_shapes),
            msg="Cannot found some feed var in program!")

    def build_paddle_program(self, target):
        self.paddle_outputs = self.exe.run(
            self.inference_program,
            feed=self.feed_data,
            fetch_list=self.fetch_targets)

    def build_cinn_program(self, target):
        convertor = PaddleModelConvertor()
        prog = convertor(self.target, self.model_dir)

        # get cinn input list
        inputs = prog.get_inputs()
        self.assertEqual(
            len(self.feed_names),
            len(inputs),
            msg="The paddle's input list not equal to cinn's input list!")

        # map the name the variable
        input_dict = {var.name(): var for var in inputs}

        cinn_inputs = []
        cinn_feed_datas = []
        for name in self.feed_names:
            cinn_name = convertor.get_cinn_name(name)

            self.assertTrue(
                cinn_name in input_dict,
                msg="Cannot find variable " + cinn_name +
                " in cinn program's input, which are " + str(
                    input_dict.items()))
            cinn_inputs.append(input_dict[cinn_name])
            cinn_feed_datas.append(self.feed_data[name])

        # get cinn output list
        output_dict = convertor.get_fetch_list()
        cinn_output = [output_dict[var.name] for var in self.fetch_targets]

        # run and get result
        self.cinn_outputs = self.get_cinn_output(prog, target, cinn_inputs,
                                                 cinn_feed_datas, cinn_output,
                                                 list())

    def test_check_results(self):
        self.check_outputs_and_grads()


if __name__ == "__main__":
    unittest.main()
