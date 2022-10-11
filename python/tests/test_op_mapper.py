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

enable_gpu = sys.argv.pop()


class TestOpMapper(OpTest):
    def setUp(self):
        if enable_gpu == "ON":
            self.target = DefaultNVGPUTarget()
            self.place = paddle.CUDAPlace(0)
        else:
            self.target = DefaultHostTarget()
            self.place = paddle.CPUPlace()

        self.init_case()

    def init_case(self):
        self.feed_data = {
            'x': self.random([10, 12, 128, 128], 'float32'),
            'y': self.random([10, 12, 128, 128], 'float32'),
        }

    def set_paddle_program(self):
        x = paddle.static.data(
            name='x', shape=[10, 12, 128, 128], dtype='float32')
        y = paddle.static.data(
            name='y', shape=[10, 12, 128, 128], dtype='float32')
        out = paddle.stack([x, y], 1)

        return ([x.name, y.name], [out])

    def build_paddle_program(self, target):
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            [self.feed_names, self.fetch_targets] = self.set_paddle_program()

        exe = paddle.static.Executor(self.place)
        exe.run(startup_program)

        self.paddle_outputs = exe.run(
            main_program, self.feed_data, fetch_list=self.fetch_targets)

        paddle.fluid.io.save_inference_model(
            "./op_mapper_test_save_model", self.feed_names, self.fetch_targets,
            exe, main_program)

    def build_cinn_program(self, target):
        convertor = PaddleModelConvertor()
        prog = convertor(self.target, "./op_mapper_test_save_model")

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


class TestOpMapperCase1(TestOpMapper):
    def init_case(self):
        self.feed_data = {
            'x': self.random([10, 12, 128, 128], 'float32'),
        }

    def set_paddle_program(self):
        x = paddle.static.data(
            name='x', shape=[10, 12, 128, 128], dtype='float32')
        out = paddle.log1p(x)

        return ([x.name], [out])


if __name__ == "__main__":
    unittest.main()
