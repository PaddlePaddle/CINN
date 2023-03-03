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

from ast import arg
import os
import logging
from typing import Any
import unittest
import numpy as np

import paddle
from paddle.fluid.framework import Variable as PaddleVariable
from paddle.fluid.layer_helper import LayerHelper

from cinn.frontend import NetBuilder, PaddleModelConvertor
from cinn.common import is_compiled_with_cuda
from cinn.framework import Scope

from tests.ops.op_test import OpTest, OpTestTool

logging.basicConfig(level=os.environ.get('LOG_LEVEL', 'INFO').upper())
logger = logging.getLogger(name="op_test")

paddle.enable_static()


class OpMapperTest(OpTest):
    def __init__(self, *args, **kwargs):
        super(OpMapperTest, self).__init__(*args, **kwargs)
        self._init_place()
        self.init_input_data()

    def _init_place(self):
        self.place = paddle.CPUPlace()
        if is_compiled_with_cuda():
            self.place = paddle.CUDAPlace(0)

    def init_input_data(self):
        self.feed_data = {}
        logger.warn("No Input Data")

    def set_op_type(self) -> str:
        """Set paddle C++ op type:\n
        The op type should be got from the paddle static program.
        Not the paddle python api name or phi api name.\n
        For example, the C++ op type of `paddle.sum` is `reduce_sum`, the code from `Paddle/python/paddle/tensor/math.py`:
        ```
        def sum(x, axis=None, dtype=None, keepdim=False, name=None):
            ...
             helper.append_op(
                type='reduce_sum',
                inputs={'X': x},
                outputs={'Out': out},
                attrs=attrs,
            )
        ```
        """
        raise Exception("Not implemented.")

    def set_op_inputs(self) -> dict:
        """Map from input parameter name to argument list, the argument should be get from paddle.static.data.\n
        For example, `concat` should return
        ```
        x1 = paddle.static.data(name='x1', shape=[1, 2], dtype='float32')
        x2 = paddle.static.data(name='x2', shape=[1, 2], dtype='float32')
        return {'X' : [x1, x2]}
        ```        """
        return dict()

    def set_op_attrs(self) -> dict:
        """Map from attribute name to attribute value:\n
        For example, `concat` should return
        ```
        return {'axis' : 0}
        ```
        """
        return dict()

    def set_op_outputs(self) -> dict:
        """Map from output parameter name to argument type, the argument type should be represented by a string.\n
        For example, if the `out_dtype` attribute of `cast` is `'float16'`, here should return
        ```
        return {'Out' : 'float16'}
        ```
        """
        raise Exception("Not implemented.")

    def __set_paddle_op(self):
        # paddle C++ op type
        self.op_type = self.set_op_type()
        # map from input param name to argument name list
        self.inputs = self.set_op_inputs()
        # map from attribute name to attribute value
        self.attrs = self.set_op_attrs()
        # map from output param name to output data type
        self.output_dtypes = self.set_op_outputs()
        # collect some important infomation
        self.input_arg_map = self.__get_arguments_map(self.inputs)
        self.fetch_targets = list()
        self.op_desc = None

    def __check_valid(self):
        self.assertIsInstance(self.op_type, str)
        self.assertNotEqual(self.op_type, "")
        self.assertIsInstance(self.inputs, dict)
        self.assertIsInstance(self.attrs, dict)
        self.assertIsInstance(self.output_dtypes, dict)
        self.assertGreater(len(self.output_dtypes), 0)

        for name, var in self.input_arg_map.items():
            self.assertIn(name, self.feed_data)
            self.assertEqual(
                var.shape,
                self.feed_data[name].shape,
                msg="The shape of input {} in feed_data error".format(
                    var.name))
            self.assertEqual(
                self.paddleddtype2nptype(var.dtype),
                str(self.feed_data[name].dtype),
                msg="The type of input {} in feed_data erroe".format(var.name))

    def __get_arguments_map(self, param_maps):
        arg_maps = dict()
        for args in param_maps.values():
            self.assertIsInstance(args, list)
            for var in args:
                self.assertIsInstance(
                    var,
                    PaddleVariable,
                    msg=
                    "The type of argument should be paddle.fluid.framework.Variable"
                )
                self.assertTrue(
                    (var.name not in arg_maps) or (arg_maps[var.name] == var),
                    msg="Argument %s is duplicated" % var.name)
                arg_maps[var.name] = var
        return arg_maps

    def __init_paddle_op(self):
        self.__set_paddle_op()
        self.__check_valid()

    def build_paddle_program(self, target):
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            self.__init_paddle_op()
            helper = LayerHelper(self.op_type)

            self.outputs = dict()
            for var_name, dtype in self.output_dtypes.items():
                out_var = helper.create_variable_for_type_inference(dtype)
                self.fetch_targets.append(out_var)
                self.outputs[var_name] = [out_var]

            self.op_desc = helper.append_op(
                type=self.op_type,
                inputs=self.inputs,
                outputs=self.outputs,
                attrs=self.attrs).desc

        logger.debug("Paddle Program:\n" + str(main_program))

        exe = paddle.static.Executor(self.place)
        exe.run(startup_program)

        logger.debug("Feed list:\n" + str(self.feed_data))

        self.paddle_outputs = exe.run(
            main_program, self.feed_data, fetch_list=self.fetch_targets)

        logger.debug("Paddle result:\n" +
                     str([{
                         self.fetch_targets[i].name: self.paddle_outputs[i]
                     } for i in range(len(self.fetch_targets))]))

    def build_cinn_program(self, target):
        scope = Scope()
        convertor = PaddleModelConvertor(target=self.target, scope=scope)

        for var_name, var in self.input_arg_map.items():
            convertor.create_input(
                dtype=self.paddleddtype2str(var.dtype),
                shape=var.shape,
                name=var_name)

        convertor.append_op(
            type=self.op_type,
            inputs=self.op_desc.inputs(),
            outputs=self.op_desc.outputs(),
            attrs=self.attrs)

        prog = convertor()

        logger.debug("CINN Program:\n" + str(prog))

        # get the CINN input list
        cinn_inputs = []
        cinn_feed_datas = []

        # map the name the variable
        if len(self.input_arg_map) > 0:
            feed_names = set(self.input_arg_map.keys())
            input_dict = convertor.get_fetch_list(fetch_list=feed_names)
            for name in feed_names:
                cinn_name = convertor.get_cinn_name(name)

                self.assertIn(
                    cinn_name,
                    input_dict,
                    msg="Cannot find variable " + cinn_name +
                    " in cinn program's input, which are " + str(
                        input_dict.items()))
                cinn_inputs.append(input_dict[cinn_name])
                cinn_feed_datas.append(self.feed_data[name])

        # get the CINN output list
        fetch_names = {var.name for var in self.fetch_targets}
        output_dict = convertor.get_fetch_list(fetch_list=fetch_names)
        cinn_output_vars = [
            output_dict[var.name] for var in self.fetch_targets
        ]

        # run and get result
        self.cinn_outputs = self.get_cinn_output(
            prog,
            target,
            cinn_inputs,
            cinn_feed_datas,
            cinn_output_vars,
            passes=list(),
            scope=scope)

        logger.debug("CINN result:\n" + str([{
            self.fetch_targets[i].name + "/" + convertor.get_cinn_name(self.fetch_targets[i].name):
            self.cinn_outputs[i]
        } for i in range(len(self.fetch_targets))]))

    @staticmethod
    def paddleddtype2nptype(dtype):
        switch_map = {
            paddle.float16: "float16",
            paddle.float32: "float32",
            paddle.float64: "float64",
            paddle.int8: "int8",
            paddle.int16: "int16",
            paddle.int32: "int32",
            paddle.int64: "int64",
            paddle.uint8: "uint8",
            paddle.bool: "bool",
        }
        assert dtype in switch_map, str(dtype) + " not support in CINN"
        return switch_map[dtype]

    @staticmethod
    def nptype2paddledtype(dtype):
        switch_map = {
            "float16": paddle.float16,
            "float32": paddle.float32,
            "float64": paddle.float64,
            "int8": paddle.int8,
            "int16": paddle.int16,
            "int32": paddle.int32,
            "int64": paddle.int64,
            "uint8": paddle.uint8,
            "bool": paddle.bool,
        }
        assert dtype in switch_map, dtype + " not support in CINN"
        return switch_map[dtype]
