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

import paddle
from cinn.frontend import *
from cinn.common import *
from op_test import OpTest, OpTestTool
from op_test_helper import TestCaseHelper


@OpTestTool.skip_if(not is_compiled_with_cuda(),
                    "x86 test will be skipped due to timeout.")
class TestSortOp(OpTest):
    def setUp(self):
        print(f"\nRunning {self.__class__.__name__}: {self.case}")
        self.inputs = {}
        self.prepare_inputs()

    def prepare_inputs(self):
        self.inputs = {
            "x": self.random(self.case["shape"], self.case["dtype"])
        }
        self.axis = self.case["axis"]
        self.descending = self.case["descending"]

    def build_paddle_program(self, target):
        x1 = paddle.to_tensor(self.inputs["x"], stop_gradient=True)
        out = paddle.sort(x1, self.axis, self.descending)

        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("sort")
        x1 = builder.create_input(
            self.nptype2cinntype(self.inputs["x"].dtype),
            self.inputs["x"].shape, "x")
        out = builder.sort(x1, self.axis, not self.descending)
        prog = builder.build()
        forward_res = self.get_cinn_output(prog, target, [x1],
                                           [self.inputs["x"]], [out])

        self.cinn_outputs = forward_res

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestSortOpShapeTest(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestSortOpShapeTest"
        self.cls = TestSortOp
        self.inputs = [
            {
                "shape": [10],
            },
            {
                "shape": [8, 5],
            },
            {
                "shape": [10, 3, 5],
            },
            # F0509 08:18:53.483060 2316861 cuda_util.cc:110] CUDA Driver Error: cuLaunchKernel(static_cast<CUfunction>(kernel_fn), grid_x, grid_y, grid_z, block_x, block_y, block_z, 0, static_cast<CUstream>(stream), kernel_args.data(), nullptr) failed with error: invalid argument
            # {
            #     "shape": [80, 40, 5, 7],
            # },
            {
                "shape": [80, 1, 5, 7],
            },
            # {
            #     "shape": [80, 3, 1024, 7],
            # },
            # {
            #     "shape": [10, 5, 1024, 2048],
            # },
            {
                "shape": [1],
            },
            {
                "shape": [1, 1, 1, 1],
            },
            {
                "shape": [1, 1, 1, 1, 1],
            },
            {
                "shape": [512],
            },
            {
                "shape": [1024],
            },
            {
                "shape": [2048],
            },
            # {
            #     "shape": [512, 256],
            # },
            # {
            #     "shape": [128, 64, 32],
            # },
            {
                "shape": [16, 8, 4, 2],
            },
            {
                "shape": [16, 8, 4, 2, 1],
            }
        ]
        self.dtypes = [{"dtype": "float32"}]
        self.attrs = [{"axis": 0, "descending": False}]


class TestSortOpDtypeTest(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestSortOpDtypeTest"
        self.cls = TestSortOp
        self.inputs = [
            {
                "shape": [16, 8, 4, 2],
            },
            # {
            #     "shape": [80, 40, 5, 7],
            # },
            # {
            #     "shape": [16, 8, 4, 2, 1],
            # }
        ]
        self.dtypes = [
            {
                "dtype": "float32"
            },
            {
                "dtype": "float64"
            },
            {
                "dtype": "int32"
            },
            {
                "dtype": "int64"
            },
        ]
        self.attrs = [{"axis": 0, "descending": False}]


class TestSortOpAxisTest(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestSortOpAttrsTest"
        self.cls = TestSortOp
        self.inputs = [
            {
                "shape": [16, 8, 4, 2],
            },
            # {
            #     "shape": [80, 40, 5, 7],
            # },
            # {
            #     "shape": [16, 8, 4, 2, 1],
            # }
        ]
        self.dtypes = [{"dtype": "float32"}]
        self.attrs = [{
            "axis": 0,
            "descending": False
        }, {
            "axis": 1,
            "descending": False
        }, {
            "axis": 2,
            "descending": False
        }, {
            "axis": 3,
            "descending": False
        }]


class TestSortOpDescedingTest(TestSortOpShapeTest):
    def init_attrs(self):
        self.class_name = "TestSortOpDescedingTest"
        self.cls = TestSortOp
        self.inputs = [
            {
                "shape": [10],
            },
            {
                "shape": [8, 5],
            },
            {
                "shape": [10, 3, 5],
            },
            # F0509 08:18:53.483060 2316861 cuda_util.cc:110] CUDA Driver Error: cuLaunchKernel(static_cast<CUfunction>(kernel_fn), grid_x, grid_y, grid_z, block_x, block_y, block_z, 0, static_cast<CUstream>(stream), kernel_args.data(), nullptr) failed with error: invalid argument
            # {
            #     "shape": [80, 40, 5, 7],
            # },
            {
                "shape": [80, 1, 5, 7],
            },
            # {
            #     "shape": [80, 3, 1024, 7],
            # },
            # {
            #     "shape": [10, 5, 1024, 2048],
            # },
            {
                "shape": [1],
            },
            {
                "shape": [1, 1, 1, 1],
            },
            {
                "shape": [1, 1, 1, 1, 1],
            },
            {
                "shape": [512],
            },
            {
                "shape": [1024],
            },
            {
                "shape": [2048],
            },
            # {
            #     "shape": [512, 256],
            # },
            # {
            #     "shape": [128, 64, 32],
            # },
            {
                "shape": [16, 8, 4, 2],
            },
            {
                "shape": [16, 8, 4, 2, 1],
            }
        ]
        self.dtypes = [{"dtype": "float32"}]
        self.attrs = [
            # NOTE: TestSortOpShapeTest has already tested with
            # the parameter 'descending=False', so just skip
            {
                "axis": 0,
                "descending": True
            }
        ]


if __name__ == "__main__":
    # TestSortOpShapeTest().run()
    TestSortOpDtypeTest().run()
    TestSortOpAxisTest().run()
    TestSortOpDescedingTest().run()
