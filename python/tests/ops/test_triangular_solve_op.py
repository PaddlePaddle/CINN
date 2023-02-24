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
import numpy as np
from op_test import OpTest, OpTestTool
import paddle
import cinn
from cinn.frontend import *
from cinn.common import *


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "triangular solve op support GPU only now."
)
class TestTriangularSolveOp(OpTest):
    def setUp(self):
        self.init_case()

    def init_case(self):
        self.inputs = {
            "input1": np.array(
                [[1.0, 1.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, -1.0]]
            ).astype(np.float32),
            "input2": np.array([[0.0], [-9.0], [5.0]]).astype(np.float32),
        }
        self.outputs = {"solution": np.array([[7.0], [-2.0], [-5.0]]).astype(np.float32)}
        self.left_side = True
        self.upper = True
        self.transpose_a = False
        self.unit_diagonal = False

    def build_paddle_program(self, target):
        solution = paddle.to_tensor(self.outputs["solution"], stop_gradient=False)
        self.paddle_outputs = [solution]

    def build_cinn_program(self, target):
        builder = NetBuilder("triangular_solve")
        input1 = builder.create_input(
            self.nptype2cinntype(self.inputs["input1"].dtype),
            self.inputs["input1"].shape,
            "input1",
        )
        input2 = builder.create_input(
            self.nptype2cinntype(self.inputs["input2"].dtype),
            self.inputs["input2"].shape,
            "input2",
        )
        out = builder.triangular_solve(
            input1,
            input2,
            self.left_side,
            self.upper,
            self.transpose_a,
            self.unit_diagonal,
        )
        prog = builder.build()
        res = self.get_cinn_output(
            prog,
            target,
            [input1, input2],
            [self.inputs["input1"], self.inputs["input2"]],
            [out],
            passes=[],
        )
        self.cinn_outputs = [res[0]]
        print(res[0])

    def test_check_results(self):
        self.check_outputs_and_grads()


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "triangular solve op support GPU only now."
)
class TestTriangularSolveOpUnitDiagonal(TestTriangularSolveOp):
    def init_case(self):
        self.inputs = {
            "input1": np.array(
                [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, -1.0]]
            ).astype(np.float32),
            "input2": np.array([[0.0], [-9.0], [5.0]]).astype(np.float32),
        }
        self.outputs = {"solution": np.array([[0.0], [-9.0], [5.0]]).astype(np.float32)}
        self.left_side = True
        self.upper = True
        self.transpose_a = False
        self.unit_diagonal = True


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "triangular solve op support GPU only now."
)
class TestTriangularSolveOpLower(TestTriangularSolveOp):
    def init_case(self):
        self.inputs = {
            "input1": np.array(
                [[1.0, 1.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, -1.0]]
            ).astype(np.float32),
            "input2": np.array([[0.0], [-9.0], [5.0]]).astype(np.float32),
        }
        self.outputs = {
            "solution": np.array([[0.0], [-4.5], [-5.0]]).astype(np.float32)
        }
        self.left_side = True
        self.upper = False
        self.transpose_a = False
        self.unit_diagonal = False


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "triangular solve op support GPU only now."
)
class TestTriangularSolveOpZeroBatchDim1(TestTriangularSolveOp):
    def init_case(self):
        self.inputs = {
            "input1": np.array(
                [[1.0, 1.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, -1.0]]
            ).astype(np.float32),
            "input2": np.array([[0.0], [-9.0], [5.0]]).astype(np.float32),
        }
        self.outputs = {
            "solution": np.array([[7.0], [-2.0], [-5.0]]).astype(np.float32)
        }
        self.left_side = True
        self.upper = True
        self.transpose_a = False
        self.unit_diagonal = False


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "triangular solve op support GPU only now."
)
class TestTriangularSolveOpZeroBatchDim2(TestTriangularSolveOp):
    def init_case(self):
        self.inputs = {
            "input1": np.array(
                [[[1.0, 1.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, -1.0]]]
            ).astype(np.float32),
            "input2": np.array([[0.0], [-9.0], [5.0]]).astype(np.float32),
        }
        self.outputs = {
            "solution": np.array([[7.0], [-2.0], [-5.0]]).astype(np.float32)
        }
        self.left_side = True
        self.upper = True
        self.transpose_a = False
        self.unit_diagonal = False


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "triangular solve op support GPU only now."
)
class TestTriangularSolveOpZeroBatchDim3(TestTriangularSolveOp):
    def init_case(self):
        self.inputs = {
            "input1": np.array(
                [[1.0, 1.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, -1.0]]
            ).astype(np.float32),
            "input2": np.array([[[0.0], [-9.0], [5.0]]]).astype(np.float32),
        }
        self.outputs = {
            "solution": np.array([[[7.0], [-2.0], [-5.0]]]).astype(np.float32)
        }
        self.left_side = True
        self.upper = True
        self.transpose_a = False
        self.unit_diagonal = False


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "triangular solve op support GPU only now."
)
class TestTriangularSolveOpTranspose(TestTriangularSolveOp):
    def init_case(self):
        self.inputs = {
            "input1": np.array(
                [[[1.0, 1.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, -1.0]]]
            ).astype(np.float32),
            "input2": np.array([[[0.0], [-9.0], [5.0]]]).astype(np.float32),
        }
        self.outputs = {
            "solution": np.array([[[0.0], [-4.5], [-9.5]]]).astype(np.float32)
        }
        self.left_side = True
        self.upper = True
        self.transpose_a = True
        self.unit_diagonal = False


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "triangular solve op support GPU only now."
)
class TestTriangularSolveOpRightSide(TestTriangularSolveOp):
    def init_case(self):
        self.inputs = {
            "input1": np.array(
                [[[1.0, 1.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, -1.0]]]
            ).astype(np.float32),
            "input2": np.array([[[0.0, -9.0, 5.0]]]).astype(np.float32),
        }
        self.outputs = {"solution": np.array([[[0.0, -4.5, -9.5]]]).astype(np.float32)}
        self.left_side = False
        self.upper = True
        self.transpose_a = False
        self.unit_diagonal = False


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "triangular solve op support GPU only now."
)
class TestTriangularSolveOpDoubleFloat(TestTriangularSolveOp):
    def init_case(self):
        self.inputs = {
            "input1": np.array(
                [
                    [
                        [0.11002420, 0.52871847, 0.58182997],
                        [0.42569950, 0.28583527, 0.27665302],
                        [0.16611369, 0.73409891, 0.79629290],
                    ]
                ]
            ).astype(np.float64),
            "input2": np.array([[[0.79183686], [0.00260521], [0.47066584]]]).astype(
                np.float64
            ),
        }
        self.outputs = {
            "solution": np.array(
                [[[7.19693364], [-13.30327987], [-0.04562798]]]
            ).astype(np.float64)
        }
        self.left_side = True
        self.upper = True
        self.transpose_a = True
        self.unit_diagonal = False


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "triangular solve op support GPU only now."
)
class TestTriangularSolveOpBatch(TestTriangularSolveOp):
    def init_case(self):
        self.inputs = {
            "input1": np.array(
                [
                    [
                        [0.54486275, 0.63678247, 0.24622563],
                        [0.10051069, 0.86657214, 0.15503791],
                        [0.54872459, 0.13460679, 0.98267227],
                    ],
                    [
                        [0.85074371, 0.76554328, 0.16651264],
                        [0.83049846, 0.53734249, 0.16128331],
                        [0.56934613, 0.01969660, 0.83272946],
                    ],
                ]
            ).astype(np.float64),
            "input2": np.array(
                [
                    [[0.87776130], [0.58458430], [0.62417507]],
                    [[0.27944639], [0.91920167], [0.72476441]],
                ]
            ).astype(np.float64),
        }
        self.outputs = {
            "solution": np.array(
                [
                    [[0.66834761], [0.56095400], [0.63518132]],
                    [[-1.14613002], [1.44940905], [0.87034799]],
                ]
            ).astype(np.float64)
        }
        self.left_side = True
        self.upper = True
        self.transpose_a = False
        self.unit_diagonal = False


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "triangular solve op support GPU only now."
)
class TestTriangularSolveOpMultipleRightHandSides(TestTriangularSolveOp):
    def init_case(self):
        self.inputs = {
            "input1": np.array(
                [
                    [
                        [0.70170873, 0.12914789, 0.72920477],
                        [0.91394228, 0.95482904, 0.19833656],
                        [0.55876005, 0.99543732, 0.84755057],
                    ],
                    [
                        [0.73824805, 0.76891583, 0.87435943],
                        [0.65631932, 0.46146169, 0.55795544],
                        [0.49111775, 0.83861703, 0.93345910],
                    ],
                ]
            ).astype(np.float64),
            "input2": np.array(
                [
                    [
                        [0.26969177, 0.06588062, 0.40519544, 0.81075680],
                        [0.81013846, 0.20625703, 0.82617444, 0.71396655],
                        [0.75928980, 0.32613993, 0.99840266, 0.84280300],
                    ],
                    [
                        [0.02779338, 0.12316120, 0.53070980, 0.93873727],
                        [0.50958878, 0.69319230, 0.37919492, 0.56692517],
                        [0.28166223, 0.16990967, 0.00446068, 0.32721943],
                    ],
                ]
            ).astype(np.float64),
        }
        self.outputs = {
            "solution": np.array(
                [
                    [
                        [-0.66854064, -0.33104106, -0.76091775, 0.02243600],
                        [0.66237611, 0.13608357, 0.62056844, 0.54118691],
                        [0.89586371, 0.38480291, 1.17798594, 0.99439848],
                    ],
                    [
                        [-1.08989978, -1.38409483, -0.13662565, 0.01827429],
                        [0.73945713, 1.28208343, 0.81594777, 0.80469664],
                        [0.30174030, 0.18202155, 0.00477865, 0.35054501],
                    ],
                ]
            ).astype(np.float64)
        }
        self.left_side = True
        self.upper = True
        self.transpose_a = False
        self.unit_diagonal = False


if __name__ == "__main__":
    unittest.main()
