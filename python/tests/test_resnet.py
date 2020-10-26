#!/usr/bin/env python3
import paddle as paddle
import paddle.fluid as fluid
from cinn.frontend import *
from cinn import Target
from cinn.framework import *
import unittest
import cinn
from cinn import runtime
from cinn import ir
from cinn import lang
from cinn.common import *
import numpy as np
import paddle.fluid as fluid
import sys

enable_gpu = sys.argv.pop()
model_dir = sys.argv.pop()


class TestLoadResnetModel(unittest.TestCase):
    def setUp(self):
        self.target = Target()
        if enable_gpu == "ON":
            self.target.arch = Target.Arch.NVGPU
        else:
            self.target.arch = Target.Arch.X86
        self.target.bits = Target.Bit.k64
        self.target.os = Target.OS.Linux

        self.model_dir = model_dir

        self.x_shape = [2, 160, 7, 7]

    def get_paddle_inference_result(self, data):
        exe = fluid.Executor(fluid.CPUPlace())

        [inference_program, feed_target_names,
         fetch_targets] = fluid.io.load_inference_model(
             dirname=self.model_dir, executor=exe)

        results = exe.run(
            inference_program,
            feed={feed_target_names[0]: data},
            fetch_list=fetch_targets)

        result = results[0]
        return result

    def apply_test(self):
        np.random.seed(0)
        x_data = np.random.random(self.x_shape).astype("float32")
        self.executor = Interpreter(["resnet_input"], [self.x_shape])
        self.executor.load_paddle_model(self.model_dir, self.target, False)
        a_t = self.executor.get_tensor("resnet_input")
        a_t.from_numpy(x_data, self.target)

        out = self.executor.get_tensor("relu_0.tmp_0")
        out.from_numpy(np.zeros(out.shape(), dtype='float32'), self.target)

        self.executor.run()

        out = out.numpy(self.target)
        target_result = self.get_paddle_inference_result(x_data)

        print("result in test_model: \n")
        out = out.reshape(-1)
        target_result = target_result.reshape(-1)
        # out.shape[0]
        for i in range(0, 20):
            if np.abs(out[i] - target_result[i]) > 1e-3:
                print("Error! ", i, "-th data has diff with target data:\n",
                      out[i], " vs: ", target_result[i], ". Diff is: ",
                      out[i] - target_result[i])
        self.assertTrue(np.allclose(out, target_result, atol=1e-3))

    def test_model(self):
        self.apply_test()
        #self.target.arch = Target.Arch.NVGPU
        #self.apply_test()


if __name__ == "__main__":
    unittest.main()
