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

model_dir = sys.argv.pop()


class TestLoadResnetModel(unittest.TestCase):
    def setUp(self):
        self.target = Target()
        self.target.arch = Target.Arch.X86
        self.target.bits = Target.Bit.k64
        self.target.os = Target.OS.Linux
        self.model_dir = model_dir
        self.x_shape = [2, 3, 224, 224]
        self.target_tensor = 'save_infer_model/scale_0'
        self.input_tensor = 'image'

    def get_paddle_inference_result(self, model_dir, data):
        config = fluid.core.AnalysisConfig(model_dir + '/__model__',
                                           model_dir + '/params')
        config.disable_gpu()
        config.switch_ir_optim(False)
        self.paddle_predictor = fluid.core.create_paddle_predictor(config)
        data = fluid.core.PaddleTensor(data)
        results = self.paddle_predictor.run([data])
        get_tensor = self.paddle_predictor.get_output_tensor(
            self.target_tensor).copy_to_cpu()
        #return results[0].as_ndarray()
        return get_tensor

    def test_model(self):
        x_data = np.random.random(self.x_shape).astype("float32")
        self.executor = Interpreter([self.input_tensor], [self.x_shape])
        print("self.mode_dir is:", self.model_dir)
        # True means load combined model
        self.executor.load_paddle_model(self.model_dir, self.target, True)
        a_t = self.executor.get_tensor(self.input_tensor)
        a_t.from_numpy(x_data, self.target)

        out = self.executor.get_tensor(self.target_tensor)
        out.from_numpy(np.zeros(out.shape(), dtype='float32'), self.target)

        self.executor.run()

        out = out.numpy(self.target)
        target_result = self.get_paddle_inference_result(
            self.model_dir, x_data)

        print("result in test_model: \n")
        out = out.reshape(-1)
        target_result = target_result.reshape(-1)
        for i in range(0, out.shape[0]):
            if np.abs(out[i] - target_result[i]) > 1e-3:
                print("Error! ", i, "-th data has diff with target data:\n",
                      out[i], " vs: ", target_result[i], ". Diff is: ",
                      out[i] - target_result[i])
        self.assertTrue(np.allclose(out, target_result, atol=1e-3))


if __name__ == "__main__":
    unittest.main()
