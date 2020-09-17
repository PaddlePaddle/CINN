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

assert len(sys.argv) == 1 + 2  # model count
multi_fc_model_dir = sys.argv.pop()
naive_model_dir = sys.argv.pop()


class TestFrontend(unittest.TestCase):
    def setUp(self):
        self.target = Target()
        self.target.arch = Target.Arch.X86
        self.target.bits = Target.Bit.k64
        self.target.os = Target.OS.Linux

    def paddle_verify(self, result):
        a = fluid.layers.data(name='A', shape=[24, 56, 56], dtype='float32')
        b = fluid.layers.data(name='B', shape=[24, 56, 56], dtype='float32')
        c = fluid.layers.elementwise_add(a, b)
        d = fluid.layers.relu(c)
        e = fluid.initializer.NumpyArrayInitializer(
            np.array(result[2]).reshape((144, 24, 1, 1)).astype("float32"))
        f = fluid.layers.conv2d(
            input=d,
            num_filters=144,
            filter_size=1,
            stride=1,
            padding=0,
            dilation=1,
            param_attr=e)
        g = fluid.layers.scale(f, scale=2.0, bias=0.5)
        res = fluid.layers.softmax(g, axis=1)

        exe = fluid.Executor(fluid.CPUPlace())
        exe.run(fluid.default_startup_program())

        x = np.array(result[0]).reshape((1, 24, 56, 56)).astype("float32")
        y = np.array(result[1]).reshape((1, 24, 56, 56)).astype("float32")
        output = exe.run(feed={"A": x, "B": y}, fetch_list=[res])
        output = np.array(output).reshape(-1)
        print("result in paddle_verify: \n")
        for i in range(0, 10):
            print(result[len(result) - 1][i], " vs: ", output[i])
        self.assertTrue(
            np.allclose(result[len(result) - 1], output, atol=1e-4))

    def test_basic(self):
        prog = Program()

        a = Variable("A").set_type(Float(32)).set_shape([1, 24, 56, 56])
        b = Variable("B").set_type(Float(32)).set_shape([1, 24, 56, 56])
        c = prog.add(a, b)
        d = prog.relu(c)
        e = Variable("E").set_type(Float(32)).set_shape([144, 24, 1, 1])
        f = prog.conv2d(d, e, {
            "stride": [1, 1],
            "dilation": 1,
            "padding": [0, 0]
        })
        print('f', f)
        g = prog.scale(f, {"scale": 2.0, "bias": 0.5})
        h = prog.softmax(g, {"axis": 1})

        self.assertEqual(prog.size(), 5)
        # print program
        for i in range(prog.size()):
            print(prog[i])
        result = prog.build_with_inputs(self.target, [a, b, e], h)

        self.paddle_verify(result)


class TestLoadPaddleModel_FC(unittest.TestCase):
    def setUp(self):
        self.target = Target()
        self.target.arch = Target.Arch.X86
        self.target.bits = Target.Bit.k64
        self.target.os = Target.OS.Linux

        self.model_dir = naive_model_dir

    def get_paddle_inference_result(self, model_dir, data):
        config = fluid.core.AnalysisConfig(model_dir)
        config.disable_gpu()
        config.switch_ir_optim(False)
        self.paddle_predictor = fluid.core.create_paddle_predictor(config)
        data = fluid.core.PaddleTensor(data)
        results = self.paddle_predictor.run([data])
        fc0_out = self.paddle_predictor.get_output_tensor(
            'fc_0.tmp_0').copy_to_cpu()

        return results[0].as_ndarray()

    def test_model(self):
        np.random.seed(0)
        self.x_shape = [4, 30]
        x_data = np.random.random(
            self.x_shape).astype("float16").astype("float32")
        print('x_data', x_data)

        self.executor = Executor(["A"], [self.x_shape])
        self.executor.load_paddle_model(self.model_dir, False)
        a_t = self.executor.get_tensor("A")
        a_t.from_numpy(x_data)

        self.executor.run()

        out = self.executor.get_tensor("var_11")
        target = self.get_paddle_inference_result(self.model_dir, x_data)

        self.assertTrue(np.allclose(out.numpy(), target, atol=1e-4))


class TestLoadPaddleModel_MultiFC(unittest.TestCase):
    def setUp(self):
        self.target = Target()
        self.target.arch = Target.Arch.X86
        self.target.bits = Target.Bit.k64
        self.target.os = Target.OS.Linux

        self.model_dir = multi_fc_model_dir

    def get_paddle_inference_result(self, model_dir, data):
        config = fluid.core.AnalysisConfig(model_dir)
        config.disable_gpu()
        config.switch_ir_optim(False)
        self.paddle_predictor = fluid.core.create_paddle_predictor(config)
        data = fluid.core.PaddleTensor(data)
        results = self.paddle_predictor.run([data])

        return results[0].as_ndarray()

    def test_model(self):
        np.random.seed(0)
        self.x_shape = [8, 64]
        x_data = np.random.random(self.x_shape).astype("float32")

        self.executor = Executor(["A"], [self.x_shape])
        self.executor.load_paddle_model(self.model_dir, False)
        a_t = self.executor.get_tensor("A")
        a_t.from_numpy(x_data)

        self.executor.run()

        out = self.executor.get_tensor("fc_5.tmp_2")
        target = self.get_paddle_inference_result(self.model_dir, x_data)

        self.assertTrue(np.allclose(out.numpy(), target, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
