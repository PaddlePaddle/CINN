#!/usr/bin/env python3
import paddle
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

assert len(sys.argv) == 1 + 2 + 1  # model and enable_gpu count
enable_gpu = sys.argv.pop()
multi_fc_model_dir = sys.argv.pop()
naive_model_dir = sys.argv.pop()


class TestFrontend(unittest.TestCase):
    def setUp(self):
        if enable_gpu == "ON":
            self.target = DefaultNVGPUTarget()
        else:
            self.target = DefaultHostTarget()

    def paddle_verify(self, result):
        paddle.enable_static()

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
        for i in range(0, output.shape[0]):
            if np.abs(output[i] - result[len(result) - 1][i]) > 1e-4:
                print("Error! ", i, "-th data has diff with target data:\n",
                      output[i], " vs: ", result[len(result) - 1][i],
                      ". Diff is: ", output[i] - result[len(result) - 1][i])
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
            "dilation": [1, 1],
            "padding": [0, 0]
        })
        g = prog.scale(f, {"scale": 2.0, "bias": 0.5})
        h = prog.softmax(g, {"axis": 1})

        self.assertEqual(prog.size(), 5)
        # print program
        for i in range(prog.size()):
            print(prog[i])
        tensor_data = [
            np.random.random([1, 24, 56, 56]).astype("float32"),
            np.random.random([1, 24, 56, 56]).astype("float32"),
            np.random.random([144, 24, 1, 1]).astype("float32")
        ]
        result = prog.build_and_get_output(self.target, [a, b, e], tensor_data,
                                           h)
        result = result.numpy(self.target).reshape(-1)
        tensor_data.append(result)
        self.paddle_verify(tensor_data)


class TestLoadPaddleModel_FC(unittest.TestCase):
    def setUp(self):
        if enable_gpu == "ON":
            self.target = DefaultNVGPUTarget()
        else:
            self.target = DefaultHostTarget()

        self.model_dir = naive_model_dir

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
        self.x_shape = [4, 30]
        x_data = np.random.random(
            self.x_shape).astype("float16").astype("float32")
        print('x_data', x_data)

        self.executor = Interpreter(["A"], [self.x_shape])
        self.executor.load_paddle_model(self.model_dir, self.target, False)
        a_t = self.executor.get_tensor("A")
        a_t.from_numpy(x_data, self.target)

        self.executor.run()

        out = self.executor.get_tensor("save_infer_model/scale_0.tmp_0")
        target_data = self.get_paddle_inference_result(self.model_dir, x_data)
        print("target_data's shape is: ", target_data.shape)
        out_np = out.numpy(self.target)
        print("cinn data's shape is: ", out_np.shape)

        self.assertTrue(np.allclose(out_np, target_data, atol=1e-4))


class TestLoadPaddleModel_MultiFC(unittest.TestCase):
    def setUp(self):
        if enable_gpu == "ON":
            self.target = DefaultNVGPUTarget()
        else:
            self.target = DefaultHostTarget()

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

        self.executor = Interpreter(["A"], [self.x_shape])
        self.executor.load_paddle_model(self.model_dir, self.target, False)
        a_t = self.executor.get_tensor("A")
        a_t.from_numpy(x_data, self.target)

        self.executor.run()

        out = self.executor.get_tensor("save_infer_model/scale_0.tmp_0")
        target = self.get_paddle_inference_result(self.model_dir, x_data)

        self.assertTrue(np.allclose(out.numpy(self.target), target, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
