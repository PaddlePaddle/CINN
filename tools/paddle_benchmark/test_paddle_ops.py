import sys, os
import numpy as np
import paddle
import paddle.fluid as fluid
import time
import argparse
from paddle.fluid.core import AnalysisConfig
from paddle.fluid.core import create_paddle_predictor


def set_config(op_name, input_shapes, enable_gpu=False):
    model_dir = "./" + op_name + "_model"
    for input_shape in input_shapes[0]:
        model_dir += "_" + str(input_shape)
    config = AnalysisConfig(model_dir)
    config.enable_profile()
    if enable_gpu:
        config.enable_use_gpu(1000, 1)
        config.gpu_device_id()
    else:
        config.disable_gpu()
        config.enable_mkldnn()
    config.switch_use_feed_fetch_ops(False)
    config.switch_specify_input_names(True)
    config.switch_ir_optim(True)

    return config


def create_model(input_names, input_shapes, input_dtypes, fn, attrs=None):
    # For paddlepaddle version >=2.0rc, we need to set paddle.enable_static()
    paddle.enable_static()
    input_args = []
    input_args_names = []
    assert len(input_names) == len(input_shapes) == len(input_dtypes)
    fn_str = "fluid.layers." + fn + "("
    dim = len(input_shapes)
    for i in range(dim - 1):
        input_args.append(
            fluid.data(
                name=input_names[i],
                shape=input_shapes[i],
                dtype=input_dtypes[i]))
        fn_str += "input_args[" + str(i) + "],"
        input_args_names.append(input_args[i].name)
    input_args.append(
        fluid.data(
            name=input_names[dim - 1],
            shape=input_shapes[dim - 1],
            dtype=input_dtypes[dim - 1]))
    input_args_names.append(input_args[dim - 1].name)
    fn_str += "input_args[" + str(dim - 1) + "]"
    if attrs is not None:
        fn_str += "," + attrs
    fn_str += ")"

    print("execute: ", fn_str)

    res = eval(fn_str)
    cpu = fluid.core.CPUPlace()
    loss = exe = fluid.Executor(cpu)
    exe.run(fluid.default_startup_program())

    model_name = "./" + fn + "_model"

    for i in range(len(input_shapes[0])):
        model_name += "_" + str(input_shapes[0][i])
    print("save model:", model_name)

    fluid.io.save_inference_model(model_name, input_args_names, [res], exe)
    print('output name is: ', res.name)


def test_benchmark(input_names, input_shapes, input_dtypes, fn, attrs=None):
    create_model(input_names, input_shapes, input_dtypes, fn, attrs)

    config = set_config(fn, input_shapes)
    predictor = create_paddle_predictor(config)

    input_names = predictor.get_input_names()
    input_tensor = predictor.get_input_tensor(input_names[0])
    fake_input = np.random.random(input_shapes[0]).astype("float32")
    print("input_shape_A", input_shapes[0])
    input_tensor.reshape(input_shapes[0])
    input_tensor.copy_from_cpu(fake_input)

    if len(input_shapes) >= 2:
        input_tensor2 = predictor.get_input_tensor(input_names[1])
        fake_input2 = np.random.random(input_shapes[1]).astype("float32")
        print("input_shape_B", input_shapes[1])
        input_tensor2.reshape(input_shapes[1])
        input_tensor2.copy_from_cpu(fake_input2)

    for _ in range(0, 10):
        predictor.zero_copy_run()
    repeat = 90
    start = time.time()
    for i in range(0, repeat):
        predictor.zero_copy_run()
    end = time.time()
    print("average execution time: ", (end - start) / repeat * 1000)
    output_names = predictor.get_output_names()
    output_tensor = predictor.get_output_tensor(output_names[0])
    output_data = output_tensor.copy_to_cpu()


def test_mul():
    input_shapes = [[1024, 1024], [1024, 1024]]
    input_names = ["mul_A", "mul_B"]
    input_dtypes = ["float32", "float32"]
    op_name = "mul"
    test_benchmark(input_names, input_shapes, input_dtypes, op_name)


def test_unary():
    input_shapes = [[1024, 2048]]
    input_names = ["A"]
    input_dtypes = ["float32"]
    for fn in [
            "exp",
            "erf",
            "sigmoid",
            "sqrt",
            "log",
            #         "log2",
            #         "log10",
            "floor",
            "ceil",
            "round",
            #         "trunc",
            "cos",
            "cosh",
            #         "tan",
            "tanh",
            "sin",
            "sinh",
            "acos",
            #         "acosh",
            "asin",
            #         "asinh",
            "atan",
            #         "atanh",
            "softmax",
            "sigmoid",
            "scale",
    ]:
        test_benchmark(input_names, input_shapes, input_dtypes, fn)


def test_binary():
    # input_shapes = [[100,32], [100,32]]
    input_shapes = [[1024, 2048], [1024, 2048]]
    input_names = ["A", "B"]
    input_dtypes = ["float32", "float32"]
    for fn in [
            "elementwise_add",
            "elementwise_mul",
    ]:
        test_benchmark(input_names, input_shapes, input_dtypes, fn)


def test_relu():
    input_shapes = [[1024, 2048]]
    input_names = ["A"]
    input_dtypes = ["float32"]
    for fn in [
            "relu",
            "relu6",
    ]:
        test_benchmark(input_names, input_shapes, input_dtypes, fn)


def test_conv2d():
    input_shapes = [[2, 512, 7, 7]]
    input_names = ["data"]
    input_dtypes = ["float32"]
    for fn in [
            "conv2d",
    ]:
        test_benchmark(input_names, input_shapes, input_dtypes, fn,
                       "num_filters=512, filter_size=3")


def test_conv2d_resnet():
    input_shapes = [[1, 3, 224, 224]]
    input_names = ["conv2d_resnet_data"]
    input_dtypes = ["float32"]
    for fn in [
            "conv2d",
    ]:
        test_benchmark(
            input_names, input_shapes, input_dtypes, fn,
            "num_filters=64, filter_size=7, stride=[2,2], padding=[3,3], groups=1, dilation=[1,1]"
        )


def test_depthwise_conv2d():
    input_shapes = [[2, 32, 112, 112]]
    input_names = ["depthwise_conv2d_data"]
    input_dtypes = ["float32"]
    for fn in [
            "conv2d",
    ]:
        test_benchmark(input_names, input_shapes, input_dtypes, fn,
                       "num_filters=32, filter_size=3,groups=1")


def test_pool2d():
    input_shapes = [[2, 32, 112, 112]]
    input_names = ["pool2d_data"]
    input_dtypes = ["float32"]
    for fn in [
            "conv2d",
    ]:
        test_benchmark(input_names, input_shapes, input_dtypes, fn,
                       "num_filters=32, filter_size=3,groups=1")


def test_pool2d():
    input_shapes = [[2, 64, 112, 112]]
    input_names = ["pool2d_data1"]
    input_dtypes = ["float32"]
    for fn in [
            "pool2d",
    ]:
        test_benchmark(
            input_names, input_shapes, input_dtypes, fn,
            "pool_size=[3,3],pool_type='max',pool_stride=[2,2],pool_padding=[1,1],global_pooling=False,ceil_mode=False,exclusive=True"
        )


def test_batchnorm():
    input_shapes = [[2, 32, 112, 112]]
    input_names = ["batchnorm_data"]
    input_dtypes = ["float32"]
    for fn in [
            "batch_norm",
    ]:
        test_benchmark(input_names, input_shapes, input_dtypes, fn)


def test_slice():
    input_shapes = [[2, 32, 113, 113]]
    input_names = ["slice_data"]
    input_dtypes = ["float32"]
    for fn in [
            "slice",
    ]:
        test_benchmark(input_names, input_shapes, input_dtypes, fn,
                       "axes=[2,3],starts=[1,1],ends=[10000000, 10000000]")


def test_dropout():
    input_shapes = [[1024, 2048]]
    input_names = ["dropout_data"]
    input_dtypes = ["float32"]
    for fn in [
            "dropout",
    ]:
        test_benchmark(
            input_names, input_shapes, input_dtypes, fn,
            "dropout_prob=0, dropout_implementation=\"upscale_in_train\"")


if __name__ == "__main__":
    test_unary()
    test_binary()
    test_mul()
    test_relu()
    test_conv2d()
    test_depthwise_conv2d()
    test_pool2d()
    test_batchnorm()
    test_slice()
    test_dropout()
    test_conv2d_resnet()
