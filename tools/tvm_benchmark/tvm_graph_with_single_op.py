import os

import numpy as np

import tvm
from tvm import te
from tvm import autotvm
from tvm import relay
import tvm.relay.testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.contrib.utils import tempdir
import tvm.contrib.graph_runtime as runtime


# To test different ops, change this single-op network.
# See ./relay_op.rst to get the op list.
def get_network():
    input_shape = [(1, 512, 7, 7), (512, 512, 3, 3)]
    output_shape = (1, 512, 7, 7)
    input_names = ["x", "y"]
    x = relay.Var(input_names[0], tvm.relay.TensorType(input_shape[0]))
    y = relay.Var(input_names[1], tvm.relay.TensorType(input_shape[1]))
    mod = relay.Function([x, y],
                         relay.nn.conv2d(
                             x,
                             y,
                             kernel_size=(3, 3),
                             padding=(1, 1),
                             strides=(1, 1)))
    params = []
    return mod, params, input_shape, output_shape, input_names


#### DEVICE CONFIG ####
target = tvm.target.cuda()
dtype = "float32"


def tune_and_evaluate():
    # extract workloads from relay program
    print("Extract tasks...")
    mod, params, input_shape, out_shape, input_names = get_network()

    print("Compile...")
    lib = relay.build_module.build(mod, target=target, params=params)

    # load parameters
    ctx = tvm.context(str(target), 0)
    module = runtime.GraphModule(lib["default"](ctx))
    for index in range(len(input_shape)):
        data_temp = tvm.nd.array(
            (np.random.uniform(size=input_shape[index])).astype(dtype))
        module.set_input(input_names[index], data_temp)
    # evaluate

    evaluator_preheat = module.module.time_evaluator(
        "run", ctx, number=50, repeat=50)
    evaluator = module.module.time_evaluator(
        "run", ctx, number=500, repeat=100)

    prof_res1 = np.array(
        evaluator_preheat().results) * 1000  # convert to millisecond
    print("[PreHeat]Mean inference time (std dev): %.4f ms (%.4f ms)" %
          (np.mean(prof_res1), np.std(prof_res1)))

    prof_res2 = np.array(evaluator().results) * 1000  # convert to millisecond
    print("[Benchmark]Mean inference time (std dev): %.4f ms (%.4f ms)" %
          (np.mean(prof_res2), np.std(prof_res2)))


tune_and_evaluate()
