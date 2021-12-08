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
"""
Run model by using CinnBuilder API
=========================================================

In this tutorial, we will introduce the ways to build and run model by using CinnBuilder API.
"""

import cinn
from cinn import frontend
from cinn import common
import numpy as np

##################################################################
#
# Define the CinnBuilder.
# -----------------------------
#
# Using CinnBuilder is a convenient way to build a model in CINN. You can build and run
# a model by invoking CinnBuilder's API as following.
#
# :code:`name`: the ID of CinnBuilder
builder = frontend.CinnBuilder(name="add_conv")

##################################################################
#
# Define the input variable of the model.
# ---------------------------------------------
#
# The input variable should be created by create_input API. Note that the variable
# here is just a placeholder, does not need the actual data.
#
# :code:`type`: the data type of input variable, now support `Void`, `Int`, `UInt`,
# `Float`, `Bool` and `String`, the parameter is the type's bit-widths, here the
# data type is `float32`.
#
# :code:`shape`: The shape of the input variable, note that here does not support
# dynamic shape, so the dimension value should be greater than 0 now.
#
# :code:`id_hint`: the name of variable, the defaule value is `""`
a = builder.create_input(
    type=common.Float(32), shape=(1, 24, 56, 56), id_hint="A")
b = builder.create_input(
    type=common.Float(32), shape=(1, 24, 56, 56), id_hint="B")
c = builder.create_input(
    type=common.Float(32), shape=(144, 24, 1, 1), id_hint="C")

##################################################################
#
# Build the model by using CinnBuilder API
# ---------------------------------------------
#
# For convenience, here we build a simple model that only consists of add and conv
# operators. Note that you can find the operator's detailed introduction in another
# document, we won't go into detail here.
#
# Generally, the API in CinnBuilder is fine-grained operator, in other words,
# the algebra or basic DL operator.
d = builder.add(a, b)
e = builder.conv(d, c)

##################################################################
#
# Generate the program
# ---------------------
#
# After the model building, the CinnBuilder will generate a CINN execution program,
# and you can get it by invoking `builder.build()` function.
prog = builder.build()

# You can print the returned program as following code:
for i in range(prog.size()):
    print(prog[i])

##################################################################
#
# Random fake input data
# -----------------------------
#
# Before running, you should read or generate some data to feed the model's input.
# In model building, we just create some placeholder, to get the model's running
# result, here we random some fake input data.
tensor_data = [
    np.random.random([1, 24, 56, 56]).astype("float32"),
    np.random.random([1, 24, 56, 56]).astype("float32"),
    np.random.random([144, 24, 1, 1]).astype("float32")
]

##################################################################
#
# Set target
# ---------------------
#
# The target identified where the model should run, now we support
# two targets:
#
# :code:`DefaultHostTarget`: the model will running at cpu.
#
# :code:`DefaultNVGPUTarget`: the model will running at nv gpu.
if common.is_compiled_with_cuda():
    target = common.DefaultNVGPUTarget()
else:
    target = common.DefaultHostTarget()

print("Model running at ", target.arch)

##################################################################
#
# Run program and print result
# -----------------------------
#
# Finally, you can running model by invoking function `build_and_get_output`.
# The `build_and_get_output` accepts the input data and finally return the results
# of model, it has four parameters:
#
# :code:`target`: the model's irunning target.
#
# :code:`tensor_inputs`: the model's input variable list.
#
# :code:`input_data`: the actual data list, the order of the list must be the same as
# `tensor_inputs`, otherwise, the resulting error.
#
# :code:`tensor_outputs`: the model's output variable list, the ordering of the model's
# result list is the same as `tensor_outputs`, here we just has one result `[e]`.
result = prog.build_and_get_output(target, [a, b, c], tensor_data, [e])

# print result
print(result[0].numpy(target))
