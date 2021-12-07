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

# sphinx_gallery_thumbnail_path = '_static/icon.png'

import cinn
from cinn import frontend
from cinn import common
import numpy as np

##################################################################
# Define the CinnBuilder
builder = frontend.CinnBuilder("add_conv2d")

##################################################################
# Define the input variable of model
a = builder.create_input(
    type=common.Float(32), shape=(1, 24, 56, 56), id_hint="A")
b = builder.create_input(
    type=common.Float(32), shape=(1, 24, 56, 56), id_hint="B")
c = builder.create_input(
    type=common.Float(32), shape=(144, 24, 1, 1), id_hint="C")

##################################################################
# Build the model computation by using CinnBuilder API
d = builder.add(a, b)
e = builder.conv(d, c)

##################################################################
# Generate the program
prog = builder.build()

# print program
for i in range(prog.size()):
    print(prog[i])

##################################################################
# Fake input data
tensor_data = [
    np.random.random([1, 24, 56, 56]).astype("float32"),
    np.random.random([1, 24, 56, 56]).astype("float32"),
    np.random.random([144, 24, 1, 1]).astype("float32")
]

##################################################################
# Run program and print result
target = common.DefaultHostTarget()
result = prog.build_and_get_output(target, [a, b, c], tensor_data, [e])

# print result
print(result[0].numpy(target))
